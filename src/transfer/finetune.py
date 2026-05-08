"""Fine-tune a USA-pretrained model on ASEAN target countries.

Strategy:
    1. Load USA-pretrained checkpoint.
    2. Fine-tune with frozen feature extractor (CNN + LSTM) for N_FROZEN epochs.
    3. Unfreeze all layers, continue training at lower LR for N_FULL epochs.
    4. Compare test R² against from-scratch baseline.

Usage:
    # Step 1: train USA baseline (if not done)
    python src/training/train.py --config experiments/configs/usa_baseline.yaml

    # Step 2: fine-tune on each ASEAN country
    python src/transfer/finetune.py --country idn
    python src/transfer/finetune.py --country vnm
    python src/transfer/finetune.py --country tha
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.cnn_lstm import CropYieldCNNLSTM, CropYieldLSTM
from src.training.train import evaluate, r2_score, rmse, set_seed, train_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

USA_CHECKPOINT = PROJECT_ROOT / "experiments" / "checkpoints" / "usa_lstm" / "best_model.pt"

FINETUNE_CFG = {
    "n_frozen_epochs": 20,     # train head only
    "n_full_epochs":   50,     # unfreeze all, lower LR
    "lr_frozen":       1e-3,
    "lr_full":         1e-4,
    "weight_decay":    1e-4,
    "batch_size":      64,
    "patience":        20,
    "seed":            42,
}

MODEL_CFG = {
    "n_features":   10,
    "cnn_channels": 64,
    "hidden_size":  256,   # must match usa_lstm.yaml
    "n_layers":     2,
    "dropout":      0.3,
}


def finetune(country: str, pretrained_path: Path) -> dict:
    """Fine-tune USA pretrained model on target country.

    Returns dict with test metrics for both transfer and scratch baselines.
    """
    set_seed(FINETUNE_CFG["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning on {country.upper()} | device: {device}")
    logger.info(f"Pretrained checkpoint: {pretrained_path}")

    # --- Data -----------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        country=country, batch_size=FINETUNE_CFG["batch_size"]
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    logger.info(f"  Samples — train: {n_train}, val: {n_val}, test: {n_test}")
    has_val = n_val > 0

    # --- Load pretrained weights ---------------------------------------------
    ckpt = torch.load(pretrained_path, map_location=device)
    # Auto-detect model type from checkpoint path name
    if "lstm" in str(pretrained_path) and "cnn" not in str(pretrained_path):
        model = CropYieldLSTM(**{k: v for k, v in MODEL_CFG.items()
                                  if k in ("n_features", "hidden_size", "n_layers", "dropout")})
    else:
        model = CropYieldCNNLSTM(**MODEL_CFG)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"  Loaded pretrained weights (USA epoch {ckpt['epoch']})")

    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints" / f"finetune_{country}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    log_rows: list[dict] = []
    best_loss = float("inf")

    # --- Phase 1: frozen feature extractor -----------------------------------
    logger.info(f"\n  Phase 1 — frozen extractor ({FINETUNE_CFG['n_frozen_epochs']} epochs)")
    model.freeze_feature_extractor()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_CFG["lr_frozen"],
        weight_decay=FINETUNE_CFG["weight_decay"],
    )
    patience_ctr = 0

    for epoch in range(1, FINETUNE_CFG["n_frozen_epochs"] + 1):
        t0 = time.time()
        train_loss  = train_epoch(model, train_loader, optimizer, None, device)
        val_metrics = evaluate(model, val_loader, device) if has_val else {"loss": train_loss, "r2": float("nan"), "rmse": float("nan")}

        log_rows.append({"phase": "frozen", "epoch": epoch, "train_loss": train_loss,
                         **{f"val_{k}": v for k, v in val_metrics.items()}})
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"    Ep {epoch:3d} | train={train_loss:.4f} val_r2={val_metrics['r2']:.4f} | {time.time()-t0:.1f}s")

        if val_metrics["loss"] < best_loss:
            best_loss    = val_metrics["loss"]
            patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= FINETUNE_CFG["patience"]:
                logger.info(f"    Early stop at epoch {epoch}")
                break

    # --- Phase 2: full fine-tuning -------------------------------------------
    logger.info(f"\n  Phase 2 — full fine-tune ({FINETUNE_CFG['n_full_epochs']} epochs)")
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_CFG["lr_full"],
        weight_decay=FINETUNE_CFG["weight_decay"],
    )
    patience_ctr = 0

    for epoch in range(1, FINETUNE_CFG["n_full_epochs"] + 1):
        t0 = time.time()
        train_loss  = train_epoch(model, train_loader, optimizer, None, device)
        val_metrics = evaluate(model, val_loader, device) if has_val else {"loss": train_loss, "r2": float("nan"), "rmse": float("nan")}

        log_rows.append({"phase": "full", "epoch": epoch, "train_loss": train_loss,
                         **{f"val_{k}": v for k, v in val_metrics.items()}})
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"    Ep {epoch:3d} | train={train_loss:.4f} val_r2={val_metrics['r2']:.4f} | {time.time()-t0:.1f}s")

        if val_metrics["loss"] < best_loss:
            best_loss    = val_metrics["loss"]
            patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= FINETUNE_CFG["patience"]:
                logger.info(f"    Early stop at epoch {epoch}")
                break

    # --- Test -----------------------------------------------------------------
    ckpt_best = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt_best["model_state"])
    transfer_metrics = evaluate(model, test_loader, device)
    logger.info(f"\n  Transfer test → R²={transfer_metrics['r2']:.4f}  RMSE={transfer_metrics['rmse']:.4f} t/ha")

    # --- From-scratch baseline ------------------------------------------------
    logger.info("\n  Training from-scratch baseline...")
    set_seed(FINETUNE_CFG["seed"])
    if "lstm" in str(pretrained_path) and "cnn" not in str(pretrained_path):
        scratch_model = CropYieldLSTM(**{k: v for k, v in MODEL_CFG.items()
                                          if k in ("n_features", "hidden_size", "n_layers", "dropout")})
    else:
        scratch_model = CropYieldCNNLSTM(**MODEL_CFG)
    scratch_model = scratch_model.to(device)
    scratch_opt   = torch.optim.AdamW(scratch_model.parameters(),
                                      lr=FINETUNE_CFG["lr_frozen"],
                                      weight_decay=FINETUNE_CFG["weight_decay"])
    scratch_best_loss = float("inf")
    scratch_best_path = ckpt_dir / "scratch_best.pt"
    patience_ctr = 0

    for epoch in range(1, FINETUNE_CFG["n_frozen_epochs"] + FINETUNE_CFG["n_full_epochs"] + 1):
        train_loss  = train_epoch(scratch_model, train_loader, scratch_opt, None, device)
        val_metrics = evaluate(scratch_model, val_loader, device) if has_val else {"loss": train_loss, "r2": float("nan"), "rmse": float("nan")}

        if val_metrics["loss"] < scratch_best_loss:
            scratch_best_loss = val_metrics["loss"]
            patience_ctr      = 0
            torch.save({"model_state": scratch_model.state_dict()}, scratch_best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= FINETUNE_CFG["patience"]:
                break

    ckpt_scratch = torch.load(scratch_best_path, map_location=device)
    scratch_model.load_state_dict(ckpt_scratch["model_state"])
    scratch_metrics = evaluate(scratch_model, test_loader, device)
    logger.info(f"  Scratch  test → R²={scratch_metrics['r2']:.4f}  RMSE={scratch_metrics['rmse']:.4f} t/ha")

    # R² improvement
    r2_delta = transfer_metrics["r2"] - scratch_metrics["r2"]
    logger.info(f"  ΔR² (transfer - scratch) = {r2_delta:+.4f}")

    # Save log
    log_dir = PROJECT_ROOT / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_dir / f"finetune_{country}_log.csv", index=False)

    results = {
        "country": country,
        "transfer_r2":   transfer_metrics["r2"],
        "transfer_rmse": transfer_metrics["rmse"],
        "scratch_r2":    scratch_metrics["r2"],
        "scratch_rmse":  scratch_metrics["rmse"],
        "r2_improvement": r2_delta,
    }

    # Append to summary table
    summary_path = PROJECT_ROOT / "experiments" / "logs" / "transfer_results.csv"
    summary_df = pd.DataFrame([results])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["country"] != country]  # replace if re-run
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Results → {summary_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, choices=["idn", "vnm", "tha", "all"])
    parser.add_argument("--pretrained", default=str(USA_CHECKPOINT))
    args = parser.parse_args()

    pretrained = Path(args.pretrained)
    if not pretrained.exists():
        logger.error(f"Pretrained checkpoint not found: {pretrained}")
        logger.error("Run: python src/training/train.py --config experiments/configs/usa_baseline.yaml")
        sys.exit(1)

    countries = ["idn", "vnm", "tha"] if args.country == "all" else [args.country]
    all_results = []
    for c in countries:
        res = finetune(c, pretrained)
        all_results.append(res)

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for r in all_results:
        logger.info(
            f"  {r['country'].upper():4s}  Transfer R²={r['transfer_r2']:.3f}  "
            f"Scratch R²={r['scratch_r2']:.3f}  ΔR²={r['r2_improvement']:+.3f}"
        )


if __name__ == "__main__":
    main()
