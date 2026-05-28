"""Training script for USA maize yield baseline model.

Logs to CSV. Optional wandb integration.

Usage:
    # USA baseline
    python src/training/05_train_usa.py --config experiments/configs/usa_baseline.yaml

    # Indonesia from-scratch (no transfer)
    python src/training/05_train_usa.py --config experiments/configs/usa_baseline.yaml \\
        --country idn --name idn_scratch
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataloaders
from src.models import CropYieldCNNLSTM, CropYieldLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    m = cfg["model"]
    if cfg["experiment"]["model"] == "cnn_lstm":
        model = CropYieldCNNLSTM(
            n_features=m["n_features"],
            cnn_channels=m["cnn_channels"],
            hidden_size=m["hidden_size"],
            n_layers=m["n_layers"],
            dropout=m["dropout"],
        )
    else:
        model = CropYieldLSTM(
            n_features=m["n_features"],
            hidden_size=m["hidden_size"],
            n_layers=m["n_layers"],
            dropout=m["dropout"],
        )
    return model.to(device)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return ((y_true - y_pred) ** 2).mean().sqrt().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    if len(loader.dataset) == 0:
        return {"loss": float("nan"), "r2": float("nan"), "rmse": float("nan")}

    model.eval()
    criterion = nn.MSELoss()
    losses, preds, truths = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(criterion(out, y).item())
        preds.append(out)
        truths.append(y)

    preds  = torch.cat(preds)
    truths = torch.cat(truths)
    return {
        "loss": float(np.mean(losses)),
        "r2":   r2_score(truths, preds),
        "rmse": rmse(truths, preds),
    }


def train_epoch(model: nn.Module, loader, optimizer, scheduler, device: torch.device) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(len(loader), 1)


def run_training(
    config_path: Path,
    country: str | None = None,
    name: str | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[nn.Module, dict]:
    """Train a model according to a YAML config.

    Args:
        config_path: Path to YAML config file.
        country: Override config's country (for ASEAN from-scratch runs).
        name: Override experiment name.
        checkpoint_path: If provided, save best model here.

    Returns:
        (trained_model, test_metrics)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if country:
        cfg["experiment"]["country"] = country
    if name:
        cfg["experiment"]["name"] = name

    country_key = cfg["experiment"]["country"]
    exp_name    = cfg["experiment"]["name"]
    t_cfg       = cfg["training"]
    set_seed(t_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Experiment: {exp_name}  |  Country: {country_key}  |  Device: {device}")

    # Data
    year_splits = {k: list(v) for k, v in cfg["data"]["year_splits"].items()}
    train_loader, val_loader, test_loader = get_dataloaders(
        country=country_key,
        batch_size=t_cfg["batch_size"],
        year_splits=year_splits if country_key == "usa" else None,
    )
    logger.info(
        f"  Train: {len(train_loader.dataset)}  "
        f"Val: {len(val_loader.dataset)}  "
        f"Test: {len(test_loader.dataset)}"
    )

    # Model
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )
    scheduler = None
    if t_cfg.get("lr_scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_cfg["epochs"]
        )

    # Checkpointing
    if checkpoint_path is None:
        ckpt_dir = PROJECT_ROOT / cfg["output"]["checkpoint_dir"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / "best_model.pt"

    # CSV log
    log_dir = PROJECT_ROOT / cfg["output"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_name}_train_log.csv"
    log_rows: list[dict] = []

    best_val_loss = float("inf")
    patience_ctr  = 0
    epochs        = t_cfg["epochs"]
    patience      = t_cfg.get("patience", epochs)
    has_val       = len(val_loader.dataset) > 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics  = evaluate(model, val_loader,  device) if has_val else {"loss": train_loss, "r2": float("nan"), "rmse": float("nan")}

        elapsed = time.time() - t0
        log_rows.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Ep {epoch:3d}/{epochs} | train_loss={train_loss:.4f} "
                f"| val_r2={val_metrics['r2']:.4f} val_rmse={val_metrics['rmse']:.4f} "
                f"| {elapsed:.1f}s"
            )

        monitor = train_loss if np.isnan(val_metrics["loss"]) else val_metrics["loss"]
        if monitor < best_val_loss:
            best_val_loss = monitor
            patience_ctr  = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, checkpoint_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # Load best and evaluate test
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        f"\n  Best epoch: {ckpt['epoch']}  "
        f"Test R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f} t/ha"
    )

    # Save log CSV
    import pandas as pd
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    logger.info(f"  Log → {log_path}")

    return model, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--country", default=None, help="Override config country")
    parser.add_argument("--name",    default=None, help="Override experiment name")
    args = parser.parse_args()

    run_training(Path(args.config), country=args.country, name=args.name)


if __name__ == "__main__":
    main()
