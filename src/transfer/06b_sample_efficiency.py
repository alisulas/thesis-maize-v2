"""Sample efficiency experiment — how many IDN samples for effective transfer?

Varies the number of Indonesian training samples (1%, 5%, 10%, 25%, 50%, 100%)
and compares transfer learning vs from-scratch performance at each level.

Output:
    experiments/logs/sample_efficiency.csv  — metrics per sample fraction
    experiments/logs/sample_efficiency.png  — learning curve plot

Usage:
    python src/transfer/06b_sample_efficiency.py
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MaizeDataset
from src.models import CropYieldLSTM
from src.training import evaluate, set_seed, train_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

USA_CHECKPOINT = PROJECT_ROOT / "experiments" / "checkpoints" / "usa_lstm" / "best_model.pt"

MODEL_CFG = {
    "n_features":  10,
    "hidden_size": 256,
    "n_layers":    2,
    "dropout":     0.3,
}

TRAIN_CFG = {
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "batch_size":   64,
    "epochs":       100,
    "patience":     20,
    "seed":         42,
}

# Fractions of training data to test
SAMPLE_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def subsample_dataset(dataset, fraction: float, seed: int = 42):
    """Return a random subset of the dataset with given fraction of samples."""
    n_total = len(dataset)
    n_sub = max(1, int(n_total * fraction))
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=n_sub, replace=False)

    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, ds, idxs):
            self.ds = ds
            self.idxs = idxs
            self.n_features = ds.n_features
            self.n_timesteps = ds.n_timesteps

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.ds[self.idxs[i]]

    return SubsetDataset(dataset, indices)


def run_experiment(fraction: float, seed: int) -> dict:
    """Train transfer + scratch for one sample fraction. Returns metrics."""
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    train_ds_full, val_ds_full, test_ds_full = None, None, None
    from src.data import MaizeDataset
    from torch.utils.data import DataLoader

    train_full = MaizeDataset("idn", "train")
    norm_stats = train_full.norm_stats
    val_full   = MaizeDataset("idn", "val",  norm_stats=norm_stats)
    test_full  = MaizeDataset("idn", "test", norm_stats=norm_stats)

    train_sub = subsample_dataset(train_full, fraction, seed)

    loader_kwargs = dict(batch_size=TRAIN_CFG["batch_size"], num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_sub, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_full, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_full, shuffle=False, **loader_kwargs)

    n_train = len(train_sub)
    has_val = len(val_full) > 0

    # ── Transfer ──────────────────────────────────────────────────────────
    ckpt = torch.load(USA_CHECKPOINT, map_location=device)
    model_transfer = CropYieldLSTM(**MODEL_CFG).to(device)
    model_transfer.load_state_dict(ckpt["model_state"])

    model_transfer.freeze_feature_extractor()
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_transfer.parameters()),
        lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"],
    )
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        train_epoch(model_transfer, train_loader, opt, None, device)
        metrics = evaluate(model_transfer, val_loader, device) if has_val else {"loss": 0.0}
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN_CFG["patience"]:
                break

    # Phase 2: unfreeze
    model_transfer.unfreeze_all()
    opt_full = torch.optim.AdamW(
        model_transfer.parameters(),
        lr=TRAIN_CFG["lr"] * 0.1,
        weight_decay=TRAIN_CFG["weight_decay"],
    )
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        train_epoch(model_transfer, train_loader, opt_full, None, device)
        metrics = evaluate(model_transfer, val_loader, device) if has_val else {"loss": 0.0}
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN_CFG["patience"]:
                break

    transfer_metrics = evaluate(model_transfer, test_loader, device)

    # ── Scratch ───────────────────────────────────────────────────────────
    set_seed(seed)
    model_scratch = CropYieldLSTM(**MODEL_CFG).to(device)
    opt_s = torch.optim.AdamW(
        model_scratch.parameters(),
        lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"],
    )
    best_loss_s = float("inf")
    patience_ctr = 0

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        train_epoch(model_scratch, train_loader, opt_s, None, device)
        metrics = evaluate(model_scratch, val_loader, device) if has_val else {"loss": 0.0}
        if metrics["loss"] < best_loss_s:
            best_loss_s = metrics["loss"]
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN_CFG["patience"]:
                break

    scratch_metrics = evaluate(model_scratch, test_loader, device)

    result = {
        "fraction":      fraction,
        "n_train":       n_train,
        "transfer_r2":   transfer_metrics["r2"],
        "transfer_rmse": transfer_metrics["rmse"],
        "scratch_r2":    scratch_metrics["r2"],
        "scratch_rmse":  scratch_metrics["rmse"],
        "delta_r2":      transfer_metrics["r2"] - scratch_metrics["r2"],
    }
    return result


def plot_results(df: pd.DataFrame) -> None:
    """Generate sample efficiency learning curve plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["n_train"], df["transfer_r2"], "o-", label="Transfer (USA→IDN)", linewidth=2)
    ax1.plot(df["n_train"], df["scratch_r2"], "s--", label="From Scratch", linewidth=2)
    ax1.set_xlabel("Jumlah Sampel Training IDN")
    ax1.set_ylabel("R² (Test)")
    ax1.set_title("Sample Efficiency: Transfer vs Scratch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["n_train"], df["delta_r2"], "D-", color="green", linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Jumlah Sampel Training IDN")
    ax2.set_ylabel("ΔR² (Transfer - Scratch)")
    ax2.set_title("Transfer Learning Gain")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PROJECT_ROOT / "experiments" / "logs" / "sample_efficiency.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Plot saved → {out_path}")


def main() -> None:
    if not USA_CHECKPOINT.exists():
        logger.error(f"USA checkpoint not found: {USA_CHECKPOINT}")
        logger.error("Run: python src/training/05_train_usa.py --config experiments/configs/usa_baseline.yaml")
        sys.exit(1)

    results = []
    for frac in SAMPLE_FRACTIONS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Fraction: {frac:.0%}  |  Seed: {TRAIN_CFG['seed']}")
        result = run_experiment(frac, TRAIN_CFG["seed"])
        logger.info(
            f"  Transfer R²={result['transfer_r2']:.4f}  "
            f"Scratch R²={result['scratch_r2']:.4f}  "
            f"ΔR²={result['delta_r2']:+.4f}  "
            f"(n={result['n_train']})"
        )
        results.append(result)

    df = pd.DataFrame(results)

    csv_path = PROJECT_ROOT / "experiments" / "logs" / "sample_efficiency.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved → {csv_path}")

    plot_results(df)

    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for _, row in df.iterrows():
        logger.info(
            f"  {row['fraction']:5.0%} (n={row['n_train']:4d})  "
            f"Transfer R²={row['transfer_r2']:.3f}  "
            f"Scratch R²={row['scratch_r2']:.3f}  "
            f"Δ={row['delta_r2']:+.3f}"
        )


if __name__ == "__main__":
    main()
