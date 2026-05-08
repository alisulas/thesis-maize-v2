"""PyTorch Dataset for maize yield prediction from MODIS time-series tensors.

Data schema per sample:
    X: (T=46, F=10) float32  — satellite features (b1-b7, ndvi, lst_day, lst_night)
    y: (1,)         float32  — yield in ton/ha

Normalization is z-score per feature across all samples.
Stats are computed from training split and applied consistently to val/test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODIS_DIR = PROJECT_ROOT / "data" / "processed" / "modis"

FEATURE_NAMES = [
    "b01", "b02", "b03", "b04", "b05", "b06", "b07",
    "ndvi", "lst_day", "lst_night",
]


class MaizeDataset(Dataset):
    """MODIS-based maize yield dataset with optional normalization.

    Args:
        country: One of 'usa', 'idn', 'vnm', 'tha'.
        split: 'train', 'val', or 'test'.
        year_splits: Dict with keys 'train', 'val', 'test' → list of years.
            Example: {'train': range(2003, 2021), 'val': [2021, 2022], 'test': [2023]}
        norm_stats: Optional (mean, std) arrays of shape (F,). If None, computed
            from this split's data (only use for training split).
    """

    DEFAULT_YEAR_SPLITS: dict[str, dict] = {
        "usa": {
            "train": list(range(2003, 2021)),
            "val":   [2021, 2022],
            "test":  [2023],
        },
        "idn": {
            "train": [2020, 2021, 2022, 2023],
            "val":   [],
            "test":  [2024],
        },
        "vnm": {
            "train": list(range(2003, 2022)),
            "val":   [2022],
            "test":  [2023],
        },
        "tha": {
            "train": [2021, 2022],
            "val":   [],
            "test":  [2023],
        },
    }

    def __init__(
        self,
        country: str,
        split: str,
        year_splits: Optional[dict] = None,
        norm_stats: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        assert country in ("usa", "idn", "vnm", "tha"), f"Unknown country: {country}"
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.country = country
        self.split = split

        data = np.load(MODIS_DIR / f"{country}_modis.npz", allow_pickle=True)
        X_all: np.ndarray = data["X"]           # (N, 46, 10)
        y_all: np.ndarray = data["y"]           # (N,)
        years_all: np.ndarray = data["years"]   # (N,)

        splits = year_splits or self.DEFAULT_YEAR_SPLITS[country]
        year_list = splits[split]

        if len(year_list) == 0:
            self.X = np.zeros((0, X_all.shape[1], X_all.shape[2]), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.float32)
            self.norm_stats = norm_stats or (np.zeros(X_all.shape[2]), np.ones(X_all.shape[2]))
            return

        mask = np.isin(years_all, year_list)
        self.X = X_all[mask]  # (N_split, 46, 10)
        self.y = y_all[mask]  # (N_split,)

        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            mean = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0)
            std  = self.X.reshape(-1, self.X.shape[-1]).std(axis=0)
            std  = np.where(std < 1e-8, 1.0, std)
            self.norm_stats = (mean, std)

        mean, std = self.norm_stats
        self.X = (self.X - mean[None, None, :]) / std[None, None, :]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])   # (46, 10)
        y = torch.tensor(self.y[idx])
        return x, y

    @property
    def n_features(self) -> int:
        return self.X.shape[-1]

    @property
    def n_timesteps(self) -> int:
        return self.X.shape[1]


def get_dataloaders(
    country: str,
    batch_size: int = 256,
    num_workers: int = 0,
    year_splits: Optional[dict] = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Convenience wrapper: returns (train_loader, val_loader, test_loader).

    Normalization stats are computed from train split and shared with val/test.
    """
    train_ds = MaizeDataset(country, "train", year_splits=year_splits)
    norm_stats = train_ds.norm_stats

    val_ds  = MaizeDataset(country, "val",  year_splits=year_splits, norm_stats=norm_stats)
    test_ds = MaizeDataset(country, "test", year_splits=year_splits, norm_stats=norm_stats)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
