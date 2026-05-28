---
name: Code Architecture & File Map
description: What each source file does and how the pipeline flows end-to-end
type: project
---

## Pipeline Flow

```
GEE extraction → raw CSVs → 03_merge_usa.py → usa_modis.npz → 04_dataset.py → 05_train_usa.py
Yield download  → parquet ↗
```

## Notebooks (urutan proses)

| Notebook | Fungsi | Script .py yang di-mirror |
|----------|--------|--------------------------|
| `00_usa_download_nass.ipynb` | Download + EDA yield USA dari NASS API | `src/data/00_download_yield_usa.py` |
| `01_usa_clean_yield.ipynb` | Cleaning yield USA step-by-step | (terintegrasi di atas) |
| `01b_idn_clean_yield.ipynb` | Cleaning yield IDN dari BDSP | — |
| `02_extract_gee.ipynb` | Setup + submit GEE extraction job | — |
| `03_merge_modis.ipynb` | Walkthrough merge MODIS + yield | `src/data/03_merge_usa.py` |
| `03b_explore_tensor.ipynb` | Validasi usa_modis.npz (6 kriteria) | — |
| `04_dataset.ipynb` | Walkthrough MaizeDataset (bisa skip) | `src/data/04_dataset.py` |
| `05_train_usa.ipynb` | Demo training 3 epoch | `src/training/05_train_usa.py` |
| `05b_train_usa_full.ipynb` | Training penuh 100 epoch di notebook | sama |
| `06_model.ipynb` | Walkthrough arsitektur CNN-LSTM detail | `src/models/04_cnn_lstm.py` |

## Source Files (`src/`)

### `src/data/`
| File | Purpose |
|------|---------|
| `00_download_yield_usa.py` | USDA NASS QuickStats API → `yield_usa_2003_2023.parquet` |
| `03_merge_usa.py` | `modis_usa_*.csv` + yield parquet → `usa_modis.npz` |
| `04_dataset.py` | `MaizeDataset` (z-score normalization, temporal split), `get_dataloaders()` |

### `src/models/`
| File | Purpose |
|------|---------|
| `04_cnn_lstm.py` | `CropYieldCNNLSTM` dan `CropYieldLSTM`. Keduanya punya `freeze_feature_extractor()` / `unfreeze_all()`. Diimport via `src/models/__init__.py`. |

### `src/training/`
| File | Purpose |
|------|---------|
| `05_train_usa.py` | Training loop: MSELoss, AdamW, cosine LR, early stopping, checkpoint + CSV log. Entry: `python src/training/05_train_usa.py --config experiments/configs/usa_baseline.yaml` |

### `src/__init__.py` pattern
Semua `__init__.py` pakai `importlib.import_module` karena nama file diawali angka (tidak valid sebagai Python identifier biasa).

## Configs (`experiments/configs/`)

| File | Model | Key params | Status |
|------|-------|-----------|--------|
| `usa_baseline.yaml` | CNN-LSTM | hidden=128, cnn_ch=64, lr=1e-3, batch=512, patience=15 | ✅ Dipakai — belum dijalankan penuh |

## Key Constants

- `N_FEATURES = 10`: [b01,b02,b03,b04,b05,b06,b07,ndvi,LST_Day_1km,LST_Night_1km]
- `N_TIMESTEPS = 46`: 8-day MODIS composites per year
- `GEE_PROJECT = "alamat-413120"` | `DRIVE_FOLDER = "thesis_maize_gee"`

## Training: Cara Menjalankan

```bash
# Training penuh (DIREKOMENDASIKAN — aman dari Jupyter mati)
python src/training/05_train_usa.py --config experiments/configs/usa_baseline.yaml

# Demo 3 epoch (verifikasi pipeline saja)
# Buka notebooks/05_train_usa.ipynb → Run All

# Training penuh di notebook (risiko: Jupyter mati → training stop)
# Buka notebooks/05b_train_usa_full.ipynb → Run All
```
