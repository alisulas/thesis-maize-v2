---
name: Code Architecture & File Map
description: What each source file does and how the pipeline flows end-to-end
type: project
---

## Pipeline Flow

```
GEE extraction → raw CSVs → merge_modis.py → .npz tensors → dataset.py → train.py / finetune.py
Yield download  → parquet/CSV ↗
```

## Source Files

### Data scripts (`src/data/`)
| File | Purpose |
|------|---------|
| `download_usa.py` | USDA NASS QuickStats API → yield parquet for USA counties |
| `download_faostat.py` | OWID/FAOSTAT → national reference CSV (4 countries) |
| `clean_indonesia.py` | Parse BPS wide-format CSV (multi-level headers) → standard yield schema |
| `clean_vietnam.py` | Clean GSO Vietnam province Excel → standard yield schema |
| `extract_modis_usa.py` | GEE: MOD09A1 + MYD11A2 → Drive CSV (USA, TIGER counties) |
| `extract_modis_asean.py` | GEE: MOD09A1 + MYD11A2 → Drive CSV (IDN/VNM/THA, GAUL L1) |
| `merge_modis.py` | 50 raw CSVs + yield files → 4 .npz tensors with region name mapping |
| `dataset.py` | `MaizeDataset` (PyTorch Dataset), z-score normalization, `get_dataloaders()` |

### Models (`src/models/`)
| File | Purpose |
|------|---------|
| `cnn_lstm.py` | `CropYieldCNNLSTM` (Conv1d + LSTM) and `CropYieldLSTM` (LSTM-only). Both have `freeze_feature_extractor()` / `unfreeze_all()` for fine-tuning. |

### Training (`src/training/`)
| File | Purpose |
|------|---------|
| `train.py` | Main training loop: MSELoss, AdamW, cosine LR, early stopping, saves checkpoint + CSV log. Entry: `python src/training/train.py --config experiments/configs/usa_lstm.yaml` |

### Transfer (`src/transfer/`)
| File | Purpose |
|------|---------|
| `finetune.py` | 2-phase fine-tuning (frozen then full) + from-scratch comparison on same data. Saves to `experiments/logs/transfer_results.csv`. Entry: `python src/transfer/finetune.py --country [idn|vnm|tha|all]` |

## Configs (`experiments/configs/`)
| File | Model | Key params | Best result |
|------|-------|-----------|-------------|
| `usa_baseline.yaml` | CNN-LSTM | hidden=128, lr=1e-3, batch=512 | Test R²=0.13 |
| `usa_lstm.yaml` | LSTM-only | hidden=256, lr=5e-4, batch=256 | Test R²=0.39 ← use this |

## Key Constants (hardcoded, consistent across scripts)
- `N_FEATURES = 10`: [b01,b02,b03,b04,b05,b06,b07,ndvi,LST_Day_1km,LST_Night_1km]
- `N_TIMESTEPS = 46`: 8-day MODIS composites per year
- `GEE_PROJECT = "alamat-413120"`
- `DRIVE_FOLDER = "thesis_maize_gee"`
- `MODEL_CFG` in finetune.py must match the pretrained checkpoint's architecture (currently hidden_size=256, matches usa_lstm.yaml)

## Collab Training Setup (not yet done)
```python
# In Colab:
from google.colab import drive; drive.mount('/content/drive')
!git clone <repo> /content/thesis_maize && cd /content/thesis_maize
!pip install -r requirements.txt
# Copy raw MODIS CSVs from Drive:
!cp /content/drive/MyDrive/thesis_maize_gee/modis_*.csv data/raw/modis/
# Commit yield parquet/CSVs to git first so they're in the clone
!python src/data/merge_modis.py
!python src/training/train.py --config experiments/configs/usa_lstm.yaml
!python src/transfer/finetune.py --country all
```
