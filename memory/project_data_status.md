---
name: Data & Pipeline Status
description: Current state of all data, tensors, models, and experiments as of 2026-05-08
type: project
---

## Yield Label Data (all done)

| Country | File | Rows | Regions | Years | Source |
|---------|------|------|---------|-------|--------|
| USA | `data/processed/usa/yield_usa_2003_2023.parquet` | 41,349 | 2,280 counties | 2003–2025 | USDA NASS API |
| IDN | `data/processed/indonesia/yield_indonesia_province_2020_2024.csv` | 190 | 38 provinces | 2020–2024 only | BPS (manually downloaded) |
| VNM | `data/processed/vietnam/yield_vietnam_province_1995_2023.csv` | 1,804 | 63 provinces | 1995–2023 | GSO (manually downloaded) |
| THA | `data/processed/thailand/thailand_province_yield_2021_2023.csv` | 126 | 43 provinces | 2021–2023 | OAE (manually downloaded) |

**Yield schema (standard):** region_id, region_name, country, year, yield_ton_ha
**Unit:** yield_ton_ha (USA bu/acre × 0.06277, IDN ku/ha ÷ 10)

**IDN note:** Pre-2020 BPS data NOT yet downloaded (different methodology).

## MODIS Satellite Data

### v1 (no cropland mask) — superseded
- 50 raw CSVs in Drive folder `thesis_maize_gee/`
- GEE task IDs: `experiments/logs/gee_tasks_asean.csv`, `gee_tasks_usa.csv`

### v2 (MCD12Q1 cropland mask, class 12) — CURRENT
- 50 raw CSVs replaced in `data/raw/modis/modis_{country}_{year}.csv`
- GEE task IDs: `experiments/logs/gee_tasks_asean_v2.csv`, `gee_tasks_usa_v2.csv`
- Drive folder: `thesis_maize_gee_v2/`
- Submitted 2026-05-08; USA files shrunk from ~33MB → ~28MB (confirms mask applied)

## Processed Tensors (v2, cropland-masked, in `data/processed/modis/`)

| File | X shape | Samples | Yield range |
|------|---------|---------|-------------|
| `usa_modis.npz` | (32296, 46, 10) | 32,296 | 0.00–16.96 t/ha |
| `idn_modis.npz` | (162, 46, 10) | 162 | 0.00–7.68 t/ha |
| `vnm_modis.npz` | (1315, 46, 10) | 1,315 | 1.48–9.02 t/ha |
| `tha_modis.npz` | (126, 46, 10) | 126 | 2.00–5.57 t/ha |

**Tensor format:** X=(N, T=46, F=10) float32, y=(N,) float32
**Features (F=10):** b01–b07 (reflectance), ndvi, LST_Day_1km, LST_Night_1km (EVI dropped)
**Zero-yield filter:** y > 0.1 applied in dataset.py (removes anomalous 0.0 t/ha samples)
**Train/val/test splits:** USA 2003-2020/2021-2022/2023 | IDN 2020-2023/–/2024 | VNM 2003-2021/2022/2023 | THA 2021/–/2023

## Experiment Results

### Mac MPS (old, superseded — from committed log files)
| Run | USA R² | Notes |
|-----|--------|-------|
| CNN-LSTM | 0.13 | usa_baseline.yaml |
| LSTM | 0.39 | usa_lstm.yaml, 107 epochs |

### Kaggle T4 GPU (REAL, 2026-05-08) — cropland-masked v2 data
| Run | Test R² | Test RMSE | Notes |
|-----|---------|-----------|-------|
| USA LSTM | **0.4416** | 1.6556 t/ha | Best epoch 41, early stop ep61 |

### Transfer Learning Results (Kaggle T4, 2026-05-08)

| Country | Transfer R² | Scratch R² | ΔR² | Interpretation |
|---------|-------------|------------|-----|----------------|
| IDN | **0.574** | 0.000 | **+0.574** | H1 strongly confirmed |
| VNM | 0.048 | -0.056 | **+0.104** | H1 weakly confirmed |
| THA | -2.726 | -0.490 | -2.236 | Overfitting (no val set, 3 yrs only) |

**Full logs:** `experiments/logs/transfer_results.csv`, `finetune_{idn,vnm,tha}_log.csv`

## Hypothesis Status

- **H1** (transfer > scratch): **CONFIRMED** — IDN (+0.574) and VNM (+0.104)
- **H2** (negative transfer from domain gap): **NOT CONFIRMED** — VNM is now positive; previous negative result was from wrong committed file
- **H3** (DANN mitigates domain gap): **NOT YET IMPLEMENTED**

## Pending / Blockers

1. **USA v2 training (NOT YET RUN)** — `usa_lstm_v2.yaml` created (hidden=512, patience=30). Run on Kaggle T4 next session. Expected: R² > 0.4416, hopefully ≥0.6.
2. **THA re-fine-tune (NOT YET RUN)** — `COUNTRY_EPOCH_OVERRIDES` in finetune.py caps THA at 10+10 epochs. Need to re-run after USA v2 checkpoint is ready.
3. **DANN implementation** — for H3 (`src/models/dann.py`). Not started.
4. **IDN pre-2020 BPS data** — separate BPS table download. Low priority.
5. **wandb integration** — not yet wired into train.py.

## Kaggle Setup (working config)

- **GPU:** T4 (sm_75) — P100 (sm_60) incompatible with PyTorch 2.10.0+cu128
- **Dataset:** `alisulashidayat/maize-yield-modis-tensors` at `/kaggle/input/datasets/alisulashidayat/maize-yield-modis-tensors/`
- **Repo:** `https://github.com/alisulas/thesis-maize-v2.git` → `/kaggle/working/thesis_maize`
- **Notebook:** `notebooks/maize-yield-transfer-learning-training.ipynb`
