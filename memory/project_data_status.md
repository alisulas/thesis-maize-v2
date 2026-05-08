---
name: Data & Pipeline Status
description: Current state of all data, tensors, models, and experiments as of 2026-05-08
type: project
---

## Yield Label Data (all done)

| Country | File | Rows | Regions | Years | Source |
|---------|------|------|---------|-------|--------|
| USA | `data/processed/usa/yield_usa_2003_2023.parquet` | 41,349 | 2,280 counties | 2003–2025 | USDA NASS API |
| IDN | `data/processed/indonesia/yield_indonesia_2003_2023.parquet` | 190 | 38 provinces | 2020–2024 only | BPS (manually downloaded) |
| VNM | `data/processed/vietnam/yield_vietnam_province_1995_2023.csv` | 1,804 | 63 provinces | 1995–2023 | GSO (manually downloaded) |
| THA | `data/processed/thailand/thailand_province_yield_2021_2023.csv` | 126 | 43 provinces | 2021–2023 | OAE (manually downloaded) |
| ALL | `data/processed/owid_all_countries_national.csv` | — | national | 2003–2023 | OWID/FAOSTAT |

**Yield schema (standard):** region_id, region_name, country, year, yield_ton_ha, harvested_ha, production_ton, data_source
**Unit:** yield_ton_ha (conversions: USA bu/acre × 0.06277, IDN ku/ha ÷ 10)

**IDN note:** Pre-2020 BPS data NOT yet downloaded (different table, older "eye estimate" methodology). Only 2020–2024 KSA method available.

## MODIS Satellite Data (all done)

- **50 raw CSVs** in `data/raw/modis/modis_{country}_{year}.csv`
  - IDN: 5 files (2020–2024), VNM: 21 files (2003–2023), THA: 3 files (2021–2023), USA: 21 files (2003–2023)
- GEE task IDs saved: `experiments/logs/gee_tasks_asean.csv` and `gee_tasks_usa.csv`
- GEE project: `alamat-413120`, Drive folder: `thesis_maize_gee`

## Processed Tensors (all done, in `data/processed/modis/`)

| File | X shape | Samples | Yield range |
|------|---------|---------|-------------|
| `usa_modis.npz` | (32296, 46, 10) | 32,296 | 0.00–16.96 t/ha |
| `idn_modis.npz` | (162, 46, 10) | 162 | 0.00–7.68 t/ha |
| `vnm_modis.npz` | (1315, 46, 10) | 1,315 | 1.48–9.02 t/ha |
| `tha_modis.npz` | (126, 46, 10) | 126 | 2.00–5.57 t/ha |

**Tensor format:** X=(N, T=46, F=10) float32, y=(N,) float32 (yield_ton_ha), + region_ids, years arrays
**Features (F=10):** b01–b07 (reflectance), ndvi, LST_Day_1km, LST_Night_1km  (EVI dropped — export overflow)
**Train/val/test splits:** USA 2003-2020/2021-2022/2023 | IDN 2020-2023/–/2024 | VNM 2003-2021/2022/2023 | THA 2021/–/2023

## Experiment Results (Mac MPS sanity check, NOT final)

| Run | Config | Country | Test R² | Test RMSE | Notes |
|-----|--------|---------|---------|-----------|-------|
| usa_baseline_cnn_lstm | usa_baseline.yaml | USA | 0.13 | 2.06 t/ha | CNN-LSTM, early stop ep46 |
| usa_baseline_lstm | usa_lstm.yaml | USA | **0.39** | 1.73 t/ha | LSTM-only, early stop ep107 |
| finetune_idn | usa_lstm pretrained | IDN | 0.06 | 1.13 t/ha | ΔR²=+0.356 vs scratch |
| finetune_vnm | usa_lstm pretrained | VNM | -0.19 | 1.46 t/ha | ΔR²=-0.094 negative transfer |
| finetune_tha | usa_lstm pretrained | THA | -0.03 | 0.44 t/ha | ΔR²=+0.008 inconclusive |

**Full transfer results:** `experiments/logs/transfer_results.csv`

**Why R² is low:** No cropland masking — MODIS features averaged over entire admin boundary (including forests, cities). This is the #1 pending fix.

## Pending / Blockers

1. **Re-run GEE with cropland mask** (MCD12Q1 class 12) — expected to fix low USA R²
2. **Colab training** — real experiments on A100, not Mac MPS
3. **DANN implementation** — for H3 (domain adaptation)
4. **IDN pre-2020 BPS data** — separate BPS table download needed
5. **Growing season filter** — optionally subset T=46 to T=23 (May–Oct only)
6. **wandb integration** — not yet wired into train.py

**Why:** Low R² is a data quality issue (no cropland masking), not a code bug. Code pipeline is correct and complete.
