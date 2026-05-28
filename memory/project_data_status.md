---
name: Data & Pipeline Status
description: Current state of all data, tensors, models, and experiments — scope USA + Indonesia only
type: project
---

## Scope

**USA (source domain) + Indonesia (target domain) only.** Vietnam dan Thailand dikeluarkan dari scope thesis ini.

## Yield Label Data

| Country | File | Rows | Regions | Years | Source | Status |
|---------|------|------|---------|-------|--------|--------|
| USA | `data/processed/usa/yield_usa_2003_2023.parquet` | 33,962 | 2,226 counties | 2003–2023 | USDA NASS API | ✅ Done |
| IDN | `data/processed/idn/` (parquet) | TBD | ~514 kabupaten | 2003–2022 | BDSP Kementan (scraping) | ⚠️ Downloaded, perlu dicek |

**Yield schema (standard):** region_id, region_name, country, year, yield_ton_ha, data_source
**Unit conversions:** USA bu/acre × 0.06277 = ton/ha | IDN ku/ha ÷ 10 = ton/ha

**USA data fix (2026-05-21→28):**
- Filtered `prodn_practice_desc` ke ALL PRODUCTION PRACTICES (hapus duplikat IRRIGATED/NON-IRRIGATED)
- Rows dengan yield < 0.1 t/ha di-drop (bukan di-NaN)
- Hasil akhir: 33,962 rows, 2,226 counties, NaN=0
- Pipeline: `notebooks/01_usa_clean_yield.ipynb` + `src/data/00_download_yield_usa.py`

**IDN known issue:** BDSP hanya sampai 2022, bukan 2025. Test split `[2023]` akan kosong untuk IDN — perlu disesuaikan ke Test 2022.

## MODIS Satellite Data

### v1 (no cropland mask) — superseded
- 50 raw CSVs in Drive folder `thesis_maize_gee/`
- GEE task IDs: `experiments/logs/gee_tasks_asean.csv`, `gee_tasks_usa.csv`

### v2 (MCD12Q1 cropland mask, class 12) — CURRENT
- 50 raw CSVs replaced in `data/raw/modis/modis_{country}_{year}.csv`
- GEE task IDs: `experiments/logs/gee_tasks_asean_v2.csv`, `gee_tasks_usa_v2.csv`
- Drive folder: `thesis_maize_gee_v2/`
- Submitted 2026-05-08; USA files shrunk from ~33MB → ~28MB (confirms mask applied)

| Country | Raw CSVs | Status | Notes |
|---------|----------|--------|-------|
| USA | `data/raw/modis/modis_usa_{2003..2023}.csv` (21 files) | ✅ Done | TIGER county boundaries |
| IDN | `data/raw/modis/modis_idn_*.csv` | ⚠️ Perlu dicek | Mungkin masih province-level (ADM1), harus kabupaten (ADM2) |

**GEE project:** `alamat-413120` | Drive folder: `thesis_maize_gee`

## Processed Tensors (`data/processed/modis/`)

| File | X shape | Samples | Counties/Kab | Years | Yield range | Status |
|------|---------|---------|-------------|-------|-------------|--------|
| `usa_modis.npz` | (33962, 46, 10) | 33,962 | 2,226 | 2003–2023 | 0.65–17.39 t/ha | ✅ Current |
| `idn_modis.npz` | — | — | — | — | — | ⚠️ Belum direbuild di kabupaten level |

**Tensor format:** X=(N, T=46, F=10) float32, y=(N,) float32 (yield_ton_ha), + region_ids, years
**Features (F=10):** b01–b07 (reflectance), ndvi, LST_Day_1km, LST_Night_1km (EVI dropped — overflow bug)

## Year Splits

```
USA: Train 2003–2020 | Val 2021–2022 | Test 2023
IDN: Train 2003–2019 | Val 2020–2021 | Test 2022  ← sesuaikan karena BDSP max 2022
```

## Experiment Results

### Mac MPS (old, superseded — smoke test tanpa cropland mask)
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

**Full logs:** `experiments/logs/transfer_results.csv`, `finetune_idn_log.csv`

## Hypothesis Status

- **H1** (transfer > scratch): **CONFIRMED** — IDN ΔR²=+0.574
- **H2** (negative transfer from domain gap): **Tidak diuji** — Vietnam/Thailand dikeluarkan dari scope
- **H3** (DANN mitigates domain gap): **NOT YET IMPLEMENTED**

## Kaggle Setup (working config)

- **GPU:** T4 (sm_75) — P100 (sm_60) incompatible with PyTorch 2.10.0+cu128
- **Dataset:** `alisulashidayat/maize-yield-modis-tensors` at `/kaggle/input/datasets/alisulashidayat/maize-yield-modis-tensors/`
- **Repo:** `https://github.com/alisulas/thesis-maize-v2.git` → `/kaggle/working/thesis_maize`
- **Notebook:** `notebooks/maize-yield-transfer-learning-training.ipynb`

## Pending / Blockers (priority order)

1. **Jalankan training USA penuh** — `python src/training/05_train_usa.py --config experiments/configs/usa_baseline.yaml`
2. **Cek dan fix IDN yield pipeline** — pastikan parquet IDN tersedia dan benar
3. **Cek IDN MODIS level** — apakah sudah kabupaten (ADM2) atau masih province (ADM1)
4. **Rebuild idn_modis.npz** jika MODIS IDN masih province-level
5. **Fix IDN year splits** — test set tidak bisa 2023 karena BDSP max 2022
6. **Fine-tune IDN** — setelah USA pretrain selesai, jalankan fine-tuning dengan freeze strategy
7. **DANN implementation** — for H3 (`src/models/dann.py`). Not started.
