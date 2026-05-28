---
name: Technical Decisions Log
description: Non-obvious design decisions made during the project with rationale
type: project
---

## Data Decisions

**EVI dropped from feature set (2026-05-08)**
MODIS EVI formula (denominator near zero at some pixels) overflows when taking the mean over an admin boundary → values reach ±1e11. Dropped entirely. Using 10 features: b01–b07, ndvi, LST_Day_1km, LST_Night_1km.
**Why:** Can't clip in post-processing reliably; re-extraction with server-side clipping is the proper fix but not urgent.

**No cropland masking in current GEE extraction (2026-05-08)**
MODIS features averaged over entire county/province. Literature always applies cropland mask (MCD12Q1 class 12 or USDA CDL for USA). This is the #1 known quality issue causing low USA R² (~0.39 vs target 0.6).
**Why:** Missed in initial GEE script design. Fix requires re-running 50 GEE export tasks.

**GAUL 2015 accepted despite missing new Papua provinces (2026-05-08)**
4 new Indonesian provinces (Papua Barat Daya, Papua Selatan, Papua Tengah, Papua Pegunungan) split dari Papua tahun 2022. GAUL 2015 tidak punya mereka. Kalimantan Utara juga missing.
**Why:** Provinsi-provinsi ini produksi jagungnya minimal. Acceptable loss; mayoritas kabupaten tetap tercakup.

**IDN data upgraded to kabupaten level 2003–2025 (2026-05-21)**
User obtained full BPS kabupaten-level yield data from 2003–2025 (~514 kabupaten). This replaces old province-level data (2020–2024 only, 38 provinces, 162 tensor samples). Kabupaten (district) is comparable to USA county in granularity.
**Why:** More data (10k+ vs 162 samples), longer time series (23 vs 5 years), better spatial resolution → stronger training signal for transfer learning.
**Impact:** MODIS must be re-extracted using GAUL ADM2 boundaries for Indonesia 2003–2025. idn_modis.npz will be rebuilt.

**NASS `prodn_practice_desc` filtered to ALL PRODUCTION PRACTICES (2026-05-21)**
NASS reports corn yield with 3 production practices: ALL, IRRIGATED, NON-IRRIGATED. Query didn't filter → 2,578 county-year pairs had duplicates. Fix: keep only `ALL PRODUCTION PRACTICES` (weighted average of irrigated + non-irrigated). This is the official USDA county-level yield figure.
**Why:** IRRIGATED and NON-IRRIGATED are subset breakdowns of the SAME county-year, not independent samples. Keeping all 3 inflates sample count and confuses the model.
**Impact:** USA yield data reduced from 41,349 → 34,180 rows (2,272 counties). 218 yield=0 rows (2003–2009 only) also converted to NaN.

**Yield=0 converted to NaN in USA cleaning (2026-05-21)**
218 rows with exact yield=0.0 t/ha found — only years 2003–2009. Not agronomically plausible (county with zero corn wouldn't be in NASS). Likely measurement/reporting artifact.
**Why:** Zero yield incorrectly trains model to predict 0 for certain conditions; converting to NaN excludes these from loss computation.
**Impact:** 218 rows (0.6%) have NaN yield. Concentrated in early years only — after 2009, 0% missing.

**Cropland mask v2 applied (2026-05-08)**
Re-extracted all 50 MODIS CSVs with MCD12Q1 IGBP class 12 mask. Output to Drive folder `thesis_maize_gee_v2/`. USA file size shrunk ~33MB → ~28MB confirming mask effective.
**Why:** No-mask version averaged over non-cropland pixels (forests, cities), diluting the agricultural signal. Expected R² improvement 0.39 → 0.60+.

**Zero-yield filter added to dataset.py (2026-05-08)**
`y > 0.1` filter applied at load time in `MaizeDataset`. Removes 206 anomalous zero-yield USA samples.
**Why:** yield = 0.0 t/ha is physically impossible for reported harvest data; these are data errors.

**Kaggle T4 GPU, not Google Colab (2026-05-08)**
Training moved to Kaggle (free, T4 GPU). P100 on Kaggle incompatible with PyTorch 2.10.0+cu128 (requires sm_70+, P100 is sm_60).
**Why:** Kaggle free tier sufficient for model size (817K params); T4 (sm_75) compatible.

## Model Decisions

**Model aktif: CNN-LSTM (usa_baseline.yaml) — belum ada hasil final (2026-05-28)**
Smoke test lama menunjukkan LSTM-only (R²=0.39) > CNN-LSTM (R²=0.13), tapi smoke test itu pakai data lama (tensor yang sudah diperbaiki) dan tanpa cropland mask. Belum bisa dijadikan kesimpulan.
Training penuh dengan data baru (33,962 sampel, usa_modis.npz fixed) belum dijalankan.
**Config aktif:** `usa_baseline.yaml` — CNN-LSTM, hidden=128, cnn_ch=64, lr=1e-3, batch=512
**Status:** Menunggu training penuh. Update bagian ini setelah hasil R² test tersedia.

**Fine-tuning: 2-phase strategy (2026-05-08)**
Phase 1 (frozen): train only FC head at lr=1e-3
Phase 2 (full): unfreeze all, lr=1e-4
Default: 20+50 epochs. Per-country overrides in `COUNTRY_EPOCH_OVERRIDES` dict in finetune.py:
- THA: 10+10 (2 training years, no val set — catastrophic overfitting at 20+50)
- IDN: 20+20 (4 training years, no val set)
- VNM: 20+50 (default — has val set, early stopping works)
**Why:** Phase 1 prevents catastrophic forgetting. THA/IDN cap is critical — using train_loss as val proxy means patience never fires, so all epochs run; with <100 samples that's certain overfitting.

**USA v2 config: hidden_size=512 (2026-05-08)**
`experiments/configs/usa_lstm_v2.yaml`: hidden=512, patience=30, 200 epochs.
**Why:** v1 (hidden=256) reached R²=0.4416 on Kaggle T4 (target ≥0.6). Hypothesis: underfitting — 32k training samples is large enough for a bigger model. Checkpoint also now saves model_cfg so fine-tuning auto-reads correct hidden_size.

## Pipeline Decisions

**One .npz per country (not one big tensor)**
Separate files: usa_modis.npz, idn_modis.npz
**Why:** USA tensor is ~300MB compressed; loading all countries at once wastes memory. Each dataset has different normalization stats (computed per-split).

**Z-score normalization computed from training split only**
Stats (mean, std per feature) computed from training data and stored in MaizeDataset; applied to val/test.
**Why:** Prevents data leakage from future years into normalization. Critical for temporal splits.

**GEOID as region_id for USA, BPS kode for Indonesia**
USA: 5-digit FIPS (state_fips + county_fips, zero-padded to 5 chars) = TIGER GEOID → direct join.
IDN: BPS kabupaten code (numeric, e.g., 1101 = Aceh Selatan) — from BPS data column.
**Why:** Stable identifiers allow reproducible merges; BPS kabupaten code is the official Indonesian standard for district-level data.
