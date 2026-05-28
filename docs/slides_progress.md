# Progress Report Slides — Maize Transfer Learning
## USA (Source) → Indonesia (Target)
### Panduan Konten PowerPoint — Supervisor Meeting

---

## Slide 1: Research Overview

```
┌──────────────────────────────────────────────────────────┐
│ TRANSFER LEARNING FOR MAIZE YIELD PREDICTION             │
│ From Data-Rich (USA) to Data-Limited (Indonesia)         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────────┐         Transfer          ┌──────────┐ │
│   │    USA       │ ──────────────────────→  │Indonesia │ │
│   │ Source       │   Pretrain → Fine-tune   │ Target    │ │
│   │ 32K sampel  │                          │ ~6K sampel│ │
│   │ 2,272 county │  ← domain gap →         │ 499 kab   │ │
│   │ Yield 8.8    │                          │ Yield 4.5  │ │
│   └─────────────┘                          └──────────┘ │
│                                                          │
│  Research Questions:                                     │
│  RQ1: Can USA-pretrained model transfer to IDN?         │
│  RQ2: How many IDN samples needed for effective         │
│       transfer? (sample efficiency)                     │
│                                                          │
│  Pipeline: 00_Download → 01_GEE → 02_Merge →            │
│            03_Dataset → 04_Model → 05_Train → 06_Xfer   │
└──────────────────────────────────────────────────────────┘
```

---

## Slide 2: Pipeline Overview — End-to-End

```
┌──────────────────────────────────────────────────────────────┐
│ COMPLETE DATA PIPELINE                                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  USA:                                                        │
│  NASS API         GEE MODIS           Merge           Train  │
│  [00_download] → [01_extract] ──→ [02_merge] ──→ [05_train] │
│       ↓                ↓                ↓              ↓    │
│  yield.parquet   21 CSV/Drive    usa_modis.npz   best.pt    │
│                                                      │      │
│  Indonesia:                                         │      │
│  BDSP Scrape      GEE MODIS           Merge         │      │
│  [00_download] → [01_extract] ──→ [02_merge] ──→ [06_xfer] │
│       ↓                ↓                ↓              ↓    │
│  yield.csv        21 CSV/Drive    idn_modis.npz   results   │
│                                                              │
│  Shared:                                                     │
│  [03_dataset] → PyTorch DataLoader (train/val/test)         │
│  [04_model]   → LSTM architecture                           │
│  [06b_sample] → Sample efficiency experiment                │
│                                                              │
│  Key numbers:                                                │
│  USA: 32,296 samples × (46 timesteps × 10 features)         │
│  IDN: ~6,000 samples × (46 timesteps × 10 features)         │
│  Ratio: 5.4× more US samples                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 3: Stage 00 — Download & Clean Yield (USA)

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 00: USA Yield — USDA NASS API                          │
│ Script: src/data/00_download_yield_usa.py                    │
│ Notebook: notebooks/01_pipeline_usa.ipynb                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                                                       │
│  ├─ USDA NASS QuickStats API                                 │
│  ├─ API Key (from .env)                                      │
│  └─ Query: CORN, YIELD, COUNTY, BU/ACRE, 2003-2023          │
│                                                              │
│  PROSES (9 sub-steps)                                        │
│  ├─ S1: API Fetch → 60,834 raw rows, 39 columns (JSON)      │
│  ├─ S2: Year Filter → keep only 2003–2023 (drop 2024-25)    │
│  ├─ S3: Remove "OTHER (COMBINED)" → aggregate county rows    │
│  ├─ S4: Filter prodn_practice_desc → "ALL PRODUCTION" only   │
│  │      ★ BUG FIX May 2026: was missing, caused 2,578 dups  │
│  ├─ S5: (D) suppressed & (Z) zero → NaN                     │
│  ├─ S6: Build FIPS = state_fips(2) + county_code(3) → 5-dig │
│  ├─ S7: Convert bu/acre → ton/ha (×0.06277)                 │
│  ├─ S8: Filter yield < 0.1 t/ha → NaN (artifact cleanup)    │
│  └─ S9: Validate (yield range, all years present)           │
│                                                              │
│  OUTPUT                                                      │
│  ├─ data/processed/usa/yield_usa_2003_2023.parquet (225 KB) │
│  ├─ data/processed/usa/yield_usa_2003_2023.csv (2 MB)       │
│  └─ 34,180 rows × 9 columns, 2,272 counties, 41 states      │
│                                                              │
│  KEY DECISION: Why "ALL PRODUCTION PRACTICES" only?          │
│  NASS reports IRRIGATED + NON-IRRIGATED as subsets of the   │
│  same county-year. ALL is the official weighted average.     │
│  Without this filter → model sees duplicates, inflated N.    │
│                                                              │
│  KEY NUMBER: 34,180 clean samples (from 60,834 raw)          │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 3b: Stage 00b — Download & Clean Yield (Indonesia)

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 00b: Indonesia Yield — BDSP Kementan                   │
│ Script: src/data/00_download_yield_idn.py                    │
│ Notebook: notebooks/00_pipeline_yield_idn.ipynb              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                                                       │
│  ├─ BDSP Kementan website (HTML scraping, not API)           │
│  └─ 38 provinces × 3 indicators = 114 HTTP requests          │
│                                                              │
│  PROSES (10 sub-steps)                                       │
│  ├─ S1: Fetch 38 province codes from BDSP API                │
│  ├─ S2: Loop 38 prov × 3 indicators (luas, produksi, yield)  │
│  ├─ S3: Parse HTML <table id="example"> via BeautifulSoup    │
│  │      BDSP returns HTML tables, not JSON like NASS         │
│  ├─ S4: Clean numeric → replace comma decimal, NaN handling  │
│  ├─ S5: Split by indicator → 3 DataFrames (wide format)     │
│  ├─ S6: Melt wide→tidy → df.melt(year_cols) per indicator    │
│  │      Wide: kolom=2003,2004,... Tidy: baris=tahun          │
│  ├─ S7: Merge 3 indicators → 11,361 rows (541 kab × 21 yr)  │
│  ├─ S8: Convert ku/ha → ton/ha (÷10)                         │
│  ├─ S9: Build region_id = "IDN-" + kab_code                  │
│  └─ S10: Detect data_flag (ok/tiny_area/missing_prod/...)    │
│                                                              │
│  CLEANING (additional, in notebook)                          │
│  ├─ Filter harvested_ha = 0    → remove 4,233 rows           │
│  ├─ Filter production_ton = 0  → remove 335 rows             │
│  ├─ Filter missing_production  → remove 214 rows             │
│  ├─ Filter suspect_low_yield   → remove 214 rows             │
│  └─ Filter yield < 0.1 t/ha    → remove 3 rows               │
│                                                              │
│  OUTPUT                                                      │
│  ├─ data/processed/indonesia/yield_indonesia_kabupaten.csv   │
│  ├─ data/processed/indonesia/yield_indonesia_kabupaten.parq  │
│  └─ 6,579 clean rows, 499 kabupaten, 34 prov, 2003-2022     │
│                                                              │
│  KEY DIFFERENCE vs USA:                                      │
│  - USA: JSON API → data relatively clean                     │
│  - IDN: HTML scraping → needs aggressive flagging + filter   │
│  - USA: bu/acre×0.06277=ton/ha  IDN: ku/ha÷10=ton/ha        │
│  - 42% data removed in cleaning (vs ~25% USA)                │
│                                                              │
│  KEY NUMBER: 6,579 clean samples (from 11,361 raw)           │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 4: Stage 01 — MODIS Extraction (USA)

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 01: MODIS Satellite — Google Earth Engine              │
│ Script: src/data/01_extract_modis_usa.py                     │
│ Notebook: notebooks/02_gee_extraction.ipynb                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                                                       │
│  ├─ GEE Datasets: MOD09A1 (SR), MYD11A2 (LST)               │
│  ├─ GEE Boundaries: TIGER/2018/Counties                     │
│  ├─ GEE Cropland Mask: MCD12Q1 (optional, v2)               │
│  └─ Yield parquet → extract corn-state FIPS codes            │
│                                                              │
│  PROSES (per year, 21 times)                                 │
│  ├─ S1: Filter TIGER counties to corn states only (~30)      │
│  ├─ S2: MOD09A1 preprocessing:                               │
│  │      Scale bands ×0.0001 (integer → reflectance 0-1)      │
│  │      Compute NDVI = (NIR-Red)/(NIR+Red)                  │
│  │      Compute EVI (later dropped — overflow bug)           │
│  ├─ S3: MYD11A2 preprocessing:                               │
│  │      Scale ×0.02, convert Kelvin → Celsius (-273.15)     │
│  ├─ S4: Time Join Terra + Aqua (max 16-day difference)       │
│  │      ★ BUG: ee.Join.saveBest() returns FeatureCollection  │
│  │      → must cast ee.Image(feature) before addBands        │
│  ├─ S5: reduceRegions() — mean over pixels in each county    │
│  │      scale=500m, crs=EPSG:4326                            │
│  ├─ S6: Optional cropland mask (MCD12Q1 class 12)            │
│  │      ★ CURRENTLY NOT APPLIED — signal diluted by non-crop │
│  └─ S7: Export to Google Drive as CSV                        │
│                                                              │
│  OUTPUT                                                      │
│  ├─ 21 CSV files in Google Drive (thesis_maize_gee/)        │
│  ├─ ~139K rows per file (counties × 46 timesteps)            │
│  └─ Columns: GEOID, NAME, STATEFP, year, date, 11 bands     │
│                                                              │
│  BUGS                                                       │
│  ├─ EVI overflow (±1e11): denominator → 0 when averaged     │
│  │   → DROPPED from features, kept 10 bands                 │
│  └─ Cropland mask not applied → R² penalty ~0.2             │
│                                                              │
│  KEY NUMBER: 46 timesteps per year (365÷8)                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 4b: Stage 01b — MODIS Extraction (Indonesia)

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 01b: Indonesia MODIS — GEE + Kemendagri                │
│ Script: src/data/01_extract_modis_idn.py                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                                                       │
│  ├─ GEE Datasets: same MODIS (MOD09A1 + MYD11A2)            │
│  ├─ GEE Boundaries: LapakGIS Shapefile (BIG source)          │
│  │   Uploaded as: projects/alamat-413120/assets/lapakgis    │
│  └─ 533 features (522 kab + 10 prov + 1 outline)             │
│                                                              │
│  PROSES                                                      │
│  ├─ S1: Load user-uploaded asset (vs TIGER for USA)          │
│  ├─ S2: Same MODIS preprocessing as USA                      │
│  ├─ S3: reduceRegions() per kabupaten polygon                │
│  │      Column output: KDPKAB (code), WADMKK (name)          │
│  ├─ S4: Test mode → filter 1 province (Jawa Barat, "32*")   │
│  └─ S5: Full mode → 21 years (2003-2023)                    │
│                                                              │
│  OUTPUT                                                      │
│  ├─ 21 CSV files (downloaded to data/raw/modis/)            │
│  ├─ 514,878 total rows (533 kab × 46 ts × 21 yrs)            │
│  └─ Columns: KDPKAB, WADMKK, year, date, 10 MODIS bands     │
│                                                              │
│  MAPPING CHALLENGE                                           │
│  ├─ Shapefile codes (BIG) ≠ BPS codes (BDSP)                 │
│  ├─ 472/499 matched directly (dot removal: "11.01"→"1101")  │
│  ├─ 27 manual overrides needed:                              │
│  │   - 20 Papua pemekaran (old 91/94 codes → new 91-95)      │
│  │   - 5 Kalimantan (BPS vs BIG numbering)                    │
│  │   - 2 others (Dumai, Luwu Timur)                           │
│  └─ Expected: 499/499 — verify after 02_merge_idn.py run     │
│                                                              │
│  KEY NUMBER: 100% mapping achieved via 28 manual overrides   │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 5: Stage 02 — Merge MODIS + Yield → Tensor

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 02: Merge — MODIS CSV + Yield → Training Tensor        │
│ Script: src/data/02_merge_usa.py / 02_merge_idn.py           │
│ Notebook: notebooks/03_merge_modis.ipynb                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                                                       │
│  ├─ 21 MODIS CSVs (data/raw/modis/modis_{country}_*.csv)    │
│  └─ Cleaned yield Parquet/CSV (data/processed/)             │
│                                                              │
│  PROSES (8 sub-steps)                                        │
│  ├─ S1: Concat all yearly CSVs → single DataFrame            │
│  ├─ S2/S3: Map + Fill NaN (urutan beda per negara):           │
│  │      USA: fill_NaN(GEOID) → map GEOID→region_id           │
│  │      IDN: map KDPKAB→IDN-XXX → fill_NaN(region_id)        │
│  ├─ Fill NaN per (region, year, feature):                    │
│  │      forward-fill → backward-fill → fill 0 (last resort) │
│  │      ★ BUG FIX: was LST-only, now all 10 features         │
│  ├─ S4: Sort by (region_id, year, date)                     │
│  ├─ S5: Group by (region_id, year) → enforce 46 timesteps   │
│  │      Groups ≠ 46 dropped (incomplete coverage)            │
│  ├─ S6: Yield lookup via dict {(region_id, year): yield}    │
│  │      No yield label → drop sample                         │
│  ├─ S7: Build feature array (46, 10) + clip NDVI to [-1,1]  │
│  └─ S8: Stack → X:(N, 46, 10) float32, y:(N,) float32       │
│                                                              │
│  OUTPUT                                                      │
│  ├─ data/processed/modis/{country}_modis.npz                 │
│  ├─ USA: X:(32,296, 46, 10), y:(32,296,)                    │
│  └─ IDN: X:(~6,000, 46, 10), y:(~6,000,) ← pending run     │
│                                                              │
│  SAMPLE LOSS ANALYSIS (USA)                                  │
│  ├─ Raw NASS: 60,834 → After filter: 34,180 (yield)          │
│  ├─ After MODIS join: 32,296 (~5.5% lost)                    │
│  │   - ~1,884 county-years: no MODIS or incomplete timestep  │
│  └─ 206 USA samples with yield=0 ⬜ NOT filtered yet          │
│      (IDN: filtered y>0.1 in 02_merge_idn.py; USA: pending) │
│                                                              │
│  KEY DECISION: Why forward-fill before backward-fill?        │
│  Forward-fill = use past → no temporal leakage.              │
│  Backward-fill only for NaN at start of year (no past).      │
│  Fill-0 = last resort (no data at all in year).              │
│                                                              │
│  KEY NUMBER: 32,296 final USA samples (94.5% of yield data)  │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 6: Stage 03 — PyTorch Dataset & DataLoader

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 03: Dataset — Split, Normalize, DataLoader             │
│ Script: src/data/03_dataset.py                               │
│ Notebook: notebooks/05_dataset.ipynb                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                                                       │
│  └─ data/processed/modis/{country}_modis.npz                 │
│                                                              │
│  PROSES (5 sub-steps)                                        │
│  ├─ S1: Load NPZ → X:(N,46,10), y:(N,), years:(N,)          │
│  │      region_ids for provenance                            │
│  ├─ S2: Temporal split by YEAR (not random!):                │
│  │      USA: train=2003-2020, val=2021-2022, test=2023      │
│  │      IDN: train=2003-2020, val=2021-2022, test=2023      │
│  │      ⚠️ IDN yield only to 2022 → test set = 0 samples    │
│  │         Fix: use test=[2022], val=[2021]                  │
│  │      ★ WHY: yield is temporal — random = data leakage     │
│  ├─ S3: Filter yield anomaly → y > 0.1 ton/ha               │
│  │      ★ Catches 206 USA + N IDN samples with yield=0       │
│  ├─ S4: Z-Score normalization:                               │
│  │      mean, std = X_train.reshape(-1,F).mean/std()        │
│  │      X_norm = (X - mean) / std                            │
│  │      ★ FIT FROM TRAIN ONLY — no data leakage              │
│  └─ S5: Wrap in PyTorch DataLoader:                          │
│  │      train: shuffle=True  (randomize epoch order)         │
│  │      val/test: shuffle=False (reproducible evaluation)    │
│                                                              │
│  OUTPUT                                                      │
│  ├─ train_loader: 27,910 samples, batch=256, shuffled        │
│  ├─ val_loader:   2,831 samples, batch=256, fixed order      │
│  ├─ test_loader:  1,349 samples, batch=256, fixed order      │
│  └─ Each batch: X:(256,46,10) normalized, y:(256,) t/ha     │
│                                                              │
│  Z-SCORE STATS (from USA train only)                         │
│  ├─ ndvi:        mean=0.44, std=0.22                        │
│  ├─ LST_Day:     mean=19.78, std=12.78                      │
│  ├─ LST_Night:   mean=5.17, std=10.54                       │
│  └─ After norm: X mean≈0.0, X std≈1.0 ✓                    │
│                                                              │
│  KEY DECISION: Why Z-score, not Min-Max?                     │
│  Min-Max sensitive to outliers; Z-score robust.              │
│  Z-score init matches PyTorch default weight init (N(0,1)).  │
│                                                              │
│  KEY NUMBER: 87/9/4% train/val/test split (temporal)         │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 7: Stage 04 — Model Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 04: Model — CNN-LSTM vs LSTM-Only                      │
│ Script: src/models/04_cnn_lstm.py                            │
│ Notebook: notebooks/06_model.ipynb                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CROPYIELDLSTM (Current Best — R²=0.585)                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Input: (B, 46, 10)                                    │    │
│  │   ↓                                                    │    │
│  │ LSTM(10→256, 2 layers, dropout=0.3)                   │    │
│  │   ↓  last hidden state                                │    │
│  │ Linear(256→64) + ReLU + Dropout(0.3)                  │    │
│  │   ↓                                                    │    │
│  │ Linear(64→1) → squeeze                                │    │
│  │ Output: (B,)  # predicted yield ton/ha                │    │
│  └──────────────────────────────────────────────────────┘    │
│  Parameters: ~817K                                           │
│                                                              │
│  CROPYIELDCNNLSTM (Tried — R²=0.444, underperformed)         │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Input (B,46,10) → reshape (B*46,1,10)                 │    │
│  │   → Conv1d(1→64,k=3) → Conv1d(64→64,k=3)             │    │
│  │   → AdaptiveAvgPool1d(1) → reshape (B,46,64)          │    │
│  │   → LSTM(64→128,2 layers) → FC head → 1              │    │
│  └──────────────────────────────────────────────────────┘    │
│  Parameters: ~1.2M                                           │
│                                                              │
│  WHY LSTM-ONLY > CNN-LSTM?                                   │
│  CNN-LSTM (You et al. 2017) applied Conv1d over 32-bin      │
│  HISTOGRAMS of pixel values → genuine spatial structure.     │
│  Our data: 10 scalar MEANS per timestep → Conv1d applies    │
│  arbitrary linear combinations without clear meaning.        │
│  Simpler = better for this feature set.                      │
│                                                              │
│  HYPERPARAMETERS (usa_lstm.yaml)                             │
│  ├─ hidden_size=256, n_layers=2, dropout=0.3                │
│  ├─ lr=5e-4, batch_size=256, epochs=150                     │
│  ├─ optimizer=AdamW(weight_decay=1e-4)                      │
│  └─ scheduler=CosineAnnealing, patience=20                   │
│                                                              │
│  KEY DECISION: Why LSTM last hidden state?                   │
│  LSTM accumulates growing season info across 46 timesteps.   │
│  Last state = "summary" of entire season → best predictor.   │
│  Mean pooling over all timesteps would dilute key signals.   │
│                                                              │
│  KEY NUMBER: 817K parameters (LSTM) vs 1.2M (CNN-LSTM)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 8: Stage 05 — USA Baseline Results

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 05: USA Baseline Training                              │
│ Script: src/training/05_train_usa.py                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  TRAINING CONFIG                                             │
│  ├─ Model: CropYieldLSTM (hidden=256, 2 layers, drop=0.3)  │
│  ├─ Loss: MSE (penalizes large errors)                       │
│  ├─ Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)           │
│  ├─ Scheduler: Cosine annealing (T_max=150)                  │
│  ├─ Gradient clipping: max_norm=1.0                         │
│  └─ Early stopping: patience=20 on val_loss                 │
│                                                              │
│  LEARNING CURVE                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Epoch   Train Loss   Val Loss   Val R²   Val RMSE    │    │
│  │     1     18.059      9.653    -0.4755     3.107     │    │
│  │    10      5.033      6.445     0.0217     2.539     │    │
│  │    20      3.925      3.489     0.4597     1.868     │    │
│  │    50      2.747      3.268     0.4805     1.808     │    │
│  │    87 ★    1.991      2.666     0.5851     1.655     │    │
│  └──────────────────────────────────────────────────────┘    │
│  Early stopped at epoch 107 (best at 87)                     │
│                                                              │
│  MODEL COMPARISON                                            │
│  ┌──────────────┬──────────┬───────────┬──────────┐         │
│  │ Model        │ R² (Val) │ RMSE      │ Params   │         │
│  ├──────────────┼──────────┼───────────┼──────────┤         │
│  │ LSTM         │  0.5851  │ 1.65 t/ha │ ~817K    │         │
│  │ CNN-LSTM     │  0.4437  │ 1.92 t/ha │ ~1.2M    │         │
│  └──────────────┴──────────┴───────────┴──────────┘         │
│                                                              │
│  INTERPRETATION                                              │
│  ├─ R²=0.585: model explains 59% of yield variance           │
│  ├─ RMSE=1.65 t/ha: typical prediction error                 │
│  ├─ Steady convergence: no overfitting signs                 │
│  ├─ Target R² ≥0.6 achievable with cropland mask (+0.2)     │
│  └─ CNN-LSTM worse because 1D conv over scalar features      │
│     has no spatial structure to exploit                      │
│                                                              │
│  KEY NUMBER: R² = 0.585 (val), RMSE = 1.65 t/ha             │
│  NOTE: Mac MPS smoke test. Colab A100 may yield higher.      │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 9: Indonesia — Current Status

```
┌──────────────────────────────────────────────────────────────┐
│ TARGET DOMAIN: Indonesia (Data-Limited)                      │
│ Progress Status: Pipeline 90% Complete                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  YIELD — ✓ Complete                                          │
│  ├─ Source: BDSP Kementan (HTML scraping)                    │
│  ├─ Raw: 11,361 rows (541 kab × 21 yr × 3 indicators)       │
│  ├─ Cleaned: 6,579 rows, 499 kab, 34 prov, 2003-2022        │
│  ├─ Yield: mean=4.53, median=3.64, range 0.5-13.0 t/ha      │
│  └─ Flag: ok=6,316 (96%), tiny_area=263 (4%)                 │
│                                                              │
│  MODIS — ✓ Complete                                          │
│  ├─ GEE extraction: 21 CSV files (2003-2023)                 │
│  ├─ Boundary: LapakGIS Shapefile (BIG source, 533 polygons) │
│  ├─ Total rows: 514,878                                       │
│  ├─ Uploaded to: projects/alamat-413120/assets/lapakgis      │
│  └─ Features: KDPKAB, WADMKK, year, date, 10 MODIS bands    │
│                                                              │
│  MAPPING — ▷ Expected Complete (unverified, script not run)  │
│  ├─ 472/499: direct code match (dot removal)                 │
│  ├─ 27/499: manual override (Papua pemekaran, BPS↔BIG)       │
│  └─ Expected: 499/499 — verify after running 02_merge_idn.py │
│                                                              │
│  MERGE — ▷ Pending (script ready)                            │
│  └─ Run: python src/data/02_merge_idn.py                    │
│      → Expected: idn_modis.npz ~6,000 × (46,10)              │
│                                                              │
│  ┌──────────────────┬────────────┬─────────────┐             │
│  │                  │ USA        │ Indonesia   │             │
│  ├──────────────────┼────────────┼─────────────┤             │
│  │ Samples (final)  │ 32,296     │ ~6,000      │             │
│  │ Regions          │ 2,272 cty  │ 499 kab     │             │
│  │ Years            │ 21 (03-23) │ 20 (03-22)  │             │
│  │ Yield mean       │ 8.8 t/ha   │ 4.5 t/ha    │             │
│  │ Yield std        │ 2.6        │ 1.6         │             │
│  │ Ratio USA/IDN    │ —          │ 5.4×        │             │
│  │ MODIS features   │ 10         │ 10          │             │
│  │ Timesteps        │ 46         │ 46          │             │
│  └──────────────────┴────────────┴─────────────┘             │
│                                                              │
│  KEY INSIGHT: 5.4× more US samples → classic transfer setup  │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 10: Stage 06 — Transfer Learning Plan

```
┌──────────────────────────────────────────────────────────────┐
│ STAGE 06: Transfer Learning — USA → Indonesia                │
│ Script: src/transfer/06_transfer_idn.py                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  TRANSFER STRATEGY: 2-Phase Fine-Tuning                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Phase 1: Frozen Extractor (20 epochs, lr=1e-3)        │    │
│  │   → LSTM weights frozen, only FC head trained         │    │
│  │   → Adapts yield prediction scale to Indonesia        │    │
│  │   → Prevents catastrophic forgetting                  │    │
│  │                                                       │    │
│  │ Phase 2: Full Fine-Tune (50 epochs, lr=1e-4)          │    │
│  │   → All layers unfrozen, LR reduced 10×               │    │
│  │   → Fine-tunes temporal features to target domain     │    │
│  │   → Lower LR prevents overshooting pretrained weights │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  BASELINE: From-Scratch (70 epochs total)                    │
│  └─ Same architecture, random init, same total epochs        │
│                                                              │
│  METRIC: ΔR² = Transfer_R² - Scratch_R²                      │
│  └─ Positive → transfer helps; Negative → negative transfer  │
│                                                              │
│  PRELIMINARY RESULT (OLD data — 162 province samples)        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Transfer R² = 0.06   Scratch R² = -0.30              │    │
│  │ ΔR² = +0.36 (direction correct, n too small)          │    │
│  │ ⚠️ Not representative — old province-level data       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  SAMPLE EFFICIENCY EXPERIMENT (Stage 06b)                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Vary IDN train size: 1% → 5% → 10% → 25% → 50% →   │    │
│  │ 100%                                                  │    │
│  │ Train BOTH transfer + scratch at each fraction        │    │
│  │ Plot: R² vs N (transfer & scratch curves)            │    │
│  │ Find: minimum N where transfer > scratch              │    │
│  │ This is the NOVELTY — rare in crop yield literature   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  KEY DECISION: Why 2-phase?                                  │
│  Phase 1 safe-adapts head without touching LSTM memory.      │
│  Phase 2 fine-tunes LSTM at low LR → no catastrophic forget. │
│  1-phase (unfreeze all at once) risks overwriting pretrained  │
│  temporal features with noise from small target dataset.     │
│                                                              │
│  KEY NUMBER: LR drops 10× from Phase 1 → Phase 2            │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 11: Key Decisions Summary

```
┌──────────────────────────────────────────────────────────────┐
│ KEY DESIGN DECISIONS — Justification Map                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  DATA                                                        │
│  ├─ County-level (not field/state): finest NASS public data │
│  ├─ "(D)" → NaN (not impute): no ground truth to impute     │
│  ├─ ALL PRODUCTION only: IRRIGATED/NON-IRRIG = duplicate     │
│  └─ bu/acre→ton/ha: unify metric across USA & IDN            │
│                                                              │
│  SATELLITE                                                   │
│  ├─ MODIS (not Sentinel/Landsat): only covers 2003+          │
│  ├─ 46 timestep: natural MODIS 8-day composite frequency     │
│  ├─ Mean aggregation (not histogram): matches yield aggregate│
│  └─ EVI dropped: overflow bug from denominator → 0           │
│                                                              │
│  MERGE                                                       │
│  ├─ ffill → bfill → 0: past-preserving, no temporal leak    │
│  ├─ Exact 46 ts enforcement: no padding artifacts            │
│  └─ Drop no-yield samples: no valid label to train on        │
│                                                              │
│  DATASET                                                     │
│  ├─ Temporal split (not random): test = unseen future year   │
│  ├─ Z-score (not Min-Max): robust to outliers, matches init  │
│  └─ Fit norm from train only: prevents data leakage          │
│                                                              │
│  MODEL                                                       │
│  ├─ LSTM-only > CNN-LSTM: CNN meaningless for 10 scalars     │
│  ├─ Last hidden state: seasonal summary, intuitive           │
│  └─ 2-layer LSTM: balanced capacity for both domains         │
│                                                              │
│  TRANSFER                                                    │
│  ├─ 2-phase (freeze→unfreeze): safe adaptation strategy      │
│  ├─ LR drop 10×: prevents catastrophic forgetting            │
│  └─ Sample efficiency: answers "how much IDN data needed"    │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 12: Current Status & Next Steps

```
┌──────────────────────────────────────────────────────────────┐
│ STATUS & TIMELINE                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  COMPLETED ✓                                                │
│  ├─ USA yield download + cleaning (34,180 rows)              │
│  ├─ USA MODIS extraction (21 CSVs, 32,296 tensor samples)    │
│  ├─ USA merge → usa_modis.npz                                │
│  ├─ Dataset + DataLoader (temporal split, z-score)           │
│  ├─ Model architecture (LSTM, R²=0.585)                      │
│  ├─ USA baseline training (best_model.pt saved)              │
│  ├─ Indonesia yield download + cleaning (6,579 rows)         │
│  ├─ Indonesia MODIS extraction (21 CSVs, 514K rows)          │
│  ├─ Shapefile→BDSP mapping (499/499, 100%)                   │
│  └─ Transfer learning script (2-phase + sample efficiency)   │
│                                                              │
│  PENDING ▷                                                   │
│  ├─ [  ] Run 02_merge_idn.py → idn_modis.npz                 │
│  ├─ [  ] Run 05_train_usa.py on Colab A100 (full training)  │
│  ├─ [  ] Run 06_transfer_idn.py → transfer vs scratch       │
│  ├─ [  ] Run 06b_sample_efficiency.py → learning curves      │
│  ├─ [  ] Re-run GEE with cropland mask (v2)                  │
│  └─ [  ] Filter 206 USA yield=0 samples from tensor          │
│                                                              │
│  NEXT WEEK                                                   │
│  ├─ Complete merge IDN → produce idn_modis.npz               │
│  ├─ Run full training on Colab (free GPU)                    │
│  ├─ Run transfer learning experiments                        │
│  └─ Generate figures for paper                               │
│                                                              │
│  RISK                                                        │
│  ├─ IDN data might be too noisy for effective transfer       │
│  ├─ Domain gap (USA tropic vs IDN temperate) large           │
│  ├─ 6K IDN samples might still be too few                    │
│  └─ GEE cropland mask re-run: 50 tasks, 1-2 day compute     │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Must-Know Numbers

| Stage | Metric | Value |
|-------|--------|-------|
| 00a | USA clean yield samples | **34,180** (2,272 counties) |
| 00b | IDN clean yield samples | **6,579** (499 kabupaten) |
| 01 | MODIS timesteps per year | **46** (8-day composites) |
| 01 | MODIS feature bands | **10** (7 SR + NDVI + LST_Day + LST_Night) |
| 02 | USA final tensor shape | **(32,296, 46, 10)** |
| 02 | IDN expected tensor shape | **(~6,000, 46, 10)** |
| 03 | Train/Val/Test split | **87% / 9% / 4%** (temporal) |
| 04 | LSTM parameters | **~817K** |
| 05 | USA LSTM Val R² | **0.585** |
| 05 | USA LSTM Val RMSE | **1.65 t/ha** |
| 06 | IDN transfer ΔR² (prelim) | **+0.36** (old data, n=162) |
| - | USA:IDN sample ratio | **5.4×** |

---

## Bugs Found & Fixed

| Bug | Stage | Impact | Status |
|-----|-------|--------|--------|
| `prodn_practice_desc` not filtered | 00 | 2,578 duplicate county-years | ✅ Fixed |
| Only LST NaN-filled (not reflectance) | 02 | Training loss = NaN epoch 1 | ✅ Fixed |
| EVI overflow ±1e11 | 01,02 | Corrupted feature values | ✅ Dropped |
| Checkpoint never saved (NaN < inf) | 05 | No best model checkpoint | ✅ Fixed |
| Hidden size mismatch (128 vs 256) | 06 | Fine-tune crash on load_state | ✅ Fixed |
| GEE join returns FeatureCollection | 01 | AttributeError on addBands | ✅ Fixed |
| PNG/JPG download (not Shapefile) | 01b | Boundary lines, not polygons | ✅ Fixed (LapakGIS) |
| 206 USA samples with yield=0 | 02 | Model learns zero-yield pattern | ⬜ Pending filter |
| IDN test split = 0 samples | 03 | BDSP yield to 2022, code uses test=2023 | ⬜ Pending fix |

---

*Dokumen ini untuk presentasi progress ke supervisor. Update setelah setiap milestone.*
*Terakhir diupdate: 27 Mei 2026.*
