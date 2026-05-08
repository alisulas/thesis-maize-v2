# Transfer Learning for Maize Yield Prediction: USA to ASEAN

> Master's thesis (S2) investigating cross-continental transfer learning for crop yield prediction — from data-rich USA (county-level, ~30k samples) to data-limited ASEAN countries (province-level, 86–1,315 samples).

## Research Question

> Can a deep learning model pretrained on US county-level maize yield data be effectively transferred to predict maize yield in ASEAN countries (Indonesia, Vietnam, Thailand) where only limited province-level data is available?

## Hypotheses

| # | Hypothesis | Status |
|---|-----------|--------|
| H1 | Transfer learning outperforms training from scratch on ASEAN data | ✅ Confirmed (IDN ΔR²=+0.574, VNM ΔR²=+0.104) |
| H2 | Climate domain gap causes negative transfer in vanilla fine-tuning | ❌ Not confirmed — VNM and IDN both show positive transfer |
| H3 | DANN domain adaptation mitigates the climate domain gap | ⬜ Not yet implemented |

---

## Methodology Overview

```
┌──────────────────────────────────────────────┐
│              DATA COLLECTION                 │
│  USA: USDA NASS yield (2003–2023, 2,280      │
│       counties, ~30k samples)                │
│  ASEAN: BPS / GSO / OAE yield               │
│  Satellite: MODIS MOD09A1 + MYD11A2         │
│  Boundaries: GADM 2015 (Level 1 & 2)        │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│         FEATURE EXTRACTION (GEE)             │
│  - 7 reflectance bands (b01–b07)             │
│  - NDVI                                      │
│  - LST Day + Night                           │
│  - Cropland mask: MCD12Q1 class 12           │
│  Output: X = (N, 46 timesteps, 10 features) │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│         PRETRAINING (USA Source)             │
│  CropYieldLSTM: hidden=512, 2 layers         │
│  Train: 2003–2020 | Val: 2021–2022           │
│  Test: 2023                                  │
│  USA Test R²: 0.4416 (v1, hidden=256)        │
│              [v2 pending, hidden=512]        │
└──────────┬───────────────────────────────────┘
           │  pretrained weights
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────────┐  ┌────────────┐
│ Vanilla    │  │   DANN     │
│ Fine-Tune  │  │ (H3, TBD)  │
│ (H1, H2)   │  │            │
└─────┬──────┘  └─────┬──────┘
      │               │
      └──────┬────────┘
             ▼
┌──────────────────────────────────────────────┐
│         TARGET DOMAIN EVALUATION            │
│  Indonesia: Test 2024  (162 train samples)  │
│  Vietnam:   Test 2023  (1,315 train samples)│
│  Thailand:  Test 2023  (126 train samples)  │
└──────────────────────────────────────────────┘
```

### Fine-Tuning Strategy (2-Phase)

```
Phase 1 (frozen, lr=1e-3)          Phase 2 (full, lr=1e-4)
─────────────────────────          ───────────────────────
LSTM weights: FROZEN          →    LSTM weights: unfrozen
FC head:      training        →    FC head:      training

Purpose: head learns ASEAN        Purpose: gentle adaptation
scale without corrupting           of LSTM to tropical
pretrained representations         climate patterns
```

Per-country epoch caps (small datasets, no val set):

| Country | Phase 1 | Phase 2 | Val set? |
|---------|---------|---------|----------|
| IDN | 20 | 20 | No |
| VNM | 20 | 50 | Yes (early stop) |
| THA | 10 | 10 | No |

---

## Current Results

### USA Baseline (Kaggle T4, v1)

| Split | R² | RMSE |
|-------|----|------|
| Train (2003–2020) | — | — |
| Val (2021–2022) | — | — |
| **Test (2023)** | **0.4416** | **1.656 t/ha** |

### Transfer Learning Results (Kaggle T4, v1 checkpoint)

| Country | Transfer R² | Scratch R² | ΔR² | Interpretation |
|---------|------------|-----------|-----|----------------|
| IDN | **0.574** | 0.000 | **+0.574** | H1 strongly confirmed |
| VNM | 0.048 | −0.056 | **+0.104** | H1 weakly confirmed |
| THA | −2.726 | −0.490 | −2.236 | Overfitting (fixed: epoch cap added) |

> Note: THA result is invalid due to overfitting (no val set, 70 epochs on 86 samples). Re-run pending with epoch cap (10+10 epochs).

---

## How I Do This Research (Step by Step)

### Step 1 — Yield Label Data Collection
Download official yield statistics from government sources:
- USA: USDA NASS QuickStats API → `yield_usa_2003_2023.parquet`
- Indonesia: BPS website (manual) → 2020–2024 (KSA methodology only)
- Vietnam: GSO website (manual) → 2003–2023
- Thailand: OAE website (manual) → 2021–2023

Standardize units to **ton/ha** and schema: `region_id, region_name, country, year, yield_ton_ha`.

### Step 2 — Satellite Feature Extraction (Google Earth Engine)
Run GEE Python API scripts that:
1. Load MODIS MOD09A1 (reflectance) + MYD11A2 (LST) for each year
2. Apply MCD12Q1 cropland mask (class 12) before averaging
3. Aggregate pixel values to admin boundary means (500m scale)
4. Export one CSV per country per year → 50 CSVs total

Output: 46 timesteps × 10 features per region-year sample.

### Step 3 — Tensor Construction
Merge satellite CSVs with yield labels by `region_id` + `year`:
- Match MODIS dates to growing season (all 46 8-day composites)
- Z-score normalize features (stats from training split only)
- Save as `.npz`: `X=(N, 46, 10)`, `y=(N,)`, `years=(N,)`

### Step 4 — USA Baseline Training
Train `CropYieldLSTM` on USA data (temporal split: 2003–2020 train, 2021–2022 val, 2023 test):
- Architecture: LSTM (hidden=512, 2 layers, dropout=0.3) → FC(512→64→1)
- Optimizer: AdamW, lr=5e-4, cosine LR decay, patience=30
- Platform: Kaggle T4 GPU (free tier)
- Checkpoint: `experiments/checkpoints/usa_lstm_v2/best_model.pt`

### Step 5 — Transfer Learning (Fine-Tuning)
Load USA checkpoint → fine-tune on each ASEAN country:
1. **Phase 1** (freeze LSTM, train head): head adapts to ASEAN yield scale
2. **Phase 2** (unfreeze all, low LR): LSTM gently adapts to tropical patterns
3. Compare against from-scratch baseline (same architecture, random init)
4. Report ΔR² = transfer R² − scratch R²

### Step 6 — DANN Domain Adaptation *(planned)*
Implement Domain Adversarial Neural Network:
- Add domain discriminator branch to shared LSTM feature extractor
- Train with gradient reversal: features become domain-invariant
- Compare against vanilla fine-tuning to test H3

### Step 7 — Analysis & Paper Writing
- Plot learning curves, prediction vs. actual scatter, feature importance
- Statistical significance tests on ΔR²
- Write paper targeting IEEE Access (Q1/Q2)

---

## Experiments Table

| # | Source | Target | Method | Status |
|---|--------|--------|--------|--------|
| 1 | — | USA | From scratch | ✅ R²=0.4416 (v1) |
| 2 | — | IDN | From scratch | ✅ R²=0.000 |
| 3 | — | VNM | From scratch | ✅ R²=−0.056 |
| 4 | — | THA | From scratch | ✅ R²=−0.490 |
| 5 | USA | IDN | Vanilla FT | ✅ R²=0.574 |
| 6 | USA | VNM | Vanilla FT | ✅ R²=0.048 |
| 7 | USA | THA | Vanilla FT | ⚠️ R²=−2.726 (re-run pending) |
| 8 | USA v2 | IDN/VNM/THA | Vanilla FT | ⬜ Pending (hidden=512) |
| 9 | USA | ASEAN | DANN | ⬜ Not yet implemented |

---

## Data Sources

| Country | Source | Resolution | Years | Samples |
|---------|--------|------------|-------|---------|
| USA | USDA NASS | County (2,280) | 2003–2023 | ~32,296 |
| Indonesia | BPS | Province (38) | 2020–2024 | 162 |
| Vietnam | GSO | Province (63) | 2003–2023 | 1,315 |
| Thailand | OAE | Province (43) | 2021–2023 | 126 |
| Satellite | MODIS MOD09A1 + MYD11A2 | 500m / 1km | 2003–2024 | — |
| Boundaries | GADM 2015 | Level 1 & 2 | Static | — |

---

## Quick Start (Kaggle)

```bash
# 1. Clone repo on Kaggle and install deps
git clone https://github.com/alisulas/thesis-maize-v2.git
cd thesis_maize
pip install -r requirements.txt

# 2. Copy .npz tensors from Kaggle dataset to data/processed/modis/
# (done automatically in notebook Cell 2)

# 3. Train USA baseline v2
python src/training/train.py --config experiments/configs/usa_lstm_v2.yaml

# 4. Fine-tune on all ASEAN countries
python src/transfer/finetune.py --country all \
    --pretrained experiments/checkpoints/usa_lstm_v2/best_model.pt
```

---

## Citation

```bibtex
@article{hidayat2026maize,
  title={Transfer Learning for Maize Yield Prediction from Data-Rich to Data-Limited Environments: A Case Study of ASEAN Countries},
  author={Hidayat, Ali Sulas},
  year={2026},
  note={Under review, IEEE Access}
}
```

## License

MIT License — see LICENSE file.
