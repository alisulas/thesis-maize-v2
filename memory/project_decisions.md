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
4 new Indonesian provinces (Papua Barat Daya IDN-92, Papua Selatan IDN-95, Papua Tengah IDN-96, Papua Pegunungan IDN-97) split from Papua in 2022. GAUL 2015 doesn't have them. Kalimantan Utara (IDN-65) also missing.
**Why:** These 5 provinces have minimal corn production. Acceptable loss for MVP; 33/38 provinces covered.

**Ha Tay (VNM) dropped (2026-05-08)**
GAUL 2015 has Ha Tay as a province but it was merged into Ha Noi in 2008. GSO yield data doesn't have it separately. Rows with Ha Tay MODIS data discarded.
**Why:** No corresponding yield label possible; can't construct valid training sample.

**IDN yield only 2020–2024 (2026-05-08)**
BPS uses KSA methodology from 2020 (satellite-based area measurement). Pre-2020 BPS data uses "eye estimate" methodology — different enough to be treated as a separate dataset. MODIS extracted for 2020–2024 to match.
**Why:** Mixing methodologies without correction would add noise. Pre-2020 download is future work.

**VNM yield 1995–2023 but MODIS only 2003–2023**
Yield file covers 1995–2023 (GSO has older data). MODIS/Terra only reliable from 2003. Tensor only contains 2003–2023 overlap.
**Why:** Can't extract satellite features pre-2003; extra yield rows are stored but not used in tensors.

## Model Decisions

**LSTM-only preferred over CNN-LSTM (2026-05-08)**
CropYieldLSTM (R²=0.39) outperformed CropYieldCNNLSTM (R²=0.13) on Mac test.
CNN-LSTM applies Conv1d over the feature axis (10 features), which doesn't have the same spatial meaning as over histogram bins (original You et al. 2017 used 32-bin histograms). With scalar mean features, CNN is applying arbitrary linear combinations of spectral bands without clear physical meaning.
**Why:** LSTM directly processes the time series; simpler is better given the small feature dimension.
**Best config:** hidden_size=256, n_layers=2, dropout=0.3, lr=5e-4, cosine LR decay

**Fine-tuning: 2-phase strategy (2026-05-08)**
Phase 1 (20 epochs): freeze LSTM, train only FC head at lr=1e-3
Phase 2 (50 epochs): unfreeze all, lr=1e-4
**Why:** Prevents destroying pretrained representations early in fine-tuning (catastrophic forgetting). Standard practice for transfer learning with small target datasets.

## Pipeline Decisions

**One .npz per country (not one big tensor)**
Separate files: usa_modis.npz, idn_modis.npz, vnm_modis.npz, tha_modis.npz
**Why:** USA tensor is ~300MB compressed; loading all countries at once wastes memory. Each dataset has different normalization stats (computed per-split).

**Z-score normalization computed from training split only**
Stats (mean, std per feature) computed from training data and stored in MaizeDataset; applied to val/test.
**Why:** Prevents data leakage from future years into normalization. Critical for temporal splits.

**GEOID as region_id for USA, IDN-XX for Indonesia**
USA: 5-digit FIPS (state_fips + county_fips, zero-padded to 5 chars) = TIGER GEOID → direct join.
IDN: "IDN-" + 2-digit BPS province code (e.g., IDN-11 = Aceh).
VNM/THA: use normalized province name as join key (no stable numeric ID in yield files).
**Why:** Stable identifiers allow reproducible merges; BPS code is the official Indonesian standard.
