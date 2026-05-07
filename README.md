# Transfer Learning for Maize Yield Prediction: USA to ASEAN

> Master's thesis project investigating cross-continental transfer learning for crop yield prediction from data-rich (USA) to data-limited (Indonesia, Vietnam, Thailand) environments.

## Abstract

Crop yield prediction is critical for food security and agricultural planning. While deep learning models have achieved strong performance in data-rich regions like the United States, their application to data-limited regions remains challenging. This work investigates whether transfer learning can bridge the gap by leveraging US county-level maize yield data to improve predictions in ASEAN countries (Indonesia, Vietnam, Thailand) where only province-level data is available. We compare from-scratch baselines against vanilla fine-tuning and Domain Adversarial Neural Networks (DANN) using MODIS satellite imagery and historical yield statistics from 2003 to 2023.

## Research Question

> Can transfer learning from US county-level maize data improve yield prediction accuracy in ASEAN countries with limited province-level training data?

## Methodology Overview
┌─────────────────────┐
            │   USA County Data   │
            │  (Source Domain)    │
            │   ~30,000 samples   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Pretrain CNN-LSTM  │
            │  (You et al. 2017)  │
            └──────────┬──────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
   ┌────────────────┐    ┌────────────────┐
   │  Vanilla FT    │    │     DANN       │
   │  (fine-tune)   │    │ (domain adapt) │
   └───────┬────────┘    └───────┬────────┘
           │                     │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │   ASEAN Province    │
           │  (Target Domain)    │
           │ Indonesia/Vietnam/  │
           │     Thailand        │
           └─────────────────────┘


### Quick Start

```bash
# 1. Download yield data for all countries
python src/data/download_yield_data.py --country all

# 2. Extract MODIS features via GEE (this will take ~2 hours)
python src/data/extract_modis.py --region usa --years 2003-2023

# 3. Train USA baseline
python src/training/train.py --config experiments/configs/usa_baseline.yaml

# 4. Run transfer learning experiments
python src/training/transfer.py --config experiments/configs/transfer_indonesia.yaml
```

## Data Sources

| Country | Source | Resolution | Years |
|---------|--------|------------|-------|
| USA | USDA NASS | County | 2003-2023 |
| Indonesia | BPS | Province | 2003-2023 |
| Vietnam | GSO | Province | 2003-2023 |
| Thailand | OAE | Province | 2003-2023 |
| Satellite | MODIS (MOD09A1, MYD11A2) | 500m / 1km | 2003-2023 |
| Boundaries | GADM | Admin levels | Static |

## Experiments

| Experiment | Source | Target | Method | Status |
|------------|--------|--------|--------|--------|
| 1 | — | USA | From scratch | TBD |
| 2 | — | Indonesia | From scratch | TBD |
| 3 | — | Vietnam | From scratch | TBD |
| 4 | — | Thailand | From scratch | TBD |
| 5 | USA | Indonesia | Vanilla FT | TBD |
| 6 | USA | Vietnam | Vanilla FT | TBD |
| 7 | USA | Thailand | Vanilla FT | TBD |
| 8 | USA | ASEAN combined | Vanilla FT | TBD |
| 9 | USA | ASEAN combined | DANN | TBD |

## Results

To be updated after experiments.

## Citation

If you use this work, please cite:

```bibtex
@article{[AliSulasHidayat]2026maize,
  title={Transfer Learning for Maize Yield Prediction from Data-Rich to Data-Limited Environments: A Case Study of ASEAN Countries},
  author={[Ali Sulas Hidayat] 
  year={2026},
  note={Under review}
}
```

## License

MIT License — see LICENSE file.
