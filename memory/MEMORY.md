# Memory Index

- [Data & Pipeline Status](project_data_status.md) — Yield data, MODIS tensors, experiment results, pending blockers. Last updated 2026-05-28
- [Technical Decisions Log](project_decisions.md) — EVI drop, no cropland mask, GAUL 2015, model choice, normalization, NASS fix
- [Code Architecture & File Map](project_architecture.md) — Semua source file, pipeline flow, cara menjalankan training
- [Coding Feedback](feedback_coding.md) — Hybrid approach: notebook=demo 3 epoch, training penuh via CLI

## Key Notebooks (urutan proses)

| Notebook | Fungsi |
|----------|--------|
| `notebooks/00_usa_download_nass.ipynb` | Download + EDA yield USA |
| `notebooks/01_usa_clean_yield.ipynb` | Cleaning yield USA |
| `notebooks/01b_idn_clean_yield.ipynb` | Cleaning yield IDN (BDSP) |
| `notebooks/03_merge_modis.ipynb` | Walkthrough merge MODIS |
| `notebooks/03b_explore_tensor.ipynb` | Validasi usa_modis.npz |
| `notebooks/05_train_usa.ipynb` | Demo 3 epoch |
| `notebooks/05b_train_usa_full.ipynb` | Training penuh di notebook |
| `notebooks/06_model.ipynb` | Walkthrough arsitektur CNN-LSTM |
