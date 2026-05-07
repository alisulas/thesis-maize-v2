# Thesis Project: Transfer Learning for Maize Yield Prediction from data-rich to data-limited environment

## Project Overview

This is a Master's thesis (S2) project investigating transfer learning for maize yield prediction from data-rich (USA) to data-limited environments (Indonesia, Vietnam, Thailand). The deliverable is a conference paper targeting **IEEE Access journal** (Q1/Q2).

**Timeline**: 12 weeks total. 

**Researcher**: Master's student, first-time deep learning practitioner, experienced in Python data analysis.

## Research Question

Can a deep learning model trained on US county-level maize yield data (data-rich) be effectively transferred to predict maize yield in ASEAN countries (Indonesia, Vietnam, Thailand) where only province-level data is available (data-limited)?

## Hypotheses

- **H1**: Transfer learning from USA outperforms training from scratch on ASEAN data due to limited target sample size
- **H2**: Domain gap between temperate (USA) and tropical (ASEAN) climate may cause negative transfer in vanilla fine-tuning
- **H3**: Domain adaptation methods (DANN) can mitigate the climate domain gap

## Scope & Constraints

### In Scope
- Single crop: **Maize only**
- Source domain: USA county-level (USDA NASS), 2003-2023
- Target domain: Indonesia (BPS), Vietnam (GSO), Thailand (OAE), province-level
- Annual aggregate yield (no seasonal split)
- Remote sensing: MODIS via Google Earth Engine
- Methods: Fine-tuning + DANN

### Out of Scope (DO NOT EXPAND)
- Other crops (rice, soybean, etc.)
- Sub-province / district-level prediction
- Real-time / in-season prediction (only end-of-season)
- Sentinel-2 high-resolution imagery (MODIS only, simpler)
- Multi-source pretraining (USA + Brazil/China) — reserved for future work

## Technical Stack

- **Language**: Python 3.10+
- **DL Framework**: PyTorch 2.x
- **Satellite Data**: Google Earth Engine Python API
- **Experiment Tracking**: Weights & Biases (wandb)
- **Notebooks**: Jupyter via Google Colab Pro
- **Local Dev**: Mac M1 8GB (development only, NOT training)
- **Training**: Google Colab Pro (GPU: A100/V100 preferred)
- **Version Control**: Git + GitHub
- **Paper**: LaTeX (IEEE Access template), Obsidian for notes
- **Package Manager**: uv (preferred) 
- **Virtual Environment**: `.venv/` (uv)



## Project Structure
thesis-maize-transfer/
├── CLAUDE.md                    # This file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .claude/
│   ├── skills/                  # Custom skills (4 total)
│   ├── agents/                  # Sub-agent definitions
│   └── hooks/                   # Pre/post hooks
├── data/
│   ├── raw/                     # Original downloads (DO NOT MODIFY)
│   ├── processed/               # Cleaned data ready for training
│   └── README.md                # Data provenance
├── notebooks/                   # Jupyter notebooks for exploration
├── src/
│   ├── data/                    # Data loading, preprocessing
│   ├── models/                  # Model architectures
│   ├── training/                # Training loops
│   ├── transfer/                # Transfer learning logic
│   └── utils/
├── experiments/
│   ├── configs/                 # YAML configs per experiment
│   ├── logs/
│   └── checkpoints/
├── paper/
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── references.bib
└── tests/                       # Unit tests

## Critical Design Decisions

1. **Spatial Resolution**: USA county-level + ASEAN province-level (mixed resolution by design — this IS the data sparsity gap)
2. **Time Period**: 2003-2023 (21 years, MODIS stable since 2003)
3. **Cropping Season**: Annual aggregate only (no seasonal split)
4. **Architecture**: CNN-LSTM (You et al. 2017) baseline → Transformer improvement
5. **Source Domain**: USA only for now (Brazil & China reserved for future iterations)
6. **Reference Implementation**: Fork from gabrieltseng/pycrop-yield-prediction (PyTorch)

## Success Metrics

- **Primary**: R² and RMSE on ASEAN test sets (2022-2023)
- **Secondary**: % improvement of transfer learning vs from-scratch baseline
- **Tertiary**: Per-country generalization analysis

### Target Performance (Realistic)
- USA baseline: R² ≥ 0.6, RMSE < 15 bu/acre
- ASEAN from-scratch: R² ≥ 0.3 (data-limited expected)
- ASEAN with transfer: R² improvement of 10-30% over from-scratch

## Coding Standards

- **Style**: Black formatter, line length 100
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google-style for all functions
- **Imports**: Absolute imports from `src/`
- **Config**: YAML files in `experiments/configs/`, no hardcoded paths
- **Reproducibility**: Set seeds (numpy, torch, random) at start of every script
- **Logging**: Use `logging` module, not `print()`, except in notebooks
- **Always activate venv before running code**: `source .venv/bin/activate` (uv) 


## Git Workflow

- **Branches**: `master` (stable) + `dev` (active work) + feature branches as needed
- **Commits**: Conventional commits format: `feat:`, `fix:`, `docs:`, `exp:`, `paper:`
- **Never commit**: Raw data >50MB, model checkpoints, .env files, credentials
- **Always commit**: Code, configs, processed data <50MB, paper drafts

## Important Patterns to Follow

### When implementing new code
1. Read existing related code first
2. Check if a relevant skill exists in `.claude/skills/`
3. Write code that matches existing patterns
4. Add unit tests for non-trivial logic
5. Update relevant config files
6. Update CLAUDE.md if architectural decisions change

### When running experiments
1. Create YAML config in `experiments/configs/`
2. Initialize wandb run with descriptive name: `{source}_{target}_{method}_{date}`
3. Log all hyperparameters
4. Save checkpoint every epoch
5. After completion, log results to `experiments/logs/results.csv`

### When stuck (>30 min on same issue)
1. STOP coding
2. Document the problem clearly in Obsidian notes
3. Discuss with claude.ai web (strategic decision-making) or supervisor
4. Don't keep trying random things — that wastes hours

## Key Datasets & References

### Datasets
- **USA Yield**: https://quickstats.nass.usda.gov/ (USDA NASS)
- **Indonesia Yield**: https://www.bps.go.id/
- **Vietnam Yield**: https://www.gso.gov.vn/en/agriculture-forestry-and-fishery/
- **Thailand Yield**: http://www.oae.go.th/
- **Boundaries**: https://gadm.org/
- **MODIS**: GEE assets `MODIS/061/MOD09A1`, `MODIS/061/MYD11A2`
- **Cropland Mask**: MapSPAM 2020 or MODIS LC

### Key Papers (must read)
1. **You et al. 2017** "Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data" — AAAI (foundational)
2. **Wang et al. 2018** "Deep Transfer Learning for Crop Yield Prediction with Remote Sensing Data" — COMPASS
3. **Khaki et al. 2021** "Simultaneous corn and soybean yield prediction from remote sensing data using deep transfer learning" — Sci. Rep.
4. **Zhang et al. 2025** "Transfer learning for improved crop yield predictions in a cross-scale pathway" — Remote Sensing of Environment

### Reference Code
- **Primary**: https://github.com/gabrieltseng/pycrop-yield-prediction (PyTorch)
- **Secondary**: https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction (TensorFlow, for reference)

## Decision Log

When making non-trivial decisions, append to this section with date and rationale.

- **[YYYY-MM-DD]** Source country: USA chosen over Brazil despite higher domain gap risk. Rationale: data cleanliness, paper precedent, supervisor preference.
- **[YYYY-MM-DD]** Reframed thesis title: "Data-Sparse" → "Data-Limited" to be more accurate.

## Out-of-Bounds Behaviors for Claude Code

When working on this project, Claude Code MUST:

- **NOT** generate full paper sections without my explicit prompt — paper writing requires my own voice
- **NOT** make architectural changes without updating CLAUDE.md decision log
- **NOT** download large datasets (>1GB) without confirming with me first
- **NOT** push to GitHub without my review of the diff
- **NOT** modify files in `data/raw/` (immutable)
- **NOT** add dependencies to `requirements.txt` without justification
- **ASK** before running expensive operations (training >10 min, GEE exports, large downloads)
- **READ** existing code patterns before generating new code
- **CHECK** if a relevant skill in `.claude/skills/` exists before starting a task

## Status & Progress

Update this section weekly.

### Current Week: [Week 1 / 12]
### Current Phase: [Foundation / Baseline / Transfer / Writing]
### Last Update: [7 May 2026]
### Blockers: [List any]
### Next Milestone: [Description + date]