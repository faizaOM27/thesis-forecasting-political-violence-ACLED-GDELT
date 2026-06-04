# Predicting Political Violence in West Africa
## Does GDELT Improve Short-Term Forecasts Beyond ACLED?

Bachelor's thesis — BSc Data Science and Society  
University of Groningen, Campus Fryslân  
Author: Faiza Omar Mohamed  
Supervisor: Dr. Noman Haleem

---

## Overview

This repository contains the full code and data pipeline for a study on forecasting political violence in West Africa using negative binomial regression models.

The project compares:

- A baseline model using only conflict-history data (ACLED)
- An extended model including international media indicators from GDELT

The goal is to assess whether media attention improves short-term conflict forecasting performance.

---

## Research Question

Do international media attention variables derived from GDELT improve short-term forecasts of political violence in West Africa beyond what conflict history alone can predict?

---

## Study Scope

The analysis covers five West African countries:

- Burkina Faso
- Ghana
- Mali
- Niger
- Nigeria

Time period: January 2015 – December 2023.

---

## Repository structure

```text
├── Notebooks/
│   ├── 01_preprocess_acled.ipynb       # Load, filter, aggregate ACLED; engineer features
│   ├── 02_preprocess_gdelt.ipynb       # Load, clean, validate, and lag GDELT variables
│   ├── 03_merge_acled_gdelt.ipynb      # Merge ACLED and GDELT panels
│   ├── 04_baseline_model_acled.ipynb   # Baseline NB model (ACLED only)
│   ├── 05_final_model_merged.ipynb     # Extended model, comparison, and country breakdown
│   └── 06_robustness_check.ipynb       # Robustness check excluding Nigeria
│
├── processed/                          # Processed datasets produced by notebooks 01–03
├── raw_data/                           # Raw ACLED and GDELT input files
│
├── .gitignore
├── LICENSE
├── README.md                   
├── model_helpers.py                    # Shared functions for modelling and evaluation
└── requirements.txt                    # Python dependencies
```


---

## Reproducibility

This project is fully reproducible.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks
Run notebooks in order from 01 to 06.

Each notebook:

- loads the required input data
- performs the relevant processing or modelling steps
- saves outputs for subsequent stages of the analysis

---

## Data

### Raw Data
Raw datasets (ACLED and GDELT) are included in raw_data/.

### Processed Data
The `processed/` folder contains cleaned and preprocessed datasets that allow the modelling notebooks (04–06) to be run directly without re-running preprocessing.

⚠️ Note:
If notebooks are run from scratch, the processed files will be overwritten and regenerated.

---

## Outputs

Some notebooks save figures, tables, and model outputs to a `Results/` directory. If this directory does not already exist, it is created automatically when the notebooks are run.

The `Results/` directory is not required to reproduce the analysis and can be regenerated at any time.

---

## License

MIT License. Free to use, modify, and distribute with attribution.
