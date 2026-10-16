# Data and Models

## Historical dataset

`data/cci_jining_coal.csv` contains 1,827 daily records and 214 columns, covering January 1, 2021 through December 31, 2025. It includes one exact duplicate record for December 31, 2025, leaving 1,826 distinct dates. The request generator removes exact duplicates and rejects conflicting values for the same date. The original CSV contents remain unchanged.

The first column contains the date, followed by 211 external factors, the historical-price feature (`历史价格`), and the target price (`CCI济宁气煤(元/吨)`, CCI Jining gas coal price in CNY per metric tonne). The data source, license, missing-value treatment, and method used to construct the historical-price feature were not supplied with the original project and remain undocumented.

The API uses English feature codes, defined by the keys in [feature-map.json](feature-map.json). Original Chinese CSV headers and feature-map values are preserved for compatibility with existing preprocessing and model artifacts. Eleven mapped field names differ from their CSV headers in wording or parentheses. `project_materials_2026-10-16/scripts/make_request.py` handles these aliases explicitly without changing the source data.

## Model inventory and forecast horizons

Each inference horizon contains 211 required LSTM models, 422 per-feature scalers, and the shared files `input_StandardScaler.pkl`, `price_StandardScaler.pkl`, and `SVR_BG.pkl`. Additional historical files are retained. Models at the repository root and those under `inference/` are kept separately; they should not be assumed to be synchronized.

The weekly LSTMs use 60 daily records to forecast 7 days. The monthly LSTMs use 90 daily records to forecast 30 days. The yearly LSTMs use 52 weekly averages to forecast 52 weeks, then interpolate the output to 365 days. The SVR uses forecast factors and recursively updated historical-price inputs to predict the target price. SHAP explanations describe the corresponding SVR inputs.

## Integrity and reproducibility

[artifacts.sha256](artifacts.sha256) lists SHA-256 digests for the existing CSV, PKL, and PT files. Verify transfer integrity from the repository root:

```bash
shasum -a 256 -c project_materials_2026-10-16/docs/artifacts.sha256
```

Existing forecasts and logs do not establish reproducibility. The exact training environment, fixed random seeds, full data provenance, and independent backtest records are missing. Identical numerical results in a new environment and future forecast performance have not been established. Load only trusted model files: deserializing PKL/Joblib files can execute code.
