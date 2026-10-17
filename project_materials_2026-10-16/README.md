# Jining coal price prediction

A coal price forecasting project for Jining, Shandong, using LSTM models for input features and SVR for the target price. It includes weekly, monthly, and yearly forecasts, SHAP factor analysis, and an asynchronous Flask inference API. The target is the CCI Jining gas coal price, measured in CNY per metric tonne.

Read the [project report](REPORT.md) for the methodology, recorded experimental results, and limitations.

## Local workspace layout

Supplemental documents, examples, dependencies, and verification tools are collected in `project_materials_2026-10-16/`. Commands below assume the original project root as the working directory. The hosted GitHub repository uses this layout too. The active CI/CD workflow is `.github/workflows/ci-cd.yml` at the repository root.

## Repository contents

| Path | Purpose |
| --- | --- |
| `svr_price.py` | SVR grid search, evaluation, and feature explanations |
| `predict_week.py` / `predict_month.py` / `predict_year.py` | Feature LSTM training, forecasting, and evaluation |
| `model/` / `lstm_models/` | Existing trained models and scalers |
| `inference/app.py` | Asynchronous task API; default port: 7020 |
| `inference/{week2week,month2month,year2year}/` | Separate inference models for each forecast horizon |
| `data/cci_jining_coal.csv` | Existing historical dataset |
| `project_materials_2026-10-16/docs/` | API, data, validation, and model inventory documentation |
| `project_materials_2026-10-16/examples/request-week.json` | Complete weekly forecast request generated from the dataset |

## Setup and startup

Python 3.11 is recommended to match the deployed Docker runtime. GitHub Actions verified loading and forward passes for all three model bundles, and a complete seven-day forecast under a 512 MB memory limit. The original training environment was not provided.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r inference/requirements.txt
python project_materials_2026-10-16/scripts/check_project.py
python inference/app.py
```

Startup loads the models for all three horizons, with 211 feature LSTMs per horizon. Allow for memory use and initialization time. On Windows, activate the environment with `.venv\Scripts\activate`. The service listens on `127.0.0.1` by default; set the `HOST` and `PORT` environment variables to change the address.

```bash
curl -X POST http://127.0.0.1:7020/tasks \
  -H 'Content-Type: application/json' \
  --data-binary @project_materials_2026-10-16/examples/request-week.json
curl http://127.0.0.1:7020/tasks/<task_id>
```

See the [API reference](docs/API.md) and [data and model notes](docs/DATA.md) for details. SHAP analysis can take substantial time.

## Excel upload and JSON conversion

The [live dashboard](https://jining-coal-price.onrender.com/) accepts `.xlsx`, `.xls`, and JSON. Download the [blank Excel template](../inference/static/downloads/coal-price-template.xlsx), fill the `数据` sheet, enter the service API key, select a horizon, and submit. The separate conversion button downloads validated JSON without creating a forecast task.

Excel needs dates, historical prices, and all 211 factors. Weekly/monthly/yearly forecasts need at least 60/90/364 consecutive daily records; the converter sorts dates and selects the latest required records. Missing fields, invalid values, gaps, and conflicting dates are rejected. A price-only spreadsheet is insufficient for the existing model. See the [API reference](docs/API.md) for limits and multipart examples.

The Excel update passed 17 automated tests, exact conversion comparison against the original JSON request, and a real seven-day Excel forecast on the deployed service. Complete monthly/yearly forecasts and accuracy remain unverified.

## Docker

```bash
docker build -f inference/dockerfile -t cci-jining ./inference
docker run --rm -p 127.0.0.1:7020:7020 cci-jining
```

Docker builds and real-model smoke tests passed on GitHub Actions. Local Docker was unavailable; the container checks ran on GitHub.

## Training

Run the training scripts from the repository root. They overwrite their corresponding models and reports; back up any outputs you need to keep.

```bash
python svr_price.py
python predict_week.py
python predict_month.py
python predict_year.py
```

The scripts retain their original experiment dates and data splits. Review those settings before retraining. Models generated at the repository root are not automatically synchronized with `inference/`. For deployment, copy the SVR, scalers, and horizon-specific LSTMs together, preserving feature order.

## Language and compatibility

The deployed dashboard and Excel validation messages use Chinese; API feature codes remain English. The original Chinese CSV headers and model feature-map values are preserved because existing preprocessing and model artifacts depend on them. API requests use stable English feature codes, listed in [feature-map.json](docs/feature-map.json). The historical target column is `CCI济宁气煤(元/吨)`, and the historical-price input column is `历史价格`.

## Checks and limitations

```bash
python project_materials_2026-10-16/scripts/check_project.py
python -m unittest discover -s project_materials_2026-10-16/tests -v
```

The [validation record](docs/VALIDATION.md) separates completed checks from unverified behavior. Task state is now persisted in SQLite under `STATE_DIR` (default: `inference/results`). Interrupted running tasks are marked failed at startup. See [deployment instructions](../deploy/README.md) for the new dashboard, authentication, and CI/CD configuration. Each completed task also writes its result to `inference/results/<task_id>.json`. The service supports one concurrent forecast and optional Bearer authentication, required by the cloud configuration. Shared task queues and multiple replicas are not implemented.

Project attribution: WeiweiDeng. No open-source license has been added on behalf of the rights holders; see [NOTICE](NOTICE) for code, data, and model rights. Local Git exclusions are maintained in `.git/info/exclude`; the saved ignore template is `project_materials_2026-10-16/.gitignore`. Logs, generated reports, runtime configuration, and caches remain excluded locally.
