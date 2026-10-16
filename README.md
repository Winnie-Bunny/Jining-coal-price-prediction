# Jining coal price prediction

A coal price forecasting project for Jining, Shandong, using LSTM models for input features and SVR for the target price. It includes weekly, monthly, and yearly forecasts, SHAP factor analysis, and an asynchronous Flask inference API. The target is the CCI Jining gas coal price, measured in CNY per metric tonne.

Read the [project report](REPORT.md) for the methodology, recorded experimental results, and limitations.

## Repository contents

| Path | Purpose |
| --- | --- |
| `svr_price.py` | SVR grid search, evaluation, and feature explanations |
| `predict_week.py` / `predict_month.py` / `predict_year.py` | Feature LSTM training, forecasting, and evaluation |
| `model/` / `lstm_models/` | Existing trained models and scalers |
| `inference/app.py` | Asynchronous task API; default port: 7020 |
| `inference/{week2week,month2month,year2year}/` | Separate inference models for each forecast horizon |
| `data/cci_jining_coal.csv` | Existing historical dataset |
| `docs/` | API, data, validation, and model inventory documentation |
| `examples/request-week.json` | Complete weekly forecast request generated from the dataset |

## Setup and startup

Python 3.11 or 3.12 is recommended. Dependency versions follow the original inference requirements. The exact environment used to train the historical models was not provided, so model loading compatibility remains unverified.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_project.py
python inference/app.py
```

Startup loads the models for all three horizons, with 211 feature LSTMs per horizon. Allow for memory use and initialization time. On Windows, activate the environment with `.venv\Scripts\activate`. The service listens on `127.0.0.1` by default; set the `HOST` and `PORT` environment variables to change the address.

```bash
curl -X POST http://127.0.0.1:7020/tasks \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request-week.json
curl http://127.0.0.1:7020/tasks/<task_id>
```

See the [API reference](docs/API.md) and [data and model notes](docs/DATA.md) for details. SHAP analysis can take substantial time.

## Docker

```bash
docker build -f inference/dockerfile -t cci-jining ./inference
docker run --rm -p 127.0.0.1:7020:7020 cci-jining
```

The Docker configuration is included, but a local image build and container startup have not been verified.

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

Documentation, project paths, and user-facing messages use English. The original Chinese CSV headers and model feature-map values are preserved because existing preprocessing and model artifacts depend on them. API requests use stable English feature codes, listed in [feature-map.json](docs/feature-map.json). The historical target column is `CCI济宁气煤(元/吨)`, and the historical-price input column is `历史价格`.

## Checks and limitations

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
```

The [validation record](docs/VALIDATION.md) separates completed checks from unverified behavior. Task state exists only in memory and is lost on restart. Each completed task also writes its result to `inference/results/<task_id>.json`. The service currently supports local experiments; authentication, a task queue, and shared state across processes are not implemented.

Project attribution: WeiweiDeng. No open-source license has been added on behalf of the rights holders; see [NOTICE](NOTICE) for code, data, and model rights. Original logs, generated reports, runtime configuration, and caches are excluded by `.gitignore` while local copies are retained.
