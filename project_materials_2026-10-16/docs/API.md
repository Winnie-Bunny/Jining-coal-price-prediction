# Asynchronous Forecast API

## Create a task

Send `POST /tasks` with the `Content-Type: application/json` header.

| Field | Type | Description |
| --- | --- | --- |
| `id` | Integer | `7`: next 7 days; `30`: next 30 days; `365`: next 365 days |
| `price_data` | Array | Daily historical records in ascending date order, without duplicate dates |
| `price_data[].date` | String | Date in `YYYY-MM-DD` format |
| `price_data[].history_price` | Number | Historical-price feature associated with that input record |
| `price_data[].<feature_code>` | Number | All 211 feature codes from [feature-map.json](feature-map.json); none may be omitted |

Weekly forecasts require at least 60 records, monthly forecasts at least 90, and yearly forecasts at least 364. The yearly model averages consecutive groups of 7 input records and uses the last 52 groups. Supply exactly 364 consecutive days of history to avoid an incomplete trailing week affecting the result. The API checks model ID, required fields, consecutive ISO dates, and finite numeric values. Annual requests require exactly 364 records. Requests over 8 MB are rejected.

The feature-map keys are English API codes. Its Chinese values match the original dataset and are retained for compatibility with existing preprocessing and model artifacts.

See `project_materials_2026-10-16/examples/request-week.json` for a complete executable example. To regenerate requests from the repository dataset:

```bash
python project_materials_2026-10-16/scripts/make_request.py --id 7 --output project_materials_2026-10-16/examples/request-week.json
python project_materials_2026-10-16/scripts/make_request.py --id 30 --output /tmp/request-month.json
python project_materials_2026-10-16/scripts/make_request.py --id 365 --output /tmp/request-year.json
```

A successful request returns HTTP 201:

```json
{"status": "running", "task_id": "<uuid>"}
```

An invalid model ID, too few records, or missing fields returns HTTP 400. Flask returns HTTP 415 for an unsupported content type and HTTP 400 for malformed JSON.

## Retrieve a task

Send `GET /tasks/<task_id>`. A successful lookup returns HTTP 200:

```json
{"status": "running", "task_id": "<uuid>", "result": {}}
```

On completion, `status` becomes `finished` and `result` contains `{"id":7,"data":[...]}`. Each prediction includes `date`, `predict_history_price`, `predict_price`, `factor_analysis`, and a `predict_<feature_code>` value for each feature. Prices are in CNY per metric tonne. Factor analysis contains absolute SHAP values sorted by descending contribution.

If processing fails, `status` becomes `error` and `result` contains `{"error":"<error description>"}`. An unknown task ID returns HTTP 404. Task state and results are persisted in SQLite under `STATE_DIR` (default: `inference/results`) and remain queryable after restart. In-flight tasks are marked failed at startup. Completed JSON files are stored in the same directory.

## Callbacks and operation

The original callback code is commented out, so the service does not currently send callbacks to external services. `inference/config/config.example.json` provides an example configuration structure. Copy it to `inference/config/config.json` and configure real addresses only if you restore callback support. Do not commit credentials.

Tasks run in daemon threads and are interrupted when the service exits. The server uses Waitress in a single process, allowing one concurrent forecast; additional submissions return HTTP 429. If API_KEY is set, both task endpoints require Authorization: Bearer <API_KEY>. REQUIRE_API_KEY=1 prevents public startup without a key. The dashboard is served at / and readiness at /health. Multi-process or multi-replica deployment requires a shared queue and database. See ../../deploy/README.md.


## Excel conversion and upload

The dashboard accepts `.xlsx`, `.xls` and `.json`. Both `POST /convert` and `POST /tasks` accept multipart form data with `id` (7, 30 or 365) and `file` (Excel). These endpoints use the same Bearer API key as JSON task requests.

`POST /convert` returns HTTP 200 with `{"request":{"id":7,"price_data":[...]},"summary":{"sheet":"数据","total_days":60,"used_days":60,"duplicates_removed":0,"start_date":"...","end_date":"..."}}`. It validates/converts only and does not create a forecast task. Send its `request` object to `POST /tasks`, or upload Excel directly to `/tasks` to receive the usual HTTP 201 task response. The dashboard's conversion button downloads the standard request JSON.

Use the worksheet named `数据`, otherwise the first worksheet. Row 1 must contain headers: 日期/date, 历史价格/history_price and all 211 factors. Chinese model labels, the original CSV aliases and English factor codes are accepted; column order is flexible. Extra target-price/comment columns are ignored. Duplicate columns mapping to the same field are rejected. Values must match the original model units; no factors or historical prices are inferred. Formula cells require previously computed cached values; the server does not evaluate formulas.

Dates may be native Excel dates or YYYY-MM-DD, YYYY/M/D, YYYY年M月D日 text. Rows are sorted, exact duplicate records are merged and conflicting duplicate dates are rejected. All supplied daily records must be consecutive. The converter selects the latest 60/90/364 days for the chosen horizon. Empty rows are skipped; missing values are rejected, never filled with zero. Max upload size is 8 MB (including multipart overhead), max 5000 data rows/512 columns, max expanded XLSX contents 32 MB. Parsing uses read-only XLSX access or on-demand XLS access and never saves raw uploads.

Blank customer template: `/static/downloads/coal-price-template.xlsx`. Its first sheet contains all 213 required headers with blank inputs; its second sheet provides Chinese filling instructions. The test fixture is historical project data and is not live market data.
