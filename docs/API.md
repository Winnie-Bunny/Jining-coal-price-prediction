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

Weekly forecasts require at least 60 records, monthly forecasts at least 90, and yearly forecasts at least 364. The yearly model averages consecutive groups of 7 input records and uses the last 52 groups. Supply exactly 364 consecutive days of history to avoid an incomplete trailing week affecting the result. The API currently checks the model ID, record count, and required fields. Callers must ensure valid dates, numeric values, and continuous daily coverage.

The feature-map keys are English API codes. Its Chinese values match the original dataset and are retained for compatibility with existing preprocessing and model artifacts.

See `examples/request-week.json` for a complete executable example. To regenerate requests from the repository dataset:

```bash
python scripts/make_request.py --id 7 --output examples/request-week.json
python scripts/make_request.py --id 30 --output /tmp/request-month.json
python scripts/make_request.py --id 365 --output /tmp/request-year.json
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

If processing fails, `status` becomes `error` and `result` contains `{"error":"<error description>"}`. An unknown task ID returns HTTP 404. The lookup endpoint can access only the current process's in-memory tasks. Completed results from earlier runs can be read manually from `inference/results/<task_id>.json`.

## Callbacks and operation

The original callback code is commented out, so the service does not currently send callbacks to external services. `inference/config/config.example.json` provides an example configuration structure. Copy it to `inference/config/config.json` and configure real addresses only if you restore callback support. Do not commit credentials.

Tasks run in daemon threads and are interrupted when the service exits. The built-in Flask server is intended for local experiments. Public deployment requires authentication, access control, queueing, and persistent task state.
