"""Load real models; optionally verify a full weekly forecast in the container."""
import argparse
import json
from pathlib import Path

import numpy as np
from app import app, wk, mt, yr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--request', type=Path)
args = parser.parse_args()
for module, lookback in ((wk, 60), (mt, 90), (yr, 52)):
    assert len(module.lstm_models) == len(module.FEATURE_KEYS) == 211
    key = module.FEATURE_KEYS[0]
    prediction = module.lstm_models[key].predict(np.zeros((1, lookback, 1), dtype=np.float32))
    assert np.isfinite(prediction).all()
    price = module.svr_model.predict(np.zeros((1, 212)))
    assert np.isfinite(price).all()
assert app.test_client().get('/health').status_code == 200
print('All three model bundles loaded; LSTM/SVR forward passes and health check passed.')
if args.request:
    body = json.loads(args.request.read_text())
    assert body['id'] == 7, 'The complete CI forecast uses the weekly sample'
    price_data, history, last_date = wk.get_data(body)
    rows = wk.optimize_model(price_data, history, last_date)
    assert len(rows) == 7
    for row in rows:
        assert np.isfinite(row['predict_price'])
        assert len(row['factor_analysis']) == 212
        assert np.isfinite(list(row['factor_analysis'].values())).all()
    print('Complete real seven-day forecast and bounded SHAP analysis passed.')
