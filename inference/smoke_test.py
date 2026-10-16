"""Exercise actual model loading and a feature/SVR forward pass for every horizon."""
import numpy as np
from app import app, wk, mt, yr

for module, lookback in ((wk, 60), (mt, 90), (yr, 52)):
    assert len(module.lstm_models) == len(module.FEATURE_KEYS) == 211
    key = module.FEATURE_KEYS[0]
    prediction = module.lstm_models[key].predict(np.zeros((1, lookback, 1), dtype=np.float32))
    assert np.isfinite(prediction).all()
    price = module.svr_model.predict(np.zeros((1, 212)))
    assert np.isfinite(price).all()
assert app.test_client().get('/health').status_code == 200
print('All three model bundles loaded; LSTM/SVR forward passes and health check passed.')
