"""Verify a live deployment; optionally submit a real weekly forecast.

The API key is read with getpass and is never printed or saved.
"""
import argparse
import datetime
import getpass
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', required=True)
parser.add_argument('--request', type=Path)
args = parser.parse_args()
base = args.url.rstrip('/')
if not base.startswith('https://'):
    raise SystemExit('Use the HTTPS deployment URL')


def get(path, key=None, data=None):
    headers = {}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data).encode()
    request = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.read()


status, _ = get('/')
assert status == 200
status, payload = get('/health')
assert status == 200 and json.loads(payload)['status'] == 'ok'
try:
    get('/tasks/nonexistent')
    raise AssertionError('Task access was not protected')
except urllib.error.HTTPError as error:
    assert error.code == 401, error.code
print('HTTPS dashboard, model readiness and unauthorized access checks passed.', flush=True)
if args.request:
    body = json.loads(args.request.read_text())
    key = getpass.getpass('API key (hidden): ')
    status, payload = get('/tasks', key, body)
    assert status == 201
    task_id = json.loads(payload)['task_id']
    print(f'Real forecast submitted: {task_id}; waiting for completion.', flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        _, payload = get('/tasks/' + task_id, key)
        result = json.loads(payload)
        if result['status'] == 'error':
            raise SystemExit('Forecast reported an error; inspect service logs.')
        if result['status'] == 'finished':
            rows = result['result']['data']
            assert len(rows) == body['id']
            previous = datetime.date.fromisoformat(body['price_data'][-1]['date'])
            for row in rows:
                current = datetime.date.fromisoformat(row['date'])
                assert (current - previous).days == 1
                assert math.isfinite(row['predict_price'])
                assert row['factor_analysis'] and all(math.isfinite(v) for v in row['factor_analysis'].values())
                previous = current
            print(f'Real forecast completed: {len(rows)} consecutive dates, finite prices and SHAP factors.', flush=True)
            break
        time.sleep(5)
    else:
        raise SystemExit('Forecast still running after 10 minutes; inspect service task status.')
