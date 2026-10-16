"""Single-process forecast API and dashboard with durable task metadata."""
import datetime
from contextlib import contextmanager
import hmac
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import week2week.predict_price as wk
import month2month.predict_price as mt
import year2year.predict_price as yr

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
api = Api(app)
file_path = str(Path(__file__).resolve().parent)
tasks = {}
worker_slot = threading.BoundedSemaphore(1)
logging.basicConfig(level=logging.INFO)


def state_dir():
    folder = Path(os.environ.get('STATE_DIR', str(Path(file_path) / 'results')))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@contextmanager
def database():
    conn = sqlite3.connect(state_dir() / 'tasks.sqlite3', timeout=15)
    try:
        with conn:
            conn.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT NOT NULL)')
            yield conn
    finally:
        conn.close()


def save_task(task_id, status, result):
    result = convert_numpy(result)
    with database() as conn:
        conn.execute('INSERT OR REPLACE INTO tasks VALUES (?, ?, ?)',
                     (task_id, status, json.dumps(result, ensure_ascii=False, allow_nan=False)))
    tasks[task_id] = {'status': status, 'result': result}


def load_task(task_id):
    with database() as conn:
        row = conn.execute('SELECT status, result FROM tasks WHERE id = ?', (task_id,)).fetchone()
    return {'status': row[0], 'result': json.loads(row[1])} if row else None


def recover_interrupted_tasks():
    with database() as conn:
        conn.execute("UPDATE tasks SET status = 'error', result = ? WHERE status = 'running'",
                     (json.dumps({'error': 'Service restarted; please submit the forecast again.'}),))


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [convert_numpy(v) for v in obj]
    return obj


@app.before_request
def authenticate():
    if request.path == '/tasks' or request.path.startswith('/tasks/'):
        expected = os.environ.get('API_KEY', '')
        if expected and not hmac.compare_digest(request.headers.get('Authorization', '').encode('utf-8'), ('Bearer ' + expected).encode('utf-8')):
            return jsonify(message='A valid API key is required'), 401


@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    if request.path.startswith('/tasks'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
def dashboard():
    return app.send_static_file('index.html')


@app.get('/health')
def health():
    expected = {'week': wk, 'month': mt, 'year': yr}
    ready = all(len(module.lstm_models) == len(module.FEATURE_KEYS) for module in expected.values())
    return jsonify(status='ok' if ready else 'unavailable'), 200 if ready else 503


def run_algorithm(data, task_id):
    try:
        module = {7: wk, 30: mt, 365: yr}[data['id']]
        price_data, hist, date_last = module.get_data(data)
        result = convert_numpy({'id': data['id'], 'data': module.optimize_model(price_data, hist, date_last)})
        path = state_dir() / f'{task_id}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
        save_task(task_id, 'finished', result)
    except Exception:
        logging.exception('Forecast failed: %s', task_id)
        save_task(task_id, 'error', {'error': 'Forecast failed. Check server logs or contact the administrator.'})
    finally:
        worker_slot.release()


def validate_records(data):
    records = data.get('price_data')
    minimum = {7: 60, 30: 90, 365: 364}[data['id']]
    if not isinstance(records, list) or not minimum <= len(records) <= 5000:
        return f'price_data requires {minimum} to 5000 records'
    if data['id'] == 365 and len(records) != 364:
        return 'The annual model requires exactly 364 consecutive daily records'
    module = {7: wk, 30: mt, 365: yr}[data['id']]
    numeric = set(module.FEATURE_KEYS) | {'history_price'}
    previous = None
    for row in records:
        if not isinstance(row, dict) or not (numeric | {'date'}).issubset(row):
            return 'Records must include date, history_price, and all feature fields'
        try:
            current = datetime.date.fromisoformat(row['date'])
        except (ValueError, TypeError):
            return 'Dates must use YYYY-MM-DD'
        if current.isoformat() != row['date']:
            return 'Dates must use YYYY-MM-DD'
        if previous and (current - previous).days != 1:
            return 'Records must be consecutive daily observations in ascending order'
        previous = current
        for key in numeric:
            try:
                valid = type(row[key]) in (int, float) and math.isfinite(row[key])
            except OverflowError:
                valid = False
            if not valid:
                return f'{key} must be a finite number'
    return None


class TaskResource(Resource):
    def get(self, task_id):
        task = load_task(task_id)
        if task is None:
            return {'message': 'Task not found'}, 404
        return dict(task, task_id=task_id), 200


class TaskCreations(Resource):
    def post(self):
        data = request.get_json()
        if not isinstance(data, dict) or type(data.get('id')) is not int or data['id'] not in (7, 30, 365):
            return {'msg': 'id must be 7, 30, or 365'}, 400
        error = validate_records(data)
        if error:
            return {'msg': error}, 400
        if not worker_slot.acquire(blocking=False):
            return {'msg': 'A forecast is already running. Please try again later.'}, 429
        task_id = str(uuid.uuid4())
        try:
            save_task(task_id, 'running', {})
            threading.Thread(target=run_algorithm, args=(data, task_id), daemon=True).start()
        except Exception:
            worker_slot.release()
            logging.exception('Could not start task')
            return {'msg': 'Could not start the forecast'}, 503
        return {'status': 'running', 'task_id': task_id}, 201


api.add_resource(TaskResource, '/tasks/<string:task_id>')
api.add_resource(TaskCreations, '/tasks')

if __name__ == '__main__':
    from waitress import serve
    if os.environ.get('REQUIRE_API_KEY') == '1' and not os.environ.get('API_KEY'):
        raise RuntimeError('API_KEY must be configured for public deployment')
    recover_interrupted_tasks()
    serve(app, host=os.environ.get('HOST', '127.0.0.1'),
          port=int(os.environ.get('PORT', '7020')), threads=4)
