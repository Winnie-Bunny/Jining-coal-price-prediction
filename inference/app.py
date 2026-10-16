import warnings
import json
import uuid
import os
import requests
import threading
import pandas as pd
import numpy as np
from flask import Flask
from flask_restful import Api, Resource, request
import week2week.predict_price as wk
import month2month.predict_price as mt
import year2year.predict_price as yr
import logging.handlers
warnings.filterwarnings("ignore")


app = Flask(__name__)
api = Api(app)


file_path = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(file_path, 'log'), exist_ok=True)
path_log = os.path.join(file_path, 'log', 'app.log')
myapp = logging.getLogger()
myapp.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(asctime)s %(filename)s %(message)s')
file_handler = logging.handlers.TimedRotatingFileHandler(path_log, when='d', interval=1, backupCount=1)
file_handler.suffix = "%Y-%m-%d.log"  # Suffix for rotated log files.
file_handler.setFormatter(formatter)
myapp.addHandler(file_handler)

# In-memory task status store.
tasks = {}


def get_url():
    file_ = os.path.dirname(os.path.abspath(__file__))
    file_ = file_ + '/config/config.json'
    with open(file_, 'r') as file:
        url = json.load(file)
    callback = url["callback"]
    token = url["token"]
    return token, callback


def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def _run_wrapper(args_tuple):
    """Unpack the request and task ID passed to the worker thread."""
    data, task_id = args_tuple
    run_algorithm(data, task_id)


def run_algorithm(data, task_id):
    final_result = {}
    result_dir = os.path.join(file_path, 'results')
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f'{task_id}.json')
    # Read the request and select the forecasting model.
    try:
        model_id = data['id']
        final_result['id'] = model_id
        if model_id == 7:  # Weekly model.
            price_data, hist, date_last = wk.get_data(data)
            optimize_result = wk.optimize_model(price_data, hist, date_last)
        elif model_id == 30:  # Monthly model.
            price_data, hist, date_last = mt.get_data(data)
            optimize_result = mt.optimize_model(price_data, hist, date_last)
        elif model_id == 365:  # Annual model.
            price_data, hist, date_last = yr.get_data(data)
            optimize_result = yr.optimize_model(price_data, hist, date_last)
        else:
            raise ValueError(f"Invalid model id: {model_id}")
        # Include the model ID and forecasts in the response.
        final_result["data"] = optimize_result

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4, default=convert_numpy)
        logging.info(f"Model {model_id} results written to {path}")
        final_result = convert_numpy(final_result)  # Convert nested NumPy values for JSON serialization.
        tasks[task_id].update({
            "status": "finished",
            "result": final_result
        })

    except Exception as e:
        logging.exception("Background task failed")
        tasks[task_id].update({
            "status": "error",
            "result": {"error": str(e)}
        })

    # Optional callback integration (currently disabled).
    # body = json.dumps(final_result, ensure_ascii=False, default=convert_numpy)  # Callback payload.
    # url_token, url_callback = get_url()
    # token_message = requests.get(url_token)  # Fetch the access token.
    # token_str = token_message.text
    # token_value = json.loads(token_str)
    # token_result = token_value['result']
    # headers = {
    #     'x-access-token': token_result,
    #     'Content-Type': 'application/json'
    # }
    # x = requests.request(method="POST", url=url_callback, data=body, verify=False, headers=headers)
    # logging.info(x.text)
    # print("Callback response received")


# Task resource endpoints.
class TaskResource(Resource):
    def get(self, task_id):
        if task_id not in tasks:
            return {'message': 'Task not found'}, 404
        # Return the task ID together with its status and result.
        return {
            'status': tasks[task_id]['status'],
            'task_id': task_id,
            'result': tasks[task_id]['result']  # Empty until the task finishes.
        }, 200


class TaskCreations(Resource):
    def post(self):
        data = request.get_json()
        if not isinstance(data, dict) or type(data.get('id')) is not int or data['id'] not in (7, 30, 365):
            return {'msg': 'id must be 7, 30, or 365'}, 400
        records = data.get('price_data')
        minimum = {7: 60, 30: 90, 365: 364}[data['id']]
        if not isinstance(records, list) or len(records) < minimum:
            return {'msg': f'price_data requires at least {minimum} records in ascending date order'}, 400
        model_module = {7: wk, 30: mt, 365: yr}[data['id']]
        required = set(model_module.FEATURE_KEYS) | {'date', 'history_price'}
        if any(not isinstance(row, dict) or not required.issubset(row) for row in records):
            return {'msg': 'price_data records must include date, history_price, and all feature fields'}, 400
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "running", "result": {}}

        threading.Thread(target=_run_wrapper, args=((data, task_id),), daemon=True).start()
        logging.info(f"Task submitted: task_id={task_id}, status={tasks[task_id]['status']}")
        return {"status": "running", "task_id": task_id}, 201


api.add_resource(TaskResource, '/tasks/<string:task_id>')
api.add_resource(TaskCreations, '/tasks')

if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'), debug=False,
            port=int(os.environ.get('PORT', '7020')))

    # http://127.0.0.1:7020/tasks
