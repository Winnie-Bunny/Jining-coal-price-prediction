"""Exercise actual Flask routes and worker lifecycle with deterministic model substitutes."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'inference'))
MATERIALS_ROOT = Path(__file__).resolve().parents[1]
HAS_DEPS = all(importlib.util.find_spec(n) for n in ('flask', 'flask_restful', 'numpy', 'pandas', 'requests'))


@unittest.skipUnless(HAS_DEPS, 'Install API dependencies to run Flask integration tests')
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_modules = {}
        mapping = json.loads((MATERIALS_ROOT / 'docs/feature-map.json').read_text())
        keys = list(mapping)
        for period in ('week2week', 'month2month', 'year2year'):
            package = types.ModuleType(period)
            module = types.ModuleType(f'{period}.predict_price')
            module.FEATURE_KEYS = keys
            module.FEATURE_MAP = mapping
            module.lstm_models = {key: object() for key in keys}
            module.get_data = lambda data: (None, None, None)
            module.optimize_model = lambda *args: [{'predict_price': 100.0}]
            package.predict_price = module
            fake_modules[period] = package
            fake_modules[module.__name__] = module
        spec = importlib.util.spec_from_file_location('cci_test_app', ROOT / 'inference/app.py')
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, fake_modules):
            spec.loader.exec_module(cls.module)
        cls.client = cls.module.app.test_client()

    def setUp(self):
        self.module.tasks.clear()
        self.folder = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(self.module, 'file_path', self.folder.name)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.folder.cleanup)

    def test_invalid_requests_do_not_create_tasks(self):
        for data in ({}, [], {'id': True}, {'id': '7'}, {'id': 8}, {'id': 7},
                     {'id': 7, 'price_data': [{}] * 60}):
            with self.subTest(data=str(data)[:80]):
                self.assertEqual(self.client.post('/tasks', json=data).status_code, 400)
        self.assertEqual(self.module.tasks, {})
        self.assertEqual(self.client.get('/tasks/unknown').status_code, 404)

    def wait_for_task(self, task_id):
        for _ in range(200):
            response = self.client.get(f'/tasks/{task_id}')
            body = response.get_json()
            if body['status'] != 'running':
                return body
            time.sleep(0.01)
        self.fail('worker did not finish')

    def test_success_writes_separate_task_results(self):
        request = json.loads((MATERIALS_ROOT / 'examples/request-week.json').read_text())
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'file_path', folder):
            ids = []
            for _ in range(2):
                response = self.client.post('/tasks', json=request)
                self.assertEqual(response.status_code, 201)
                task_id = response.get_json()['task_id']
                ids.append(task_id)
                body = self.wait_for_task(task_id)
                self.assertEqual(body['status'], 'finished')
                self.assertEqual(body['result']['data'][0]['predict_price'], 100.0)
                result_file = Path(folder) / 'results' / f'{task_id}.json'
                self.assertEqual(json.loads(result_file.read_text()), body['result'])
            self.assertNotEqual(*ids)

    def test_worker_failure_is_queryable(self):
        request = json.loads((MATERIALS_ROOT / 'examples/request-week.json').read_text())
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'file_path', folder), \
                patch.object(self.module.wk, 'get_data', side_effect=ValueError('model failure')):
            with self.assertLogs(level='ERROR'):
                response = self.client.post('/tasks', json=request)
                body = self.wait_for_task(response.get_json()['task_id'])
            self.assertEqual(body['status'], 'error')
            self.assertEqual(body['result'], {'error': 'Forecast failed. Check server logs or contact the administrator.'})

    def test_dashboard_health_and_authentication(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        with self.client.get('/static/app.js') as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/health').status_code, 200)
        with patch.dict('os.environ', {'API_KEY': 'test-secret'}):
            self.assertEqual(self.client.get('/tasks/unknown').status_code, 401)
            self.assertEqual(self.client.post('/tasks', json={}).status_code, 401)
            self.assertEqual(self.client.get('/tasks/unknown', headers={'Authorization': 'Bearer test-secret'}).status_code, 404)
            self.assertEqual(self.client.get('/health').status_code, 200)

    def test_persisted_results_and_restart_recovery(self):
        self.module.save_task('finished-task', 'finished', {'id': 7, 'data': []})
        self.module.save_task('interrupted-task', 'running', {})
        self.module.tasks.clear()
        self.module.recover_interrupted_tasks()
        finished = self.client.get('/tasks/finished-task').get_json()
        self.assertEqual(finished['status'], 'finished')
        self.assertEqual(finished['result']['id'], 7)
        interrupted = self.client.get('/tasks/interrupted-task').get_json()
        self.assertEqual(interrupted['status'], 'error')

    def test_invalid_dates_and_values_and_capacity(self):
        original = json.loads((MATERIALS_ROOT / 'examples/request-week.json').read_text())
        import copy
        for field, value in (('date', 'bad-date'), ('history_price', '100'), ('history_price', True), ('history_price', float('nan'))):
            data = copy.deepcopy(original)
            data['price_data'][0][field] = value
            self.assertEqual(self.client.post('/tasks', json=data).status_code, 400)
        data = copy.deepcopy(original)
        data['price_data'][1]['date'] = data['price_data'][0]['date']
        self.assertEqual(self.client.post('/tasks', json=data).status_code, 400)
        self.module.worker_slot.acquire()
        try:
            self.assertEqual(self.client.post('/tasks', json=original).status_code, 429)
        finally:
            self.module.worker_slot.release()

    def test_excel_conversion_and_forecast_upload(self):
        from io import BytesIO
        content = (MATERIALS_ROOT / 'tests/fixtures/week.xlsx').read_bytes()
        response = self.client.post('/convert', data={'id': '7', 'file': (BytesIO(content), 'customer.xlsx')})
        self.assertEqual(response.status_code, 200, response.get_json())
        converted = response.get_json()
        expected = json.loads((MATERIALS_ROOT / 'examples/request-week.json').read_text())
        self.assertEqual(converted['request'], expected)
        self.assertEqual(converted['summary']['used_days'], 60)
        with self.client.get('/static/downloads/coal-price-template.xlsx') as template:
            self.assertEqual(template.status_code, 200)
        response = self.client.post('/tasks', data={'id': '7', 'file': (BytesIO(content), 'customer.xlsx')})
        self.assertEqual(response.status_code, 201, response.get_json())
        self.assertEqual(self.wait_for_task(response.get_json()['task_id'])['status'], 'finished')

    def test_excel_errors_and_auth_do_not_create_tasks(self):
        from io import BytesIO
        for endpoint in ('/convert', '/tasks'):
            response = self.client.post(endpoint, data={'id':'7', 'file':(BytesIO(b'bad'), 'bad.xlsx')})
            self.assertEqual(response.status_code, 400)
            with patch.dict('os.environ', {'API_KEY':'test-key'}):
                response = self.client.post(endpoint, data={'id':'7', 'file':(BytesIO(b'bad'), 'bad.xlsx')})
                self.assertEqual(response.status_code, 401)
        self.assertEqual(self.module.tasks, {})


if __name__ == '__main__':
    unittest.main()
