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

ROOT = Path(__file__).resolve().parents[1]
HAS_DEPS = all(importlib.util.find_spec(n) for n in ('flask', 'flask_restful', 'numpy', 'pandas', 'requests'))


@unittest.skipUnless(HAS_DEPS, 'Install API dependencies to run Flask integration tests')
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_modules = {}
        keys = list(json.loads((ROOT / 'docs/feature-map.json').read_text()))
        for period in ('week2week', 'month2month', 'year2year'):
            package = types.ModuleType(period)
            module = types.ModuleType(f'{period}.predict_price')
            module.FEATURE_KEYS = keys
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
        request = json.loads((ROOT / 'examples/request-week.json').read_text())
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
        request = json.loads((ROOT / 'examples/request-week.json').read_text())
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'file_path', folder), \
                patch.object(self.module.wk, 'get_data', side_effect=ValueError('model failure')):
            response = self.client.post('/tasks', json=request)
            body = self.wait_for_task(response.get_json()['task_id'])
            self.assertEqual(body['status'], 'error')
            self.assertEqual(body['result'], {'error': 'model failure'})


if __name__ == '__main__':
    unittest.main()
