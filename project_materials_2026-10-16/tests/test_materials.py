import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from make_request import make_request


class RequestMaterialTests(unittest.TestCase):
    def test_all_periods_have_complete_continuous_numeric_records(self):
        for model_id, count in ((7, 60), (30, 90), (365, 364)):
            with self.subTest(model_id=model_id):
                request = make_request(model_id)
                self.assertEqual(request['id'], model_id)
                self.assertEqual(len(request['price_data']), count)
                previous = None
                for row in request['price_data']:
                    self.assertEqual(len(row), 213)
                    current = datetime.date.fromisoformat(row['date'])
                    if previous:
                        self.assertEqual((current - previous).days, 1)
                    self.assertTrue(all(isinstance(v, float) for k, v in row.items() if k != 'date'))
                    previous = current


if __name__ == '__main__':
    unittest.main()
