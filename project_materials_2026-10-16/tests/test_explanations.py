"""Verify explanation memory limits with a lightweight SHAP substitute."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class ExplanationTests(unittest.TestCase):
    def load_helper(self, mock_explainer):
        spec = importlib.util.spec_from_file_location('test_factor_helper', ROOT / 'inference/forecast_explanations.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'shap': types.SimpleNamespace(KernelExplainer=mock_explainer)}):
            spec.loader.exec_module(module)
        return module

    def test_bounds_background_and_sample_budget(self):
        explainer = Mock()
        explainer.shap_values.return_value = np.zeros((7, 212))
        factory = Mock(return_value=explainer)
        helper = self.load_helper(factory)
        model = Mock()
        original = np.arange(95 * 212).reshape(95, 212)
        inputs = np.zeros((7, 212))
        with patch.dict(os.environ, {'SHAP_BACKGROUND_LIMIT': '16', 'SHAP_NSAMPLES': '256'}):
            output = helper.explain_factors(model, original, inputs)
        passed = factory.call_args.args[1]
        self.assertEqual(passed.shape, (16, 212))
        np.testing.assert_array_equal(passed[0], original[0])
        np.testing.assert_array_equal(passed[-1], original[-1])
        self.assertEqual(explainer.shap_values.call_args.kwargs, {'nsamples': 256, 'silent': True})
        self.assertEqual(output.shape, (7, 212))

    def test_invalid_resource_limits_are_rejected(self):
        factory = Mock()
        helper = self.load_helper(factory)
        with patch.dict(os.environ, {'SHAP_BACKGROUND_LIMIT': '0'}):
            with self.assertRaises(ValueError):
                helper.explain_factors(Mock(), np.zeros((95, 212)), np.zeros((7, 212)))
        factory.assert_not_called()
