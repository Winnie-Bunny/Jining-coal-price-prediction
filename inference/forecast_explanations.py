"""Bound Kernel SHAP's synthetic matrix for small deployment instances.

These limits affect approximate factor explanations only. They do not change
LSTM outputs, SVR prices or recursively generated historical prices.
"""
import os

import numpy as np
import shap


def explain_factors(model, background, inputs):
    limit = int(os.environ.get('SHAP_BACKGROUND_LIMIT', '16'))
    samples = int(os.environ.get('SHAP_NSAMPLES', '256'))
    if not 1 <= limit <= 95 or not 1 <= samples <= 4096:
        raise ValueError('SHAP_BACKGROUND_LIMIT must be 1..95 and SHAP_NSAMPLES 1..4096')
    background = np.asarray(background)
    if len(background) > limit:
        indices = np.linspace(0, len(background) - 1, limit, dtype=int)
        background = background[indices]
    explainer = shap.KernelExplainer(model.predict, background)
    return explainer.shap_values(inputs, nsamples=samples, silent=True)
