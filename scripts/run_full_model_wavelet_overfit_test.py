import argparse
import math

import pytest

from openpi.training import config as training_config
from scripts import run_full_model_wavelet_overfit


def _args(**overrides):
    values = {
        "steps": 10,
        "log_interval": 2,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "max_final_ratio": 0.5,
        "max_action_mse": 1e-3,
        "min_band_grad_norm": 1e-12,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_full_model_gate_accepts_plan1_normalized_subband_config():
    config = training_config.get_config("plan1_subband_l2_norm")
    run_full_model_wavelet_overfit._validate_config(config, _args())


def test_full_model_gate_rejects_non_subband_config():
    config = training_config.get_config("plan1_pi05_libero_baseline")
    with pytest.raises(ValueError, match="not an NH-WaFM"):
        run_full_model_wavelet_overfit._validate_config(config, _args())


def test_gate_ratios_cover_all_required_metrics():
    initial = {key: 2.0 for key in run_full_model_wavelet_overfit._GATE_METRICS}
    final = {key: 0.5 for key in run_full_model_wavelet_overfit._GATE_METRICS}
    ratios = run_full_model_wavelet_overfit._ratios(initial, final)

    assert ratios == {key: 0.25 for key in run_full_model_wavelet_overfit._GATE_METRICS}
    assert all(math.isfinite(value) for value in ratios.values())
