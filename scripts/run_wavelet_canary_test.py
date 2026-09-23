import argparse

import pytest

from scripts import run_wavelet_canary


def _args(**overrides):
    values = {
        "steps": 1000,
        "canary_budget": 2000,
        "warmup_steps": 200,
        "batch_size": 2,
        "log_interval": 50,
        "save_interval": 1000,
        "peak_learning_rate": 5e-5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_canary_accepts_debug_short_warmup():
    run_wavelet_canary._validate_args(_args())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("canary_budget", 1000, "canary_budget"),
        ("canary_budget", 3001, "canary_budget"),
        ("warmup_steps", 501, "warmup_steps"),
        ("steps", 2001, "steps cannot exceed"),
    ],
)
def test_canary_rejects_out_of_scope_debug_budget(field, value, message):
    with pytest.raises(ValueError, match=message):
        run_wavelet_canary._validate_args(_args(**{field: value}))
