import dataclasses
import logging

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.training import weight_loaders


def test_checkpoint_partial_load(monkeypatch, caplog):
    initialized = {
        "core": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "wavelet_flow_head": {"kernel": np.full((2, 1), 7.0, dtype=np.float32)},
        "expert": {"projection_lora_a": np.full((2, 1), 9.0, dtype=np.float32)},
    }
    checkpoint = {
        "core": {"kernel": np.ones((2, 2), dtype=np.float64)},
        "obsolete_head": {"kernel": np.ones((1,), dtype=np.float32)},
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(
        _model,
        "restore_params",
        lambda _path, *, restore_type: checkpoint,
    )

    with caplog.at_level(logging.WARNING):
        merged = weight_loaders.CheckpointWeightLoader("checkpoint/params").load(initialized)

    np.testing.assert_array_equal(merged["core"]["kernel"], checkpoint["core"]["kernel"])
    assert merged["core"]["kernel"].dtype == np.float32
    np.testing.assert_array_equal(
        merged["wavelet_flow_head"]["kernel"],
        initialized["wavelet_flow_head"]["kernel"],
    )
    np.testing.assert_array_equal(
        merged["expert"]["projection_lora_a"],
        initialized["expert"]["projection_lora_a"],
    )
    assert "obsolete_head" not in merged
    assert "wavelet_flow_head/kernel" in caplog.text
    assert "expert/projection_lora_a" in caplog.text
    assert "obsolete_head/kernel" in caplog.text


def test_merge_params_rejects_missing_non_optional():
    initialized = {
        "core": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "wavelet_flow_head": {"kernel": np.zeros((2, 1), dtype=np.float32)},
    }
    checkpoint = {"wavelet_flow_head": {"kernel": np.ones((2, 1), dtype=np.float32)}}

    with pytest.raises(ValueError, match=r"(?s)missing required parameters.*core/kernel"):
        weight_loaders._merge_params(  # noqa: SLF001
            checkpoint,
            initialized,
            missing_regex=r".*(lora|wavelet_flow_head).*",
        )


def test_merge_params_rejects_common_shape_mismatch():
    initialized = {"core": {"kernel": np.zeros((2, 2), dtype=np.float32)}}
    checkpoint = {"core": {"kernel": np.zeros((2, 3), dtype=np.float32)}}

    with pytest.raises(
        ValueError,
        match=r"(?s)shape mismatch.*core/kernel.*expected \(2, 2\), got \(2, 3\)",
    ):
        weight_loaders._merge_params(  # noqa: SLF001
            checkpoint,
            initialized,
            missing_regex=r".*lora.*",
        )


@dataclasses.dataclass(frozen=True)
class _TinyConfig(_model.BaseModelConfig):
    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    def create(self, rng):
        raise AssertionError("nnx.eval_shape is replaced in this unit test")

    def inputs_spec(self, *, batch_size: int = 1):
        raise NotImplementedError


class _FakeState:
    def __init__(self, initialized):
        self.initialized = initialized
        self.replaced = None

    def to_pure_dict(self):
        return self.initialized

    def replace_by_pure_dict(self, params):
        self.replaced = params


def test_base_model_config_load_preserves_only_optional_initialization(monkeypatch, caplog):
    initialized = {
        "core": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "wavelet_flow_head": {"kernel": np.full((2, 1), 3.0, dtype=np.float32)},
        "expert": {"projection_lora_b": np.full((1, 2), 5.0, dtype=np.float32)},
    }
    checkpoint = {
        "core": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "legacy_only": {"kernel": np.ones((1,), dtype=np.float32)},
    }
    state = _FakeState(initialized)
    monkeypatch.setattr(_model.nnx, "eval_shape", lambda create, rng: object())
    monkeypatch.setattr(_model.nnx, "split", lambda model: ("graphdef", state))
    monkeypatch.setattr(_model.nnx, "merge", lambda graphdef, new_state: (graphdef, new_state))

    config = _TinyConfig(action_dim=1, action_horizon=1, max_token_len=1)
    with caplog.at_level(logging.WARNING, logger="openpi"):
        result = config.load(checkpoint)

    assert result == ("graphdef", state)
    np.testing.assert_array_equal(state.replaced["core"]["kernel"], checkpoint["core"]["kernel"])
    np.testing.assert_array_equal(
        state.replaced["wavelet_flow_head"]["kernel"],
        initialized["wavelet_flow_head"]["kernel"],
    )
    np.testing.assert_array_equal(
        state.replaced["expert"]["projection_lora_b"],
        initialized["expert"]["projection_lora_b"],
    )
    assert "legacy_only" not in state.replaced
    assert "legacy_only/kernel" in caplog.text
    assert "wavelet_flow_head/kernel" in caplog.text
    assert "expert/projection_lora_b" in caplog.text
