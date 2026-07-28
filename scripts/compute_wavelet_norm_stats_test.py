import pathlib
import types

import pytest

from . import compute_wavelet_norm_stats


@pytest.mark.parametrize("norm_stats", [None, {"state": object()}])
def test_main_rejects_missing_action_norm_stats(monkeypatch, norm_stats):
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            wavelet_levels=2,
            wavelet_norm_stats_path="./assets/wavelet_norm_stats_l2.json",
        ),
        assets_dirs=pathlib.Path("assets/test"),
    )
    data_config = types.SimpleNamespace(
        norm_stats=norm_stats,
        repo_id="physical-intelligence/libero",
    )

    class _FakeDataLoader:
        def data_config(self):
            return data_config

        def __iter__(self):
            raise AssertionError("Missing action norm_stats must fail before dataset iteration")

    monkeypatch.setattr(compute_wavelet_norm_stats._config, "get_config", lambda _name: config)  # noqa: SLF001
    monkeypatch.setattr(
        compute_wavelet_norm_stats._data_loader,  # noqa: SLF001
        "create_data_loader",
        lambda *_args, **_kwargs: _FakeDataLoader(),
    )

    with pytest.raises(FileNotFoundError, match="did not load OpenPI action norm_stats"):
        compute_wavelet_norm_stats.main("plan1_subband_l2_norm", num_batches=1)


def test_resolve_output_path_prefers_model_configured_path():
    configured_path = pathlib.Path("./assets/shared/wavelet_norm_stats_l2.json")
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            wavelet_levels=2,
            wavelet_norm_stats_path=str(configured_path),
        ),
        assets_dirs=pathlib.Path("assets/config_name"),
    )
    data_config = types.SimpleNamespace(repo_id="physical-intelligence/libero")

    result = compute_wavelet_norm_stats._resolve_output_path(  # noqa: SLF001
        config,
        data_config,
        output_path=None,
        requested_levels=2,
    )

    assert result == configured_path


def test_resolve_output_path_preserves_config_assets_fallback():
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            wavelet_levels=2,
            wavelet_norm_stats_path=None,
        ),
        assets_dirs=pathlib.Path("assets/config_name"),
    )
    data_config = types.SimpleNamespace(repo_id="organization/dataset")

    result = compute_wavelet_norm_stats._resolve_output_path(  # noqa: SLF001
        config,
        data_config,
        output_path=None,
        requested_levels=2,
    )

    assert result == pathlib.Path("assets/config_name/organization/dataset/wavelet_norm_stats_l2.json")


def test_resolve_output_path_requires_explicit_path_for_levels_override():
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            wavelet_levels=2,
            wavelet_norm_stats_path="./assets/shared/wavelet_norm_stats_l2.json",
        ),
        assets_dirs=pathlib.Path("assets/config_name"),
    )
    data_config = types.SimpleNamespace(repo_id="physical-intelligence/libero")

    with pytest.raises(ValueError, match="Pass --output-path explicitly"):
        compute_wavelet_norm_stats._resolve_output_path(  # noqa: SLF001
            config,
            data_config,
            output_path=None,
            requested_levels=1,
        )
