import dataclasses
import json
import pathlib

from openpi.training import config as _config


_STAGE1_CONFIGS = (
    "plan1_pi05_libero_baseline",
    "plan1_legacy_wafm_l2",
    "plan1_subband_l2_no_norm",
    "plan1_subband_l2_norm",
)


def test_plan1_stage1_configs_share_training_setup():
    configs = [_config.get_config(name) for name in _STAGE1_CONFIGS]
    baseline = configs[0]

    for field in dataclasses.fields(baseline):
        if field.name in {"name", "model"}:
            continue
        values = [getattr(config, field.name) for config in configs]
        if field.name == "freeze_filter":
            assert all(type(value) is type(values[0]) and repr(value) == repr(values[0]) for value in values)
        else:
            assert all(value == values[0] for value in values)

    for config in configs:
        assert isinstance(config.data, _config.LeRobotLiberoDataConfig)
        assert config.data.repo_id == "physical-intelligence/libero"
        assert config.data.assets == _config.AssetsConfig(
            assets_dir="./assets/pi05_libero",
            asset_id="physical-intelligence/libero",
        )
        assert config.data.base_config is not None
        assert config.data.base_config.prompt_from_task
        assert not config.data.extra_delta_transform

        assert config.batch_size == 256
        assert config.ema_decay == 0.999
        assert config.seed == 42
        assert config.num_train_steps == 30_000

    for field in dataclasses.fields(baseline.model):
        if (
            field.name == "use_wavelet_flow_head"
            or field.name.startswith("wavelet_")
            or field.name.startswith("lambda_wavelet_")
        ):
            continue
        assert all(getattr(config.model, field.name) == getattr(baseline.model, field.name) for config in configs)


def test_plan1_stage1_model_ablation_boundaries():
    baseline, legacy, subband_no_norm, subband_norm = (
        _config.get_config(name).model for name in _STAGE1_CONFIGS
    )

    assert not baseline.use_wavelet_flow_head

    assert legacy.use_wavelet_flow_head
    assert legacy.wavelet_flow_impl == "legacy_head"
    assert legacy.wavelet_levels == 2
    assert not legacy.wavelet_use_band_gate

    for model in (subband_no_norm, subband_norm):
        assert model.use_wavelet_flow_head
        assert model.wavelet_flow_impl == "subband_flow"
        assert model.wavelet_levels == 2
        assert not model.wavelet_hierarchical_coupling
        assert not model.wavelet_shared_noise
        assert model.wavelet_conditioning_mode == "temporal_pooling"
        assert not model.wavelet_use_band_gate
        assert model.lambda_wavelet_flow_loss == 0.0
        assert not model.wavelet_use_action_reconstruction_loss
        assert not model.wavelet_use_cross_band_consistency

    assert not subband_no_norm.wavelet_band_normalization
    assert subband_no_norm.wavelet_norm_stats_path is None

    assert subband_norm.wavelet_band_normalization
    assert subband_norm.wavelet_norm_stats_fallback == "error"
    assert (
        subband_norm.wavelet_norm_stats_path
        == "./assets/pi05_libero/physical-intelligence/libero/wavelet_norm_stats_l2.json"
    )
    subband_differences = {
        field.name
        for field in dataclasses.fields(subband_no_norm)
        if getattr(subband_no_norm, field.name) != getattr(subband_norm, field.name)
    }
    assert subband_differences == {
        "wavelet_band_normalization",
        "wavelet_norm_stats_path",
    }


def test_plan1_manifest_matches_registered_stage1_configs():
    manifest_path = pathlib.Path(__file__).parents[3] / "configs" / "plan1_experiments.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage1 = next(stage for stage in manifest["stages"] if stage["stage"] == 1)
    registered = {
        experiment["experiment_id"]: experiment["train_config"]
        for experiment in stage1["experiments"]
        if experiment["registered"]
    }

    assert registered == {
        "p1_pi05_baseline": "plan1_pi05_libero_baseline",
        "p1_legacy_wafm": "plan1_legacy_wafm_l2",
        "p1_subband_no_norm": "plan1_subband_l2_no_norm",
        "p1_subband_with_norm": "plan1_subband_l2_norm",
    }
    runtime = manifest["runtime_integration"]
    assert runtime["stage1_train_configs"] == registered
    assert runtime["shared_action_norm_stats_dir"] == "./assets/pi05_libero/physical-intelligence/libero"
    assert (
        runtime["wavelet_norm_stats_l2_path"]
        == _config.get_config("plan1_subband_l2_norm").model.wavelet_norm_stats_path
    )
