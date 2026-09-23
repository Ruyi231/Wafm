import dataclasses

import jax.numpy as jnp
import pytest

from scripts import run_wavelet_overfit


def test_make_head_and_batch_accepts_fixed_actions_and_stats():
    config = run_wavelet_overfit.OverfitConfig(
        steps=1,
        batch_size=2,
        action_horizon=8,
        action_dim=3,
        token_dim=4,
        levels=2,
        bottleneck_dim=8,
    )
    actions = run_wavelet_overfit._synthetic_actions(config)
    approx, details, _ = run_wavelet_overfit.wavelet_flow_head.multi_level_haar_dwt(actions, config.levels)
    means, stds = run_wavelet_overfit._band_stats(approx, tuple(details), config.norm_eps)

    head, batch = run_wavelet_overfit._make_head_and_batch(
        config,
        actions=actions,
        band_means=means,
        band_stds=stds,
    )

    assert head.action_horizon == 8
    assert batch.state_approx.shape[0] == 2
    assert len(batch.state_details) == 2


def test_make_head_and_batch_rejects_mismatched_fixed_actions():
    config = run_wavelet_overfit.OverfitConfig(batch_size=2, action_horizon=8, action_dim=3)
    wrong_actions = jnp.zeros((1, 8, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Fixed actions must have shape"):
        run_wavelet_overfit._make_head_and_batch(config, actions=wrong_actions)


def test_result_payload_records_real_provenance(tmp_path):
    config = dataclasses.replace(
        run_wavelet_overfit.OverfitConfig(),
        output_dir=tmp_path,
        steps=1,
        log_interval=1,
        batch_size=2,
        action_horizon=8,
        action_dim=2,
        token_dim=4,
        bottleneck_dim=8,
        require_improvement=False,
    )
    result = run_wavelet_overfit.run_overfit(
        config,
        provenance={"data_source": "fixed_real_lerobot_batch", "fixed_actions_sha256": "abc"},
    )

    payload = __import__("json").loads(result["json_path"].read_text())
    assert payload["provenance"]["data_source"] == "fixed_real_lerobot_batch"
    assert payload["provenance"]["fixed_actions_sha256"] == "abc"
