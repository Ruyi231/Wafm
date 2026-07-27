import json

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import wavelet_normalization


def test_wavelet_norm_roundtrip(tmp_path):
    actions = np.random.default_rng(0).normal(size=(6, 11, 4)).astype(np.float32)
    stats = wavelet_normalization.compute_wavelet_norm_stats(
        (actions[:2], actions[2:]),
        levels=3,
        eps=1e-5,
        source_config="unit_test",
    )
    approx, details, actual_levels = wavelet_normalization.decompose_wavelet_bands(actions, levels=3)

    normalized_approx, normalized_details = wavelet_normalization.normalize_wavelet_bands(approx, details, stats)
    restored_approx, restored_details = wavelet_normalization.denormalize_wavelet_state_bands(
        normalized_approx, normalized_details, stats
    )
    restored_actions = wavelet_normalization.reconstruct_wavelet_bands(
        restored_approx,
        restored_details,
        target_length=actions.shape[1],
    )

    assert actual_levels == 3
    assert stats.means.shape == (actual_levels + 1, actions.shape[-1])
    assert stats.stds.shape == (actual_levels + 1, actions.shape[-1])
    np.testing.assert_allclose(restored_approx, approx, rtol=1e-5, atol=1e-5)
    for restored_detail, detail in zip(restored_details, details, strict=True):
        np.testing.assert_allclose(restored_detail, detail, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(restored_actions, actions, rtol=1e-5, atol=1e-5)

    stats_path = wavelet_normalization.save_wavelet_norm_stats(tmp_path / "stats.json", stats)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert list(payload["bands"]) == ["A_3", "D_3", "D_2", "D_1"]
    assert payload["metadata"]["input_action_normalization"] == "openpi_norm_stats"

    loaded = wavelet_normalization.load_wavelet_norm_stats(
        stats_path,
        expected_levels=3,
        expected_action_dim=4,
        expected_action_horizon=11,
        eps=1e-5,
    )
    np.testing.assert_allclose(loaded.means, stats.means)
    np.testing.assert_allclose(loaded.stds, stats.stds)


def test_wavelet_velocity_denormalization():
    stats = wavelet_normalization.WaveletNormStats(
        approx=wavelet_normalization.WaveletBandStats(
            mean=np.array([10.0, -5.0]),
            std=np.array([2.0, 3.0]),
            coefficient_count=2,
        ),
        details=(
            wavelet_normalization.WaveletBandStats(
                mean=np.array([20.0, 30.0]),
                std=np.array([4.0, 5.0]),
                coefficient_count=2,
            ),
        ),
        levels=1,
        requested_levels=1,
        action_dim=2,
        eps=0.25,
        action_horizon=4,
        padded_horizon=4,
        sample_count=1,
    )
    ones = jnp.ones((1, 2, 2), dtype=jnp.float32)

    velocity_approx, velocity_details = wavelet_normalization.denormalize_wavelet_velocity_bands(ones, (ones,), stats)
    state_approx, state_details = wavelet_normalization.denormalize_wavelet_state_bands(ones, (ones,), stats)

    np.testing.assert_allclose(
        velocity_approx,
        np.broadcast_to(np.array([2.25, 3.25]), velocity_approx.shape),
    )
    np.testing.assert_allclose(
        velocity_details[0],
        np.broadcast_to(np.array([4.25, 5.25]), velocity_details[0].shape),
    )
    np.testing.assert_allclose(
        state_approx,
        np.broadcast_to(np.array([12.25, -1.75]), state_approx.shape),
    )
    np.testing.assert_allclose(
        state_details[0],
        np.broadcast_to(np.array([24.25, 35.25]), state_details[0].shape),
    )


@pytest.mark.parametrize(
    ("horizon", "requested_levels", "expected_levels"),
    [
        (2, 3, 1),
        (5, 1, 1),
        (5, 3, 3),
        (10, 3, 3),
        (11, 2, 2),
    ],
)
def test_multilevel_wavelet_shapes(horizon, requested_levels, expected_levels):
    actions = np.random.default_rng(horizon).normal(size=(2, horizon, 3)).astype(np.float32)
    approx, details, actual_levels = wavelet_normalization.decompose_wavelet_bands(actions, requested_levels)
    padded_horizon = horizon + (-horizon) % (2**actual_levels)

    assert actual_levels == expected_levels
    assert approx.shape == (2, padded_horizon // (2**actual_levels), 3)
    assert len(details) == actual_levels
    for index, detail in enumerate(details, start=1):
        assert detail.shape == (2, padded_horizon // (2**index), 3)

    restored = wavelet_normalization.reconstruct_wavelet_bands(approx, details, target_length=horizon)
    assert restored.shape == actions.shape
    np.testing.assert_allclose(restored, actions, rtol=1e-5, atol=1e-5)


def test_near_zero_variance_is_epsilon_protected():
    actions = np.full((4, 6, 3), 2.0, dtype=np.float32)
    stats = wavelet_normalization.compute_wavelet_norm_stats(
        (actions,),
        levels=2,
        eps=1e-4,
    )
    approx, details, _ = wavelet_normalization.decompose_wavelet_bands(actions, levels=2)

    assert np.all(stats.stds <= 1e-12)
    normalized_approx, normalized_details = wavelet_normalization.normalize_wavelet_bands(approx, details, stats)
    assert bool(jnp.all(jnp.isfinite(normalized_approx)))
    assert all(bool(jnp.all(jnp.isfinite(detail))) for detail in normalized_details)

    restored_approx, restored_details = wavelet_normalization.denormalize_wavelet_state_bands(
        normalized_approx, normalized_details, stats
    )
    restored = wavelet_normalization.reconstruct_wavelet_bands(
        restored_approx, restored_details, target_length=actions.shape[1]
    )
    np.testing.assert_allclose(restored, actions, rtol=1e-5, atol=1e-5)


def test_identity_fallback_is_explicit_and_strict(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="explicitly enable identity fallback"):
        wavelet_normalization.load_wavelet_norm_stats(
            missing_path,
            expected_levels=2,
            expected_action_dim=3,
            eps=1e-5,
        )

    stats = wavelet_normalization.load_wavelet_norm_stats(
        missing_path,
        expected_levels=2,
        expected_action_dim=3,
        expected_action_horizon=5,
        eps=1e-5,
        allow_identity_fallback=True,
    )
    actions = np.random.default_rng(5).normal(size=(2, 5, 3)).astype(np.float32)
    approx, details, _ = wavelet_normalization.decompose_wavelet_bands(actions, levels=2)
    normalized_approx, normalized_details = wavelet_normalization.normalize_wavelet_bands(approx, details, stats)
    np.testing.assert_array_equal(normalized_approx, approx)
    for normalized_detail, detail in zip(normalized_details, details, strict=True):
        np.testing.assert_array_equal(normalized_detail, detail)

    stats_path = wavelet_normalization.save_wavelet_norm_stats(tmp_path / "identity.json", stats)
    with pytest.raises(ValueError, match="action_dim mismatch"):
        wavelet_normalization.load_wavelet_norm_stats(
            stats_path,
            expected_levels=2,
            expected_action_dim=4,
        )
