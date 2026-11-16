import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import wavelet_flow_loss


@pytest.mark.parametrize("horizon", [1, 10, 11, 16])
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_reconstruction_and_loss_against_independent_matrix(horizon, levels):
    rng = np.random.default_rng(7)
    pred = rng.normal(size=(2, horizon, 7)).astype(np.float32)
    target = rng.normal(size=pred.shape).astype(np.float32)
    actual_levels = min(levels, (horizon - 1).bit_length())
    padded_length = int(np.ceil(horizon / 2**actual_levels)) * 2**actual_levels
    matrix = np.eye(padded_length)
    bands = []
    for _ in range(actual_levels):
        bands.append((matrix[::2] - matrix[1::2]) / np.sqrt(2))
        matrix = (matrix[::2] + matrix[1::2]) / np.sqrt(2)
    padded_error = np.pad(pred - target, ((0, 0), (0, padded_length - horizon), (0, 0)), mode="edge")
    expected = sum(np.mean(np.einsum("nt,bta->bna", band, padded_error) ** 2) for band in [matrix, *bands])
    loss = jax.jit(wavelet_flow_loss.output_wavelet_flow_loss, static_argnums=2)(pred, target, levels)
    np.testing.assert_allclose(loss, expected, atol=2e-6, rtol=2e-6)
    approx, details = wavelet_flow_loss.haar_decompose(jnp.asarray(pred), levels)
    np.testing.assert_allclose(wavelet_flow_loss.haar_reconstruct(approx, details, horizon), pred, atol=1e-6)


def test_band_means_are_summed_and_loss_has_gradients():
    pred = jnp.ones((2, 10, 32))
    target = jnp.zeros_like(pred)
    # A2=2, details=0: summing band means gives 4, unlike a global MSE.
    loss, grad = jax.value_and_grad(wavelet_flow_loss.output_wavelet_flow_loss)(pred, target, 2)
    np.testing.assert_allclose(loss, 4, atol=1e-5)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0)
    np.testing.assert_allclose(wavelet_flow_loss.output_wavelet_flow_loss(pred, pred, 2), 0)


def test_final_output_supervision_excludes_discarded_internal_boundary():
    padded = jnp.zeros((1, 12, 1)).at[:, -1].set(1)
    approx, details = wavelet_flow_loss.haar_decompose(padded, 2)
    internal_loss = jnp.mean(approx**2) + sum(jnp.mean(d**2) for d in details)
    final = wavelet_flow_loss.haar_reconstruct(approx, details, 10)
    assert internal_loss > 0
    np.testing.assert_allclose(
        wavelet_flow_loss.output_wavelet_flow_loss(final, jnp.zeros_like(final), 2), 0, atol=1e-12
    )


@pytest.mark.parametrize("shape", [(1, 0, 3), (2, 3)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError, match="nonempty"):
        wavelet_flow_loss.haar_decompose(jnp.zeros(shape), 2)


def test_invalid_depth_and_mismatched_targets():
    with pytest.raises(ValueError, match="positive"):
        wavelet_flow_loss.haar_decompose(jnp.ones((1, 10, 3)), 0)
    with pytest.raises(ValueError, match="shapes differ"):
        wavelet_flow_loss.output_wavelet_flow_loss(jnp.ones((1, 10, 3)), jnp.ones((1, 11, 3)), 2)
