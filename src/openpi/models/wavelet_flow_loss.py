"""Haar transforms and a shared auxiliary objective on final velocity outputs."""

import math

import jax
import jax.numpy as jnp


def haar_decompose(x: jax.Array, levels: int) -> tuple[jax.Array, list[jax.Array]]:
    """Return A_L and [D1, ..., D_L], using one right-side edge extension.

    Cap depth as in the original WFM implementation. This avoids introducing
    arbitrary extra levels when a requested depth exceeds the input horizon.
    """
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("Expected a nonempty [batch, horizon, channels] array")
    if levels < 1:
        raise ValueError("wavelet_levels must be positive")
    actual_levels = min(levels, (x.shape[1] - 1).bit_length())
    pad = (-x.shape[1]) % (2**actual_levels)
    approx = jnp.pad(x, ((0, 0), (0, pad), (0, 0)), mode="edge") if pad else x
    details = []
    for _ in range(actual_levels):
        even, odd = approx[:, ::2], approx[:, 1::2]
        approx = (even + odd) / math.sqrt(2.0)
        details.append((even - odd) / math.sqrt(2.0))
    return approx, details


def haar_reconstruct(approx: jax.Array, details: list[jax.Array], horizon: int) -> jax.Array:
    """Invert the complete decomposition and crop to the external horizon."""
    for detail in reversed(details):
        even = (approx + detail) / math.sqrt(2.0)
        odd = (approx - detail) / math.sqrt(2.0)
        approx = jnp.stack((even, odd), axis=2).reshape(approx.shape[0], -1, approx.shape[-1])
    return approx[:, :horizon]


def output_wavelet_flow_loss(prediction: jax.Array, target: jax.Array, levels: int) -> jax.Array:
    """Sum per-band MSEs, applying the SAME transform to both final [B,T,A] arrays.

    This is not a global coefficient MSE: each band's mean has equal weight.
    For WFM, re-transform its cropped output rather than its internal padded
    coefficients. These differ at the boundary when T is not divisible by 2**L.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Velocity shapes differ: {prediction.shape} vs {target.shape}")
    pred_a, pred_d = haar_decompose(prediction.astype(jnp.float32), levels)
    true_a, true_d = haar_decompose(target.astype(jnp.float32), levels)
    loss = jnp.mean(jnp.square(pred_a - true_a))
    for pred, true in zip(pred_d, true_d, strict=True):
        loss = loss + jnp.mean(jnp.square(pred - true))
    return loss
