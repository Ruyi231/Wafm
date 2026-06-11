import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import wavelet_flow_head


def test_haar_dwt_idwt_reconstruction_even_and_odd_lengths():
    for time_len in (10, 11):
        x = jax.random.normal(jax.random.key(time_len), (2, time_len, 7))
        approx, details, actual_levels = wavelet_flow_head.multi_level_haar_dwt(x, levels=3)
        recon = wavelet_flow_head.multi_level_haar_idwt(approx, details, target_length=time_len)
        assert actual_levels <= 3
        assert recon.shape == x.shape
        assert jnp.max(jnp.abs(recon - x)) < 1e-5


def test_wavelet_subband_flow_head_shapes():
    rng = jax.random.key(0)
    noisy_actions = jax.random.normal(rng, (2, 10, 7))
    action_tokens = jax.random.normal(jax.random.key(1), (2, 10, 32))
    head = wavelet_flow_head.WaveletSubbandFlowHead(
        action_dim=7,
        token_dim=32,
        levels=2,
        bottleneck_dim=16,
        use_band_gate=True,
        rngs=nnx.Rngs(rng),
    )

    v_x, info = head(noisy_actions, action_tokens)
    z_approx, z_details, actual_levels = wavelet_flow_head.multi_level_haar_dwt(noisy_actions, levels=2)

    assert v_x.shape == noisy_actions.shape
    assert info["wavelet_flow_approx"].shape == z_approx.shape
    assert len(info["wavelet_flow_details"]) == len(z_details)
    assert info["wavelet_levels"] == actual_levels
    assert info["wavelet_gates"].shape == (2, actual_levels + 1)
    for v_detail, z_detail in zip(info["wavelet_flow_details"], z_details, strict=True):
        assert v_detail.shape == z_detail.shape
