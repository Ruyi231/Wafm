import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import wavelet_flow_head


def test_haar_roundtrip():
    known = jnp.asarray([[[1.0], [3.0]]])
    known_approx, known_detail = wavelet_flow_head.haar_dwt_1d(known)
    assert jnp.allclose(known_approx, 2.0 * jnp.sqrt(2.0))
    assert jnp.allclose(known_detail, -jnp.sqrt(2.0))

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


def test_hierarchical_head_shapes():
    rng = jax.random.key(3)
    actions = jax.random.normal(rng, (2, 11, 7))
    tokens = jax.random.normal(jax.random.key(4), (2, 11, 32))
    approx, details, _ = wavelet_flow_head.multi_level_haar_dwt(actions, levels=3)
    head = wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead(
        action_dim=7,
        token_dim=32,
        action_horizon=11,
        levels=3,
        bottleneck_dim=16,
        hierarchical_coupling=True,
        detach_coarse_condition=False,
        conditioning_mode="temporal_pooling",
        rngs=nnx.Rngs(rng),
    )

    predicted_approx, predicted_details, info = head(
        approx,
        tuple(details),
        tokens,
        jnp.asarray([0.25, 0.75]),
    )

    assert predicted_approx.shape == approx.shape
    assert tuple(x.shape for x in predicted_details) == tuple(x.shape for x in details)
    assert info["wavelet_prediction_order"] == ("A", "D3", "D2", "D1")
    assert info["wavelet_gates"] is None


def test_hierarchical_coupling_and_detach_semantics():
    rng = jax.random.key(12)
    actions = jax.random.normal(rng, (1, 8, 3))
    tokens = jax.random.normal(jax.random.key(13), (1, 8, 8))
    approx, details, _ = wavelet_flow_head.multi_level_haar_dwt(actions, levels=2)
    head = wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead(
        action_dim=3,
        token_dim=8,
        action_horizon=8,
        levels=2,
        bottleneck_dim=12,
        hierarchical_coupling=True,
        detach_coarse_condition=False,
        conditioning_mode="temporal_pooling",
        rngs=nnx.Rngs(rng),
    )
    timestep = jnp.asarray([0.5])

    def fine_loss(approx_input):
        _, predicted_details, _ = head(approx_input, tuple(details), tokens, timestep)
        return jnp.mean(jnp.square(predicted_details[0]))

    coupled_grad = jax.grad(fine_loss)(approx)
    assert jnp.linalg.norm(coupled_grad) > 0

    head.detach_coarse_condition = True
    detached_grad = jax.grad(fine_loss)(approx)
    assert jnp.allclose(detached_grad, 0.0)

    head.detach_coarse_condition = False
    head.hierarchical_coupling = False
    independent_grad = jax.grad(fine_loss)(approx)
    assert jnp.allclose(independent_grad, 0.0)


def test_band_query_head_shapes_and_velocity_denormalization():
    rng = jax.random.key(5)
    actions = jax.random.normal(rng, (1, 10, 3))
    tokens = jax.random.normal(jax.random.key(6), (1, 10, 8))
    approx, details, _ = wavelet_flow_head.multi_level_haar_dwt(actions, levels=2)
    means = jnp.asarray([[10.0, -2.0, 3.0], [4.0, 5.0, 6.0], [-7.0, 8.0, 9.0]])
    stds = jnp.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    head = wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead(
        action_dim=3,
        token_dim=8,
        action_horizon=10,
        levels=2,
        bottleneck_dim=12,
        hierarchical_coupling=True,
        detach_coarse_condition=True,
        conditioning_mode="band_query",
        band_means=means,
        band_stds=stds,
        norm_eps=0.25,
        rngs=nnx.Rngs(rng),
    )

    predicted_approx, predicted_details, _ = head(
        approx,
        tuple(details),
        tokens,
        jnp.asarray([0.5]),
    )
    assert predicted_approx.shape == approx.shape
    assert tuple(x.shape for x in predicted_details) == tuple(x.shape for x in details)

    velocity_approx, velocity_details = head.denormalize_velocity_bands(
        jnp.ones_like(approx),
        tuple(jnp.ones_like(detail) for detail in details),
    )
    assert jnp.allclose(velocity_approx[0, 0], stds[0] + 0.25)
    assert jnp.allclose(velocity_details[0][0, 0], stds[1] + 0.25)
    assert jnp.allclose(velocity_details[1][0, 0], stds[2] + 0.25)


def test_subband_flow_target():
    data_approx = jnp.asarray([[[1.0, 2.0]]])
    data_details = (jnp.asarray([[[3.0, 4.0]]]),)
    noise_approx = jnp.asarray([[[5.0, 8.0]]])
    noise_details = (jnp.asarray([[[7.0, 10.0]]]),)
    state_approx, state_details, target_approx, target_details = wavelet_flow_head.subband_flow_bridge(
        data_approx,
        data_details,
        noise_approx,
        noise_details,
        jnp.asarray([0.25]),
    )

    assert jnp.allclose(state_approx, 0.75 * data_approx + 0.25 * noise_approx)
    assert jnp.allclose(state_details[0], 0.75 * data_details[0] + 0.25 * noise_details[0])
    assert jnp.array_equal(target_approx, noise_approx - data_approx)
    assert jnp.array_equal(target_details[0], noise_details[0] - data_details[0])


def test_shared_noise_physical_velocity_roundtrip():
    actions = jax.random.normal(jax.random.key(20), (1, 10, 3))
    _, padded_horizon, _, _ = wavelet_flow_head.wavelet_layout(10, 2)
    action_noise = jax.random.normal(jax.random.key(21), (1, padded_horizon, 3))
    data_approx, data_details, _ = wavelet_flow_head.multi_level_haar_dwt(actions, levels=2)
    noise_approx, noise_details, _ = wavelet_flow_head.multi_level_haar_dwt(action_noise, levels=2)
    means = jnp.asarray([[2.0, -3.0, 4.0], [1.0, 5.0, -2.0], [-4.0, 3.0, 0.5]])
    stds = jnp.asarray([[0.5, 2.0, 1.5], [3.0, 0.75, 2.5], [1.25, 4.0, 0.8]])
    head = wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead(
        action_dim=3,
        token_dim=8,
        action_horizon=10,
        levels=2,
        bottleneck_dim=12,
        hierarchical_coupling=False,
        detach_coarse_condition=False,
        conditioning_mode="temporal_pooling",
        band_means=means,
        band_stds=stds,
        norm_eps=1e-4,
        rngs=nnx.Rngs(jax.random.key(22)),
    )
    normalized_data = head.normalize_bands(data_approx, tuple(data_details))
    normalized_noise = head.normalize_bands(noise_approx, tuple(noise_details))
    _, _, target_approx, target_details = wavelet_flow_head.subband_flow_bridge(
        *normalized_data,
        *normalized_noise,
        jnp.asarray([0.5]),
    )
    physical_approx, physical_details = head.denormalize_velocity_bands(target_approx, target_details)
    physical_velocity = wavelet_flow_head.multi_level_haar_idwt(
        physical_approx,
        physical_details,
        target_length=actions.shape[1],
    )

    assert jnp.allclose(physical_velocity, action_noise[:, : actions.shape[1]] - actions, atol=1e-5)
