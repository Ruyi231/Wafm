from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert not model.use_wavelet_flow_head
    assert model.wavelet_flow_head is None

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_legacy_mode_unchanged():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=10,
        use_wavelet_flow_head=True,
        wavelet_levels=2,
        wavelet_flow_bottleneck_dim=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # module_jit treats keyword booleans as dynamic values; training calls
    # return_metrics=True as a static literal inside the jitted train step.
    loss, metrics = model.compute_loss(key, obs, act, return_metrics=True)
    assert loss.shape == (batch_size, config.action_horizon)
    assert "loss_action_flow" in metrics
    assert "loss_wavelet_flow" in metrics
    assert model.wavelet_flow_impl == "legacy_head"

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


@pytest.mark.parametrize("num_steps", [1, 5, 10])
def test_subband_sampling_one_step(num_steps):
    key = jax.random.key(num_steps)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=10,
        use_wavelet_flow_head=True,
        wavelet_flow_impl="subband_flow",
        wavelet_levels=2,
        wavelet_flow_bottleneck_dim=16,
        wavelet_band_normalization=False,
        wavelet_hierarchical_coupling=True,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=num_steps)

    assert actions.shape == (1, config.action_horizon, config.action_dim)
    assert jax.numpy.all(jax.numpy.isfinite(actions))


def test_no_nan_forward_backward():
    key = jax.random.key(11)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=10,
        use_wavelet_flow_head=True,
        wavelet_flow_impl="subband_flow",
        wavelet_levels=2,
        wavelet_flow_bottleneck_dim=16,
        wavelet_band_normalization=False,
        wavelet_hierarchical_coupling=True,
        wavelet_use_action_reconstruction_loss=True,
        lambda_wavelet_recon_loss=0.25,
        wavelet_use_cross_band_consistency=True,
        wavelet_cross_band_consistency_weight=0.1,
    )
    model = config.create(key)
    obs, act = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    def scalar_loss(module):
        return jax.numpy.mean(module.compute_loss(key, obs, act))

    loss, grads = nnx.value_and_grad(scalar_loss)(model)

    assert jax.numpy.isfinite(loss)
    leaves = jax.tree.leaves(grads)
    assert leaves
    assert all(bool(jax.numpy.all(jax.numpy.isfinite(leaf))) for leaf in leaves)


@pytest.mark.parametrize(
    "override",
    [
        {"wavelet_band_norm_eps": float("nan")},
        {"wavelet_band_loss_weights": (1.0, float("nan"), 1.0)},
        {"wavelet_cross_band_consistency_weight": float("nan")},
        {
            "use_wavelet_flow_head": True,
            "wavelet_flow_impl": "subband_flow",
            "wavelet_use_action_reconstruction_loss": True,
            "lambda_wavelet_recon_loss": float("nan"),
        },
    ],
)
def test_nh_wafm_config_rejects_nonfinite_weights(override):
    with pytest.raises(ValueError, match="finite|positive"):
        pi0_config.Pi0Config(action_horizon=10, wavelet_levels=2, **override)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
