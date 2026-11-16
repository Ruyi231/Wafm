import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import raw_velocity_head
from openpi.models import wavelet_flow_loss
from openpi.models import wavelet_velocity_head


def test_capacity_matched_heads_and_gradients():
    raw = raw_velocity_head.RawFiLMVelocityHead(
        action_dim=32, token_dim=1024, bottleneck_dim=128, num_experts=3, rngs=nnx.Rngs(0)
    )
    wfm = wavelet_velocity_head.WaveletFiLMVelocityHead(
        action_dim=32, token_dim=1024, bottleneck_dim=128, levels=2, rngs=nnx.Rngs(0)
    )
    for head in (raw, wfm):
        assert sum(x.size for x in jax.tree.leaves(nnx.state(head, nnx.Param))) == 812_256

        def loss_fn(model):
            velocity = model(jnp.ones((2, 10, 32)), jnp.ones((2, 10, 1024)))
            assert velocity.shape == (2, 10, 32)
            return wavelet_flow_loss.output_wavelet_flow_loss(velocity, jnp.zeros_like(velocity), 2)

        loss, grads = nnx.value_and_grad(loss_fn)(head)
        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
        assert any(jnp.any(x != 0) for x in jax.tree.leaves(grads))
