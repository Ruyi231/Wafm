"""Ungated WFM head sharing exactly the raw control's FiLM building block."""

import flax.nnx as nnx
import jax

from openpi.models import raw_velocity_head
from openpi.models import wavelet_flow_loss


class WaveletFiLMVelocityHead(nnx.Module):
    def __init__(self, *, action_dim: int, token_dim: int, bottleneck_dim: int, levels: int, rngs: nnx.Rngs):
        if levels < 1:
            raise ValueError("wavelet_levels must be positive")
        self.levels = levels
        self.approx_head = raw_velocity_head.FiLMVelocityExpert(action_dim, token_dim, bottleneck_dim, rngs=rngs)
        self.detail_heads = nnx.Dict(
            **{
                f"D{i}": raw_velocity_head.FiLMVelocityExpert(action_dim, token_dim, bottleneck_dim, rngs=rngs)
                for i in range(1, levels + 1)
            }
        )

    def __call__(self, noisy_actions: jax.Array, action_tokens: jax.Array) -> jax.Array:
        approx, details = wavelet_flow_loss.haar_decompose(noisy_actions, self.levels)
        cond_a, cond_d = wavelet_flow_loss.haar_decompose(action_tokens, self.levels)
        velocity_a = self.approx_head(approx, cond_a)
        velocity_d = [
            self.detail_heads[f"D{i + 1}"](detail, cond)
            for i, (detail, cond) in enumerate(zip(details, cond_d, strict=True))
        ]
        return wavelet_flow_loss.haar_reconstruct(velocity_a, velocity_d, noisy_actions.shape[1])
