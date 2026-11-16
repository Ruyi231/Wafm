from __future__ import annotations

import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax.numpy as jnp

if TYPE_CHECKING:
    from openpi.shared import array_typing as at


def raw_film_parameter_count(
    action_dim: int,
    token_dim: int,
    bottleneck_dim: int,
    num_experts: int,
) -> int:
    """Returns the effective parameter count of a raw-domain FiLM head."""
    per_expert = (
        action_dim * bottleneck_dim
        + bottleneck_dim
        + token_dim * (2 * bottleneck_dim)
        + 2 * bottleneck_dim
        + bottleneck_dim * action_dim
        + action_dim
    )
    return num_experts * per_expert


class FiLMVelocityExpert(nnx.Module):
    """A raw-domain velocity expert conditioned by action-token features."""

    def __init__(self, action_dim: int, token_dim: int, bottleneck_dim: int, *, rngs: nnx.Rngs):
        self.state_proj = nnx.Linear(action_dim, bottleneck_dim, rngs=rngs)
        self.film_proj = nnx.Linear(token_dim, 2 * bottleneck_dim, rngs=rngs)
        self.out_proj = nnx.Linear(bottleneck_dim, action_dim, rngs=rngs)

    def __call__(
        self,
        noisy_actions: at.Float[at.Array, "b t a"],
        action_tokens: at.Float[at.Array, "b t h"],
    ) -> at.Float[at.Array, "b t a"]:
        state_features = self.state_proj(noisy_actions)
        gamma, beta = jnp.split(self.film_proj(action_tokens), 2, axis=-1)

        # Match the modulation scale used by the WFM FiLM predictors while staying entirely in the raw action domain.
        gamma = 0.1 * jnp.tanh(gamma)
        beta = 0.1 * beta
        hidden = nnx.gelu((1.0 + gamma) * state_features + beta)
        return self.out_proj(hidden)


class RawFiLMVelocityHead(nnx.Module):
    """Capacity-matched raw-domain velocity head with one or more independent experts."""

    def __init__(
        self,
        *,
        action_dim: int,
        token_dim: int,
        bottleneck_dim: int,
        num_experts: int,
        rngs: nnx.Rngs,
    ):
        if bottleneck_dim < 1:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        self.num_experts = num_experts
        self.experts = nnx.Dict(
            **{
                f"expert_{index}": FiLMVelocityExpert(action_dim, token_dim, bottleneck_dim, rngs=rngs)
                for index in range(num_experts)
            }
        )

    def __call__(
        self,
        noisy_actions: at.Float[at.Array, "b t a"],
        action_tokens: at.Float[at.Array, "b t h"],
    ) -> at.Float[at.Array, "b t a"]:
        expert_outputs = [
            self.experts[f"expert_{index}"](noisy_actions, action_tokens) for index in range(self.num_experts)
        ]
        # Variance-preserving fixed aggregation adds no trainable parameters.
        return jnp.sum(jnp.stack(expert_outputs, axis=0), axis=0) / math.sqrt(self.num_experts)
