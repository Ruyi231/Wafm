import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import raw_velocity_head


@pytest.mark.parametrize(
    ("bottleneck_dim", "num_experts", "expected_parameters"),
    [(384, 1, 812_192), (128, 3, 812_256)],
)
def test_raw_film_velocity_head_shape_and_parameter_count(
    bottleneck_dim: int,
    num_experts: int,
    expected_parameters: int,
):
    head = raw_velocity_head.RawFiLMVelocityHead(
        action_dim=32,
        token_dim=1024,
        bottleneck_dim=bottleneck_dim,
        num_experts=num_experts,
        rngs=nnx.Rngs(0),
    )
    noisy_actions = jnp.zeros((2, 10, 32), dtype=jnp.float32)
    action_tokens = jnp.zeros((2, 10, 1024), dtype=jnp.float32)

    output = head(noisy_actions, action_tokens)

    assert output.shape == noisy_actions.shape
    assert raw_velocity_head.raw_film_parameter_count(32, 1024, bottleneck_dim, num_experts) == expected_parameters
    actual_parameters = sum(leaf.size for leaf in jax.tree.leaves(nnx.state(head, nnx.Param)))
    assert actual_parameters == expected_parameters


@pytest.mark.parametrize(
    ("bottleneck_dim", "num_experts", "error_match"),
    [(0, 1, "bottleneck_dim must be positive"), (128, 0, "num_experts must be positive")],
)
def test_raw_film_velocity_head_rejects_invalid_dimensions(bottleneck_dim: int, num_experts: int, error_match: str):
    with pytest.raises(ValueError, match=error_match):
        raw_velocity_head.RawFiLMVelocityHead(
            action_dim=32,
            token_dim=1024,
            bottleneck_dim=bottleneck_dim,
            num_experts=num_experts,
            rngs=nnx.Rngs(0),
        )
