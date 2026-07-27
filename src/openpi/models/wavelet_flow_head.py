from collections.abc import Sequence
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at

# _SQRT2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))
_SQRT2 = math.sqrt(2.0)


def _pad_time_to_even(x: at.Array) -> at.Array:
    """Fallback padding for one DWT level; multilevel DWT pads once at entry."""
    time_len = x.shape[1]
    if time_len % 2 == 0:
        return x
    return jnp.pad(x, ((0, 0), (0, 1), (0, 0)), mode="edge")


@at.typecheck
def haar_dwt_1d(
    x: at.Float[at.Array, "b t a"],
) -> tuple[at.Float[at.Array, "b th a"], at.Float[at.Array, "b th a"]]:
    """对 action horizon 维做一级正交 Haar-DWT。"""
    x = _pad_time_to_even(x)
    even = x[:, 0::2, :]
    odd = x[:, 1::2, :]
    approx = (even + odd) / _SQRT2
    detail = (even - odd) / _SQRT2
    return approx, detail


@at.typecheck
def haar_idwt_1d(
    approx: at.Float[at.Array, "b th a"],
    detail: at.Float[at.Array, "b th a"],
) -> at.Float[at.Array, "b t a"]:
    """Apply one Haar IDWT level by interleaving even and odd samples."""
    even = (approx + detail) / _SQRT2
    odd = (approx - detail) / _SQRT2
    return jnp.stack([even, odd], axis=2).reshape(approx.shape[0], approx.shape[1] * 2, approx.shape[2])


def multi_level_haar_dwt(
    x: at.Float[at.Array, "b t a"], levels: int
) -> tuple[at.Float[at.Array, "b tl a"], list[at.Float[at.Array, "b td a"]], int]:
    """Return A_L, [D1, ..., D_L], and the effective number of Haar levels."""
    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}")
    if x.shape[1] < 2 or levels == 0:
        return x, [], 0

    actual_levels = min(levels, max(0, (x.shape[1] - 1).bit_length()))
    multiple = 2**actual_levels
    pad_len = (-x.shape[1]) % multiple
    approx = x
    if pad_len:
        # Edge-pad the full action chunk once; IDWT crops back to target_length.
        approx = jnp.pad(approx, ((0, 0), (0, pad_len), (0, 0)), mode="edge")

    details = []
    completed_levels = 0
    for _ in range(actual_levels):
        if approx.shape[1] < 2:
            break
        approx, detail = haar_dwt_1d(approx)
        details.append(detail)
        completed_levels += 1
    return approx, details, completed_levels


def multi_level_haar_idwt(
    approx: at.Float[at.Array, "b tl a"],
    details: Sequence[at.Float[at.Array, "b td a"]],
    target_length: int | None = None,
) -> at.Float[at.Array, "b t a"]:
    """Reconstruct [D1, ..., D_L] from coarse to fine and optionally crop."""
    x = approx
    for detail in reversed(details):
        x = haar_idwt_1d(x, detail)
    if target_length is not None:
        x = x[:, :target_length, :]
    return x


def wavelet_layout(action_horizon: int, levels: int) -> tuple[int, int, int, tuple[int, ...]]:
    """Return effective levels, padded horizon, A_L length and [D1, ..., D_L] lengths."""
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}")
    effective_levels = min(levels, max(0, (action_horizon - 1).bit_length()))
    padded_horizon = ((action_horizon + 2**effective_levels - 1) // 2**effective_levels) * 2**effective_levels
    approx_length = padded_horizon // 2**effective_levels
    detail_lengths = tuple(padded_horizon // 2**level for level in range(1, effective_levels + 1))
    return effective_levels, padded_horizon, approx_length, detail_lengths


def _pool_tokens_to_length(
    action_tokens: at.Float[at.Array, "b t h"], *, padded_horizon: int, target_length: int
) -> at.Float[at.Array, "b ts h"]:
    """Average-pool action tokens to a wavelet scale without applying a fixed DWT."""
    if padded_horizon < action_tokens.shape[1]:
        raise ValueError(f"padded_horizon ({padded_horizon}) must cover token length ({action_tokens.shape[1]})")
    if padded_horizon % target_length != 0:
        raise ValueError(f"target_length {target_length} must divide padded_horizon {padded_horizon}")
    pad_len = padded_horizon - action_tokens.shape[1]
    padded = action_tokens
    if pad_len:
        padded = jnp.pad(padded, ((0, 0), (0, pad_len), (0, 0)), mode="edge")
    pool_width = padded_horizon // target_length
    return padded.reshape(padded.shape[0], target_length, pool_width, padded.shape[-1]).mean(axis=2)


def _resize_temporal(x: at.Array, target_length: int) -> at.Array:
    """Nearest-neighbour upsample a coarser band to a finer, statically known length."""
    if x.shape[1] == target_length:
        return x
    if target_length % x.shape[1] != 0:
        raise ValueError(f"Cannot resize temporal length {x.shape[1]} to {target_length}")
    return jnp.repeat(x, target_length // x.shape[1], axis=1)


def subband_flow_bridge(
    data_approx: at.Array,
    data_details: Sequence[at.Array],
    noise_approx: at.Array,
    noise_details: Sequence[at.Array],
    timestep: at.Array,
) -> tuple[at.Array, tuple[at.Array, ...], at.Array, tuple[at.Array, ...]]:
    """Construct the probability path and target directly in subband space."""
    if len(data_details) != len(noise_details):
        raise ValueError(f"Data/noise detail counts differ: {len(data_details)} versus {len(noise_details)}")
    time_expanded = timestep[..., None, None]
    state_approx = time_expanded * noise_approx + (1 - time_expanded) * data_approx
    state_details = tuple(
        time_expanded * noise_detail + (1 - time_expanded) * data_detail
        for noise_detail, data_detail in zip(noise_details, data_details, strict=True)
    )
    target_approx = noise_approx - data_approx
    target_details = tuple(
        noise_detail - data_detail for noise_detail, data_detail in zip(noise_details, data_details, strict=True)
    )
    return state_approx, state_details, target_approx, target_details


class _BandQueryConditioner(nnx.Module):
    """Single-head cross-attention from learned band queries to current action tokens."""

    def __init__(self, token_dim: int, target_length: int, *, max_queries: int = 4, rngs: nnx.Rngs):
        self.target_length = target_length
        self.num_queries = min(target_length, max_queries)
        self.queries = nnx.Embed(self.num_queries, token_dim, rngs=rngs)
        self.query_proj = nnx.Linear(token_dim, token_dim, rngs=rngs)
        self.key_proj = nnx.Linear(token_dim, token_dim, rngs=rngs)
        self.value_proj = nnx.Linear(token_dim, token_dim, rngs=rngs)
        self.out_proj = nnx.Linear(token_dim, token_dim, rngs=rngs)

    def __call__(self, action_tokens: at.Float[at.Array, "b t h"]) -> at.Float[at.Array, "b ts h"]:
        query = self.queries(jnp.arange(self.num_queries))
        query = jnp.broadcast_to(query[None, ...], (action_tokens.shape[0], *query.shape))
        query = self.query_proj(query)
        key = self.key_proj(action_tokens)
        value = self.value_proj(action_tokens)
        scale = math.sqrt(query.shape[-1])
        weights = jax.nn.softmax(jnp.einsum("bqh,bkh->bqk", query, key) / scale, axis=-1)
        query_condition = self.out_proj(jnp.einsum("bqk,bkh->bqh", weights, value))
        if self.num_queries == self.target_length:
            return query_condition
        target_indices = jnp.arange(self.target_length) * self.num_queries // self.target_length
        return query_condition[:, target_indices, :]


class _NormalizedSubbandFiLMHead(nnx.Module):
    """Predict a normalized subband velocity with explicit time and optional coarse conditions."""

    def __init__(self, action_dim: int, token_dim: int, bottleneck_dim: int, *, rngs: nnx.Rngs):
        self.state_proj = nnx.Linear(action_dim, bottleneck_dim, rngs=rngs)
        # Three explicit time features and two action-dimensional coarse slots:
        # the approximation prediction and the immediately coarser detail.
        condition_dim = token_dim + 3 + 2 * action_dim
        self.film_proj = nnx.Linear(condition_dim, 2 * bottleneck_dim, rngs=rngs)
        self.out_proj = nnx.Linear(bottleneck_dim, action_dim, rngs=rngs)

    def __call__(
        self,
        state: at.Float[at.Array, "b ts a"],
        token_condition: at.Float[at.Array, "b ts h"],
        timestep: at.Float[at.Array, " b"],
        approx_condition: at.Float[at.Array, "b ts a"],
        detail_condition: at.Float[at.Array, "b ts a"],
    ) -> at.Float[at.Array, "b ts a"]:
        time_features = jnp.stack(
            [timestep, jnp.sin(jnp.pi * timestep), jnp.cos(jnp.pi * timestep)],
            axis=-1,
        )
        time_features = jnp.broadcast_to(
            time_features[:, None, :], (*token_condition.shape[:2], time_features.shape[-1])
        )
        condition = jnp.concatenate(
            [token_condition, time_features, approx_condition, detail_condition],
            axis=-1,
        )
        gamma, beta = jnp.split(self.film_proj(condition), 2, axis=-1)
        gamma = 0.1 * jnp.tanh(gamma)
        beta = 0.1 * beta
        hidden = (1.0 + gamma) * self.state_proj(state) + beta
        return self.out_proj(nnx.gelu(hidden))


class WaveletNormStat(nnx.Variable):
    """Non-trainable per-band statistics stored with the model state."""


class NormalizedHierarchicalWaveletFlowHead(nnx.Module):
    """NH-WaFM head operating directly on normalized Haar subband states.

    Detail tuples use storage order ``(D1, ..., D_L)``. Prediction is performed
    in the fixed coarse-to-fine order ``A_L, D_L, ..., D1``.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        token_dim: int,
        action_horizon: int,
        levels: int,
        bottleneck_dim: int,
        hierarchical_coupling: bool,
        detach_coarse_condition: bool,
        conditioning_mode: str,
        band_means: at.Array | None = None,
        band_stds: at.Array | None = None,
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs,
    ):
        effective_levels, padded_horizon, approx_length, detail_lengths = wavelet_layout(action_horizon, levels)
        if effective_levels < 1:
            raise ValueError("NormalizedHierarchicalWaveletFlowHead requires an action horizon of at least 2")
        if conditioning_mode not in ("temporal_pooling", "band_query"):
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode!r}")

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.levels = effective_levels
        self.padded_horizon = padded_horizon
        self.approx_length = approx_length
        self.detail_lengths = detail_lengths
        self.hierarchical_coupling = hierarchical_coupling
        self.detach_coarse_condition = detach_coarse_condition
        self.conditioning_mode = conditioning_mode
        if norm_eps < 0:
            raise ValueError(f"norm_eps must be non-negative, got {norm_eps}")
        expected_stats_shape = (effective_levels + 1, action_dim)
        if band_means is None:
            band_means = jnp.zeros(expected_stats_shape, dtype=jnp.float32)
        if band_stds is None:
            band_stds = jnp.ones(expected_stats_shape, dtype=jnp.float32)
        band_means = jnp.asarray(band_means, dtype=jnp.float32)
        band_stds = jnp.asarray(band_stds, dtype=jnp.float32)
        if band_means.shape != expected_stats_shape or band_stds.shape != expected_stats_shape:
            raise ValueError(
                f"Band statistics must have shape {expected_stats_shape} in [A, D1, ..., D_L] order; "
                f"got means={band_means.shape}, stds={band_stds.shape}"
            )
        self.band_means = WaveletNormStat(band_means)
        self.band_stds = WaveletNormStat(band_stds)
        self.norm_eps = norm_eps

        self.approx_head = _NormalizedSubbandFiLMHead(action_dim, token_dim, bottleneck_dim, rngs=rngs)
        self.detail_heads = nnx.Dict(
            **{
                f"D{level}": _NormalizedSubbandFiLMHead(action_dim, token_dim, bottleneck_dim, rngs=rngs)
                for level in range(1, effective_levels + 1)
            }
        )
        self.band_queries = None
        if conditioning_mode == "band_query":
            self.band_queries = nnx.Dict(
                A=_BandQueryConditioner(token_dim, approx_length, rngs=rngs),
                **{
                    f"D{level}": _BandQueryConditioner(token_dim, detail_lengths[level - 1], rngs=rngs)
                    for level in range(1, effective_levels + 1)
                },
            )

    def normalize_bands(
        self,
        approx: at.Float[at.Array, "b tl a"],
        details: Sequence[at.Float[at.Array, "b td a"]],
    ) -> tuple[at.Float[at.Array, "b tl a"], tuple[at.Float[at.Array, "b td a"], ...]]:
        if len(details) != self.levels:
            raise ValueError(f"Expected {self.levels} detail bands, got {len(details)}")
        means = self.band_means.value
        scales = self.band_stds.value + self.norm_eps
        normalized_approx = (approx - means[0]) / scales[0]
        normalized_details = tuple(
            (detail - means[level]) / scales[level] for level, detail in enumerate(details, start=1)
        )
        return normalized_approx, normalized_details

    def denormalize_state_bands(
        self,
        approx: at.Float[at.Array, "b tl a"],
        details: Sequence[at.Float[at.Array, "b td a"]],
    ) -> tuple[at.Float[at.Array, "b tl a"], tuple[at.Float[at.Array, "b td a"], ...]]:
        if len(details) != self.levels:
            raise ValueError(f"Expected {self.levels} detail bands, got {len(details)}")
        means = self.band_means.value
        scales = self.band_stds.value + self.norm_eps
        restored_approx = approx * scales[0] + means[0]
        restored_details = tuple(detail * scales[level] + means[level] for level, detail in enumerate(details, start=1))
        return restored_approx, restored_details

    def denormalize_velocity_bands(
        self,
        approx_velocity: at.Float[at.Array, "b tl a"],
        detail_velocities: Sequence[at.Float[at.Array, "b td a"]],
    ) -> tuple[at.Float[at.Array, "b tl a"], tuple[at.Float[at.Array, "b td a"], ...]]:
        """Invert normalized velocities with scale only; state means must never be added."""
        if len(detail_velocities) != self.levels:
            raise ValueError(f"Expected {self.levels} detail bands, got {len(detail_velocities)}")
        scales = self.band_stds.value + self.norm_eps
        restored_approx = approx_velocity * scales[0]
        restored_details = tuple(velocity * scales[level] for level, velocity in enumerate(detail_velocities, start=1))
        return restored_approx, restored_details

    def _condition(
        self, band_name: str, target_length: int, action_tokens: at.Float[at.Array, "b t h"]
    ) -> at.Float[at.Array, "b ts h"]:
        if self.conditioning_mode == "temporal_pooling":
            return _pool_tokens_to_length(
                action_tokens,
                padded_horizon=self.padded_horizon,
                target_length=target_length,
            )
        if self.band_queries is None:
            raise ValueError("band_query conditioning was selected but query modules were not initialized")
        return self.band_queries[band_name](action_tokens)

    def __call__(
        self,
        normalized_approx: at.Float[at.Array, "b tl a"],
        normalized_details: Sequence[at.Float[at.Array, "b td a"]],
        action_tokens: at.Float[at.Array, "b t h"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b tl a"],
        tuple[at.Float[at.Array, "b td a"], ...],
        dict[str, object],
    ]:
        if len(normalized_details) != self.levels:
            raise ValueError(f"Expected {self.levels} detail bands, got {len(normalized_details)}")

        zeros_approx = jnp.zeros_like(normalized_approx)
        approx_condition = self._condition("A", normalized_approx.shape[1], action_tokens)
        predicted_approx = self.approx_head(
            normalized_approx,
            approx_condition,
            timestep,
            zeros_approx,
            zeros_approx,
        )

        predicted_details: list[at.Array | None] = [None] * self.levels
        token_conditions: list[at.Array | None] = [None] * self.levels
        previous_coarse: at.Array | None = None
        coarse_approx = predicted_approx
        if self.detach_coarse_condition:
            coarse_approx = jax.lax.stop_gradient(coarse_approx)

        for level in range(self.levels, 0, -1):
            detail_state = normalized_details[level - 1]
            token_condition = self._condition(f"D{level}", detail_state.shape[1], action_tokens)
            if self.hierarchical_coupling:
                approx_for_level = _resize_temporal(coarse_approx, detail_state.shape[1])
                if previous_coarse is None:
                    detail_for_level = jnp.zeros_like(detail_state)
                else:
                    detail_for_level = _resize_temporal(previous_coarse, detail_state.shape[1])
            else:
                approx_for_level = jnp.zeros_like(detail_state)
                detail_for_level = jnp.zeros_like(detail_state)

            prediction = self.detail_heads[f"D{level}"](
                detail_state,
                token_condition,
                timestep,
                approx_for_level,
                detail_for_level,
            )
            predicted_details[level - 1] = prediction
            token_conditions[level - 1] = token_condition
            previous_coarse = jax.lax.stop_gradient(prediction) if self.detach_coarse_condition else prediction

        details_tuple = tuple(prediction for prediction in predicted_details if prediction is not None)
        token_tuple = tuple(condition for condition in token_conditions if condition is not None)
        if len(details_tuple) != self.levels or len(token_tuple) != self.levels:
            raise AssertionError("All statically configured wavelet bands must be predicted")
        info: dict[str, object] = {
            "wavelet_flow_approx": predicted_approx,
            "wavelet_flow_details": details_tuple,
            "wavelet_state_approx": normalized_approx,
            "wavelet_state_details": tuple(normalized_details),
            "wavelet_token_approx": approx_condition,
            "wavelet_token_details": token_tuple,
            "wavelet_levels": self.levels,
            "wavelet_prediction_order": ("A", *tuple(f"D{i}" for i in range(self.levels, 0, -1))),
            "wavelet_gates": None,
        }
        return predicted_approx, details_tuple, info


class _SubbandFiLMHead(nnx.Module):
    """Legacy independent FiLM head for one wavelet subband."""

    def __init__(self, action_dim: int, cond_dim: int, bottleneck_dim: int, *, rngs: nnx.Rngs):
        self.state_proj = nnx.Linear(action_dim, bottleneck_dim, rngs=rngs)
        self.film_proj = nnx.Linear(cond_dim, 2 * bottleneck_dim, rngs=rngs)
        self.out_proj = nnx.Linear(bottleneck_dim, action_dim, rngs=rngs)

    def __call__(
        self,
        z_subband: at.Float[at.Array, "b ts a"],
        cond_band: at.Float[at.Array, "b ts h"],
    ) -> at.Float[at.Array, "b ts a"]:
        h_z = self.state_proj(z_subband)
        gamma_beta = self.film_proj(cond_band)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)

        # In replace mode this head owns the velocity; bound early FiLM modulation.
        gamma = 0.1 * jnp.tanh(gamma)
        beta = 0.1 * beta

        h = (1.0 + gamma) * h_z + beta
        h = nnx.gelu(h)
        return self.out_proj(h)


class WaveletSubbandFlowHead(nnx.Module):
    """Legacy head that reconstructs an action velocity from Haar subbands."""

    def __init__(
        self,
        *,
        action_dim: int,
        token_dim: int,
        levels: int,
        bottleneck_dim: int,
        use_band_gate: bool,
        rngs: nnx.Rngs,
    ):
        if levels < 1:
            raise ValueError(f"WaveletSubbandFlowHead expects levels >= 1, got {levels}")
        self.levels = levels
        self.use_band_gate = use_band_gate
        self.approx_head = _SubbandFiLMHead(action_dim, token_dim, bottleneck_dim, rngs=rngs)
        self.detail_heads = nnx.Dict(
            **{
                f"D{i}": _SubbandFiLMHead(action_dim, token_dim, bottleneck_dim, rngs=rngs)
                for i in range(1, levels + 1)
            }
        )
        self.gate_proj = nnx.Linear(token_dim, levels + 1, rngs=rngs) if use_band_gate else None

    def __call__(
        self,
        noisy_actions: at.Float[at.Array, "b t a"],
        action_tokens: at.Float[at.Array, "b t h"],
    ) -> tuple[at.Float[at.Array, "b t a"], dict[str, at.Array | list[at.Array] | int | None]]:
        z_approx, z_details, actual_levels = multi_level_haar_dwt(noisy_actions, self.levels)
        h_approx, h_details, _ = multi_level_haar_dwt(action_tokens, actual_levels)

        v_approx = self.approx_head(z_approx, h_approx)
        v_details = [
            self.detail_heads[f"D{i + 1}"](z_detail, h_detail)
            for i, (z_detail, h_detail) in enumerate(zip(z_details, h_details, strict=True))
        ]

        gates = None
        if self.use_band_gate:
            # The legacy gate is global per band; FiLM handles within-band channels.
            cond_global = jnp.mean(action_tokens, axis=1)
            assert self.gate_proj is not None
            gates = nnx.sigmoid(self.gate_proj(cond_global)[:, : actual_levels + 1])
            v_approx = gates[:, 0, None, None] * v_approx
            v_details = [gates[:, i + 1, None, None] * v_detail for i, v_detail in enumerate(v_details)]

        v_x = multi_level_haar_idwt(v_approx, v_details, target_length=noisy_actions.shape[1])
        info: dict[str, at.Array | list[at.Array] | int | None] = {
            "wavelet_flow_approx": v_approx,
            "wavelet_flow_details": v_details,
            "wavelet_state_approx": z_approx,
            "wavelet_state_details": z_details,
            "wavelet_token_approx": h_approx,
            "wavelet_token_details": h_details,
            "wavelet_levels": actual_levels,
            "wavelet_gates": gates,
        }
        if gates is not None:
            info["wavelet_gate_mean"] = jnp.mean(gates)
            info["wavelet_gate_A_mean"] = jnp.mean(gates[:, 0])
            for i in range(actual_levels):
                info[f"wavelet_gate_D{i + 1}_mean"] = jnp.mean(gates[:, i + 1])
        return v_x, info
