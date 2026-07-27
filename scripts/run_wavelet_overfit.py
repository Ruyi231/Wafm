"""CPU-friendly synthetic overfit gate for the NH-WaFM subband head.

This harness deliberately trains only ``NormalizedHierarchicalWaveletFlowHead``.
It is not a proxy for end-to-end Pi0 training; it is a fast stop-loss check that
each normalized subband target and the reconstructed action-domain velocity can
be fitted on one fixed synthetic batch.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import pathlib
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import wavelet_flow_head


@dataclasses.dataclass(frozen=True)
class OverfitConfig:
    output_dir: pathlib.Path = pathlib.Path("outputs/plan1_wavelet_overfit")
    seed: int = 0
    steps: int = 400
    log_interval: int = 25
    batch_size: int = 8
    action_horizon: int = 16
    action_dim: int = 3
    token_dim: int = 12
    levels: int = 2
    bottleneck_dim: int = 48
    learning_rate: float = 3e-3
    norm_eps: float = 1e-6
    conditioning_mode: str = "temporal_pooling"
    hierarchical_coupling: bool = True
    detach_coarse_condition: bool = False
    max_final_ratio: float = 0.5
    require_improvement: bool = True
    platform: str = "cpu"


class _FixedBatch(NamedTuple):
    state_approx: jax.Array
    state_details: tuple[jax.Array, ...]
    action_tokens: jax.Array
    timestep: jax.Array
    target_approx: jax.Array
    target_details: tuple[jax.Array, ...]


class _Metrics(NamedTuple):
    loss_total: jax.Array
    loss_approx: jax.Array
    loss_details: tuple[jax.Array, ...]
    action_idwt_mse: jax.Array


def _validate_config(config: OverfitConfig) -> None:
    if config.steps < 1:
        raise ValueError(f"steps must be positive, got {config.steps}")
    if config.log_interval < 1:
        raise ValueError(f"log_interval must be positive, got {config.log_interval}")
    if config.batch_size < 1 or config.action_horizon < 2 or config.action_dim < 1:
        raise ValueError("batch_size and action_dim must be positive, and action_horizon must be at least 2")
    if config.levels < 1:
        raise ValueError(f"levels must be at least 1, got {config.levels}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if not 0 < config.max_final_ratio < 1:
        raise ValueError(f"max_final_ratio must be in (0, 1), got {config.max_final_ratio}")
    if config.conditioning_mode not in ("temporal_pooling", "band_query"):
        raise ValueError(f"Unsupported conditioning_mode: {config.conditioning_mode!r}")
    if config.platform not in ("cpu", "default"):
        raise ValueError(f"platform must be 'cpu' or 'default', got {config.platform!r}")


def _synthetic_actions(config: OverfitConfig) -> jax.Array:
    """Create a deterministic batch with low- and high-frequency content."""
    time = jnp.linspace(0.0, 1.0, config.action_horizon, dtype=jnp.float32)
    batch_phase = jnp.arange(config.batch_size, dtype=jnp.float32)[:, None, None] * 0.17
    dimensions = jnp.arange(1, config.action_dim + 1, dtype=jnp.float32)[None, None, :]
    time = time[None, :, None]
    low = jnp.sin(2.0 * jnp.pi * dimensions * time + batch_phase)
    high = 0.3 * jnp.cos(2.0 * jnp.pi * (dimensions + 2.0) * time - 0.5 * batch_phase)
    trend = 0.2 * (time - 0.5) * dimensions
    return low + high + trend


def _band_stats(approx: jax.Array, details: tuple[jax.Array, ...], eps: float) -> tuple[jax.Array, jax.Array]:
    bands = (approx, *details)
    means = jnp.stack([jnp.mean(band, axis=(0, 1)) for band in bands])
    stds = jnp.stack([jnp.std(band, axis=(0, 1)) for band in bands])
    return means, jnp.maximum(stds, eps)


def _make_head_and_batch(
    config: OverfitConfig,
) -> tuple[wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead, _FixedBatch]:
    root_key = jax.random.key(config.seed)
    head_key, noise_key, token_key, time_key = jax.random.split(root_key, 4)

    actions = _synthetic_actions(config)
    data_approx, data_details_list, actual_levels = wavelet_flow_head.multi_level_haar_dwt(actions, config.levels)
    if actual_levels != config.levels:
        raise ValueError(
            f"Requested {config.levels} levels but horizon {config.action_horizon} supports {actual_levels}; "
            "lower --levels or increase --action-horizon"
        )
    data_details = tuple(data_details_list)
    band_means, band_stds = _band_stats(data_approx, data_details, config.norm_eps)

    head = wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead(
        action_dim=config.action_dim,
        token_dim=config.token_dim,
        action_horizon=config.action_horizon,
        levels=config.levels,
        bottleneck_dim=config.bottleneck_dim,
        hierarchical_coupling=config.hierarchical_coupling,
        detach_coarse_condition=config.detach_coarse_condition,
        conditioning_mode=config.conditioning_mode,
        band_means=band_means,
        band_stds=band_stds,
        norm_eps=config.norm_eps,
        rngs=nnx.Rngs(head_key),
    )

    normalized_approx, normalized_details = head.normalize_bands(data_approx, data_details)
    noise_keys = jax.random.split(noise_key, config.levels + 1)
    noise_approx = jax.random.normal(noise_keys[0], normalized_approx.shape)
    noise_details = tuple(
        jax.random.normal(key, detail.shape) for key, detail in zip(noise_keys[1:], normalized_details, strict=True)
    )
    timestep = jax.random.uniform(
        time_key,
        (config.batch_size,),
        minval=0.15,
        maxval=0.85,
    )
    state_approx, state_details, target_approx, target_details = wavelet_flow_head.subband_flow_bridge(
        normalized_approx,
        normalized_details,
        noise_approx,
        noise_details,
        timestep,
    )

    # Fixed, sample-specific tokens make this a strict memorization/optimization
    # check without involving PaliGemma or historical context.
    action_tokens = jax.random.normal(
        token_key,
        (config.batch_size, config.action_horizon, config.token_dim),
    )
    batch = _FixedBatch(
        state_approx=state_approx,
        state_details=state_details,
        action_tokens=action_tokens,
        timestep=timestep,
        target_approx=target_approx,
        target_details=target_details,
    )
    return head, batch


def _loss_and_metrics(
    head: wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead,
    batch: _FixedBatch,
) -> tuple[jax.Array, _Metrics]:
    pred_approx, pred_details, _ = head(
        batch.state_approx,
        batch.state_details,
        batch.action_tokens,
        batch.timestep,
    )
    loss_approx = jnp.mean(jnp.square(pred_approx - batch.target_approx))
    loss_details = tuple(
        jnp.mean(jnp.square(prediction - target))
        for prediction, target in zip(pred_details, batch.target_details, strict=True)
    )
    loss_total = (loss_approx + sum(loss_details)) / (1 + len(loss_details))

    # Velocity denormalization is scale-only. The head method intentionally
    # does not add the state means.
    pred_velocity_approx, pred_velocity_details = head.denormalize_velocity_bands(
        pred_approx,
        pred_details,
    )
    target_velocity_approx, target_velocity_details = head.denormalize_velocity_bands(
        batch.target_approx,
        batch.target_details,
    )
    pred_action_velocity = wavelet_flow_head.multi_level_haar_idwt(
        pred_velocity_approx,
        pred_velocity_details,
        target_length=head.action_horizon,
    )
    target_action_velocity = wavelet_flow_head.multi_level_haar_idwt(
        target_velocity_approx,
        target_velocity_details,
        target_length=head.action_horizon,
    )
    action_idwt_mse = jnp.mean(jnp.square(pred_action_velocity - target_action_velocity))
    metrics = _Metrics(
        loss_total=loss_total,
        loss_approx=loss_approx,
        loss_details=loss_details,
        action_idwt_mse=action_idwt_mse,
    )
    return loss_total, metrics


@nnx.jit
def _train_step(optimizer: nnx.Optimizer, batch: _FixedBatch) -> _Metrics:
    def loss_fn(
        head: wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead,
    ) -> tuple[jax.Array, _Metrics]:
        return _loss_and_metrics(head, batch)

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
    optimizer.update(grads)
    return metrics


@nnx.jit
def _evaluate(
    head: wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead,
    batch: _FixedBatch,
) -> _Metrics:
    return _loss_and_metrics(head, batch)[1]


def _metrics_row(step: int, metrics: _Metrics) -> dict[str, int | float]:
    row: dict[str, int | float] = {
        "step": step,
        "loss_total": float(metrics.loss_total),
        "action_idwt_mse": float(metrics.action_idwt_mse),
        "loss_band_A": float(metrics.loss_approx),
    }
    row.update({f"loss_band_D{level}": float(loss) for level, loss in enumerate(metrics.loss_details, start=1)})
    return row


def _ratios(initial: dict[str, int | float], final: dict[str, int | float]) -> dict[str, float]:
    ratio_keys = [key for key in initial if key.startswith("loss_band_")]
    ratio_keys.append("action_idwt_mse")
    return {key: float(final[key]) / max(float(initial[key]), 1e-12) for key in ratio_keys}


def _write_results(
    config: OverfitConfig,
    history: list[dict[str, int | float]],
    ratios: dict[str, float],
    *,
    passed: bool,
) -> tuple[pathlib.Path, pathlib.Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.output_dir / "wavelet_overfit.json"
    csv_path = config.output_dir / "wavelet_overfit_history.csv"
    payload = {
        "status": "passed" if passed else "failed",
        "criterion": f"all band losses and action_idwt_mse final/initial <= {config.max_final_ratio}",
        "config": {
            **dataclasses.asdict(config),
            "output_dir": str(config.output_dir),
        },
        "initial": history[0],
        "final": history[-1],
        "final_to_initial_ratio": ratios,
        "history": history,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return json_path, csv_path


def run_overfit(config: OverfitConfig) -> dict[str, object]:
    """Train on one fixed batch and return the measured stop-loss result."""
    _validate_config(config)
    if config.platform == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    head, batch = _make_head_and_batch(config)
    optimizer = nnx.Optimizer(head, optax.adam(config.learning_rate), wrt=nnx.Param)
    initial = _metrics_row(0, _evaluate(head, batch))
    history = [initial]

    for step in range(1, config.steps + 1):
        _train_step(optimizer, batch)
        if step % config.log_interval == 0 or step == config.steps:
            row = _metrics_row(step, _evaluate(head, batch))
            history.append(row)
            print(f"step={step:04d} loss={row['loss_total']:.6f} action_idwt_mse={row['action_idwt_mse']:.6f}")

    ratios = _ratios(history[0], history[-1])
    passed = all(math.isfinite(value) and value <= config.max_final_ratio for value in ratios.values())
    json_path, csv_path = _write_results(config, history, ratios, passed=passed)
    result: dict[str, object] = {
        "passed": passed,
        "initial": history[0],
        "final": history[-1],
        "ratios": ratios,
        "json_path": json_path,
        "csv_path": csv_path,
    }
    if config.require_improvement and not passed:
        raise RuntimeError(
            "NH-WaFM synthetic overfit gate failed; do not start large-scale training. "
            f"Measured final/initial ratios: {ratios}. Results: {json_path}"
        )
    return result


def _parse_args() -> OverfitConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=OverfitConfig.output_dir)
    parser.add_argument("--seed", type=int, default=OverfitConfig.seed)
    parser.add_argument("--steps", type=int, default=OverfitConfig.steps)
    parser.add_argument("--log-interval", type=int, default=OverfitConfig.log_interval)
    parser.add_argument("--batch-size", type=int, default=OverfitConfig.batch_size)
    parser.add_argument("--action-horizon", type=int, default=OverfitConfig.action_horizon)
    parser.add_argument("--action-dim", type=int, default=OverfitConfig.action_dim)
    parser.add_argument("--token-dim", type=int, default=OverfitConfig.token_dim)
    parser.add_argument("--levels", type=int, default=OverfitConfig.levels)
    parser.add_argument("--bottleneck-dim", type=int, default=OverfitConfig.bottleneck_dim)
    parser.add_argument("--learning-rate", type=float, default=OverfitConfig.learning_rate)
    parser.add_argument("--norm-eps", type=float, default=OverfitConfig.norm_eps)
    parser.add_argument(
        "--conditioning-mode",
        choices=("temporal_pooling", "band_query"),
        default=OverfitConfig.conditioning_mode,
    )
    parser.add_argument(
        "--hierarchical-coupling",
        action=argparse.BooleanOptionalAction,
        default=OverfitConfig.hierarchical_coupling,
    )
    parser.add_argument(
        "--detach-coarse-condition",
        action=argparse.BooleanOptionalAction,
        default=OverfitConfig.detach_coarse_condition,
    )
    parser.add_argument("--max-final-ratio", type=float, default=OverfitConfig.max_final_ratio)
    parser.add_argument(
        "--require-improvement",
        action=argparse.BooleanOptionalAction,
        default=OverfitConfig.require_improvement,
    )
    parser.add_argument("--platform", choices=("cpu", "default"), default=OverfitConfig.platform)
    return OverfitConfig(**vars(parser.parse_args()))


def main() -> None:
    config = _parse_args()
    result = run_overfit(config)
    print(f"status={'passed' if result['passed'] else 'failed'}")
    print(f"json={result['json_path']}")
    print(f"csv={result['csv_path']}")


if __name__ == "__main__":
    main()
