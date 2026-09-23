"""Run an end-to-end NH-WaFM overfit gate on one fixed real LIBERO batch.

Unlike the lightweight head-only gate, this runner initializes the complete
Pi0.5 model from the configured base checkpoint and updates every parameter
that the selected training config marks trainable. Training keeps the real
batch fixed while using the normal per-step flow noise. Measurements use one
fixed evaluation RNG so that loss ratios reflect learning rather than a change
in sampled flow state or timestep.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import hashlib
import json
import math
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.training import config as training_config
from openpi.training import data_loader as data_loader_lib
from openpi.training import optimizer as optimizer_lib
from openpi.training import sharding
from scripts import train as train_lib


_GATE_METRICS = (
    "loss_band_A",
    "loss_band_D1",
    "loss_band_D2",
    "loss_action_reconstruction",
)
_BAND_GRAD_METRICS = (
    "gradient_norm_each_band_head_A",
    "gradient_norm_each_band_head_D1",
    "gradient_norm_each_band_head_D2",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="plan1_subband_l2_norm")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("plan1_results/server_validation/full_model_real_overfit"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-final-ratio", type=float, default=0.5)
    parser.add_argument("--max-action-mse", type=float, default=1e-3)
    parser.add_argument("--min-band-grad-norm", type=float, default=1e-12)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--deterministic-target", action="store_true")
    parser.add_argument("--require-strict-threshold", action="store_true")
    return parser.parse_args()


def _validate_config(config: training_config.TrainConfig, args: argparse.Namespace) -> None:
    model = config.model
    if model.wavelet_flow_impl != "subband_flow" or not model.use_wavelet_flow_head:
        raise ValueError(f"Config {config.name!r} is not an NH-WaFM subband-flow config")
    if not model.wavelet_band_normalization:
        raise ValueError(f"Config {config.name!r} does not enable wavelet band normalization")
    if args.steps <= 0 or args.log_interval <= 0 or args.batch_size <= 0:
        raise ValueError("steps, log_interval, and batch_size must be positive")
    if args.learning_rate <= 0 or not 0 < args.max_final_ratio <= 1:
        raise ValueError("learning_rate must be positive and max_final_ratio must be in (0, 1]")
    if args.max_action_mse <= 0 or args.min_band_grad_norm < 0:
        raise ValueError("max_action_mse must be positive and min_band_grad_norm must be nonnegative")
    expected_metrics = {"loss_band_A", *(f"loss_band_D{i}" for i in range(1, model.wavelet_levels + 1))}
    if expected_metrics != set(_GATE_METRICS[:-1]):
        raise ValueError(f"This gate expects exactly two DWT levels, got {model.wavelet_levels}")


def _evaluation_step(
    eval_rng: jax.Array,
    state,
    batch: tuple[model_lib.Observation, model_lib.Actions],
    *,
    train_preprocessing: bool,
) -> dict[str, jax.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train() if train_preprocessing else model.eval()
    observation, actions = batch[:2]
    loss, metrics = model.compute_loss(
        eval_rng,
        observation,
        actions,
        train=train_preprocessing,
        return_metrics=True,
    )
    return {"loss": jnp.mean(loss), **metrics}


def _host_metrics(step: int, metrics: dict[str, jax.Array]) -> dict[str, float | int]:
    values = jax.device_get(metrics)
    row: dict[str, float | int] = {"step": step}
    for key, value in values.items():
        scalar = float(np.asarray(value))
        if not math.isfinite(scalar):
            raise FloatingPointError(f"Metric {key} is not finite at step {step}: {scalar}")
        row[key] = scalar
    return row


def _ratios(initial: dict[str, float | int], final: dict[str, float | int]) -> dict[str, float]:
    return {key: float(final[key]) / max(float(initial[key]), 1e-12) for key in _GATE_METRICS}


def _gradient_metrics(info: dict[str, jax.Array]) -> dict[str, float]:
    values = jax.device_get(info)
    result = {}
    for key in _BAND_GRAD_METRICS:
        value = float(np.asarray(values[key]))
        if not math.isfinite(value):
            raise FloatingPointError(f"Gradient metric {key} is not finite: {value}")
        result[key] = value
    return result


def _write_results(
    output_dir: pathlib.Path,
    history: list[dict[str, float | int]],
    ratios: dict[str, float],
    provenance: dict[str, object],
    *,
    threshold: float,
    deterministic_target: bool,
    max_action_mse: float,
    min_band_grad_norm: float,
) -> tuple[pathlib.Path, pathlib.Path, bool]:
    strict_threshold_met = all(math.isfinite(value) and value <= threshold for value in ratios.values())
    observed_gradients = [float(row[key]) for row in history[1:] for key in _BAND_GRAD_METRICS]
    gradients_nonzero = bool(observed_gradients) and min(observed_gradients) > min_band_grad_norm
    action_near_zero = float(history[-1]["loss_action_reconstruction"]) <= max_action_mse
    deterministic_fit_passed = strict_threshold_met and gradients_nonzero and action_near_zero
    if deterministic_target:
        status = "deterministic_overfit_passed" if deterministic_fit_passed else "deterministic_overfit_failed"
        passed = deterministic_fit_passed
    else:
        status = "stochastic_training_chain_passed"
        passed = strict_threshold_met
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "status": status,
        "strict_diagnostic_threshold": threshold,
        "strict_diagnostic_threshold_met": strict_threshold_met,
        "deterministic_target": deterministic_target,
        "deterministic_fit_passed": deterministic_fit_passed if deterministic_target else None,
        "deterministic_criteria": {
            "all_gate_metric_final_to_initial_ratios_at_most": threshold,
            "final_action_reconstruction_mse_at_most": max_action_mse,
            "each_observed_band_gradient_norm_greater_than": min_band_grad_norm,
        },
        "action_near_zero": action_near_zero,
        "band_gradients_nonzero": gradients_nonzero,
        "gate_metrics": list(_GATE_METRICS),
        "initial": history[0],
        "final": history[-1],
        "final_to_initial_ratio": ratios,
        "history": history,
        "provenance": provenance,
    }
    json_path = output_dir / "full_model_wavelet_overfit.json"
    csv_path = output_dir / "full_model_wavelet_overfit_history.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    return json_path, csv_path, passed


def main() -> None:
    args = _parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"The full-model overfit gate requires GPU, got {jax.default_backend()!r}")

    base_config = training_config.get_config(args.config_name)
    _validate_config(base_config, args)
    config = dataclasses.replace(
        base_config,
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=args.steps,
        lr_schedule=optimizer_lib.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=args.learning_rate,
            decay_steps=max(args.steps, 1),
            decay_lr=args.learning_rate,
        ),
        wandb_enabled=False,
        fsdp_devices=args.fsdp_devices if args.fsdp_devices is not None else base_config.fsdp_devices,
    )
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by {jax.device_count()} devices")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    eval_rng = train_rng if args.deterministic_target else jax.random.key(args.eval_seed)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader_lib.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=1,
    )
    batch = next(iter(loader))
    actions = np.asarray(jax.device_get(batch[1]), dtype=np.float32)
    if not np.all(np.isfinite(actions)):
        raise FloatingPointError("The fixed normalized real action batch contains NaN or Inf")

    state, state_sharding = train_lib.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    ptrain_step = jax.jit(
        functools.partial(train_lib.train_step, config, fold_in_step=not args.deterministic_target),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(_evaluation_step, train_preprocessing=args.deterministic_target),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    initial = _host_metrics(0, peval_step(eval_rng, state, batch))
    initial.update({key: 0.0 for key in _BAND_GRAD_METRICS})
    history = [initial]
    print(
        f"step=0000 loss={history[-1]['loss']:.6f} "
        f"action_idwt_mse={history[-1]['loss_action_reconstruction']:.6f}",
        flush=True,
    )
    for step in range(1, args.steps + 1):
        with sharding.set_mesh(mesh):
            state, train_info = ptrain_step(train_rng, state, batch)
        if step % args.log_interval == 0 or step == args.steps:
            row = _host_metrics(step, peval_step(eval_rng, state, batch))
            row.update(_gradient_metrics(train_info))
            history.append(row)
            print(
                f"step={step:04d} loss={row['loss']:.6f} "
                f"action_idwt_mse={row['loss_action_reconstruction']:.6f}",
                flush=True,
            )

    ratios = _ratios(history[0], history[-1])
    stats_path = pathlib.Path(config.model.wavelet_norm_stats_path)
    provenance = {
        "data_source": "fixed_real_lerobot_batch",
        "train_config": config.name,
        "dataset_repo_id": loader.data_config().repo_id,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "eval_seed": args.eval_seed,
        "training_rng_fixed": args.deterministic_target,
        "evaluation_rng_matches_training_rng": args.deterministic_target,
        "flow_noise_fixed": args.deterministic_target,
        "flow_timestep_fixed": args.deterministic_target,
        "training_preprocessing_fixed": args.deterministic_target,
        "fixed_actions_shape": list(actions.shape),
        "fixed_actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "wavelet_norm_stats_path": str(stats_path),
        "wavelet_norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "full_model_trainable_filter": repr(config.trainable_filter),
        "checkpoint_saved": False,
        "device_count": jax.device_count(),
        "fsdp_devices": config.fsdp_devices,
    }
    json_path, csv_path, passed = _write_results(
        args.output_dir,
        history,
        ratios,
        provenance,
        threshold=args.max_final_ratio,
        deterministic_target=args.deterministic_target,
        max_action_mse=args.max_action_mse,
        min_band_grad_norm=args.min_band_grad_norm,
    )
    status = "passed" if passed else "strict_threshold_not_met"
    print(f"status={status}", flush=True)
    print(f"ratios={ratios}", flush=True)
    print(f"json={json_path}", flush=True)
    print(f"csv={csv_path}", flush=True)
    if args.deterministic_target and not passed:
        raise RuntimeError(f"Deterministic full-model overfit gate failed: {ratios}")
    if args.require_strict_threshold and not passed:
        raise RuntimeError(f"Strict additional diagnostic threshold was not met: {ratios}")


if __name__ == "__main__":
    main()
