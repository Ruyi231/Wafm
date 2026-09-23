"""Run the single-config NH-WaFM GPU canary with fixed validation metrics.

This is a debugging run, not a formal Stage 1 comparison. It uses the complete
``plan1_subband_l2_norm`` model with a short warmup, trains on the normal
shuffled LIBERO stream, evaluates one fixed validation batch with a fixed RNG,
records band-head gradients and timing, and supports an explicit checkpoint
resume phase.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import json
import math
import pathlib
import time

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import checkpoints
from openpi.training import config as training_config
from openpi.training import data_loader as data_loader_lib
from openpi.training import optimizer as optimizer_lib
from openpi.training import sharding
from scripts import run_full_model_wavelet_overfit as overfit_lib
from scripts import train as train_lib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="plan1_subband_l2_norm")
    parser.add_argument("--exp-name", default="canary_short_warmup")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("plan1_results/server_validation/subband_l2_norm_canary"),
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--canary-budget", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--peak-learning-rate", type=float, default=5e-5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.canary_budget < 2000 or args.canary_budget > 3000:
        raise ValueError("steps must be positive and canary_budget must be in [2000, 3000]")
    if args.steps > args.canary_budget:
        raise ValueError("steps cannot exceed canary_budget")
    if not 0 <= args.warmup_steps <= 500:
        raise ValueError("warmup_steps must be in [0, 500] for the debugging canary")
    if args.batch_size <= 0 or args.log_interval <= 0 or args.save_interval <= 0:
        raise ValueError("batch_size, log_interval, and save_interval must be positive")
    if args.peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be positive")


def _finite_host_metrics(metrics: dict[str, jax.Array]) -> dict[str, float]:
    host = jax.device_get(metrics)
    result = {}
    for key, value in host.items():
        scalar = float(np.asarray(value))
        if not math.isfinite(scalar):
            raise FloatingPointError(f"Canary metric {key} is not finite: {scalar}")
        result[key] = scalar
    return result


def _load_history(output_dir: pathlib.Path, *, resume: bool) -> list[dict[str, float | int | str]]:
    json_path = output_dir / "wavelet_canary.json"
    if resume:
        if not json_path.exists():
            raise FileNotFoundError(f"Cannot resume without prior canary results: {json_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))["history"]
    if output_dir.exists():
        raise FileExistsError(f"Canary output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    return []


def _write_results(
    output_dir: pathlib.Path,
    history: list[dict[str, float | int | str]],
    provenance: dict[str, object],
    *,
    complete: bool,
    checkpoint_step: int | None,
) -> None:
    validation_rows = [row for row in history if row["phase"] == "validation"]
    training_rows = [row for row in history if row["phase"] == "training"]
    initial = validation_rows[0]
    final = validation_rows[-1]
    ratios = {
        key: float(final[key]) / max(float(initial[key]), 1e-12)
        for key in ("loss_band_A", "loss_band_D1", "loss_band_D2", "loss_action_reconstruction")
    }
    gradients_nonzero = bool(training_rows) and all(
        float(row[key]) > 0
        for row in training_rows
        for key in overfit_lib._BAND_GRAD_METRICS  # noqa: SLF001
    )
    checkpoint_complete = checkpoint_step == int(provenance["completed_steps"]) - 1
    passed = complete and checkpoint_complete and all(value < 1 for value in ratios.values()) and gradients_nonzero
    payload = {
        "status": "canary_passed" if passed else ("canary_failed" if complete else "canary_partial_complete"),
        "debug_only": True,
        "formal_stage1_result": False,
        "history": history,
        "fixed_validation_final_to_initial_ratio": ratios,
        "band_gradients_nonzero": gradients_nonzero,
        "checkpoint_complete": checkpoint_complete,
        "latest_checkpoint_step": checkpoint_step,
        "provenance": provenance,
    }
    json_path = output_dir / "wavelet_canary.json"
    csv_path = output_dir / "wavelet_canary_history.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    if complete and not passed:
        raise RuntimeError(f"Single-config GPU canary failed: validation ratios={ratios}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"The canary requires GPU, got {jax.default_backend()!r}")

    base_config = training_config.get_config(args.config_name)
    if args.config_name != "plan1_subband_l2_norm":
        raise ValueError("This canary intentionally permits only plan1_subband_l2_norm")
    config = dataclasses.replace(
        base_config,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=args.steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=None,
        lr_schedule=optimizer_lib.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_learning_rate,
            decay_steps=args.canary_budget,
            decay_lr=args.peak_learning_rate,
        ),
        overwrite=not args.resume,
        resume=args.resume,
        wandb_enabled=False,
    )
    history = _load_history(args.output_dir, resume=args.resume)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    validation_rng = jax.random.key(2026)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    if resuming != args.resume:
        raise RuntimeError(f"Checkpoint resume mismatch: requested={args.resume}, detected={resuming}")

    train_loader = data_loader_lib.create_data_loader(config, sharding=data_sharding, shuffle=True)
    fixed_loader = data_loader_lib.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=1,
    )
    train_iter = iter(train_loader)
    fixed_batch = next(iter(fixed_loader))
    state, state_sharding = train_lib.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    if resuming:
        state = checkpoints.restore_state(checkpoint_manager, state, train_loader)

    ptrain_step = jax.jit(
        functools.partial(train_lib.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(overfit_lib._evaluation_step, train_preprocessing=False),  # noqa: SLF001
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(state.step)
    if not history:
        initial = overfit_lib._host_metrics(start_step, peval_step(validation_rng, state, fixed_batch))  # noqa: SLF001
        initial["phase"] = "validation"
        history.append(initial)
    infos = []
    interval_train_seconds = 0.0
    interval_steps = 0
    for step in range(start_step, args.steps):
        batch = next(train_iter)
        started = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info = ptrain_step(train_rng, state, batch)
        jax.block_until_ready(state)
        interval_train_seconds += time.perf_counter() - started
        interval_steps += 1
        infos.append(info)

        completed_step = step + 1
        if completed_step % args.log_interval == 0 or completed_step == args.steps:
            mean_info = _finite_host_metrics(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            train_row: dict[str, float | int | str] = {
                "step": completed_step,
                "phase": "training",
                "seconds_per_step": interval_train_seconds / interval_steps,
                **mean_info,
            }
            history.append(train_row)
            validation = overfit_lib._host_metrics(
                completed_step,
                peval_step(validation_rng, state, fixed_batch),
            )
            validation["phase"] = "validation"
            history.append(validation)
            print(
                f"step={completed_step:04d} train_loss={train_row['loss']:.6f} "
                f"val_band={validation['loss_band_total']:.6f} "
                f"val_action={validation['loss_action_reconstruction']:.6f} "
                f"seconds_per_step={train_row['seconds_per_step']:.4f}",
                flush=True,
            )
            infos = []
            interval_train_seconds = 0.0
            interval_steps = 0

        if (step % args.save_interval == 0 and step > start_step) or step == args.steps - 1:
            checkpoints.save_state(checkpoint_manager, state, train_loader, step)

    checkpoint_manager.wait_until_finished()
    latest_step = checkpoint_manager.latest_step()
    memory_stats = jax.devices()[0].memory_stats() or {}
    provenance = {
        "train_config": args.config_name,
        "debug_short_warmup": True,
        "warmup_steps": args.warmup_steps,
        "canary_budget": args.canary_budget,
        "batch_size": args.batch_size,
        "peak_learning_rate": args.peak_learning_rate,
        "checkpoint_dir": str(config.checkpoint_dir),
        "resume_phase": args.resume,
        "resume_verified": args.resume and start_step > 0,
        "restored_start_step": start_step,
        "completed_steps": args.steps,
        "gpu_device": str(jax.devices()[0]),
        "gpu_peak_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
    }
    _write_results(
        args.output_dir,
        history,
        provenance,
        complete=args.steps == args.canary_budget,
        checkpoint_step=latest_step,
    )
    print(f"checkpoint_step={latest_step}", flush=True)
    print(f"json={args.output_dir / 'wavelet_canary.json'}", flush=True)


if __name__ == "__main__":
    main()
