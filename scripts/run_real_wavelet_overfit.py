"""Overfit the NH-WaFM head on one fixed, normalized real-data action batch.

This complements the synthetic head gate and the end-to-end two-step training
smoke. It loads one batch through the regular OpenPI data pipeline, reuses the
formal global wavelet statistics, and repeatedly optimizes the lightweight
NH-WaFM head on those fixed real actions.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import wavelet_normalization
from openpi.training import config as training_config
from openpi.training import data_loader as data_loader_lib
import run_wavelet_overfit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="plan1_subband_l2_norm")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("plan1_results/server_validation/real_wavelet_overfit"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=12)
    parser.add_argument("--bottleneck-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-final-ratio", type=float, default=0.5)
    parser.add_argument("--platform", choices=("cpu", "default"), default="cpu")
    return parser.parse_args()


def _load_fixed_actions(args: argparse.Namespace):
    train_config = training_config.get_config(args.config_name)
    if train_config.model.wavelet_flow_impl != "subband_flow":
        raise ValueError(f"Config {args.config_name!r} is not a subband-flow config")
    if not train_config.model.wavelet_band_normalization:
        raise ValueError(f"Config {args.config_name!r} does not enable wavelet band normalization")

    loader_config = dataclasses.replace(train_config, batch_size=args.batch_size, num_workers=0)
    loader = data_loader_lib.create_data_loader(
        loader_config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=False,
    )
    data_config = loader.data_config()
    if data_config.norm_stats is None or "actions" not in data_config.norm_stats:
        raise FileNotFoundError("The real-data loader did not load OpenPI action norm_stats")

    batch = next(iter(loader))
    actions = np.asarray(batch[1], dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != args.batch_size:
        raise ValueError(f"Expected a fixed action batch of size {args.batch_size}, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("The fixed real action batch contains NaN or Inf")

    stats_path = pathlib.Path(train_config.model.wavelet_norm_stats_path)
    stats = wavelet_normalization.load_wavelet_norm_stats(
        stats_path,
        expected_levels=train_config.model.wavelet_levels,
        expected_action_dim=actions.shape[2],
        expected_action_horizon=actions.shape[1],
        eps=train_config.model.wavelet_band_norm_eps,
    )
    provenance = {
        "data_source": "fixed_real_lerobot_batch",
        "train_config": args.config_name,
        "dataset_repo_id": data_config.repo_id,
        "action_norm_stats_loaded": True,
        "wavelet_norm_stats_path": str(stats_path),
        "wavelet_norm_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "fixed_actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "fixed_actions_shape": list(actions.shape),
    }
    return train_config, jnp.asarray(actions), jnp.asarray(stats.means), jnp.asarray(stats.stds), provenance


def main() -> None:
    args = _parse_args()
    if args.platform == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    train_config, actions, band_means, band_stds, provenance = _load_fixed_actions(args)
    overfit_config = run_wavelet_overfit.OverfitConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        steps=args.steps,
        log_interval=args.log_interval,
        batch_size=actions.shape[0],
        action_horizon=actions.shape[1],
        action_dim=actions.shape[2],
        token_dim=args.token_dim,
        levels=train_config.model.wavelet_levels,
        bottleneck_dim=args.bottleneck_dim,
        learning_rate=args.learning_rate,
        norm_eps=train_config.model.wavelet_band_norm_eps,
        conditioning_mode=train_config.model.wavelet_conditioning_mode,
        hierarchical_coupling=train_config.model.wavelet_hierarchical_coupling,
        detach_coarse_condition=train_config.model.wavelet_detach_coarse_condition,
        max_final_ratio=args.max_final_ratio,
        platform=args.platform,
    )
    result = run_wavelet_overfit.run_overfit(
        overfit_config,
        actions=actions,
        band_means=band_means,
        band_stds=band_stds,
        provenance=provenance,
    )
    print(f"status={'passed' if result['passed'] else 'failed'}")
    print(f"json={result['json_path']}")
    print(f"csv={result['csv_path']}")


if __name__ == "__main__":
    main()
