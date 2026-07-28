"""Compute NH-WaFM subband statistics from normalized training actions.

Unlike ``compute_norm_stats.py``, this script deliberately uses the regular
``create_data_loader`` pipeline. Therefore each yielded action has already
passed through OpenPI ``norm_stats`` normalization and all model transforms.
"""

from collections.abc import Iterable
import pathlib
from typing import Any

import tqdm
import tyro

from openpi.models import wavelet_normalization
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _normalized_actions(data_loader: Iterable[Any], num_batches: int) -> Iterable[Any]:
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing wavelet stats"):
        if not isinstance(batch, tuple) or len(batch) not in (2, 3):
            raise TypeError(
                "create_data_loader must yield (observation, actions) or "
                f"(observation, actions, extras), got {type(batch).__name__}"
            )
        yield batch[1]


def _require_action_norm_stats(data_config: Any, config_name: str) -> None:
    """Reject statistics computed from actions that skipped OpenPI normalization."""
    norm_stats = data_config.norm_stats
    if norm_stats is None or "actions" not in norm_stats:
        raise FileNotFoundError(
            f"Config {config_name!r} did not load OpenPI action norm_stats. "
            "NH-WaFM wavelet statistics must be computed after action normalization. "
            "Run scripts/compute_norm_stats.py for the action-statistics asset owner, "
            "then verify the configured data.assets directory before retrying."
        )


def _resolve_output_path(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    output_path: pathlib.Path | None,
    requested_levels: int,
) -> pathlib.Path:
    """Resolve an explicit path before falling back to config-owned assets."""
    if output_path is not None:
        return output_path

    configured_path = getattr(config.model, "wavelet_norm_stats_path", None)
    if configured_path is not None:
        configured_levels = getattr(config.model, "wavelet_levels", None)
        if configured_levels != requested_levels:
            raise ValueError(
                "The model-configured wavelet_norm_stats_path can only be used with "
                f"wavelet_levels={configured_levels}; got --levels={requested_levels}. "
                "Pass --output-path explicitly for a levels ablation."
            )
        return pathlib.Path(configured_path)

    dataset_path = pathlib.Path(data_config.repo_id) if data_config.repo_id else pathlib.Path("dataset")
    return config.assets_dirs / dataset_path / f"wavelet_norm_stats_l{requested_levels}.json"


def main(
    config_name: str,
    output_path: pathlib.Path | None = None,
    levels: int | None = None,
    num_batches: int = 1000,
    eps: float = 1e-6,
) -> None:
    """Compute and save per-band, per-action-dimension mean and std."""
    if num_batches < 1:
        raise ValueError(f"num_batches must be positive, got {num_batches}")

    config = _config.get_config(config_name)
    configured_levels = getattr(config.model, "wavelet_levels", None)
    if levels is None:
        if configured_levels is None:
            raise ValueError(f"Config {config_name!r} has no model.wavelet_levels; pass --levels explicitly")
        levels = configured_levels

    # skip_norm_stats intentionally remains False. This is the key difference
    # from the action-domain norm-statistics script.
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=num_batches,
        skip_norm_stats=False,
    )
    data_config = data_loader.data_config()
    _require_action_norm_stats(data_config, config_name)
    stats = wavelet_normalization.compute_wavelet_norm_stats(
        _normalized_actions(data_loader, num_batches),
        levels,
        eps,
        source_config=config_name,
    )

    output_path = _resolve_output_path(config, data_config, output_path, stats.requested_levels)
    output_path = wavelet_normalization.save_wavelet_norm_stats(output_path, stats)

    print(f"Wrote wavelet normalization statistics to: {output_path}")
    print(
        f"samples={stats.sample_count}, horizon={stats.action_horizon}, "
        f"padded_horizon={stats.padded_horizon}, action_dim={stats.action_dim}, "
        f"levels={stats.levels}"
    )
    for name, band in stats.bands.items():
        print(f"{name}: coefficient_count={band.coefficient_count}")


if __name__ == "__main__":
    tyro.cli(main)
