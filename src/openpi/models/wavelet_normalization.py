"""Wavelet subband statistics and normalization utilities for NH-WaFM.

The detail tuple used by this module always follows the existing OpenPI DWT
convention ``(D1, D2, ..., DL)``. JSON files are written in the human-facing
coarse-to-fine order ``A_L, D_L, ..., D1``.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import logging
import math
import numbers
import pathlib
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from openpi.models import wavelet_flow_head

_FORMAT_VERSION = 1
_INPUT_NORMALIZATION = "openpi_norm_stats"
_PADDING_MODE = "edge"
_WAVELET = "haar"
_LOGGER = logging.getLogger(__name__)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _effective_levels(action_horizon: int, requested_levels: int) -> int:
    """Mirror ``multi_level_haar_dwt`` level clamping for a static horizon."""
    action_horizon = _positive_int(action_horizon, "action_horizon")
    requested_levels = _positive_int(requested_levels, "requested_levels")
    if action_horizon < 2:
        raise ValueError("Wavelet statistics require action_horizon >= 2")
    return min(requested_levels, (action_horizon - 1).bit_length())


def _padded_horizon(action_horizon: int, levels: int) -> int:
    multiple = 2**levels
    return action_horizon + (-action_horizon) % multiple


def _coarse_to_fine_names(levels: int) -> tuple[str, ...]:
    levels = _positive_int(levels, "levels")
    return (f"A_{levels}", *(f"D_{level}" for level in range(levels, 0, -1)))


def _expected_band_length(padded_horizon: int, band_name: str, levels: int) -> int:
    if band_name == f"A_{levels}":
        scale = levels
    elif band_name.startswith("D_"):
        try:
            scale = int(band_name[2:])
        except ValueError as exc:
            raise ValueError(f"Invalid detail band name: {band_name!r}") from exc
        if not 1 <= scale <= levels:
            raise ValueError(f"Detail band {band_name!r} is outside levels 1..{levels}")
    else:
        raise ValueError(f"Invalid wavelet band name: {band_name!r}")
    return padded_horizon // (2**scale)


@dataclasses.dataclass(frozen=True, eq=False)
class WaveletBandStats:
    """Per-action-dimension statistics for one wavelet band."""

    mean: np.ndarray
    std: np.ndarray
    coefficient_count: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        coefficient_count = _nonnegative_int(self.coefficient_count, "coefficient_count")
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(f"Band mean/std must be rank 1, got shapes {mean.shape} and {std.shape}")
        if mean.shape != std.shape:
            raise ValueError(f"Band mean/std shapes differ: {mean.shape} versus {std.shape}")
        if mean.size == 0:
            raise ValueError("Band statistics cannot have an empty action dimension")
        if not np.all(np.isfinite(mean)):
            raise ValueError("Band mean contains NaN or Inf")
        if not np.all(np.isfinite(std)):
            raise ValueError("Band std contains NaN or Inf")
        if np.any(std < 0.0):
            raise ValueError("Band std must be non-negative")
        mean = mean.copy()
        std = std.copy()
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "coefficient_count", coefficient_count)


@dataclasses.dataclass(frozen=True, eq=False)
class WaveletNormStats:
    """Validated NH-WaFM statistics.

    ``details`` is always ordered ``(D1, ..., DL)``. ``levels`` is the
    effective DWT level count, while ``requested_levels`` records the original
    configuration value in case a very short horizon caused level clamping.
    """

    approx: WaveletBandStats
    details: tuple[WaveletBandStats, ...]
    levels: int
    requested_levels: int
    action_dim: int
    eps: float
    action_horizon: int | None = None
    padded_horizon: int | None = None
    sample_count: int = 0
    source_config: str | None = None
    is_identity: bool = False

    def __post_init__(self) -> None:
        levels = _positive_int(self.levels, "levels")
        requested_levels = _positive_int(self.requested_levels, "requested_levels")
        action_dim = _positive_int(self.action_dim, "action_dim")
        eps = _positive_finite(self.eps, "eps")
        sample_count = _nonnegative_int(self.sample_count, "sample_count")
        details = tuple(self.details)

        if levels > requested_levels:
            raise ValueError(f"Effective levels {levels} cannot exceed requested levels {requested_levels}")
        if len(details) != levels:
            raise ValueError(f"Expected {levels} detail statistics in D1..DL order, got {len(details)}")
        if self.approx.mean.shape != (action_dim,):
            raise ValueError(
                f"A_{levels} statistics have action dimension {self.approx.mean.shape}, expected ({action_dim},)"
            )
        for index, detail in enumerate(details, start=1):
            if detail.mean.shape != (action_dim,):
                raise ValueError(
                    f"D_{index} statistics have action dimension {detail.mean.shape}, expected ({action_dim},)"
                )

        action_horizon = self.action_horizon
        padded_horizon = self.padded_horizon
        if action_horizon is None:
            if padded_horizon is not None:
                raise ValueError("padded_horizon requires action_horizon")
        else:
            action_horizon = _positive_int(action_horizon, "action_horizon")
            effective_levels = _effective_levels(action_horizon, requested_levels)
            if levels != effective_levels:
                raise ValueError(
                    f"Effective levels mismatch: metadata has {levels}, but horizon {action_horizon} "
                    f"and requested levels {requested_levels} imply {effective_levels}"
                )
            expected_padded_horizon = _padded_horizon(action_horizon, levels)
            if padded_horizon is None:
                padded_horizon = expected_padded_horizon
            else:
                padded_horizon = _positive_int(padded_horizon, "padded_horizon")
                if padded_horizon != expected_padded_horizon:
                    raise ValueError(
                        f"padded_horizon is {padded_horizon}, expected {expected_padded_horizon} "
                        f"for horizon {action_horizon} and {levels} levels"
                    )

        if not self.is_identity and sample_count == 0:
            raise ValueError("Non-identity statistics require sample_count > 0")
        if self.is_identity and sample_count != 0:
            raise ValueError("Identity statistics must have sample_count == 0")

        if action_horizon is not None and not self.is_identity:
            assert padded_horizon is not None
            expected_approx_count = sample_count * _expected_band_length(padded_horizon, f"A_{levels}", levels)
            if self.approx.coefficient_count != expected_approx_count:
                raise ValueError(
                    f"A_{levels} coefficient_count is {self.approx.coefficient_count}, expected {expected_approx_count}"
                )
            for index, detail in enumerate(details, start=1):
                expected_count = sample_count * _expected_band_length(padded_horizon, f"D_{index}", levels)
                if detail.coefficient_count != expected_count:
                    raise ValueError(
                        f"D_{index} coefficient_count is {detail.coefficient_count}, expected {expected_count}"
                    )

        if self.source_config is not None and not isinstance(self.source_config, str):
            raise TypeError(f"source_config must be a string or None, got {type(self.source_config).__name__}")

        object.__setattr__(self, "details", details)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "requested_levels", requested_levels)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "action_horizon", action_horizon)
        object.__setattr__(self, "padded_horizon", padded_horizon)
        object.__setattr__(self, "sample_count", sample_count)

    @property
    def bands(self) -> Mapping[str, WaveletBandStats]:
        """Return a read-only mapping in JSON coarse-to-fine order."""
        ordered = {f"A_{self.levels}": self.approx}
        ordered.update({f"D_{index}": self.details[index - 1] for index in range(self.levels, 0, -1)})
        return MappingProxyType(ordered)

    @property
    def means(self) -> np.ndarray:
        """Statistics stacked as ``[A, D1, ..., DL]`` for model variables."""
        values = np.stack((self.approx.mean, *(detail.mean for detail in self.details)))
        values.setflags(write=False)
        return values

    @property
    def stds(self) -> np.ndarray:
        """Statistics stacked as ``[A, D1, ..., DL]`` for model variables."""
        values = np.stack((self.approx.std, *(detail.std for detail in self.details)))
        values.setflags(write=False)
        return values

    def with_eps(self, eps: float) -> "WaveletNormStats":
        """Return the same statistics with a validated runtime epsilon."""
        return dataclasses.replace(self, eps=_positive_finite(eps, "eps"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": _FORMAT_VERSION,
            "metadata": {
                "wavelet": _WAVELET,
                "padding_mode": _PADDING_MODE,
                "input_action_normalization": _INPUT_NORMALIZATION,
                "requested_levels": self.requested_levels,
                "levels": self.levels,
                "action_horizon": self.action_horizon,
                "padded_horizon": self.padded_horizon,
                "action_dim": self.action_dim,
                "epsilon": self.eps,
                "band_order": list(_coarse_to_fine_names(self.levels)),
                "sample_count": self.sample_count,
                "source_config": self.source_config,
                "identity": self.is_identity,
            },
            "bands": {
                name: {
                    "mean": band.mean.tolist(),
                    "std": band.std.tolist(),
                    "coefficient_count": band.coefficient_count,
                }
                for name, band in self.bands.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WaveletNormStats":
        if not isinstance(payload, Mapping):
            raise TypeError(f"Wavelet stats root must be an object, got {type(payload).__name__}")
        expected_root_keys = {"format_version", "metadata", "bands"}
        if set(payload) != expected_root_keys:
            raise ValueError(f"Wavelet stats root keys must be {sorted(expected_root_keys)}, got {sorted(payload)}")
        if payload["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported wavelet stats format_version {payload['format_version']!r}; expected {_FORMAT_VERSION}"
            )

        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a JSON object")
        expected_metadata_keys = {
            "action_dim",
            "action_horizon",
            "band_order",
            "epsilon",
            "identity",
            "input_action_normalization",
            "levels",
            "padded_horizon",
            "padding_mode",
            "requested_levels",
            "sample_count",
            "source_config",
            "wavelet",
        }
        if set(metadata) != expected_metadata_keys:
            raise ValueError(f"metadata keys must be {sorted(expected_metadata_keys)}, got {sorted(metadata)}")
        if metadata["wavelet"] != _WAVELET:
            raise ValueError(f"Expected wavelet {_WAVELET!r}, got {metadata['wavelet']!r}")
        if metadata["padding_mode"] != _PADDING_MODE:
            raise ValueError(f"Expected padding_mode {_PADDING_MODE!r}, got {metadata['padding_mode']!r}")
        if metadata["input_action_normalization"] != _INPUT_NORMALIZATION:
            raise ValueError(
                "Wavelet statistics must be computed after OpenPI norm_stats normalization; "
                f"got {metadata['input_action_normalization']!r}"
            )
        if not isinstance(metadata["identity"], bool):
            raise TypeError("metadata.identity must be a boolean")

        levels = _positive_int(metadata["levels"], "metadata.levels")
        expected_order = _coarse_to_fine_names(levels)
        band_order = metadata["band_order"]
        if not isinstance(band_order, list) or tuple(band_order) != expected_order:
            raise ValueError(f"band_order must be {list(expected_order)}, got {band_order!r}")

        raw_bands = payload["bands"]
        if not isinstance(raw_bands, Mapping):
            raise TypeError("bands must be a JSON object")
        if set(raw_bands) != set(expected_order):
            raise ValueError(f"bands must contain exactly {list(expected_order)}, got {sorted(raw_bands)}")

        def parse_band(name: str) -> WaveletBandStats:
            raw_band = raw_bands[name]
            if not isinstance(raw_band, Mapping):
                raise TypeError(f"Band {name} must be a JSON object")
            expected_band_keys = {"mean", "std", "coefficient_count"}
            if set(raw_band) != expected_band_keys:
                raise ValueError(f"Band {name} keys must be {sorted(expected_band_keys)}, got {sorted(raw_band)}")
            return WaveletBandStats(
                mean=np.asarray(raw_band["mean"], dtype=np.float64),
                std=np.asarray(raw_band["std"], dtype=np.float64),
                coefficient_count=raw_band["coefficient_count"],
            )

        return cls(
            approx=parse_band(f"A_{levels}"),
            details=tuple(parse_band(f"D_{index}") for index in range(1, levels + 1)),
            levels=levels,
            requested_levels=metadata["requested_levels"],
            action_dim=metadata["action_dim"],
            eps=metadata["epsilon"],
            action_horizon=metadata["action_horizon"],
            padded_horizon=metadata["padded_horizon"],
            sample_count=metadata["sample_count"],
            source_config=metadata["source_config"],
            is_identity=metadata["identity"],
        )


def identity_wavelet_norm_stats(
    levels: int,
    action_dim: int,
    eps: float,
    action_horizon: int | None = None,
) -> WaveletNormStats:
    """Create an explicit identity fallback.

    Identity mode bypasses all arithmetic, rather than using ``mean=0`` and
    ``std=1`` (which would scale by ``1 + eps``).
    """
    requested_levels = _positive_int(levels, "levels")
    action_dim = _positive_int(action_dim, "action_dim")
    eps = _positive_finite(eps, "eps")
    effective_levels = (
        _effective_levels(action_horizon, requested_levels) if action_horizon is not None else requested_levels
    )
    zeros = np.zeros((action_dim,), dtype=np.float64)
    ones = np.ones((action_dim,), dtype=np.float64)

    def identity_band() -> WaveletBandStats:
        return WaveletBandStats(mean=zeros, std=ones, coefficient_count=0)

    return WaveletNormStats(
        approx=identity_band(),
        details=tuple(identity_band() for _ in range(effective_levels)),
        levels=effective_levels,
        requested_levels=requested_levels,
        action_dim=action_dim,
        eps=eps,
        action_horizon=action_horizon,
        padded_horizon=(_padded_horizon(action_horizon, effective_levels) if action_horizon is not None else None),
        sample_count=0,
        source_config=None,
        is_identity=True,
    )


def save_wavelet_norm_stats(path: str | pathlib.Path, stats: WaveletNormStats) -> pathlib.Path:
    """Save validated statistics as UTF-8 JSON."""
    if not isinstance(stats, WaveletNormStats):
        raise TypeError(f"stats must be WaveletNormStats, got {type(stats).__name__}")
    output_path = pathlib.Path(path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"Wavelet statistics path is a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_wavelet_norm_stats(
    path: str | pathlib.Path | None,
    expected_levels: int | None = None,
    expected_action_dim: int | None = None,
    eps: float | None = None,
    *,
    expected_action_horizon: int | None = None,
    allow_identity_fallback: bool = False,
) -> WaveletNormStats:
    """Load and strictly validate a statistics file.

    A missing path is an error unless ``allow_identity_fallback`` is explicitly
    enabled. Invalid or incompatible files never fall back silently.
    """
    stats_path = None if path is None else pathlib.Path(path)
    if stats_path is None or not stats_path.is_file():
        missing_description = "no path was provided" if stats_path is None else f"{stats_path} does not exist"
        if not allow_identity_fallback:
            raise FileNotFoundError(
                f"Wavelet normalization statistics are required, but {missing_description}. "
                "Run scripts/compute_wavelet_norm_stats.py or explicitly enable identity fallback."
            )
        if expected_levels is None or expected_action_dim is None:
            raise ValueError("Identity fallback requires expected_levels and expected_action_dim")
        fallback_eps = 1e-6 if eps is None else eps
        _LOGGER.warning(
            "Wavelet normalization statistics unavailable (%s); using explicit identity fallback.",
            missing_description,
        )
        return identity_wavelet_norm_stats(
            expected_levels,
            expected_action_dim,
            fallback_eps,
            action_horizon=expected_action_horizon,
        )

    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read wavelet statistics from {stats_path}: {exc}") from exc
    stats = WaveletNormStats.from_dict(payload)

    if expected_levels is not None:
        expected_levels = _positive_int(expected_levels, "expected_levels")
        if stats.requested_levels != expected_levels:
            raise ValueError(
                f"Wavelet stats requested_levels mismatch: file has {stats.requested_levels}, "
                f"model expects {expected_levels}"
            )
    if expected_action_dim is not None:
        expected_action_dim = _positive_int(expected_action_dim, "expected_action_dim")
        if stats.action_dim != expected_action_dim:
            raise ValueError(
                f"Wavelet stats action_dim mismatch: file has {stats.action_dim}, model expects {expected_action_dim}"
            )
    if expected_action_horizon is not None:
        expected_action_horizon = _positive_int(expected_action_horizon, "expected_action_horizon")
        if stats.action_horizon != expected_action_horizon:
            raise ValueError(
                f"Wavelet stats action_horizon mismatch: file has {stats.action_horizon}, "
                f"model expects {expected_action_horizon}"
            )
    if eps is not None:
        stats = stats.with_eps(eps)
    return stats


def _validate_float_band(value: Any, *, name: str, stats: WaveletNormStats, index: int | None) -> jnp.ndarray:
    value = jnp.asarray(value)
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least time and action dimensions, got shape {value.shape}")
    if value.shape[-1] != stats.action_dim:
        raise ValueError(f"{name} has action dimension {value.shape[-1]}, expected {stats.action_dim}")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}")
    if stats.padded_horizon is not None:
        band_name = f"A_{stats.levels}" if index is None else f"D_{index}"
        expected_length = _expected_band_length(stats.padded_horizon, band_name, stats.levels)
        if value.shape[-2] != expected_length:
            raise ValueError(
                f"{name} has temporal length {value.shape[-2]}, expected {expected_length} for {band_name}"
            )
    return value


def _validate_wavelet_bands(
    approx: Any,
    details: Sequence[Any],
    stats: WaveletNormStats,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    if not isinstance(stats, WaveletNormStats):
        raise TypeError(f"stats must be WaveletNormStats, got {type(stats).__name__}")
    details = tuple(details)
    if len(details) != stats.levels:
        raise ValueError(f"Expected {stats.levels} detail arrays in D1..DL order, got {len(details)}")
    approx_array = _validate_float_band(approx, name=f"A_{stats.levels}", stats=stats, index=None)
    detail_arrays = tuple(
        _validate_float_band(detail, name=f"D_{index}", stats=stats, index=index)
        for index, detail in enumerate(details, start=1)
    )
    batch_shape = approx_array.shape[:-2]
    if any(detail.shape[:-2] != batch_shape for detail in detail_arrays):
        raise ValueError("Approximation and detail bands must have identical leading batch dimensions")
    return approx_array, detail_arrays


def _band_transform(
    value: jnp.ndarray,
    band: WaveletBandStats,
    eps: float,
    *,
    operation: str,
) -> jnp.ndarray:
    mean = jnp.asarray(band.mean, dtype=value.dtype)
    scale = jnp.asarray(band.std, dtype=value.dtype) + jnp.asarray(eps, dtype=value.dtype)
    if operation == "normalize_state":
        return (value - mean) / scale
    if operation == "denormalize_state":
        return value * scale + mean
    if operation == "denormalize_velocity":
        # A velocity is a difference of states. Adding the state mean here is incorrect.
        return value * scale
    raise AssertionError(f"Unknown band transform operation: {operation}")


def _transform_wavelet_bands(
    approx: Any,
    details: Sequence[Any],
    stats: WaveletNormStats,
    eps: float | None,
    *,
    operation: str,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    approx_array, detail_arrays = _validate_wavelet_bands(approx, details, stats)
    if stats.is_identity:
        return approx_array, detail_arrays
    runtime_eps = stats.eps if eps is None else _positive_finite(eps, "eps")
    return (
        _band_transform(approx_array, stats.approx, runtime_eps, operation=operation),
        tuple(
            _band_transform(detail, band_stats, runtime_eps, operation=operation)
            for detail, band_stats in zip(detail_arrays, stats.details, strict=True)
        ),
    )


def normalize_wavelet_bands(
    approx: Any,
    details: Sequence[Any],
    stats: WaveletNormStats,
    eps: float | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """Normalize wavelet states with ``(state - mean) / (std + eps)``."""
    return _transform_wavelet_bands(approx, details, stats, eps, operation="normalize_state")


def denormalize_wavelet_state_bands(
    approx: Any,
    details: Sequence[Any],
    stats: WaveletNormStats,
    eps: float | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """Denormalize wavelet states with ``state * (std + eps) + mean``."""
    return _transform_wavelet_bands(approx, details, stats, eps, operation="denormalize_state")


def denormalize_wavelet_velocity_bands(
    approx: Any,
    details: Sequence[Any],
    stats: WaveletNormStats,
    eps: float | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """Denormalize velocities with ``velocity * (std + eps)`` and no mean."""
    return _transform_wavelet_bands(approx, details, stats, eps, operation="denormalize_velocity")


def decompose_wavelet_bands(
    actions: Any,
    levels: int,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], int]:
    """Apply the exact legacy Haar DWT padding and detail-order convention."""
    levels = _positive_int(levels, "levels")
    actions = jnp.asarray(actions)
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [batch, horizon, action_dim], got {actions.shape}")
    if actions.shape[1] < 2:
        raise ValueError(f"actions horizon must be at least 2, got {actions.shape[1]}")
    if actions.shape[2] < 1:
        raise ValueError("actions must have a non-empty action dimension")
    if not jnp.issubdtype(actions.dtype, jnp.floating):
        raise TypeError(f"actions must have a floating dtype, got {actions.dtype}")
    approx, details, actual_levels = wavelet_flow_head.multi_level_haar_dwt(actions, levels)
    return approx, tuple(details), actual_levels


def reconstruct_wavelet_bands(
    approx: Any,
    details: Sequence[Any],
    *,
    target_length: int,
) -> jnp.ndarray:
    """Apply the exact legacy Haar IDWT and crop to ``target_length``."""
    target_length = _positive_int(target_length, "target_length")
    approx = jnp.asarray(approx)
    details = tuple(jnp.asarray(detail) for detail in details)
    if approx.ndim != 3:
        raise ValueError(f"approx must have shape [batch, time, action_dim], got {approx.shape}")
    if not details:
        raise ValueError("At least one detail band is required")
    if any(detail.ndim != 3 for detail in details):
        raise ValueError("Every detail band must have shape [batch, time, action_dim]")
    if any(detail.shape[0] != approx.shape[0] or detail.shape[-1] != approx.shape[-1] for detail in details):
        raise ValueError("Approximation and detail bands must share batch and action dimensions")
    reconstructed_length = details[0].shape[1] * 2
    if target_length > reconstructed_length:
        raise ValueError(f"target_length {target_length} exceeds reconstructed padded length {reconstructed_length}")
    return wavelet_flow_head.multi_level_haar_idwt(approx, details, target_length=target_length)


@dataclasses.dataclass
class _RunningMoments:
    action_dim: int
    count: int = 0
    mean: np.ndarray = dataclasses.field(init=False)
    m2: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros((self.action_dim,), dtype=np.float64)
        self.m2 = np.zeros((self.action_dim,), dtype=np.float64)

    def update(self, values: Any, band_name: str) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != self.action_dim:
            raise ValueError(f"{band_name} must have shape [batch, time, {self.action_dim}], got {values.shape}")
        flat = values.reshape(-1, self.action_dim)
        if not np.all(np.isfinite(flat)):
            raise ValueError(f"{band_name} contains NaN or Inf")
        batch_count = flat.shape[0]
        if batch_count == 0:
            raise ValueError(f"{band_name} batch is empty")
        batch_mean = np.mean(flat, axis=0, dtype=np.float64)
        centered = flat - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total_count)
        self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total_count)
        self.count = total_count

    def finalize(self) -> WaveletBandStats:
        if self.count == 0:
            raise ValueError("Cannot finalize empty wavelet statistics")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return WaveletBandStats(
            mean=self.mean,
            std=np.sqrt(variance),
            coefficient_count=self.count,
        )


def compute_wavelet_norm_stats(
    action_batches: Iterable[Any],
    levels: int,
    eps: float,
    *,
    source_config: str | None = None,
) -> WaveletNormStats:
    """Compute streaming per-band, per-action-dimension statistics.

    Inputs must already be normalized by the regular OpenPI ``norm_stats``
    transform. The function intentionally does not perform action
    normalization itself.
    """
    requested_levels = _positive_int(levels, "levels")
    eps = _positive_finite(eps, "eps")
    running: dict[str, _RunningMoments] | None = None
    action_horizon: int | None = None
    action_dim: int | None = None
    actual_levels: int | None = None
    sample_count = 0

    for batch_index, batch in enumerate(action_batches):
        actions = np.asarray(batch)
        if actions.ndim != 3:
            raise ValueError(
                f"Action batch {batch_index} must have shape [batch, horizon, action_dim], got {actions.shape}"
            )
        if actions.shape[0] == 0:
            raise ValueError(f"Action batch {batch_index} is empty")
        if not np.issubdtype(actions.dtype, np.floating):
            raise TypeError(f"Action batch {batch_index} must be floating, got {actions.dtype}")
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"Action batch {batch_index} contains NaN or Inf")

        if running is None:
            action_horizon = _positive_int(actions.shape[1], "action_horizon")
            if action_horizon < 2:
                raise ValueError("Wavelet statistics require action_horizon >= 2")
            action_dim = _positive_int(actions.shape[2], "action_dim")
            expected_levels = _effective_levels(action_horizon, requested_levels)
            running = {name: _RunningMoments(action_dim) for name in _coarse_to_fine_names(expected_levels)}
        elif actions.shape[1:] != (action_horizon, action_dim):
            raise ValueError(
                f"Action batch {batch_index} shape {actions.shape[1:]} differs from "
                f"the first batch shape {(action_horizon, action_dim)}"
            )

        approx, details, batch_levels = decompose_wavelet_bands(actions, requested_levels)
        if actual_levels is None:
            actual_levels = batch_levels
        elif batch_levels != actual_levels:
            raise ValueError(f"Action batch {batch_index} produced {batch_levels} levels, expected {actual_levels}")
        assert running is not None
        running[f"A_{batch_levels}"].update(approx, f"A_{batch_levels}")
        for index, detail in enumerate(details, start=1):
            running[f"D_{index}"].update(detail, f"D_{index}")
        sample_count += actions.shape[0]

    if running is None or action_horizon is None or action_dim is None or actual_levels is None:
        raise ValueError("No action batches were provided")

    return WaveletNormStats(
        approx=running[f"A_{actual_levels}"].finalize(),
        details=tuple(running[f"D_{index}"].finalize() for index in range(1, actual_levels + 1)),
        levels=actual_levels,
        requested_levels=requested_levels,
        action_dim=action_dim,
        eps=eps,
        action_horizon=action_horizon,
        padded_horizon=_padded_horizon(action_horizon, actual_levels),
        sample_count=sample_count,
        source_config=source_config,
        is_identity=False,
    )
