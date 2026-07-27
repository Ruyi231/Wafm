import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA / optional wavelet-head weights.
        return _merge_params(loaded_params, params, missing_regex=".*(lora|wavelet_flow_head).*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    pattern = re.compile(missing_regex)
    common = flat_loaded.keys() & flat_ref.keys()
    shape_mismatches = [
        f"{key}: expected {flat_ref[key].shape}, got {flat_loaded[key].shape}"
        for key in sorted(common)
        if flat_loaded[key].shape != flat_ref[key].shape
    ]
    if shape_mismatches:
        raise ValueError(
            "Checkpoint parameter shape mismatch:\n" + "\n".join(f"  - {item}" for item in shape_mismatches)
        )

    missing = sorted(flat_ref.keys() - flat_loaded.keys())
    allowed_missing = [key for key in missing if pattern.fullmatch(key)]
    required_missing = [key for key in missing if not pattern.fullmatch(key)]
    if required_missing:
        raise ValueError(
            "Checkpoint is missing required parameters:\n" + "\n".join(f"  - {key}" for key in required_missing)
        )

    extra = sorted(flat_loaded.keys() - flat_ref.keys())
    if extra:
        logger.warning(
            "Dropping %d checkpoint parameter(s) not present in the current model:\n%s",
            len(extra),
            "\n".join(f"  - {key}" for key in extra),
        )
    if allowed_missing:
        logger.warning(
            "Checkpoint is missing %d permitted parameter(s); retaining their initialized values:\n%s",
            len(allowed_missing),
            "\n".join(f"  - {key}" for key in allowed_missing),
        )

    result = {
        key: value.astype(flat_ref[key].dtype) if value.dtype != flat_ref[key].dtype else value
        for key, value in flat_loaded.items()
        if key in flat_ref
    }
    result.update({key: flat_ref[key] for key in allowed_missing})

    return flax.traverse_util.unflatten_dict(result, sep="/")
