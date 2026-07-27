import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Wavelet flow is opt-in; disabled keeps the original action_out_proj path.
    use_wavelet_flow_head: bool = False
    wavelet_flow_mode: str = "replace"
    # Existing configs default to the legacy action-domain bridge. NH-WaFM is
    # selected explicitly with "subband_flow".
    wavelet_flow_impl: str = "legacy_head"
    wavelet_levels: int = 3
    wavelet_flow_bottleneck_dim: int = 128
    wavelet_use_band_gate: bool = False
    lambda_wavelet_flow_loss: float = 0.1
    lambda_wavelet_recon_loss: float = 0.0
    lambda_wavelet_sparse_gate: float = 0.0
    lambda_wavelet_gate_supervision: float = 0.0
    wavelet_use_gripper_transition_label: bool = False
    wavelet_gripper_action_index: int | None = None

    # NH-WaFM options. These defaults are inert for pi0.5 and legacy WaFM.
    wavelet_band_normalization: bool = False
    wavelet_band_norm_eps: float = 1e-6
    wavelet_norm_stats_path: str | None = None
    wavelet_norm_stats_fallback: str = "error"
    wavelet_hierarchical_coupling: bool = False
    # False is the canonical NH-WaFM prior: independent N(0, I) in each
    # normalized band. True is an action-domain shared-noise ablation.
    wavelet_shared_noise: bool = False
    wavelet_band_loss_weights: tuple[float, ...] | None = None
    wavelet_detach_coarse_condition: bool = False
    wavelet_use_action_reconstruction_loss: bool = False
    wavelet_use_cross_band_consistency: bool = False
    wavelet_cross_band_consistency_weight: float = 0.0
    wavelet_conditioning_mode: str = "temporal_pooling"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.wavelet_flow_impl not in ("legacy_head", "subband_flow"):
            raise ValueError(
                f"wavelet_flow_impl must be 'legacy_head' or 'subband_flow', got {self.wavelet_flow_impl!r}"
            )
        if self.wavelet_levels < 1:
            raise ValueError(f"wavelet_levels must be at least 1, got {self.wavelet_levels}")
        if not math.isfinite(self.wavelet_band_norm_eps) or self.wavelet_band_norm_eps <= 0:
            raise ValueError(f"wavelet_band_norm_eps must be positive, got {self.wavelet_band_norm_eps}")
        if self.wavelet_norm_stats_fallback not in ("error", "identity"):
            raise ValueError(
                f"wavelet_norm_stats_fallback must be 'error' or 'identity', got {self.wavelet_norm_stats_fallback!r}"
            )
        if self.wavelet_conditioning_mode not in ("temporal_pooling", "band_query"):
            raise ValueError(
                "wavelet_conditioning_mode must be 'temporal_pooling' or 'band_query', "
                f"got {self.wavelet_conditioning_mode!r}"
            )
        if self.wavelet_band_loss_weights is not None:
            effective_levels = min(self.wavelet_levels, max(0, (self.action_horizon - 1).bit_length()))
            expected_bands = effective_levels + 1
            if len(self.wavelet_band_loss_weights) != expected_bands:
                raise ValueError(
                    "wavelet_band_loss_weights uses prediction order [A_L, D_L, ..., D_1] and must contain "
                    f"{expected_bands} entries, got {len(self.wavelet_band_loss_weights)}"
                )
            if any(not math.isfinite(weight) or weight < 0 for weight in self.wavelet_band_loss_weights):
                raise ValueError("wavelet_band_loss_weights must be finite and non-negative")
        if (
            not math.isfinite(self.wavelet_cross_band_consistency_weight)
            or self.wavelet_cross_band_consistency_weight < 0
        ):
            raise ValueError("wavelet_cross_band_consistency_weight must be finite and non-negative")
        if self.use_wavelet_flow_head and self.wavelet_flow_impl == "subband_flow":
            if self.action_horizon < 2:
                raise ValueError("subband_flow requires action_horizon >= 2")
            if self.wavelet_use_action_reconstruction_loss and (
                not math.isfinite(self.lambda_wavelet_recon_loss) or self.lambda_wavelet_recon_loss <= 0
            ):
                raise ValueError(
                    "wavelet_use_action_reconstruction_loss=True requires a finite lambda_wavelet_recon_loss > 0"
                )
            if self.wavelet_use_cross_band_consistency and self.wavelet_cross_band_consistency_weight <= 0:
                raise ValueError(
                    "wavelet_use_cross_band_consistency=True requires wavelet_cross_band_consistency_weight > 0"
                )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0  # noqa: PLC0415

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
