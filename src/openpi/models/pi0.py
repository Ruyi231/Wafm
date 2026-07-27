import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import wavelet_flow_head as _wavelet_flow_head
from openpi.models import wavelet_normalization as _wavelet_normalization
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_wavelet_flow_head = config.use_wavelet_flow_head
        self.wavelet_flow_mode = config.wavelet_flow_mode
        self.wavelet_flow_impl = config.wavelet_flow_impl
        self.wavelet_levels = config.wavelet_levels
        self.lambda_wavelet_flow_loss = config.lambda_wavelet_flow_loss
        self.lambda_wavelet_recon_loss = config.lambda_wavelet_recon_loss
        self.lambda_wavelet_sparse_gate = config.lambda_wavelet_sparse_gate
        self.lambda_wavelet_gate_supervision = config.lambda_wavelet_gate_supervision
        self.wavelet_use_gripper_transition_label = config.wavelet_use_gripper_transition_label
        self.wavelet_gripper_action_index = config.wavelet_gripper_action_index
        self.wavelet_band_normalization = config.wavelet_band_normalization
        self.wavelet_band_norm_eps = config.wavelet_band_norm_eps
        self.wavelet_shared_noise = config.wavelet_shared_noise
        self.wavelet_band_loss_weights = config.wavelet_band_loss_weights
        self.wavelet_use_action_reconstruction_loss = config.wavelet_use_action_reconstruction_loss
        self.wavelet_use_cross_band_consistency = config.wavelet_use_cross_band_consistency
        self.wavelet_cross_band_consistency_weight = config.wavelet_cross_band_consistency_weight
        if (
            self.use_wavelet_flow_head
            and self.wavelet_flow_impl == "legacy_head"
            and self.wavelet_flow_mode != "replace"
        ):
            raise ValueError(f"Only wavelet_flow_mode='replace' is supported, got {self.wavelet_flow_mode!r}")
        if self.use_wavelet_flow_head and not self.pi05:
            logger.warning(
                "Wavelet flow head is primarily intended for pi0.5; continuing because the token path is compatible."
            )
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.wavelet_flow_head = None
        if self.use_wavelet_flow_head:
            if self.wavelet_flow_impl == "legacy_head":
                self.wavelet_flow_head = _wavelet_flow_head.WaveletSubbandFlowHead(
                    action_dim=config.action_dim,
                    token_dim=action_expert_config.width,
                    levels=config.wavelet_levels,
                    bottleneck_dim=config.wavelet_flow_bottleneck_dim,
                    use_band_gate=config.wavelet_use_band_gate,
                    rngs=rngs,
                )
            else:
                effective_levels, _, _, _ = _wavelet_flow_head.wavelet_layout(
                    config.action_horizon, config.wavelet_levels
                )
                stats = self._load_wavelet_norm_stats(config)
                self.wavelet_flow_head = _wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead(
                    action_dim=config.action_dim,
                    token_dim=action_expert_config.width,
                    action_horizon=config.action_horizon,
                    levels=effective_levels,
                    bottleneck_dim=config.wavelet_flow_bottleneck_dim,
                    hierarchical_coupling=config.wavelet_hierarchical_coupling,
                    detach_coarse_condition=config.wavelet_detach_coarse_condition,
                    conditioning_mode=config.wavelet_conditioning_mode,
                    band_means=stats.means,
                    band_stds=stats.stds,
                    norm_eps=0.0 if stats.is_identity else stats.eps,
                    rngs=rngs,
                )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _load_wavelet_norm_stats(self, config: pi0_config.Pi0Config) -> _wavelet_normalization.WaveletNormStats:
        if not config.wavelet_band_normalization:
            return _wavelet_normalization.identity_wavelet_norm_stats(
                config.wavelet_levels,
                config.action_dim,
                config.wavelet_band_norm_eps,
                action_horizon=config.action_horizon,
            )
        if config.wavelet_norm_stats_path is None:
            if config.wavelet_norm_stats_fallback == "error":
                raise ValueError(
                    "wavelet_band_normalization=True requires wavelet_norm_stats_path. "
                    "Run scripts/compute_wavelet_norm_stats.py or explicitly set "
                    "wavelet_norm_stats_fallback='identity' for an ablation."
                )
            logger.warning("NH-WaFM band normalization requested without statistics; using explicit identity fallback.")
            return _wavelet_normalization.identity_wavelet_norm_stats(
                config.wavelet_levels,
                config.action_dim,
                config.wavelet_band_norm_eps,
                action_horizon=config.action_horizon,
            )
        return _wavelet_normalization.load_wavelet_norm_stats(
            config.wavelet_norm_stats_path,
            expected_levels=config.wavelet_levels,
            expected_action_dim=config.action_dim,
            expected_action_horizon=config.action_horizon,
            eps=config.wavelet_band_norm_eps,
            allow_identity_fallback=config.wavelet_norm_stats_fallback == "identity",
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _predict_action_velocity(
        self,
        noisy_actions: _model.Actions,
        action_token_outputs: at.Float[at.Array, "b ah emb"],
    ) -> tuple[_model.Actions, dict | None]:
        if not self.use_wavelet_flow_head:
            return self.action_out_proj(action_token_outputs), None
        if self.wavelet_flow_impl == "subband_flow":
            raise ValueError("subband_flow predicts normalized band velocities through _predict_subband_velocity")
        if self.wavelet_flow_head is None:
            raise ValueError("use_wavelet_flow_head=True but wavelet_flow_head was not initialized")
        # Legacy replace mode gets the full action velocity from the wavelet head.
        return self.wavelet_flow_head(noisy_actions, action_token_outputs)

    def _subband_head(self) -> _wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead:
        if not isinstance(
            self.wavelet_flow_head,
            _wavelet_flow_head.NormalizedHierarchicalWaveletFlowHead,
        ):
            raise ValueError("wavelet_flow_impl='subband_flow' requires the normalized hierarchical head")
        return self.wavelet_flow_head

    def _sample_subband_noise(
        self,
        rng: at.KeyArrayLike,
        actions: _model.Actions,
    ) -> tuple[at.Array, tuple[at.Array, ...]]:
        head = self._subband_head()
        if self.wavelet_shared_noise:
            # Sampling on the padded horizon keeps the orthonormal Haar
            # coefficients Gaussian even when action_horizon is not a 2**L
            # multiple. Cropping the inverse transform recovers the first
            # action_horizon samples exactly.
            action_noise = jax.random.normal(
                rng,
                (actions.shape[0], head.padded_horizon, actions.shape[-1]),
                dtype=actions.dtype,
            )
            approx, details, actual_levels = _wavelet_flow_head.multi_level_haar_dwt(action_noise, head.levels)
            if actual_levels != head.levels:
                raise AssertionError(f"Expected {head.levels} DWT levels, got {actual_levels}")
            # This branch preserves one action-domain Gaussian draw shared by
            # all bands. Its DWT coefficients are physical states, so they
            # must enter the same normalized coordinates as the data bands.
            return head.normalize_bands(approx, tuple(details))

        keys = jax.random.split(rng, head.levels + 1)
        approx = jax.random.normal(
            keys[0],
            (actions.shape[0], head.approx_length, actions.shape[-1]),
            dtype=actions.dtype,
        )
        details = tuple(
            jax.random.normal(
                keys[level],
                (actions.shape[0], head.detail_lengths[level - 1], actions.shape[-1]),
                dtype=actions.dtype,
            )
            for level in range(1, head.levels + 1)
        )
        return approx, details

    def _subband_state_to_actions(
        self,
        normalized_approx: at.Array,
        normalized_details: tuple[at.Array, ...],
    ) -> _model.Actions:
        head = self._subband_head()
        approx, details = head.denormalize_state_bands(normalized_approx, normalized_details)
        return _wavelet_flow_head.multi_level_haar_idwt(approx, details, target_length=self.action_horizon)

    def _subband_velocity_to_actions(
        self,
        normalized_approx_velocity: at.Array,
        normalized_detail_velocities: tuple[at.Array, ...],
    ) -> _model.Actions:
        head = self._subband_head()
        approx_velocity, detail_velocities = head.denormalize_velocity_bands(
            normalized_approx_velocity,
            normalized_detail_velocities,
        )
        return _wavelet_flow_head.multi_level_haar_idwt(
            approx_velocity,
            detail_velocities,
            target_length=self.action_horizon,
        )

    def _predict_subband_velocity(
        self,
        normalized_approx: at.Array,
        normalized_details: tuple[at.Array, ...],
        action_token_outputs: at.Float[at.Array, "b ah emb"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Array, tuple[at.Array, ...], dict[str, object]]:
        return self._subband_head()(
            normalized_approx,
            normalized_details,
            action_token_outputs,
            timestep,
        )

    def _subband_losses(
        self,
        predicted_approx: at.Array,
        predicted_details: tuple[at.Array, ...],
        target_approx: at.Array,
        target_details: tuple[at.Array, ...],
    ) -> tuple[at.Array, at.Array, dict[str, at.Array]]:
        head = self._subband_head()
        if len(predicted_details) != head.levels or len(target_details) != head.levels:
            raise ValueError("Subband loss received a detail tuple with the wrong static length")
        weights = (
            (1.0,) * (head.levels + 1) if self.wavelet_band_loss_weights is None else self.wavelet_band_loss_weights
        )

        approx_error = jnp.square(predicted_approx - target_approx)
        per_example = weights[0] * jnp.mean(approx_error, axis=(1, 2))
        metrics: dict[str, at.Array] = {
            "loss_band_A": jnp.mean(approx_error),
            "band_target_energy_A": jnp.mean(jnp.square(target_approx)),
            "band_pred_energy_A": jnp.mean(jnp.square(predicted_approx)),
        }

        for level in range(head.levels, 0, -1):
            predicted = predicted_details[level - 1]
            target = target_details[level - 1]
            error = jnp.square(predicted - target)
            weight_index = head.levels - level + 1
            per_example = per_example + weights[weight_index] * jnp.mean(error, axis=(1, 2))
            metrics[f"loss_band_D{level}"] = jnp.mean(error)
            metrics[f"band_target_energy_D{level}"] = jnp.mean(jnp.square(target))
            metrics[f"band_pred_energy_D{level}"] = jnp.mean(jnp.square(predicted))

        # Cross-band consistency is the physical (denormalized) squared-energy
        # allocation over the orthonormal Haar coefficients. Summing rather
        # than averaging accounts for the different coefficient counts.
        physical_predicted_approx, physical_predicted_details = head.denormalize_velocity_bands(
            predicted_approx,
            predicted_details,
        )
        physical_target_approx, physical_target_details = head.denormalize_velocity_bands(
            target_approx,
            target_details,
        )
        predicted_energy = jnp.stack(
            (
                jnp.sum(jnp.square(physical_predicted_approx), axis=(1, 2)),
                *(
                    jnp.sum(jnp.square(physical_predicted_details[level - 1]), axis=(1, 2))
                    for level in range(head.levels, 0, -1)
                ),
            ),
            axis=-1,
        )
        target_energy = jnp.stack(
            (
                jnp.sum(jnp.square(physical_target_approx), axis=(1, 2)),
                *(
                    jnp.sum(jnp.square(physical_target_details[level - 1]), axis=(1, 2))
                    for level in range(head.levels, 0, -1)
                ),
            ),
            axis=-1,
        )
        energy_eps = jnp.asarray(1e-8, dtype=predicted_energy.dtype)
        predicted_fraction = predicted_energy / (jnp.sum(predicted_energy, axis=-1, keepdims=True) + energy_eps)
        target_fraction = target_energy / (jnp.sum(target_energy, axis=-1, keepdims=True) + energy_eps)
        cross_per_example = jnp.mean(jnp.square(predicted_fraction - target_fraction), axis=-1)
        metrics["loss_cross_band_consistency"] = jnp.mean(cross_per_example)
        metrics["loss_band_total"] = jnp.mean(per_example)
        return per_example, cross_per_example, metrics

    def _wavelet_flow_loss(
        self,
        u_t: _model.Actions,
        wavelet_info: dict | None,
    ) -> at.Float[at.Array, ""]:
        if wavelet_info is None:
            return jnp.asarray(0.0, dtype=u_t.dtype)
        actual_levels = int(wavelet_info["wavelet_levels"])
        u_approx, u_details, _ = _wavelet_flow_head.multi_level_haar_dwt(u_t, actual_levels)
        loss = jnp.mean(jnp.square(wavelet_info["wavelet_flow_approx"] - u_approx))
        for v_detail, u_detail in zip(wavelet_info["wavelet_flow_details"], u_details, strict=True):
            loss = loss + jnp.mean(jnp.square(v_detail - u_detail))
        return loss

    def _wavelet_sparse_gate_loss(self, wavelet_info: dict | None) -> at.Float[at.Array, ""]:
        if wavelet_info is None or wavelet_info.get("wavelet_gates") is None:
            return jnp.asarray(0.0)
        return jnp.mean(wavelet_info["wavelet_gates"])

    def _normalize_wavelet_gate_label(self, label: at.Array, gates: at.Array) -> at.Array:
        num_bands = gates.shape[1]
        if label.ndim == 1:
            return jnp.broadcast_to(label[:, None], gates.shape)
        if label.ndim == 2 and label.shape[1] == 1:
            return jnp.broadcast_to(label, gates.shape)
        if label.ndim == 2 and label.shape[1] == num_bands:
            return label
        if label.ndim == 2:
            # Reduce an informative-frame score [B, T] globally, then broadcast to bands.
            score = jnp.max(label, axis=1, keepdims=True)
            return jnp.broadcast_to(score, gates.shape)
        return jnp.zeros_like(gates)

    def _wavelet_gate_supervision_loss(
        self, actions: _model.Actions, wavelet_info: dict | None, extras: dict | None = None
    ) -> at.Float[at.Array, ""]:
        if wavelet_info is None or wavelet_info.get("wavelet_gates") is None:
            return jnp.asarray(0.0, dtype=actions.dtype)

        gates = wavelet_info["wavelet_gates"]
        labels = None
        if extras is not None:
            for key in ("wavelet_gate_label", "frameskip_chunk_label", "frameskip_score"):
                if key in extras:
                    labels = self._normalize_wavelet_gate_label(jnp.asarray(extras[key], dtype=gates.dtype), gates)
                    break

        if self.wavelet_use_gripper_transition_label and self.wavelet_gripper_action_index is not None:
            gripper_index = self.wavelet_gripper_action_index
            if 0 <= gripper_index < actions.shape[-1] and gates.shape[1] >= 2:
                gripper = actions[..., gripper_index]
                transition = jnp.abs(gripper[:, 1:] - gripper[:, :-1])
                score = jnp.max(transition, axis=1)
                score = score / (jnp.max(score) + 1e-6)
                gripper_labels = jnp.zeros_like(gates).at[:, 1].set(score)
                labels = gripper_labels if labels is None else jnp.clip(0.5 * labels + 0.5 * gripper_labels, 0.0, 1.0)

        if labels is None:
            return jnp.asarray(0.0, dtype=actions.dtype)
        return jnp.mean(jnp.square(gates - labels))

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        return_metrics: bool = False,
        extras: dict | None = None,
    ) -> at.Float[at.Array, "*b ah"] | tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        subband_mode = self.use_wavelet_flow_head and self.wavelet_flow_impl == "subband_flow"
        if subband_mode:
            head = self._subband_head()
            data_approx, data_details, actual_levels = _wavelet_flow_head.multi_level_haar_dwt(
                actions,
                head.levels,
            )
            if actual_levels != head.levels:
                raise AssertionError(f"Expected {head.levels} DWT levels, got {actual_levels}")
            normalized_data_approx, normalized_data_details = head.normalize_bands(
                data_approx,
                tuple(data_details),
            )
            noise_approx, noise_details = self._sample_subband_noise(noise_rng, actions)
            (
                normalized_state_approx,
                normalized_state_details,
                target_approx,
                target_details,
            ) = _wavelet_flow_head.subband_flow_bridge(
                normalized_data_approx,
                normalized_data_details,
                noise_approx,
                noise_details,
                time,
            )
            x_t = self._subband_state_to_actions(normalized_state_approx, normalized_state_details)
        else:
            noise = jax.random.normal(noise_rng, actions.shape)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        del prefix_out
        action_token_outputs = suffix_out[:, -self.action_horizon :]

        if subband_mode:
            predicted_approx, predicted_details, _ = self._predict_subband_velocity(
                normalized_state_approx,
                normalized_state_details,
                action_token_outputs,
                time,
            )
            band_per_example, cross_per_example, metrics = self._subband_losses(
                predicted_approx,
                predicted_details,
                target_approx,
                target_details,
            )
            loss = jnp.broadcast_to(band_per_example[:, None], actions.shape[:-1])

            predicted_action_velocity = self._subband_velocity_to_actions(
                predicted_approx,
                predicted_details,
            )
            target_action_velocity = self._subband_velocity_to_actions(
                target_approx,
                target_details,
            )
            action_reconstruction = jnp.mean(
                jnp.square(predicted_action_velocity - target_action_velocity),
                axis=-1,
            )
            metrics["loss_action_reconstruction"] = jnp.mean(action_reconstruction)
            if self.wavelet_use_action_reconstruction_loss:
                loss = loss + self.lambda_wavelet_recon_loss * action_reconstruction
            if self.wavelet_use_cross_band_consistency:
                loss = loss + self.wavelet_cross_band_consistency_weight * cross_per_example[:, None]
            metrics["loss_action_flow"] = jnp.mean(action_reconstruction)
            if return_metrics:
                return loss, metrics
            return loss

        v_t, wavelet_info = self._predict_action_velocity(x_t, action_token_outputs)

        loss_action = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        loss = loss_action
        metrics: dict[str, at.Array] = {}
        if self.use_wavelet_flow_head:
            loss_wavelet = self._wavelet_flow_loss(u_t, wavelet_info)
            loss_sparse_gate = self._wavelet_sparse_gate_loss(wavelet_info)
            loss_gate_supervision = self._wavelet_gate_supervision_loss(actions, wavelet_info, extras)
            loss = (
                loss
                + self.lambda_wavelet_flow_loss * loss_wavelet
                + self.lambda_wavelet_sparse_gate * loss_sparse_gate
                + self.lambda_wavelet_gate_supervision * loss_gate_supervision
            )
            metrics = {
                "loss_action_flow": jnp.mean(loss_action),
                "loss_wavelet_flow": loss_wavelet,
                "loss_wavelet_sparse_gate": loss_sparse_gate,
                "loss_wavelet_gate_supervision": loss_gate_supervision,
            }
            if wavelet_info is not None and wavelet_info.get("wavelet_gate_mean") is not None:
                metrics["wavelet_gate_mean"] = wavelet_info["wavelet_gate_mean"]
            if wavelet_info is not None:
                for key in (
                    "wavelet_gate_A_mean",
                    "wavelet_gate_D1_mean",
                    "wavelet_gate_D2_mean",
                    "wavelet_gate_D3_mean",
                ):
                    if key in wavelet_info:
                        metrics[key] = wavelet_info[key]

        if return_metrics:
            return loss, metrics
        return loss

    def _sample_actions_subband(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""],
        noise: at.Float[at.Array, "b ah ad"] | None,
    ) -> _model.Actions:
        head = self._subband_head()
        batch_size = observation.state.shape[0]
        action_template = jnp.zeros(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=observation.state.dtype,
        )
        if noise is None:
            normalized_approx, normalized_details = self._sample_subband_noise(rng, action_template)
        else:
            if noise.shape != action_template.shape:
                raise ValueError(f"noise must have shape {action_template.shape}, got {noise.shape}")
            noise_approx, details, actual_levels = _wavelet_flow_head.multi_level_haar_dwt(
                noise,
                head.levels,
            )
            if actual_levels != head.levels:
                raise AssertionError(f"Expected {head.levels} DWT levels, got {actual_levels}")
            normalized_approx, normalized_details = head.normalize_bands(noise_approx, tuple(details))

        dt = -1.0 / num_steps
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def step(carry):
            current_approx, current_details, time = carry
            current_actions = self._subband_state_to_actions(current_approx, current_details)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                current_actions,
                jnp.broadcast_to(time, batch_size),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cached_prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cached_prefix_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            if prefix_out is not None:
                raise AssertionError("Cached prefix output must be None during suffix decoding")
            action_token_outputs = suffix_out[:, -self.action_horizon :]
            approx_velocity, detail_velocities, _ = self._predict_subband_velocity(
                current_approx,
                current_details,
                action_token_outputs,
                jnp.broadcast_to(time, batch_size),
            )
            next_approx = current_approx + dt * approx_velocity
            next_details = tuple(
                detail + dt * velocity for detail, velocity in zip(current_details, detail_velocities, strict=True)
            )
            return next_approx, next_details, time + dt

        def cond(carry):
            _, _, time = carry
            return time >= -dt / 2

        initial_time = jnp.asarray(1.0, dtype=normalized_approx.dtype)
        final_approx, final_details, _ = jax.lax.while_loop(
            cond,
            step,
            (normalized_approx, normalized_details, initial_time),
        )
        return self._subband_state_to_actions(final_approx, final_details)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        if self.use_wavelet_flow_head and self.wavelet_flow_impl == "subband_flow":
            return self._sample_actions_subband(
                rng,
                observation,
                num_steps=num_steps,
                noise=noise,
            )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            action_token_outputs = suffix_out[:, -self.action_horizon :]
            v_t, _ = self._predict_action_velocity(x_t, action_token_outputs)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
