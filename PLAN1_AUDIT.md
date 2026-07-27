# Plan 1 pre-implementation audit

## Scope and audit status

This audit covers the JAX/Flax NNX implementation on branch
`codex/nh-wafm-plan1`, based on remote branch `origin/decrete_band` at commit
`bf54f86` (`add calvin test`). No NH-WaFM model code was changed before this
document was completed.

The repository was fetched as a shallow clone. The working tree was clean at
the start of the audit.

## 1. Training entry point and configuration system

- The JAX training entry point is `scripts/train.py`.
- `scripts/train.py` obtains a `TrainConfig` through
  `openpi.training.config.cli()`, which uses
  `tyro.extras.overridable_config_cli`.
- All named experiments are registered in the `_CONFIGS` list in
  `src/openpi/training/config.py`; `_CONFIGS_DICT` is the name lookup table.
- `TrainConfig` owns the model config, data factory, optimizer, schedule,
  checkpoint paths, batch size, training length, logging, FSDP and resume
  settings.
- `Pi0Config` in `src/openpi/models/pi0_config.py` owns the current wavelet
  feature flags.
- The data pipeline order is:

  1. dataset repacking;
  2. robot/data-specific transforms;
  3. OpenPI normalization (`transforms.Normalize`);
  4. model-specific transforms, including prompt/image processing and
     state/action dimension padding.

  Therefore a wavelet-statistics script must use the normal training data
  loader (or reproduce this exact ordering) and collect `batch["actions"]`
  after step 3. The current `scripts/compute_norm_stats.py` intentionally
  collects data before OpenPI normalization and is not sufficient for wavelet
  band statistics.

- `train.sh` currently launches
  `wafm_l2_nogate_loss_calvin_ABC_D_with_state` with two GPUs and offline W&B.

## 2. Current `Pi0.compute_loss()` probability path

`Pi0.compute_loss()` currently:

1. splits the RNG into preprocessing, noise and time keys;
2. preprocesses the observation;
3. samples action-domain Gaussian noise with the same shape as the action
   tensor;
4. samples one beta-distributed time per batch item:
   `Beta(1.5, 1) * 0.999 + 0.001`;
5. constructs the action-domain path
   `x_t = t * noise + (1 - t) * actions`;
6. constructs the action-domain target velocity
   `u_t = noise - actions`;
7. embeds `x_t` as action-expert suffix tokens and runs PaliGemma/action expert;
8. projects the action-expert outputs through either the original
   `action_out_proj` or the legacy `WaveletSubbandFlowHead`;
9. always computes action-domain MSE, averaged over action dimension and
   returned with shape `[batch, action_horizon]`.

In legacy WaFM mode, an additional loss DWT-transforms the already constructed
action-domain target `u_t` and compares its bands against the head bands.
Consequently the current implementation does **not** construct the flow bridge
in wavelet space.

The config contains `lambda_wavelet_recon_loss`, but the current loss path does
not use it.

## 3. Current `Pi0.sample_actions()` ODE integration

- Sampling starts from action-domain Gaussian noise shaped
  `[batch, action_horizon, action_dim]`.
- It uses explicit Euler integration from `t=1` to `t=0` with
  `dt = -1 / num_steps`.
- The prefix KV cache is computed once.
- Each `jax.lax.while_loop` iteration embeds the current action-domain `x_t`,
  runs the action expert, predicts an action-domain velocity, and applies
  `x_t <- x_t + dt * v_t`.
- The loop carry is `(x_t, time)`.
- Legacy WaFM still maintains only action-domain state. It DWT-transforms
  `x_t` inside the output head on every step, reconstructs a velocity using
  IDWT, and updates action-domain `x_t`.

Thus a real `subband_flow` mode needs a separate fixed-structure loop carry
containing normalized bands; it cannot reuse this state transition unchanged.

## 4. Current `WaveletSubbandFlowHead`

Inputs:

- noisy actions: `[B, H, A]`;
- action-expert token outputs: `[B, H, E]`.

Transform and band shapes:

- the input horizon `H` is padded once by edge replication to
  `P = ceil(H / 2^L) * 2^L`;
- `A_L` has shape `[B, P / 2^L, A]`;
- `D_i` has shape `[B, P / 2^i, A]`, for `i=1..L`;
- action tokens are passed through the same fixed Haar DWT, producing matching
  temporal lengths and feature dimension `E`;
- details are stored in code order `[D_1, D_2, ..., D_L]`.

Architecture and outputs:

- one independent FiLM MLP predicts `A_L`;
- one independent FiLM MLP predicts each `D_i`;
- the FiLM condition for a band is the Haar-transformed action token at the
  same band;
- there is no coarse-to-fine dependency between predicted bands;
- the timestep is not an explicit head input. It is only implicit in the
  action-expert output (and, for pi0.5, its adaRMS conditioning);
- an optional global sigmoid gate multiplies whole-band velocities;
- IDWT produces an action-domain velocity `[B, H, A]`;
- the auxiliary info dictionary exposes band states, token bands, predicted
  velocities, level count and optional gates.

The existing fixed DWT of action tokens is a temporal transform, not evidence
that the tokens have learned band-specific semantics. It must remain only in
the legacy path.

## 5. Padding and cropping

- `multi_level_haar_dwt()` clamps the effective number of levels for very
  short inputs, pads the full horizon once to a multiple of `2**levels`, and
  uses edge padding.
- `haar_dwt_1d()` also defensively edge-pads an odd single-level input.
- `multi_level_haar_idwt()` reconstructs details in reverse order and crops
  `x[:, :target_length, :]`.
- Existing tests cover round trips for horizons 10 and 11 at level 3.

NH-WaFM statistics, training and sampling must reuse these exact rules. The
original horizon must remain static model metadata so cropping is JIT-stable.

## 6. Checkpoint loading compatibility

There are three different paths:

1. **Fine-tuning initialization**:
   `CheckpointWeightLoader` restores a checkpoint and `_merge_params()` fills
   missing parameters matching `.*(lora|wavelet_flow_head).*` from the newly
   initialized reference tree. This can initialize new modules if they remain
   under `wavelet_flow_head`.
2. **Training resume**:
   Orbax restores the full saved training state against the current tree.
   This is suitable for resuming an unchanged architecture, not migrating an
   old optimizer/model tree into a new architecture.
3. **Policy/inference loading**:
   `BaseModelConfig.load()` intersects extra checkpoint parameters, but then
   requires the resulting parameter tree to exactly equal the newly created
   model tree. A pi0.5 or legacy WaFM checkpoint that lacks newly configured
   NH-WaFM parameters will fail this check.

Current shortcomings:

- allowed missing optional parameters are not listed in logs;
- checkpoint-only extra keys are dropped without a clear per-key log;
- the inference loader does not initialize allowed missing new modules;
- there is no unit test for partial checkpoint loading.

Compatibility work must add explicit missing/extra-key reporting and restrict
partial initialization to named optional module paths. Shape mismatches for
existing keys must remain errors.

## 7. Python, JAX, Flax NNX and type constraints

Declared project environment:

- Python: `>=3.11`; Ruff target is Python 3.11;
- JAX: `jax[cuda12]==0.5.3`;
- Flax: `0.10.2`, using `flax.nnx` and `flax.nnx.bridge`;
- Orbax checkpoint: `0.11.13`;
- jaxtyping: `0.2.36`;
- beartype: `0.19.0`;
- NumPy: `>=1.22.4,<2.0.0`.

The RLDS optional dependency group specifically requires Python 3.11 because
TensorFlow CPU 2.15 only provides the required wheel there.

Type/runtime constraints:

- public model methods use `openpi.shared.array_typing` annotations and
  `@at.typecheck`;
- array typing is runtime-checked with jaxtyping/beartype unless explicitly
  disabled;
- `jax.lax.while_loop` carries must have an invariant pytree structure, shapes
  and dtypes;
- band containers inside JIT code must be fixed tuples/static module
  structures, not dynamically growing Python lists;
- the repository Ruff line length is 120 and imports are single-line sorted.

Current audit machine:

- `python` is CPython 3.12.7 from `D:\anaconda3\python.exe`;
- JAX and Flax are not installed in that interpreter;
- `uv` is not available on `PATH`.

Accordingly, source-level inspection is complete, but JAX/NNX tests cannot be
claimed as executed until a compatible environment is made available.

## 8. Existing training, evaluation and experiment configuration

Training:

- JAX: `scripts/train.py`;
- PyTorch: `scripts/train_pytorch.py` (the current wavelet implementation is
  JAX-only);
- normalization: `scripts/compute_norm_stats.py`;
- smoke training test: `scripts/train_test.py` runs and resumes two/four debug
  steps on fake data.

Evaluation/serving:

- `scripts/serve_policy.py` creates a policy from a named config and
  checkpoint;
- `examples/libero/main.py` runs LIBERO rollouts and reports task and aggregate
  success rate;
- `examples/libero/compose.yml` provides the recommended server/client
  workflow;
- `server.sh` currently serves a CALVIN WaFM checkpoint, but this checkout
  does not contain a complete CALVIN benchmark driver comparable to the
  LIBERO client;
- policy adapters exist for LIBERO, CALVIN, ALOHA and DROID.

Relevant existing named configs include:

- original `pi05_libero`;
- legacy WaFM level 1/2/3, gate/no-gate and wavelet-loss ablations for LIBERO;
- legacy WaFM CALVIN configurations with and without state;
- `debug`, `debug_restore` and `debug_pi05`.

The current repository does not provide a ready RoboTwin experiment.

## 9. Existing test coverage and gaps

Existing wavelet tests verify:

- Haar DWT/IDWT reconstruction for even and odd horizons;
- legacy head output and gate shapes.

Existing model tests verify:

- original pi0 loss and 10-step sampling;
- legacy pi0.5 WaFM loss metrics and 2-step sampling;
- a manual original checkpoint restore.

Missing tests map directly to Plan 1 requirements:

- band normalization and velocity denormalization;
- subband probability-path target construction;
- independent/hierarchical/detached hierarchical heads;
- temporal-pooling and band-query conditions;
- one/five/ten-step subband sampling;
- partial checkpoint initialization and explicit reporting;
- no-NaN forward/backward;
- small-data overfit.

## 10. Implementation guardrails derived from the audit

The implementation will preserve three explicit modes:

- original pi0.5: `use_wavelet_flow_head=False`;
- legacy WaFM: `wavelet_flow_impl="legacy_head"`;
- NH-WaFM: `wavelet_flow_impl="subband_flow"`.

The legacy default and parameter names will remain unchanged. NH-WaFM
parameters will live under the optional `wavelet_flow_head` subtree where
possible. Gate losses stay available only to legacy configs and default off
for NH-WaFM.

NH-WaFM will:

- construct normalized state and velocity targets per band before the head;
- denormalize velocities with `velocity = normalized_velocity * std` (no mean
  addition);
- predict in `A_L -> D_L -> ... -> D_1` order;
- use fixed tuples and static band metadata in JIT/ODE code;
- preserve action-domain output shape and crop semantics;
- fail clearly when band normalization is enabled without valid statistics,
  unless an explicit fallback mode is configured.

## 11. Audit completion decision

The repository is suitable for a backward-compatible NH-WaFM implementation,
but checkpoint inference loading and local test environment availability are
known blockers that must be handled explicitly. Large-scale training must not
start before the required unit tests and synthetic overfit test pass.
