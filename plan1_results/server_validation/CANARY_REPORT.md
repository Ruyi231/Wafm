# Plan 1 `plan1_subband_l2_norm` GPU Canary

## Scope

This is a debugging canary for the single normalized level-2 subband-flow
configuration. It is not a formal Stage 1 comparison result.

- config: `plan1_subband_l2_norm`
- dataset: `/nfs/lizhenhao/huggingface/lerobot/libero_full`
- total steps: 2,000
- batch size: 2
- debug warmup: 200 steps
- peak learning rate: `5e-5`
- GPU: one NVIDIA H200
- checkpoint phases: steps 0–999, then resume from state step 1,000 and train to step 2,000

## Outcome

Status: **canary passed**.

| Fixed validation metric | Initial | Step 2,000 | Final / initial |
|:---|---:|---:|---:|
| `loss_band_A` | 0.968391 | 0.562126 | 0.580474 |
| `loss_band_D1` | 1.272390 | 0.757968 | 0.595704 |
| `loss_band_D2` | 1.394368 | 0.893866 | 0.641055 |
| `loss_action_reconstruction` | 0.092473 | 0.059268 | 0.640920 |

The fixed validation batch used a fixed evaluation RNG. All four metrics
improved over the complete run. Random training-flow loss also trended down,
with normal short-run noise.

All recorded band-head gradients were finite and nonzero:

| Gradient norm | Minimum | Maximum | Final interval |
|:---|---:|---:|---:|
| A | 0.801120 | 2.917430 | 2.917430 |
| D1 | 0.634430 | 1.998788 | 1.998788 |
| D2 | 0.840266 | 2.122114 | 2.122114 |

No NaN or Inf was observed. After compilation, median step time was
`0.2042 s` and mean step time was `0.2042 s`. Runtime observation showed about
`108153 MiB` device allocation; JAX reported `69735172608` peak bytes in use.

Checkpoint step `999` was successfully restored: the second phase reported
`restored_start_step=1000` and completed to checkpoint `1999`. Orbax retained
the latest 42 GB checkpoint and removed `999` under the configured retention
policy. The nonfatal `Missing metrics for step 999` warning is the same Orbax
metrics-file warning seen in the earlier resume smoke; model and optimizer
state restoration succeeded.

## Interpretation

The deterministic overfit confirmation and this stochastic canary together
support proceeding to formal Stage 1 correctness experiments. They do not
establish LIBERO task success rate or comparative superiority. Formal runs must
restore the shared original warmup and training budget and run sequentially:

1. `plan1_pi05_libero_baseline`
2. `plan1_legacy_wafm_l2`
3. `plan1_subband_l2_no_norm`
4. `plan1_subband_l2_norm`
