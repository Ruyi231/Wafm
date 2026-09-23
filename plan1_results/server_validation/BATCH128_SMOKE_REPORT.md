# `plan1_subband_l2_norm` Batch-128 GPU Smoke

## Scope

This validation checks whether the complete normalized level-2 subband-flow
model can execute one real-data forward/backward update at global batch size
128. It is a runtime and memory smoke test, not a convergence or performance
experiment.

- config: `plan1_subband_l2_norm`
- global batch size: 128
- action shape: `(128, 10, 32)`
- image shape: three views of `(128, 224, 224, 3)`
- devices: H200 GPU 0 and GPU 2
- JAX device count: 2
- FSDP devices: 2
- checkpoint saved: no

## Result

Status: **passed**.

The complete initial evaluation, backward pass, optimizer update, and
post-update evaluation finished without OOM, NaN, or Inf.

| Metric | Before update | After update |
|:---|---:|---:|
| total band loss | 3.998513 | 3.986695 |
| A loss | 1.314303 | 1.310264 |
| D1 loss | 1.295793 | 1.292799 |
| D2 loss | 1.388417 | 1.383632 |
| IDWT action reconstruction | 0.129601 | 0.129278 |

Post-update band-head gradient norms were finite and nonzero:

- A: `0.285259`
- D1: `0.268229`
- D2: `0.313018`

Runtime observation showed approximately `109309 MiB` allocated on GPU 0 and
`109299 MiB` on GPU 2 during compilation/execution. Both devices were released
after completion.

JAX reported nonfatal unused-buffer-donation and slow constant-folding
warnings. They increased compile time but did not stop the update or produce
invalid values. A single-step loss decrease is not evidence of model gain;
gain must be established by the controlled Stage 1 training and rollout
comparison.
