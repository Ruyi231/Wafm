# Plan 1 Full-Model Fixed-Batch Overfit Gate

## Scope

This gate uses the complete Pi0.5 NH-WaFM model, the real normalized LIBERO
data pipeline, the configured `pi05_base` checkpoint, and the formal level-2
wavelet statistics. A batch of two real action chunks is held fixed. Training
retains the normal per-step flow noise, flow timestep sampling, and training
preprocessing. Evaluation uses a fixed RNG and `train=False`, so every recorded
point measures the same flow state and timestep.

The runner deliberately does not save another 42 GB checkpoint. Checkpoint
save and resume were validated separately by the real GPU smoke test.

## Fixed acceptance criterion

Before running the gate, the acceptance threshold was fixed as:

```text
final / initial <= 0.5
```

for all of:

- `loss_band_A`
- `loss_band_D1`
- `loss_band_D2`
- `loss_action_reconstruction`

The threshold was not relaxed after observing the results.

## Results

| Steps | A ratio | D1 ratio | D2 ratio | IDWT action ratio | Status |
|---:|---:|---:|---:|---:|:---|
| 100 | 0.915846 | 0.880855 | 0.856351 | 0.837105 | strict threshold not met |
| 400 | 0.712002 | 0.750394 | 0.738248 | 0.136840 | strict threshold not met |
| 1000 | 0.570120 | 0.628269 | 0.661425 | 0.101555 | strict threshold not met |

For the 1000-step run, total standardized band loss decreased from
`3.635147` to `2.273772`, while IDWT action reconstruction error decreased
from `0.092473` to `0.009391`. All three band losses learned, and no NaN or Inf
was observed, but none reached the fixed 50% ratio.

## Correct interpretation

The `final / initial <= 0.5` rule was an additional strict diagnostic chosen
for this runner; it was not an original Plan 1 stop-loss condition. Therefore
these runs do **not** establish that the model cannot overfit and do not mark
Plan 1 as failed.

The supported conclusions are:

- the complete training chain runs;
- action-domain fitting is substantial;
- every standardized band loss decreases stably;
- the additional strict threshold was not met;
- a deterministic full-model overfit test is required before deciding whether
  parameter isolation is necessary.

Only the real batch and evaluation RNG were fixed in these runs. Training still
resampled flow noise and timestep and used training preprocessing, so the model
was learning a stochastic family of flow bridges rather than repeatedly fitting
one identical supervised target.

## Deterministic confirmation

A follow-up 400-step run fixed the real batch, flow noise, flow timestep,
training preprocessing result, training RNG, and evaluation RNG. Evaluation
used the same fixed RNG and `train=True` preprocessing path as optimization.

| Metric | Initial | Final | Final / initial |
|:---|---:|---:|---:|
| `loss_band_A` | 1.071530 | 0.000158 | 0.000148 |
| `loss_band_D1` | 1.038757 | 0.000160 | 0.000155 |
| `loss_band_D2` | 1.197088 | 0.000109 | 0.000091 |
| `loss_action_reconstruction` | 0.097275 | 0.00000583 | 0.000060 |

All observed band-head gradient norms were finite and nonzero. Across logged
steps their minima were `0.02638` for A, `0.02628` for D1, and `0.02218` for
D2. No NaN or Inf was observed.

The deterministic full-model overfit gate therefore **passed**. This confirms
that the model, normalized wavelet targets, band heads, IDWT reconstruction,
and complete training path can fit one fixed supervised flow target. The
head-only parameter-isolation experiment is not required. The next permitted
experiment is the single-config `plan1_subband_l2_norm` GPU canary; this result
does not itself establish task-level performance.

## Reproduction

```bash
cd /nfs/lizhenhao/Project/work1
source env.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. uv run --offline \
  scripts/run_full_model_wavelet_overfit.py \
  --config-name plan1_subband_l2_norm \
  --output-dir plan1_results/server_validation/full_model_real_overfit_1000 \
  --steps 1000 \
  --log-interval 50 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --max-final-ratio 0.5
```

The command is expected to exit nonzero when the gate fails, after writing its
JSON and CSV evidence.
