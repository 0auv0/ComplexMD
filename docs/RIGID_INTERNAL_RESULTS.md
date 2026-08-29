# Rigid/internal projection: full MISATO results

Date: 2026-08-16

## Method and causal scope

All protein and ligand frames first use the existing frame-0 pocket-aligned
canonical coordinates. The additional Kabsch operation is not another change
of protein coordinate system. Given the previous predicted ligand `X_prev`
and the conditional Flow proposal `X_flow`, it extracts one ligand-rigid
motion inside the already canonical pocket frame:

```text
X_rigid = X_prev R + b
X_next = X_rigid + alpha (X_flow - X_rigid)
```

The final coordinates are written once. The fit uses only `X_prev`, `X_flow`
and the ligand mask; no future target coordinates enter inference. The Flow
proposal's translation and rotation relative to the pocket are retained,
while `alpha` controls its non-rigid Cartesian residual.

## Validation-only selection

The earlier test-128 run was treated as a mechanism diagnostic only. Final
hyperparameters were selected independently on the first 128 complexes of the
MISATO validation split.

At Flow base scale 1, the lower-is-better structural objective was the mean of
Matching relative to persistence and instability `(100 - Stability)` relative
to persistence over T1/T2/T3:

| Internal scale `alpha` | Validation structural score |
|---:|---:|
| **0** | **1.0000** |
| 0.25 | 1.7604 |
| 1.0 (original Flow) | 3.5239 |

With `alpha=0`, the lower-is-better motion objective was the mean of
persistence-normalized RMSE, RMSF-MAE and error-growth slope:

| Flow base scale | Validation motion score |
|---:|---:|
| 0 | 1.0246 |
| 0.25 | 1.0184 |
| **0.5** | **1.0016** |
| 1.0 | 1.1457 |

The frozen full-test setting was therefore `alpha=0`, Flow base scale 0.5,
10-step Heun integration, seed 42.

## Full-test protocol and audit

- Test complexes: 1,357.
- T1: observe 50 frames, predict 50.
- T2: observe 80 frames, predict 20.
- T3: observe 20 frames, predict 80.
- Model records: 4,071; persistence records: 4,071.
- All three scenarios contain indices 0--1,356 exactly once.
- All 31 serialized metrics are finite.
- Aggregates were independently recomputed from records and match the JSON.
- The identifiers exactly match the retained full Flow and DDIM evaluations.
- Runtime: approximately 4 hours 47 minutes on A100 GPU 6.

The table reports raw metrics. Lower is better except Stability, where higher
is better.

### T1

| Model | RMSE | Matching | Stability (%) | Internal-distance RMSE | Bond-length RMSE | RMSF MAE | Step P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rigid Flow BindMD** | **1.8422** | **0.4549** | **84.31** | **0.4746** | **0.0474** | 1.0067 | 0.3313 |
| Original Flow BindMD | 2.8918 | 1.3026 | 42.25 | 1.3660 | 1.0801 | **1.0019** | 2.2212 |
| DDIM BindMD | 3.7673 | 1.7548 | 36.11 | 1.8621 | 1.6856 | 1.4387 | 1.8937 |
| NeuralMD | 3.9233 | 0.5114 | 81.46 | 0.5336 | 0.0963 | 2.5087 | 0.0232 |
| Persistence | 1.8034 | 0.4549 | 84.31 | 0.4746 | 0.0474 | 1.2869 | 0.0000 |

### T2

| Model | RMSE | Matching | Stability (%) | Internal-distance RMSE | Bond-length RMSE | RMSF MAE | Step P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rigid Flow BindMD** | **1.4468** | **0.4096** | **85.72** | **0.4259** | **0.0471** | 0.8663 | 0.3457 |
| Original Flow BindMD | 2.1223 | 0.9272 | 53.61 | 0.9596 | 0.6839 | **0.6279** | 2.0978 |
| DDIM BindMD | 2.6153 | 1.1641 | 47.06 | 1.2173 | 0.9533 | 0.8368 | 1.7816 |
| NeuralMD | 2.7992 | 0.4303 | 84.71 | 0.4472 | 0.0608 | 1.8390 | 0.0231 |
| Persistence | 1.4854 | 0.4096 | 85.72 | 0.4259 | 0.0471 | 1.0516 | 0.0000 |

### T3

| Model | RMSE | Matching | Stability (%) | Internal-distance RMSE | Bond-length RMSE | RMSF MAE | Step P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rigid Flow BindMD** | **2.1150** | **0.4890** | **83.09** | **0.5106** | **0.0472** | **1.0682** | 0.3272 |
| Original Flow BindMD | 3.3832 | 1.5786 | 36.97 | 1.6653 | 1.3933 | 1.2521 | 2.2651 |
| DDIM BindMD | 4.5376 | 2.2273 | 31.24 | 2.3860 | 2.3270 | 1.8925 | 1.9690 |
| NeuralMD | 4.8086 | 0.5919 | 78.24 | 0.6204 | 0.1406 | 3.0014 | 0.0227 |
| Persistence | 1.9875 | 0.4890 | 83.09 | 0.5106 | 0.0472 | 1.4372 | 0.0000 |

## Main comparisons

Relative to the original Flow BindMD, rigid projection changes:

| Scenario | RMSE | Matching | Stability | Internal distance | Bond length | Step P95 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | -36.3% | -65.1% | +42.06 pp | -65.3% | -95.6% | -85.1% |
| T2 | -31.8% | -55.8% | +32.11 pp | -55.6% | -93.1% | -83.5% |
| T3 | -37.5% | -69.0% | +46.12 pp | -69.3% | -96.6% | -85.6% |

The paired improvements over original Flow are broad rather than aggregate
outliers: rigid Flow has lower per-complex RMSE on 98.1%, 98.2% and 96.9% of
T1/T2/T3 complexes; higher Stability on 100% in all scenarios. The paired
95% normal intervals for mean RMSE improvement are `[1.016, 1.083]`,
`[0.653, 0.698]` and `[1.228, 1.308]` Angstrom respectively.

Relative to NeuralMD, rigid Flow lowers RMSE by 53.0%, 48.3% and 56.0%; lowers
Matching by 11.1%, 4.8% and 17.4%; and raises Stability by 2.85, 1.02 and 4.85
percentage points for T1/T2/T3. RMSF MAE is lower by 59.9%, 52.9% and 64.4%.
Absolute-coordinate comparisons with NeuralMD still require caution because
its released evaluator does not use BindMD's protein-aligned canonical frame;
rigid-invariant structural metrics are the safest direct comparison.

## Interpretation and limitations

The gain is real, but its source must be stated precisely. With `alpha=0`,
every rollout frame retains the previous ligand's pair distances. Matching,
Stability, internal-distance RMSE and inferred bond-length RMSE therefore
equal the persistence baseline up to float32 Kabsch error. This experiment
proves that independent per-atom Flow deformation caused most of the original
geometry collapse; it does **not** prove that the current checkpoint learned
accurate internal conformational dynamics.

The method is not persistence. Step P95 is about 0.33 Angstrom rather than
zero because the Flow still predicts ligand translation and rotation relative
to the pocket. This is 83.5--85.6% smaller than original Flow, yet much larger
than NeuralMD's approximately 0.023 Angstrom. RMSF error is 21.8%, 17.6% and
25.7% lower than persistence for T1/T2/T3, indicating useful non-static rigid
motion. Coordinate RMSE is 2.2% worse than persistence on T1, 2.6% better on
T2, and 6.4% worse on T3, so persistence is not uniformly beaten.

RMSF also exposes a trade-off: compared with original Flow it is essentially
unchanged on T1, 38.0% worse on T2, and 14.7% better on T3. The validation
objective selected a balance among coordinate RMSE, RMSF and error growth,
not a setting that dominates every dynamics proxy.

The appropriate next trained architecture is an explicit factorization:

1. predict pocket-conditioned SE(3) translation and incremental rotation;
2. predict a smaller torsional/internal residual using ligand bond topology;
3. reconstruct Cartesian coordinates once;
4. train rigid, torsion, bond/angle and multi-step rollout losses separately.

That retains the demonstrated stability benefit while allowing genuine
conformational change instead of hard-freezing internal geometry.

## Artifacts

```text
configs/bindmd_full_flow_rigid.yaml
bindmd/models/flow.py
scripts/evaluate.py
scripts/select_rigid_hparams.py
scripts/compare_rigid_results.py
tests/test_flow.py
outputs/rigid_validation_128/selection.json
outputs/misato_aligned_full_flow_rigid/bindmd_rigid_test.json
outputs/misato_aligned_full_flow_rigid/comparison_rigid.json
outputs/misato_aligned_full_flow_rigid/logs/evaluate.log
```
