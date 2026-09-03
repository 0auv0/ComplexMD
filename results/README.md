# Audited 6+6 results

These are raw test-split metrics on all 1,357 MISATO test complexes. Lower is
better for coordinate error, matching and RMSF MAE; higher is better for
stability. The official competition normalization constants and hidden labels
are unavailable, so no unofficial scalar competition score is reported.

| Scenario | Model | coordinate error (A) | matching | stability (%) | RMSF MAE (A) |
|---|---:|---:|---:|---:|---:|
| T1, 10->10 | ComplexMD 6+6 | 1.5565 | 0.3908 | 86.5590 | 0.4372 |
| T1, 10->10 | NeuralMD | 3.9233 | 0.5114 | 81.4633 | 2.5087 |
| T2, 80->20 | ComplexMD 6+6 | 1.8751 | 0.4079 | 85.8105 | 0.5708 |
| T2, 80->20 | NeuralMD | 2.7992 | 0.4303 | 84.7050 | 1.8390 |

`final_6plus6/T1.json` and `final_6plus6/T2.json` contain the complete metric
records. T3 evaluation was sharded and was not complete when this release was
frozen, so a full T3 aggregate is intentionally not claimed here.

The reported `coordinate error` follows the retained NeuralMD reduction: mean
per-atom Euclidean coordinate error. `matching` is internal pair-distance
RMSE. `stability` is the percentage of internal pair distances within 0.5 A;
it is a geometric proxy rather than an energy-conservation claim.

