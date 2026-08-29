# Frame-0 pocket alignment ablation

## Controlled setup

- Seed: 42
- Train/holdout complexes: the same 8/4 identifiers as the clean baseline
- Windows: the same 24 train and 12 holdout windows
- Model: 6 joint space-time blocks, width 128, 4 heads, SE(3)-equivariant vector head
- Optimization: 1,000 steps, batch size 4, learning rate 3e-4
- Sampling: 5-step DDIM, 5-frame rollout after frame 60
- Only experimental variable: the coordinate preprocessing described below

The older result JSON omits `diffusion_schedule` and `max_displacement` from its
serialized config. The corresponding `BindMD` defaults are `linear` and 5.0,
which equal the explicit values in the aligned run.

## Coordinate preprocessing

1. Select the pocket once from frame 0 using the 12 Angstrom ligand-to-CA crop.
2. For each frame, Kabsch-fit the same pocket N/CA/C atoms to frame 0.
3. Apply that exact proper rigid transform to the ligand in the same frame.
4. Keep the pocket conditioning coordinates fixed at frame 0.
5. Put the first selected pocket residue's N atom at the origin. Define x with
   N-to-CA, y from the component of CA-to-C orthogonal to x, and z by a
   right-handed cross product. This orientation is computed once from frame 0.

The architecture itself is unchanged for this ablation: the existing pocket
encoder, pocket cross-attention in every joint space-time block, and the
SE(3)-equivariant ligand/pocket vector head are retained. This isolates the
effect of removing reference-frame drift.

## Drift diagnostics

| Split | Raw mean ligand step (A) | Aligned mean ligand step (A) | Raw worst step (A) | Aligned worst step (A) |
|---|---:|---:|---:|---:|
| Train (8) | 1.304 | 0.896 | 68.097 | 3.298 |
| Holdout (4) | 2.510 | 0.837 | 34.363 | 2.463 |

The mean pocket-fit RMSD is 0.581 A on train and 0.484 A on holdout. These
nonzero values represent protein conformational motion left after removal of
the shared rigid-body component.

## Trained five-frame rollout comparison

Lower is better except NeuralMD stability, where higher is better.

| Split | Metric | Unaligned | Frame-0 aligned | Change |
|---|---|---:|---:|---:|
| Train | NeuralMD RMSE | 2.3468 | 1.7909 | -23.69% |
| Train | NeuralMD matching | 1.2951 | 1.1693 | -9.72% |
| Train | NeuralMD stability | 38.6253 | 43.5552 | +12.76% |
| Train | Geometry ligand RMSD | 2.5371 | 2.0487 | -19.25% |
| Train | Last-frame ligand RMSD | 3.7631 | 2.6599 | -29.32% |
| Train | Error-growth slope | 2.3184 | 1.2575 | -45.76% |
| Holdout | NeuralMD RMSE | 2.4244 | 1.9549 | -19.36% |
| Holdout | NeuralMD matching | 1.5204 | 1.4055 | -7.56% |
| Holdout | NeuralMD stability | 33.5520 | 35.5797 | +6.04% |
| Holdout | Geometry ligand RMSD | 2.5738 | 2.1213 | -17.58% |
| Holdout | Last-frame ligand RMSD | 3.3952 | 2.8614 | -15.72% |
| Holdout | Error-growth slope | 2.1086 | 1.5587 | -26.08% |

Holdout one-step coordinate RMSE changes from 0.7924 to 0.7791 A (-1.67%),
while the holdout diffusion objective changes from 0.8643 to 0.8009 (-7.34%).
The largest gains are therefore in autoregressive rollout stability rather than
teacher-forced one-step prediction, which is the expected signature of removing
frame-to-frame coordinate drift.

## Important limitation

The aligned model still trails the persistence baseline on the main rollout
metrics (for example, holdout NeuralMD RMSE 1.9549 versus 0.8094). Alignment is
therefore a necessary data correction, not a complete dynamics solution. The
experiment has only one seed and four holdout complexes, so it should be
repeated at full scale and across seeds before drawing statistical conclusions.

## Artifacts

- Aligned cache: `outputs/small_sample_misato_aligned/train_subset_aligned.pt`
- Alignment metadata: `outputs/small_sample_misato_aligned/train_subset_aligned.json`
- Result: `outputs/small_sample_misato_aligned/result.json`
- Checkpoint: `outputs/small_sample_misato_aligned/small_sample_last.pt`
