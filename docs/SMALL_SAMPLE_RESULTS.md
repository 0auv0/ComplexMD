# MISATO train-split small-sample experiment

Date: 2026-08-05

## Split and protocol

The cache was sampled strictly from `train_MD.txt` with seed 42. None of the
selected identifiers occurs in the MISATO validation or test split.

- Training complexes: `6ME2`, `2AVO`, `5F37`, `4YVC`, `5O1D`, `6MIM`,
  `3QTX`, `5HMY`.
- Train-split holdout complexes: `4PRJ`, `6FMN`, `2BB7`, `3B65`.
- Controlled fixed windows: target frames 20, 40, and 60.
- Training examples: 24; holdout examples: 12.
- Short rollout: observe through frame 59, predict frames 60–64.
- Model: the 2.20 M parameter BindMD base model.
- Optimization: 1,000 steps, batch size 4, AdamW, learning rate 3e-4.
- Runtime: 71.1 seconds on A100 GPU 6; peak allocated memory 1.30 GB.

The primary result uses the stable 100-step linear diffusion schedule and a
5-step DDIM sampler.

## Primary result

| Measurement | Random initialization | Trained | Persistence |
|---|---:|---:|---:|
| Train diffusion objective | 0.7588 | **0.3907** | — |
| Holdout diffusion objective | 0.8567 | 0.8643 | — |
| Train one-step coordinate RMSE | 1.5125 | **0.7831** | 0.8431 |
| Holdout one-step coordinate RMSE | 1.3835 | 0.7924 | **0.6238** |
| Train 5-frame NeuralMD RMSE | 3.8287 | 2.3468 | **1.6903** |
| Holdout 5-frame NeuralMD RMSE | 3.5530 | 2.4244 | **1.2907** |
| Train 5-frame matching | 3.2361 | 1.2951 | **0.3715** |
| Holdout 5-frame matching | 2.9506 | 1.5204 | **0.3180** |
| Train 5-frame stability (%) | 16.96 | 38.63 | **87.23** |
| Holdout 5-frame stability (%) | 16.76 | 33.55 | **88.94** |

The model is trainable: the controlled train objective fell by 48.5%, and the
teacher-forced train one-step RMSE became 7.1% better than persistence. The
same model substantially improved over random initialization on holdout
one-step and short rollout metrics.

It is not yet a competitive trajectory predictor. Holdout one-step sampling
remained 27% worse than persistence, and autoregressive rollout accumulated
error quickly. Internal-distance matching and stability show that arbitrary
per-atom displacement diffusion distorts ligand geometry.

## Periodic-boundary artifact

The first experiment used target frame 80 and exposed nonphysical coordinate
jumps in the processed data:

- `4PRJ`: approximately 33 Å at frame 80 and 34 Å at frame 87.
- `4YVC`: approximately 68 Å at frame 48.

These are consistent with an unwrapped ligand crossing a periodic boundary.
Because BindMD centres coordinates on the protein and has no simulation box
vectors, it cannot infer the wrap translation. Frame 80 was therefore removed
from the controlled comparison. Full training must unwrap trajectories, or
exclude/repair discontinuous transitions before window sampling.

## Diffusion schedule diagnostic

The original 100-step linear schedule has
`alpha_bar[-1] = 0.3636`, so its sampling prior is not truly pure noise.
A cosine schedule reaches `2.43e-7`, but epsilon prediction errors at the
highest noise level are amplified severely during clean-coordinate recovery.

Without displacement clipping, the cosine model produced unusable samples.
After applying a 5 Å per-atom clean-displacement bound, its train one-step RMSE
was 0.7226, but holdout one-step RMSE was 0.8222 and holdout rollout RMSE was
3.5056. This was worse than the clipped linear model, so linear remains the
default pending a v-prediction or clean-displacement-prediction decoder.

## Recommended next changes

1. Unwrap ligand trajectories against protein/box coordinates before creating
   training windows.
2. Predict rigid-body translation/rotation plus torsional or internal
   deformation, rather than independent Cartesian atom displacements.
3. Add explicit bond topology and stronger bond/angle constraints.
4. Parameterize the transition as a correction to persistence or recent
   velocity, giving the model a strong deterministic mean path.
5. Train with multi-step scheduled rollouts, not only teacher-forced
   next-frame diffusion.
6. Replace epsilon prediction with v-prediction or bounded clean-displacement
   prediction before revisiting a full cosine schedule.

## Artifacts

```text
outputs/small_sample_misato/train_subset.pt
outputs/small_sample_misato_clean/result.json
outputs/small_sample_misato_clean/small_sample_last.pt
outputs/small_sample_misato_clean_cosine/result.json
outputs/small_sample_misato_clean_cosine/small_sample_last.pt
outputs/small_sample_clipped_reeval.json
```

