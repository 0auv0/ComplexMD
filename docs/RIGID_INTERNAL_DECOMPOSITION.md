# Rigid-motion and internal-deformation projection

## Why this is not a second coordinate alignment

MISATO preprocessing and ligand projection solve two different problems.

1. **Pocket/frame-0 alignment** maps every protein frame and its ligand into
   one protein-defined canonical coordinate system. It removes simulation-box
   drift and the protein's shared rigid-body motion.
2. **Ligand rigid/internal decomposition** operates entirely inside that
   already aligned system. It asks which part of a Flow proposal moves the
   ligand as a rigid object relative to the pocket and which part changes the
   ligand's internal geometry.

The second operation does not move the structure through two coordinate
systems and does not add two physical displacements. It is a decomposition of
one proposed next structure followed by one final coordinate write.

## Projection

Let `X_prev` be the previous predicted ligand frame and `X_flow` the endpoint
proposed by conditional Rectified Flow. Masked Kabsch fitting gives the proper
rotation `R` and translation `b` that best map `X_prev` to `X_flow`:

```text
X_rigid = X_prev R + b
X_internal = X_flow - X_rigid
X_next = X_rigid + alpha X_internal
```

`X_rigid` retains the proposal's ligand translation and rotation relative to
the fixed pocket. `X_internal` contains non-rigid Cartesian deformation.
`alpha=1` reproduces the original Flow endpoint exactly; `alpha=0` preserves
all pairwise distances of the previous ligand frame while still allowing
Flow-predicted translation and rotation. Intermediate values damp, rather
than remove, internal deformation.

The fit uses only the previous predicted frame, the current Flow proposal and
the ligand atom mask. No future target frame enters inference.

## Validation-only selection

Hyperparameters are selected on a fixed 128-complex validation subset. The
earlier 128-complex test experiment is diagnostic only and is excluded from
selection.

1. At Flow base scale 1, choose `alpha` from `{0, 0.25, 1}` using the mean of
   Matching relative to persistence and instability `(100 - Stability)`
   relative to persistence over T1/T2/T3.
2. At the selected `alpha`, choose Flow base scale from `{0, 0.25, 0.5, 1}`
   using the mean persistence-normalized RMSE, RMSF-MAE and error-growth slope
   over T1/T2/T3.
3. Freeze both values and evaluate the full 1,357-complex test set once.

This is an inference-time projection of the existing trained Flow checkpoint.
It tests whether geometry-aware factorization fixes autoregressive distortion
before committing to a separately trained SE(3)/torsion decoder.
