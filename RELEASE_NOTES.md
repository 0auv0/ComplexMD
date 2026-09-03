# Frozen release notes - 2026-09-03

Final submission candidate: ComplexMD v3, 12-frame 6+6 temporal split.

- Selected checkpoint: `epoch_004.pt`
- Packaged checkpoint: `weights/complexmd_v3_6plus6_epoch004.pt`
- Weight SHA256: `9493faa931d305ec3a78b4c14a1e6a3257d400fc9114a935bdab9606c81901ee`
- Final inference torsion confidence threshold: 0.75
- Final inference torsion step limit: 5 degrees
- Protein pose translation/rotation scales: 0.25/0.25
- Sampling: conditional Flow Matching, Heun, 10 steps
- Optional T4: excluded

The experimental 8+4-from-scratch v4 checkpoint is intentionally excluded
because its ligand-coordinate rollout error was worse than the 6+6 model.
Datasets, evaluation trajectories, caches, generated XTC files and server logs
are also excluded from this source/reproducibility release.

