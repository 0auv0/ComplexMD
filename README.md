# ComplexMD

ComplexMD is a research baseline for the GOAI 2026 protein–small-molecule
complex trajectory task. The internal Python package remains named `bindmd`
so existing checkpoints remain directly loadable.

The central choice is deliberate: **ComplexMD predicts future coordinates
directly. It does not predict forces, and it does not integrate Newton's
equations.**

## Method

BindMD combines three ideas:

- NeuralMD-compatible MISATO tensors and the released NeuralMD evaluation
  reductions, so comparisons use the same data contract.
- ConfRover-style frame-autoregressive conditional generation: generate one
  next frame with DDIM or conditional Rectified Flow, append it to history,
  and repeat.
- STAR-MD-style joint space-time processing: atom-frame tokens interact in a
  single causal attention operation, with separate rotary phases for atom and
  frame indices and small contextual perturbations during training.

For a history \(x_{t-H:t}\), the generated variable is the next-frame
displacement

\[
\Delta x_{t+1} = x_{t+1} - x_t.
\]

The vector-field/denoising network sees historical ligand coordinates, a
candidate displacement along the probability path, ligand atom identity/mass,
and a cropped fixed protein pocket.
Its vector output is assembled from relative ligand–ligand, ligand–backbone,
and temporal directions. This makes the output SE(3)-equivariant while all
attention features remain invariant.

This implementation is inspired by, but not a copy of:

- [STAR-MD](https://arxiv.org/abs/2602.02128), especially causal joint
  spatio-temporal attention, two-axis position encoding, and contextual noise.
- [ConfRover](https://github.com/ByteDance-Seed/ConfRover), especially
  autoregressive frame conditioning and a diffusion structure decoder.
- [NeuralMD](https://github.com/chao1224/NeuralMD), for the semi-flexible
  protein–ligand data representation and baseline metrics.

## What differs from those systems

STAR-MD models protein residue frames and uses OpenFold representations.
BindMD instead models ligand heavy atoms conditioned on fixed protein N/CA/C
backbone geometry. ConfRover alternates a spatial Pairformer with a temporal
language model; BindMD uses one joint atom-frame attention graph. NeuralMD
learns acceleration/force-like dynamics and numerically integrates them;
BindMD samples coordinate displacements.

The current implementation uses a bounded history window instead of a KV
cache. This keeps the first baseline compact and makes T1/T2/T3 evaluation
unambiguous.

## Repository layout

```text
bindmd/data/misato.py        NeuralMD processed-data adapter and causal centering
bindmd/data/goai.py          GOAI loading, frame fixing, full-complex restoration
bindmd/models/layers.py      joint space-time attention, 2D RoPE, pocket encoder
bindmd/models/bindmd.py      equivariant denoiser, training objective, rollout
bindmd/models/flow.py        conditional Flow Matching and rigid/internal projection
bindmd/evaluation/metrics.py NeuralMD metrics plus Geo/Phys/Dyn/Stab proxies
scripts/train.py             training entry point
scripts/evaluate.py          T1/T2/T3 rollout evaluation
scripts/predict_goai.py      competition-format all-atom XTC generation
scripts/select_rigid_hparams.py validation-only rigid/Flow hyperparameter selection
scripts/compare_rigid_results.py unified Flow/DDIM/NeuralMD/persistence comparison
tests/                       shape, metric, and SE(3)-equivariance tests
```

## Data

BindMD consumes the files produced by NeuralMD's
`DatasetMISATOSemiFlexibleMultiTrajectory`:

```text
MISATO/
└── processed_semi_flexible/
    ├── geometric_data_processed_train.pt
    ├── geometric_data_processed_val.pt
    └── geometric_data_processed_test.pt
```

The configured server location is:

```text
/data/shared/zwr/GOAI/NeuralMD/data_runtime/MISATO
```

NeuralMD's cached coordinates were globally centred using all 100 frames.
BindMD subtracts the fixed protein CA centroid from both protein and ligand
coordinates at load time. That cancels the cached global offset and prevents
future ligand frames from supplying the model's coordinate origin. Pocket
cropping uses only the last observed frame.

At present, `val` and `test` are processed on the server; process the NeuralMD
`train` split before a full run.

## Installation and checks

The existing Geom3D environment already contains the required PyTorch and PyG
versions:

```bash
cd /data/shared/zwr/GOAI/BindMD
/data2/users/zwruu45/.conda_envs/Geom3D/bin/pip install -e . --no-deps
/data2/users/zwruu45/.conda_envs/Geom3D/bin/python -m pytest -q
```

Inspect one real complex:

```bash
/data2/users/zwruu45/.conda_envs/Geom3D/bin/python scripts/inspect_data.py \
  --root /data/shared/zwr/GOAI/NeuralMD/data_runtime/MISATO \
  --split val
```

Run a short overfit/smoke training job on the processed validation data:

```bash
/data2/users/zwruu45/.conda_envs/Geom3D/bin/python scripts/train.py \
  --config configs/bindmd_base.yaml --split val --max-steps 20
```

For full training, omit `--split val --max-steps 20` after the train cache has
been generated.

Run the reproducible MISATO train-split small-sample experiment:

```bash
PYTHONPATH=.:/data:/data/shared/zwr/GOAI/NeuralMD \
  /data2/users/zwruu45/.conda_envs/Geom3D/bin/python \
  scripts/prepare_small_misato.py --config configs/small_sample.yaml

CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=.:/data:/data/shared/zwr/GOAI/NeuralMD \
  /data2/users/zwruu45/.conda_envs/Geom3D/bin/python \
  scripts/small_sample_experiment.py --config configs/small_sample_clean.yaml
```

The measured setup, metrics, periodic-boundary artifact, and conclusions are
documented in [`docs/SMALL_SAMPLE_RESULTS.md`](docs/SMALL_SAMPLE_RESULTS.md).

## Evaluation

```bash
/data2/users/zwruu45/.conda_envs/Geom3D/bin/python scripts/evaluate.py \
  --config configs/bindmd_base.yaml \
  --checkpoint outputs/checkpoints/last.pt \
  --scenario all
```

The scenarios are:

- T1 proxy: observe 50 frames, predict 50.
- T2: observe 80 frames, predict 20.
- T3: observe 20 frames, predict 80.

Every evaluation JSON contains the original NeuralMD MAE, RMSE, matching,
stability, ligand-collision, and binding-collision reductions, followed by raw
Geo/Phys/Dyn/Stab proxy metrics. Official competition normalization constants
are not public, so no unofficial scalar "competition score" is fabricated.

### Public GOAI package and XTC output

The public adapter Kabsch-aligns every observed pocket backbone to frame 0.
Inside the model, all protein atoms are therefore represented by the complete
frame-0 rigid template, and the ligand is generated in the same fixed frame.
Before writing the final XTC, an observed-only global protein pose is applied
to both the protein and ligand. The default `hold_last` policy uses the last
observed pose; `constant_velocity` extrapolates a clipped pose velocity fitted
only from observations. Neither policy reads future targets.

The ligand-heavy prediction determines a single rigid SE(3) pose, which is
also applied to ligand hydrogens. T4 ions persist from the last observation.
The output retains the PDB atom order, contains exactly `n_pred` frames, and is
written in XTC's nm convention.

```bash
PYTHONPATH=. /data2/users/zwruu45/.conda_envs/Geom3D/bin/python \
  scripts/predict_goai.py \
  --input-root GOAI_eval_public \
  --output-dir outputs/goai_public \
  --tier T1 \
  --checkpoint outputs/misato_aligned_full_flow/checkpoints/last.pt \
  --config configs/bindmd_full_flow_rigid.yaml \
  --pose-mode hold_last
```

Use `--ligand-mode persistence` for a checkpoint-free output-contract smoke
test. XTC I/O supports either `mdtraj` or `MDAnalysis`.

The full 1,357-complex rigid/internal Flow experiment, validation-only
hyperparameter protocol, audited T1/T2/T3 tables, and comparison with original
Flow, DDIM, NeuralMD and persistence are documented in
[`docs/RIGID_INTERNAL_RESULTS.md`](docs/RIGID_INTERNAL_RESULTS.md). The method
separates a ligand's rigid translation/rotation relative to the aligned pocket
from its internal Cartesian deformation; it is not a second protein-frame
alignment.

## Current limitations

- Protein internal coordinates are fixed to frame 0. Future global pose is
  held at the last observation unless observed-only extrapolation is selected.
- MISATO processed tensors contain ligand heavy atoms and protein backbone but
  no explicit ligand bond graph, charges, stereochemistry, hydrogens, or
  force-field parameters. Bond and clash metrics are therefore proxies.
- Autoregressive Flow/DDIM sampling is much slower than a deterministic
  one-step head, especially for T3. Distillation, cached history and dynamic
  batching are natural next steps.
- The best current inference projection uses zero internal-deformation scale.
  It preserves ligand geometry and still predicts rigid motion, but cannot
  model genuine torsional changes. A trained SE(3)-plus-torsion decoder is the
  next architectural step.
