# BindMD design notes

## Factorization

BindMD uses

\[
p(x_{1:L}\mid P)=\prod_t p(x_{t+1}\mid x_{t-H:t},P),
\]

where \(P\) is a fixed, causally cropped protein pocket. Each factor is a
conditional diffusion distribution over the ligand displacement.

## Information flow

1. Encode residue identity and invariant N/CA/C geometry.
2. Create ligand atom-frame tokens for clean history and the noisy current
   candidate.
3. Apply causal joint space-time self-attention with distance bias and 2D
   atom/time rotary encoding.
4. Cross-attend every ligand token to fixed pocket tokens.
5. Convert scalar token features to a vector denoising field using relative
   coordinate directions.
6. DDIM-sample a displacement and append the resulting frame to history.

## Why no force head

The prediction interval in the benchmark is coarse relative to an MD
integration step. At that scale, unresolved degrees of freedom induce memory
and stochasticity. A learned acceleration followed by an explicit integrator
imposes a restrictive Markovian second-order model. BindMD instead learns the
coarse conditional transition distribution directly.

## Objective

The main objective is diffusion-noise MSE. A small all-pairs distance loss on
the reconstructed clean frame discourages immediate ligand geometry
distortion. Context coordinates receive up to 0.1 Å random perturbation during
training to expose the denoiser to the slightly imperfect histories encountered
during rollout.

Energy labels are retained by the data adapter but intentionally excluded from
the default objective. They can later be added as an auxiliary critic without
turning energy gradients into the primary trajectory mechanism.
