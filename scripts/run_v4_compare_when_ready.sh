#!/usr/bin/env bash
set -euo pipefail

project=/data/shared/zwr/GOAI/ComplexMD
python_bin=/data2/users/zwruu45/.conda_envs/Geom3D/bin/python
run_dir=outputs/misato_rigid_fragments_v4_window12_current8_history4_scratch
candidate_dir=${run_dir}/evaluation/final_conf075
output=${candidate_dir}/comparison.json

cd "$project"
while true; do
  if [ -s "$candidate_dir/T1.json" ] && [ -s "$candidate_dir/T2.json" ] && [ -s "$candidate_dir/T3.json" ]; then
    break
  fi
  sleep 60
done

"$python_bin" scripts/compare_window_results.py \
  --candidate-dir "$candidate_dir" \
  --v2-dir outputs/misato_rigid_fragments_v2/evaluation/final \
  --v3-dir outputs/misato_rigid_fragments_v3_window12_confidence/evaluation/final_conf075 \
  --neuralmd /data/shared/zwr/GOAI/NeuralMD/outputs/neuralmd_baseline_rerun_20260811/test_sde_seed42_all.json \
  --output "$output"
