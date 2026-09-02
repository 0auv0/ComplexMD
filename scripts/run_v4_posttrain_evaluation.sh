#!/usr/bin/env bash
set -euo pipefail

project=/data/shared/zwr/GOAI/ComplexMD
gpu=1
python_bin=/data2/users/zwruu45/.conda_envs/Geom3D/bin/python
train_pid=3615333
run_dir=outputs/misato_rigid_fragments_v4_window12_current8_history4_scratch
checkpoint_dir=${run_dir}/checkpoints
selection_dir=${run_dir}/evaluation/selection
final_dir=${run_dir}/evaluation/final_conf075
config=configs/complexmd_hierarchical_rigid_fragments_v4_window12_current8_history4_scratch.yaml

cd "$project"
mkdir -p "$selection_dir" "$final_dir"
echo "$(date -Is) waiting for scratch-training PID ${train_pid}"
while kill -0 "$train_pid" 2>/dev/null; do
  sleep 60
done

echo "$(date -Is) training process ended; starting checkpoint validation"
shopt -s nullglob
checkpoints=("$checkpoint_dir"/epoch_*.pt)
if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "no epoch checkpoints were produced" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  stem=$(basename "$checkpoint" .pt)
  output="$selection_dir/${stem}_val20.json"
  if [ -s "$output" ]; then
    continue
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$project" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split val --scenario T1 --max-complexes 20 \
      --sampling-steps 10 --torsion-step-limit-deg 5 \
      --torsion-confidence-threshold 0.75 \
      --pose-translation-scale 0.25 --pose-rotation-scale 0.25 \
      --output "$output" \
      > "$selection_dir/${stem}_val20.log" 2>&1
done

"$python_bin" scripts/select_window_checkpoint.py \
  --evaluation-dir "$selection_dir" \
  --checkpoint-dir "$checkpoint_dir" \
  --output "$selection_dir/selection.json"

best_checkpoint=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["checkpoint"])' "$selection_dir/selection.json")
echo "$(date -Is) selected ${best_checkpoint}; starting full evaluation"

for tier in T1 T2 T3; do
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$project" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate.py \
      --config "$config" \
      --checkpoint "$best_checkpoint" \
      --split test --scenario "$tier" \
      --sampling-steps 10 --torsion-step-limit-deg 5 \
      --torsion-confidence-threshold 0.75 \
      --pose-translation-scale 0.25 --pose-rotation-scale 0.25 \
      --output "$final_dir/${tier}.json" \
      > "$final_dir/${tier}.log" 2>&1
done

echo "$(date -Is) full T1/T2/T3 evaluation completed"
