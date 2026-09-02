#!/usr/bin/env bash
set -euo pipefail

project=/data/shared/zwr/GOAI/ComplexMD
gpu=1
memory_limit_mb=7000
python_bin=/data2/users/zwruu45/.conda_envs/Geom3D/bin/python
config=configs/complexmd_hierarchical_rigid_fragments_v4_window12_current8_history4_scratch.yaml

cd "$project"
mkdir -p outputs/misato_rigid_fragments_v4_window12_current8_history4_scratch

echo "$(date -Is) queued scratch training on GPU ${gpu}; waiting for memory <= ${memory_limit_mb} MB"
while true; do
  memory_used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  utilization=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date -Is) gpu=${gpu} memory_used_mb=${memory_used} utilization=${utilization}%"
  if [ "$memory_used" -le "$memory_limit_mb" ]; then
    break
  fi
  sleep 60
done

echo "$(date -Is) starting scratch training; no initialize/resume checkpoint"
exec env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$project" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u scripts/train.py --config "$config"
