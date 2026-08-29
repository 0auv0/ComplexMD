#!/usr/bin/env bash
set -euo pipefail

cd /data/shared/zwr/GOAI/BindMD
run_dir=outputs/misato_aligned_full_flow
python_bin=/data2/users/zwruu45/.conda_envs/Geom3D/bin/python
export PYTHONPATH=.:/data/shared/zwr/GOAI/NeuralMD:/data
mkdir -p "$run_dir/logs"

CUDA_VISIBLE_DEVICES=1 "$python_bin" scripts/train.py \
    --config configs/bindmd_full_flow.yaml \
    --split train \
    > "$run_dir/logs/train.log" 2>&1

CUDA_VISIBLE_DEVICES=1 "$python_bin" scripts/evaluate.py \
    --config configs/bindmd_full_flow.yaml \
    --checkpoint "$run_dir/checkpoints/last.pt" \
    --scenario all \
    --output "$run_dir/bindmd_flow_test.json" \
    > "$run_dir/logs/evaluate.log" 2>&1

"$python_bin" scripts/compare_full_results.py \
    --bindmd "$run_dir/bindmd_flow_test.json" \
    --output "$run_dir/comparison_flow.json" \
    > "$run_dir/logs/compare.log" 2>&1
