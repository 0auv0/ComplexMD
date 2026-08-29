#!/usr/bin/env bash
set -euo pipefail

cd /data/shared/zwr/GOAI/BindMD
run_dir=outputs/misato_aligned_full
python_bin=/data2/users/zwruu45/.conda_envs/Geom3D/bin/python
export PYTHONPATH=.:/data/shared/zwr/GOAI/NeuralMD:/data

prepare_pattern="^/data2/users/zwruu45/.conda_envs/Geom3D/bin/python scripts/prepare_aligned_misato.py --config configs/bindmd_full_aligned.yaml"
while pgrep -f "$prepare_pattern" >/dev/null; do
    sleep 30
done

for split in train val test; do
    test -s "$run_dir/cache/aligned_${split}.pt"
done

CUDA_VISIBLE_DEVICES=2 "$python_bin" scripts/train.py \
    --config configs/bindmd_full_aligned.yaml \
    --split train \
    > "$run_dir/logs/train.log" 2>&1

CUDA_VISIBLE_DEVICES=2 "$python_bin" scripts/evaluate.py \
    --config configs/bindmd_full_aligned.yaml \
    --checkpoint "$run_dir/checkpoints/last.pt" \
    --scenario all \
    --output "$run_dir/bindmd_test.json" \
    > "$run_dir/logs/evaluate.log" 2>&1

"$python_bin" scripts/compare_full_results.py \
    --bindmd "$run_dir/bindmd_test.json" \
    --output "$run_dir/comparison.json" \
    > "$run_dir/logs/compare.log" 2>&1
