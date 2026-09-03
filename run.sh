#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="$ROOT_DIR/weights/complexmd_v3_6plus6_epoch004.pt"
CONFIG="$ROOT_DIR/configs/complexmd_v3_6plus6_submission.yaml"
OUTPUT_ROOT="$ROOT_DIR/predictions"
LOG_ROOT="$ROOT_DIR/run_logs"

find_input_root() {
  local candidate
  if [[ -n "${GOAI_INPUT_ROOT:-}" ]]; then
    candidate="$GOAI_INPUT_ROOT"
    if [[ -f "$candidate/protocol.json" && -d "$candidate/T1" && -d "$candidate/T2" && -d "$candidate/T3" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  for candidate in \
    "$ROOT_DIR/evaluation_data" \
    "$ROOT_DIR/GOAI_eval_public" \
    "$ROOT_DIR/public" \
    "$ROOT_DIR/data/evaluation_data" \
    "$ROOT_DIR/../GOAI_eval_public"; do
    if [[ -f "$candidate/protocol.json" && -d "$candidate/T1" && -d "$candidate/T2" && -d "$candidate/T3" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

INPUT_ROOT="$(find_input_root || true)"
if [[ -z "$INPUT_ROOT" ]]; then
  echo "ERROR: evaluation data were not found." >&2
  echo "Place the unpacked package at ./evaluation_data (recommended) or set GOAI_INPUT_ROOT." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT/manifests"

"$PYTHON_BIN" scripts/submission_preflight.py \
  --input-root "$INPUT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG"

for tier in T1 T2 T3; do
  "$PYTHON_BIN" scripts/predict_goai.py \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT" \
    --manifest-dir "$LOG_ROOT/manifests" \
    --tier "$tier" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --history-frames 12 \
    --sampling-steps 10 \
    --topology-source auto \
    --ligand-projection fragments \
    --pose-mode model \
    --pose-translation-scale 0.25 \
    --pose-rotation-scale 0.25 \
    --seed 42 \
    --device cuda
done

"$PYTHON_BIN" scripts/validate_goai_submission.py \
  --prediction-root "$OUTPUT_ROOT" \
  --evaluation-root "$INPUT_ROOT" \
  --output "$LOG_ROOT/submission_validation.json"

echo "ComplexMD inference completed successfully."
echo "Predictions: $OUTPUT_ROOT/{T1,T2,T3}/*_pred.xtc"
echo "Validation:  $LOG_ROOT/submission_validation.json"

