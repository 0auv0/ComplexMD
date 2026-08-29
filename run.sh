#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLEXMD_PYTHON="${COMPLEXMD_PYTHON:-python}"
CHECKPOINT="$ROOT_DIR/checkpoints/epoch_015.pt"
CONFIG="$ROOT_DIR/configs/complexmd_inference.yaml"
OUTPUT_ROOT="${COMPLEXMD_OUTPUT_ROOT:-$ROOT_DIR/predictions}"
EXPECTED_SHA256="96fff72a87d7c9a7b24f59501a317f8443b2a1fb612b7c6e2602a1739f871616"

if [[ -n "${GOAI_INPUT_ROOT:-}" ]]; then
  INPUT_ROOT="$GOAI_INPUT_ROOT"
elif [[ -f "$ROOT_DIR/GOAI_eval_public/protocol.json" ]]; then
  INPUT_ROOT="$ROOT_DIR/GOAI_eval_public"
elif [[ -f "/data/GOAI_eval_public/protocol.json" ]]; then
  INPUT_ROOT="/data/GOAI_eval_public"
else
  echo "ERROR: GOAI evaluation data not found." >&2
  echo "Place GOAI_eval_public beside run.sh or set GOAI_INPUT_ROOT." >&2
  exit 2
fi

[[ -f "$CHECKPOINT" ]] || { echo "ERROR: missing $CHECKPOINT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "ERROR: missing $CONFIG" >&2; exit 2; }
for tier in T1 T2 T3; do
  [[ -f "$INPUT_ROOT/$tier/ids.txt" ]] || {
    echo "ERROR: missing $INPUT_ROOT/$tier/ids.txt" >&2
    exit 2
  }
done

ACTUAL_SHA256="$($COMPLEXMD_PYTHON - "$CHECKPOINT" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
with pathlib.Path(sys.argv[1]).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: checkpoint SHA256 mismatch." >&2
  echo "expected=$EXPECTED_SHA256" >&2
  echo "actual=$ACTUAL_SHA256" >&2
  exit 2
fi

"$COMPLEXMD_PYTHON" - <<'PY'
import torch

print(f"Python/PyTorch self-check: torch={torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU is required for the full reproduction run")
print(f"CUDA self-check: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

for tier in T1 T2 T3; do
  "$COMPLEXMD_PYTHON" "$ROOT_DIR/scripts/predict_goai.py" \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT" \
    --tier "$tier" \
    --ligand-mode model \
    --pose-mode model \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --sampling-steps 10 \
    --pose-translation-scale 0.25 \
    --pose-rotation-scale 0.25 \
    --seed 42 \
    --device cuda
done

"$COMPLEXMD_PYTHON" "$ROOT_DIR/scripts/validate_goai_submission.py" \
  --prediction-root "$OUTPUT_ROOT" \
  --evaluation-root "$INPUT_ROOT" \
  --output "$OUTPUT_ROOT/validation.json"

"$COMPLEXMD_PYTHON" "$ROOT_DIR/scripts/package_submission.py" \
  --prediction-root "$OUTPUT_ROOT" \
  --output "$ROOT_DIR/GOAI_pred_COMPLEXMD.zip"

echo "ComplexMD reproduction completed successfully."
echo "Predictions: $OUTPUT_ROOT"
echo "Material-A archive: $ROOT_DIR/GOAI_pred_COMPLEXMD.zip"

