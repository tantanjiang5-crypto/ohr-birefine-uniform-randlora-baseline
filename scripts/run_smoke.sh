#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_DIR [PHYSICAL_GPU]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"
GPU="${3:-0}"
export PYTHONPATH="$REPO_ROOT/baseline/code:$REPO_ROOT/baseline/vendor/DAG-RandLoRA-v1.0.0:${PYTHONPATH:-}"
export SAM1_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["sam_checkpoint"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES="$GPU" python -u "$REPO_ROOT/baseline/code/scripts/train_randlora_pair.py" \
  --config "$CONFIG" --mode V1 --device cuda:0 --output-dir "$OUTPUT" \
  --max-optimizer-steps 2 --max-val-images 16
