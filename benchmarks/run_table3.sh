#!/usr/bin/env bash
# Reproduce Table 3 (TorchKM vs ThunderSVM on a7a/a8a/w7a).
#
# Usage:
#   bash benchmarks/run_table3.sh [DATA_DIR] [THUNDERSVM_PYTHON_PATH]
#
# Defaults: DATA_DIR=./libsvm_data, ThunderSVM path empty (column auto-skips).
# If a specific GPU is needed, prefix with CUDA_VISIBLE_DEVICES=<idx>.

set -euo pipefail

DATA_DIR="${1:-./libsvm_data}"
THUNDERSVM_PATH="${2:-}"
BASE_URL="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

mkdir -p "$DATA_DIR"
for f in a7a a7a.t a8a a8a.t w7a w7a.t; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "Downloading $f -> $DATA_DIR/$f"
    curl -fsSL "$BASE_URL/$f" -o "$DATA_DIR/$f"
  fi
done

EXTRA=()
if [ -n "$THUNDERSVM_PATH" ]; then
  EXTRA+=(--thundersvm-path "$THUNDERSVM_PATH")
fi

python benchmarks/table3_benchmarks.py \
  --data-dir "$DATA_DIR" \
  --repeats 10 \
  --device cuda \
  "${EXTRA[@]}"
