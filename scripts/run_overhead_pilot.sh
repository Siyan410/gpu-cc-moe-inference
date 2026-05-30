#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/root/models/Qwen3-30B-A3B-Instruct-2507}"
CC_MODE="${CC_MODE:-unknown}"
SUITE="${SUITE:-pilot}"
BATCH_SIZES="${BATCH_SIZES:-1}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
CONDA_ENV="${CONDA_ENV:-CC}"
RUN_ID="${RUN_ID:-overhead-${CC_MODE}-${SUITE}-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-runs/${RUN_ID}}"

EXTRA_ARGS=()
if [[ "${MEASURE_TTFT_PROXY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--measure-ttft-proxy)
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

conda run -n "${CONDA_ENV}" python -m gpu_cc_moe_inference.overhead run-transformers \
  --run-id "${RUN_ID}" \
  --out "${OUT}" \
  --model "${MODEL}" \
  --cc-mode "${CC_MODE}" \
  --suite "${SUITE}" \
  --batch-sizes "${BATCH_SIZES}" \
  --repeats "${REPEATS}" \
  --warmup "${WARMUP}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}"

echo "Wrote ${OUT}/overhead_measurements.jsonl"
echo "Wrote ${OUT}/overhead_summary.json"
