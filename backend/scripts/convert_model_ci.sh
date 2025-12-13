#!/usr/bin/env bash
set -euo pipefail

# Script: convert_model_ci.sh
# Usage: convert_model_ci.sh <model_filename.onnx> <output_dir>

MODEL_NAME=${1:-Kim_Vocal_2.onnx}
OUTDIR=${2:-backend/models_openvino}

echo "🔍 Searching for model: ${MODEL_NAME}"

SEARCH_PATHS=(
  "$HF_HOME"
  "${HOME}/.cache"
  "./backend/models"
  "./models"
  "./"
)

MODEL_PATH=""
for p in "${SEARCH_PATHS[@]}"; do
  if [ -z "${p}" ]; then
    continue
  fi
  if [ -d "${p}" ]; then
    echo "  searching in ${p}..."
    found=$(find "${p}" -type f -name "${MODEL_NAME}" -print -quit 2>/dev/null || true)
    if [ -n "${found}" ]; then
      MODEL_PATH="${found}"
      break
    fi
  fi
done

if [ -z "${MODEL_PATH}" ]; then
  echo "⚠️ Model ${MODEL_NAME} not found in search paths."
  echo "You can pre-place the model in ./backend/models or set HF_HOME to its cache." >&2
  exit 1
fi

echo "✅ Found model: ${MODEL_PATH}"
mkdir -p "${OUTDIR}"

echo "🔁 Converting to OpenVINO IR (FP16)..."
python -m openvino.tools.mo --input_model "${MODEL_PATH}" --compress_to_fp16 --output_dir "${OUTDIR}"

echo "✅ Converted artifacts in: ${OUTDIR}"
