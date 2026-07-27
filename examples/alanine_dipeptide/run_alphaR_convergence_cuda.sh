#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${1:-50}"
MICROSTATES="${MICROSTATES:-80}"
CONFIG="config_cuda_alphaR_convergence.yaml"
RUN_DIR="runs/alanine_dipeptide_alphaR_convergence"

#CMD=(trails-md --config "$CONFIG" --iterations "$ITERATIONS" --log-level INFO)
#if [[ "${RESUME:-}" == "1" || "${RESUME:-}" == "latest" ]]; then
#  CMD+=(--resume latest)
#fi

#"${CMD[@]}"

python analyze_alphaR_committor_convergence.py \
  --run-dir "$RUN_DIR" \
  --n-microstates "$MICROSTATES"
