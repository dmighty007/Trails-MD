#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${1:-150}"
MICROSTATES="${MICROSTATES:-100}"
CONFIG="config_alphaL_explore_cuda.yaml"
RUN_DIR="runs/alad_alphaL_explore_36x36"

trails-md --config "$CONFIG" --iterations "$ITERATIONS" --log-level INFO

python ../alanine_dipeptide/analyze_alphaR_committor_convergence.py \
  --run-dir "$RUN_DIR" \
  --target alphaL \
  --n-microstates "$MICROSTATES" \
  --local-grid-size 36
