#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${1:-150}"
MICROSTATES="${MICROSTATES:-100}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="config_alphaL_explore_cuda.yaml"
CONFIG="config_alphaL_explore_cuda_${STAMP}.yaml"
RUN_DIR="runs/alad_alphaL_explore_36x36_${STAMP}"

python - "$BASE_CONFIG" "$CONFIG" "$RUN_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

base, out, run_dir = map(Path, sys.argv[1:])
data = yaml.safe_load(base.read_text(encoding="utf-8"))
data["outdir"] = str(run_dir)
data.setdefault("spawning", {})["resolution_check_patience"] = 0
out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
print(out)
PY

trails-md --config "$CONFIG" --iterations "$ITERATIONS" --log-level INFO

python ../alanine_dipeptide/analyze_alphaR_committor_convergence.py \
  --run-dir "$RUN_DIR" \
  --target alphaL \
  --n-microstates "$MICROSTATES" \
  --local-grid-size 36

echo "Run directory: $RUN_DIR"
echo "Config: $CONFIG"
