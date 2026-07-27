#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="examples/AlaD/config_alphaL_committor_cuda.yaml"
BASE_RUN="examples/AlaD/runs/alad_alphaL_explore_36x36"
RUN_DIR="examples/AlaD/runs/alad_alphaL_committor_refine_from145"
RESUME_ITER="${RESUME_ITER:-145}"
ITERATIONS="${ITERATIONS:-12}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refinement run directory already exists: ${RUN_DIR}" >&2
  echo "Choose a new outdir in ${CONFIG} or move the existing directory aside." >&2
  exit 1
fi

cp -a "${BASE_RUN}" "${RUN_DIR}"

RUN_ABS="$(realpath "${RUN_DIR}")"
BASE_ABS="$(realpath "${BASE_RUN}")"
LOG="${RUN_DIR}/output.log"
TMP_LOG="${LOG}.tmp"
{
  printf '# Trails-MD run log\n'
  printf '# outdir=%s\n' "${RUN_ABS}"
  printf '# md_engine=openmm\n'
  printf '# space_mode=fixed\n'
  printf '# spawn_scheme=committor\n'
  printf '# spawn_type=hard\n'
  printf '# walker=16\n'
  printf '# step=1000\n'
  printf '# stride=50\n'
  printf '# dt=0.002\n'
  printf '# n_bins=[36, 36]\n'
  printf '# min_values=[-3.141592653589793, -3.141592653589793]\n'
  printf '# max_values=[3.141592653589793, 3.141592653589793]\n'
  printf 'iteration\trunner_s\tanalysis_s\ttotal_s\tsuccessful_walkers\tfailed_walkers\tframes_this_iteration\tcumulative_frames\toccupied_bins\ttotal_bins\texploration_fraction\tspawn_indices\tcvs_file\ttrajectory_dir\tcheckpoint_dir\n'
  awk -v resume="${RESUME_ITER}" -v src="${BASE_ABS}" -v dst="${RUN_ABS}" '
    BEGIN { FS = OFS = "\t" }
    /^[0-9]+\t/ && $1 <= resume {
      gsub(src, dst)
      print
    }
  ' "${LOG}"
} > "${TMP_LOG}"
mv "${TMP_LOG}" "${LOG}"

python -m trails_md.cli \
  --config "${CONFIG}" \
  --resume "${RESUME_ITER}" \
  --iterations "${ITERATIONS}" \
  --log-level INFO
