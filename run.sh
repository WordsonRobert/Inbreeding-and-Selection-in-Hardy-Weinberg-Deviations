#!/usr/bin/env bash
# Reproduce everything in results/ with one command.
#
#   bash run.sh                 full run, replicate scale 2.5 (the published run; ~3.5 h on 16 threads)
#   SCALE=1 bash run.sh         same experiments with 2.5x fewer replicates (~1.4 h on 16 threads)
#   QUICK=1 bash run.sh         smoke test only (~6-10 min on 2 cores), writes results_quick/
#   WORKERS=8 bash run.sh       number of CPU worker processes (default: all cores minus one)
#   NO_VENV=1 bash run.sh       use the active python/conda environment instead of creating .venv
#   OUT=my_results bash run.sh  write somewhere other than results/
#
# The full run overwrites results/. Interrupted? Run the same command again: stages whose
# summary file already exists are skipped (--resume).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ -z "${NO_VENV:-}" ]; then
  [ -d .venv ] || $PY -m venv .venv || { echo "venv failed: rerun with NO_VENV=1 inside your own environment"; exit 1; }
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m pip install -q --upgrade pip || true
python -m pip install -q -r requirements.txt
python -c "import torch" 2>/dev/null \
  && echo "torch found: E11 (neural-ODE comparison) will run" \
  || echo "torch not found: E11 (neural-ODE comparison) is skipped; everything else runs"

ARGS=()
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "$WORKERS")

if [ -n "${QUICK:-}" ]; then
  python run_all.py --mode quick --out "${OUT:-results_quick}" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee run_quick.log
else
  python run_all.py --mode full --scale "${SCALE:-2.5}" --ode-reps "${ODE_REPS:-6}" --resume \
    --out "${OUT:-results}" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "${OUT:-results}_run.log"
  echo "done: ${OUT:-results}/SUMMARY.md"
fi
