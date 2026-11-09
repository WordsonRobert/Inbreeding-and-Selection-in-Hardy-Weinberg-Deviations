#!/bin/bash
# SLURM version of run.sh (sbatch slurm_job.sh). Edit partition / CPUs / time for your cluster.
#SBATCH --job-name=bastion
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=bastion_%j.log
##SBATCH --partition=<partition>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
if [ -z "${NO_VENV:-}" ]; then
  [ -d .venv ] || python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -q -r requirements.txt
fi
python run_all.py --mode full --scale "${SCALE:-2.5}" --ode-reps "${ODE_REPS:-6}" --resume \
  --workers "${SLURM_CPUS_PER_TASK:-32}" --out "${OUT:-results}"
