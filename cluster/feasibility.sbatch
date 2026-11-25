#!/bin/bash
#SBATCH --job-name=hbvlab-feas
#SBATCH --partition=myriad
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%j.out
# Feasibility check on a CPU node: API reachability from compute nodes, then HBV_Lab timing on one catchment.
set -uo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
for u in https://api.anthropic.com https://api.openai.com https://generativelanguage.googleapis.com https://openrouter.ai; do
  printf "%s " "$u"; curl -s -o /dev/null -m 10 -w "%{http_code}\n" "$u" || echo "FAIL (no route)"
done
lscpu | grep "Model name"
.venv/bin/python scripts/feasibility.py "$@"
echo "end $(date -Is)"
