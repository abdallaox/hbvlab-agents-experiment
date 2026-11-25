#!/bin/bash
#SBATCH --job-name=hbvlab-agent
#SBATCH --partition=myriad
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%j.out
# Claude agent on the showcase catchments (config/showcase.txt), 3 repeats, 3 runs at a time, stop at $30 total.
set -euo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
.venv/bin/python harness/run_showcase.py --repeats 3 --parallel 3 --max-cost 30
echo "end $(date -Is)"
