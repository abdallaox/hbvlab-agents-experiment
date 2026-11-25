#!/bin/bash
#SBATCH --job-name=hbvlab-cpu
#SBATCH --partition=myriad
#SBATCH --qos=1cpu72h
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%j.out
# Any single-core Python script, e.g.  sbatch --job-name=cmp-opt cluster/cpu.sbatch scripts/compare_optimisers.py
set -euo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
.venv/bin/python "$@"
echo "end $(date -Is)"
