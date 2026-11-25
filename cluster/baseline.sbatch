#!/bin/bash
#SBATCH --job-name=hbvlab-dds
#SBATCH --partition=myriad
#SBATCH --qos=1cpu72h
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --array=0-63
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%A_%a.out
# DDS baseline, one catchment per array task (3 seeds each):  sbatch cluster/baseline.sbatch
set -euo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
echo "host $(hostname)  task $SLURM_ARRAY_TASK_ID  start $(date -Is)"
.venv/bin/python scripts/run_baseline.py --n 64 --task-index "$SLURM_ARRAY_TASK_ID" 2>&1 | grep -v "^Time step\|^Warmup\|^Loaded"
echo "end $(date -Is)"
