#!/bin/bash
#SBATCH --job-name=hbvlab-ctrl
#SBATCH --partition=myriad
#SBATCH --qos=1cpu72h
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%A_%a.out
# DDS range controls for the pilot catchments: task = gauge index x range set (wide, agent)
set -euo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
GAUGES=(12001 33019 54003 39014)
RANGES=(wide agent)
G=${GAUGES[$((SLURM_ARRAY_TASK_ID / 2))]}
R=${RANGES[$((SLURM_ARRAY_TASK_ID % 2))]}
echo "host $(hostname)  gauge $G  ranges $R  start $(date -Is)"
.venv/bin/python scripts/run_baseline.py --gauges "$G" --ranges "$R" 2>&1 | grep -v "^Time step\|^Warmup\|^Loaded"
echo "end $(date -Is)"
