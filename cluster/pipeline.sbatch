#!/bin/bash
#SBATCH --job-name=hbvlab-pipe
#SBATCH --partition=myriad
#SBATCH --qos=1cpu72h
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --array=0-668%150
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%A_%a.out
# Pipeline baseline on all 669 catchments, one per array task: DDS, the agent's ranges and run budget, 3 seeds
# (settings in config/agents.yml, `baseline`).   sbatch cluster/pipeline.sbatch
set -euo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
echo "host $(hostname)  task $SLURM_ARRAY_TASK_ID  start $(date -Is)"
.venv/bin/python scripts/run_baseline.py --all --task-index "$SLURM_ARRAY_TASK_ID" 2>&1 \
  | grep -v "^Time step\|^Warmup\|^Loaded\|Calibration failed"
echo "end $(date -Is)"
