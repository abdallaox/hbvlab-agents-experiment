#!/bin/bash
#SBATCH --job-name=hbvlab-rep
#SBATCH --partition=myriad
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/Scratch/runs/slurm/%x_%j.out
# Claude agent once on a representative sample: rounds 1-6 of config/subset_ordered.csv (96 catchments, 6 per
# aridity x baseflow-index quartile class), 4 runs at a time. Runs already done (showcase r1) are skipped.
# Stops starting runs at $25 total under runs/agents/claude-api.
set -euo pipefail
cd ~/hbvlab-agents-paper
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
gauges=$(.venv/bin/python -c "import pandas as pd; s=pd.read_csv('config/subset_ordered.csv'); print(' '.join(map(str, s[s['round']<=6].sort_values('order').gauge_id)))")
.venv/bin/python harness/run_showcase.py --gauges $gauges --repeats 1 --parallel 4 --max-cost 25
echo "end $(date -Is)"
