"""The "pipeline" baseline: DDS calibration of HBV_Lab, no agent, on the same inputs, ranges and run budget as the agents.

For each catchment and seed: load data/agent_inputs/<gauge>/forcing.csv (training period only, first year warm-up),
calibrate NSE with DDS, and write runs/baseline/<name>/<gauge>/s<seed>/submission.json in the agents' format, so
scripts/evaluate_runs.py scores both the same way.

  python scripts/run_baseline.py --gauges 12001 39001 --seeds 0 1 2
  python scripts/run_baseline.py --all --task-index $SLURM_ARRAY_TASK_ID      # one catchment per array task
Defaults (name, ranges, iterations, seeds) come from config/agents.yml, `baseline`.

--ranges sets the search ranges:
  default  HBV_Lab's default ranges
  wide     the experiment's starting ranges (config/agents.yml, `ranges`), as given to the agents
  agent    each catchment's final ranges from an agent's submission, --agent-run runs/agents/<agent>/{gauge}/r1
The pilot controls (24 Nov 2026) used --name dds (default ranges), dds_wide and dds_agentranges, 2000 runs each.
"""
import argparse
import json
import os
import time

import pandas as pd
import yaml

from HBV_Lab import HBVModel, __version__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "agents.yml"), encoding="utf-8"))["baseline"]

# The experiment's starting ranges (config/agents.yml), the same for every catchment
WIDE = {k: tuple(v) for k, v in yaml.safe_load(open(os.path.join(ROOT, "config", "agents.yml"),
                                                  encoding="utf-8"))["ranges"].items()}
p = argparse.ArgumentParser()
p.add_argument("--gauges", type=int, nargs="*")
p.add_argument("--n", type=int, default=64, help="first n catchments of config/subset_ordered.csv")
p.add_argument("--all", action="store_true", help="all catchments in config/basins_all.txt")
p.add_argument("--task-index", type=int, help="run only the catchment at this 0-based position")
p.add_argument("--seeds", type=int, nargs="*", default=cfg["seeds"])
p.add_argument("--iterations", type=int, default=cfg["iterations"])
p.add_argument("--inputs", default=os.path.join(ROOT, "data", "agent_inputs"))
p.add_argument("--ranges", choices=["default", "wide", "agent"], default=cfg["ranges"])
p.add_argument("--name", default=cfg["name"], help="output folder under runs/baseline")
p.add_argument("--agent-run", default=os.path.join(ROOT, "runs", "agents", "claude-code-opus", "{gauge}", "r1"))
p.add_argument("--out")
a = p.parse_args()
a.out = a.out or os.path.join(ROOT, "runs", "baseline", a.name)


def apply_ranges(m, g):
    if a.ranges == "default":
        return
    if a.ranges == "wide":
        rng = WIDE
    else:
        rng = json.load(open(os.path.join(a.agent_run.format(gauge=g), "submission.json")))["ranges"]
    upd = {}
    for grp, ps in m.params.items():
        for k, v in ps.items():
            lo, hi = rng[k]
            upd.setdefault(grp, {})[k] = {"min": lo, "max": hi, "default": min(max(v["default"], lo), hi)}
    m.set_parameters(upd)

if a.gauges:
    gauges = a.gauges
elif a.all:
    gauges = pd.read_csv(os.path.join(ROOT, "config", "basins_all.txt"), header=None)[0].astype(int).tolist()
else:
    gauges = pd.read_csv(os.path.join(ROOT, "config", "subset_ordered.csv"))["gauge_id"].head(a.n).tolist()
if a.task_index is not None:
    gauges = [gauges[a.task_index]]

for g in gauges:
    for seed in a.seeds:
        out = os.path.join(a.out, str(g), f"s{seed}")
        if os.path.exists(os.path.join(out, "submission.json")) or os.path.exists(os.path.join(out, "failed.json")):
            continue
        os.makedirs(out, exist_ok=True)
        m = HBVModel()
        m.load_data(file_path=os.path.join(a.inputs, str(g), "forcing.csv"), obs_q_column="Discharge",
                    warmup_end="19710930")
        apply_ranges(m, g)
        t0 = time.time()
        try:
            m.calibrate(method="DDS", objective="NSE", iterations=a.iterations, seed=seed, verbose=False,
                        plot_results=False)
        except ValueError as e:  # e.g. no observed flow in the training period
            json.dump({"gauge": g, "seed": seed, "error": str(e)}, open(os.path.join(out, "failed.json"), "w"))
            print(g, seed, "failed:", e, flush=True)
            continue
        sub = {
            "model_id": "baseline", "diagnosis": "", "method": "DDS", "iterations": a.iterations, "seed": seed,
            "range_set": a.ranges,
            "hbv_lab": __version__,
            "parameters": {k: float(v["default"]) for grp in m.params.values() for k, v in grp.items()},
            "ranges": {k: [float(v["min"]), float(v["max"])] for grp in m.params.values() for k, v in grp.items()},
            "initial_states": {k: float(v) for k, v in m.states.items()},
            "metrics_reported": {k: float(v) for k, v in m.performance_metrics.items()},
            "model_runs_used": a.iterations + 1, "wall_seconds": round(time.time() - t0, 1),
        }
        json.dump(sub, open(os.path.join(out, "submission.json"), "w"), indent=1)
        print(g, seed, round(sub["metrics_reported"]["NSE"], 3), sub["wall_seconds"], "s", flush=True)
