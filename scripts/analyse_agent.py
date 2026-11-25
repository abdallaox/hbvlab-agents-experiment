"""Agent showcase analysis: agent (claude-api, 3 repeats) against the pipeline (3 seeds) and LC-HBV per catchment,
tools used, range and initial-state changes, cost. Writes results/agent_per_catchment.csv and
results/agent_behaviour.csv and prints a summary.
"""
import collections
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sc = pd.read_csv(os.path.join(ROOT, "results", "runs_scored.csv"))
bench = pd.read_csv(os.path.join(ROOT, "results", "benchmark_per_catchment.csv"), index_col="gauge")
ag = sc[sc.agent == "claude-api"].copy()
pl = sc[sc.agent == "pipeline"]
attrs = pd.read_csv(os.path.join(ROOT, "config", "subset_ordered.csv"))

beh = []
for path in sorted(glob.glob(os.path.join(ROOT, "runs", "agents", "claude-api", "*", "r*"))):
    gauge, rep = int(os.path.basename(os.path.dirname(path))), os.path.basename(path)
    calls = collections.Counter()
    cal_methods, init_sets, range_sets, thinking_chars = [], 0, 0, 0
    for line in open(os.path.join(path, "transcript.jsonl")):
        d = json.loads(line)
        if d["kind"] == "tool":
            calls[d["name"]] += 1
            if d["name"] == "calibrate":
                cal_methods.append(f"{d['args'].get('method', '?')}:{d['args'].get('iterations', '?')}")
            init_sets += d["name"] == "set_initial_conditions"
            range_sets += d["name"] == "set_parameter_ranges"
        elif d["kind"] == "response":
            for b in (d["message"] if isinstance(d["message"], list) else d["message"].get("content", [])):
                if b.get("type") == "thinking":
                    thinking_chars += len(b.get("thinking", ""))
    sub = json.load(open(os.path.join(path, "submission.json")))
    beh.append({"gauge": gauge, "repeat": rep, "n_calibrate": calls["calibrate"], "calibrations": " ".join(cal_methods),
                "set_ranges": range_sets, "set_initial": init_sets, "clone": calls["clone_model"],
                "compare": calls["compare_models"], "uncertainty": calls["evaluate_uncertainty"],
                "run_model": calls["run_model"], "set_parameters": calls["set_parameters"],
                "tools": " ".join(f"{k}:{v}" for k, v in sorted(calls.items())), "thinking_chars": thinking_chars,
                "diagnosis": sub.get("diagnosis", "")})
beh = pd.DataFrame(beh)
ag["gauge"] = ag.gauge.astype(int)
beh = beh.merge(ag[["gauge", "repeat", "NSE_cal", "NSE_test", "KGE_test", "tool_calls", "model_runs", "cost_usd",
                    "wall_seconds", "ranges_changed", "at_bound_list"]], on=["gauge", "repeat"])
beh.to_csv(os.path.join(ROOT, "results", "agent_behaviour.csv"), index=False)

g = beh.groupby("gauge")
per = pd.DataFrame({"agent_NSE": g.NSE_test.median(), "agent_NSE_min": g.NSE_test.min(), "agent_NSE_max": g.NSE_test.max(),
                    "agent_NSE_cal": g.NSE_cal.median(),
                    "pipeline_NSE": pl.groupby("gauge").NSE_test.median(), "pipeline_NSE_cal": pl.groupby("gauge").NSE_cal.median(),
                    "pipeline_NSE_min": pl.groupby("gauge").NSE_test.min(), "pipeline_NSE_max": pl.groupby("gauge").NSE_test.max()}
                   ).loc[beh.gauge.unique()]
per["LC-HBV_NSE"] = bench["LC-HBV_NSE"].reindex(per.index)
per["diff_agent_minus_pipeline"] = per.agent_NSE - per.pipeline_NSE
per["ranges_changed_runs"] = g.apply(lambda x: (x.ranges_changed.fillna("") != "").sum())
per["initial_set_runs"] = g.apply(lambda x: (x.set_initial > 0).sum())
per["cost_usd"] = g.cost_usd.sum()
per = per.join(attrs.set_index("gauge_id")[[c for c in ("gauge_name", "aridity", "baseflow_index") if c in attrs.columns]]
               if "gauge_id" in attrs.columns else pd.DataFrame(index=per.index))
per.index.name = "gauge"
per.to_csv(os.path.join(ROOT, "results", "agent_per_catchment.csv"))
pd.set_option("display.width", 250)
print(per.round(3).drop(columns=[c for c in ("aridity", "baseflow_index") if c in per]).to_string())
print()
print("medians: agent", round(per.agent_NSE.median(), 3), " pipeline", round(per.pipeline_NSE.median(), 3),
      " LC-HBV", round(per["LC-HBV_NSE"].median(), 3))
d = per.diff_agent_minus_pipeline
print("agent - pipeline: median", round(d.median(), 3), " agent better (>0.01)", (d > 0.01).sum(),
      " tie (|d|<=0.01)", (d.abs() <= 0.01).sum(), " worse (<-0.01)", (d < -0.01).sum())
print("runs: n", len(beh), " cost total $", round(beh.cost_usd.sum(), 2), " per run median $", round(beh.cost_usd.median(), 3),
      " max $", round(beh.cost_usd.max(), 3))
print("tool calls median", beh.tool_calls.median(), " range", beh.tool_calls.min(), "-", beh.tool_calls.max(),
      "  model runs median", beh.model_runs.median(), "  wall median s", beh.wall_seconds.median())
print("runs that changed ranges", (beh.ranges_changed.fillna("") != "").sum(), " set initial states",
      (beh.set_initial > 0).sum(), " used uncertainty", (beh.uncertainty > 0).sum(), " cloned", (beh.clone > 0).sum())
print("calibrations per run median", beh.n_calibrate.median())
print("ranges changed:", collections.Counter(" ".join(beh.ranges_changed.fillna("")).split()).most_common())
print("repeat spread (max-min test NSE) median", round((g.NSE_test.max() - g.NSE_test.min()).median(), 3),
      " max", round((g.NSE_test.max() - g.NSE_test.min()).max(), 3))
