"""Agent against pipeline on the representative sample: rounds 1-6 of config/subset_ordered.csv (95 catchments,
about 6 per aridity x baseflow-index quartile class). Agent: repeat 1 (one run per catchment). Pipeline: median over
its 3 seeds per catchment, plus its seed range. Metrics from results/runs_scored.csv (benchmark definitions).

Writes results/agent_representative.csv (per catchment) and prints a summary.
Run after evaluate_runs.py, compare_benchmarks.py and analyse_agent.py.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sub = pd.read_csv(os.path.join(ROOT, "config", "subset_ordered.csv"))
sample = sub[sub["round"] <= 6].set_index("gauge_id")
sc = pd.read_csv(os.path.join(ROOT, "results", "runs_scored.csv"))
bench = pd.read_csv(os.path.join(ROOT, "results", "benchmark_per_catchment.csv"), index_col="gauge")
beh = pd.read_csv(os.path.join(ROOT, "results", "agent_behaviour.csv"))
M = ["NSE", "KGE", "PBIAS", "FHV", "FMS", "FLV"]

ag = sc[(sc.agent == "claude-api") & (sc["repeat"] == "r1") & sc.gauge.isin(sample.index)].set_index("gauge")
pl = sc[(sc.agent == "pipeline") & sc.gauge.isin(sample.index)].groupby("gauge")
per = pd.DataFrame(index=sample.index)
for m in M:
    per[f"agent_{m}"] = ag[f"{m}_test"]
    per[f"pipeline_{m}"] = pl[f"{m}_test"].median()
per["pipeline_NSE_min"], per["pipeline_NSE_max"] = pl.NSE_test.min(), pl.NSE_test.max()
per["agent_NSE_cal"], per["pipeline_NSE_cal"] = ag.NSE_cal, pl.NSE_cal.median()
per["LC-HBV_NSE"] = bench["LC-HBV_NSE"].reindex(per.index)
per["diff_NSE"] = per.agent_NSE - per.pipeline_NSE
per[["aridity", "baseflow_index", "aridity_q", "bfi_q"]] = sample[["aridity", "baseflow_index", "aridity_q", "bfi_q"]]
b1 = beh[beh["repeat"] == "r1"].set_index("gauge")
for c in ("tool_calls", "model_runs", "cost_usd", "wall_seconds", "set_ranges", "set_initial", "n_calibrate",
          "ranges_changed"):
    per[c] = b1[c].reindex(per.index)
per.index.name = "gauge"
per.to_csv(os.path.join(ROOT, "results", "agent_representative.csv"))

d = per.dropna(subset=["agent_NSE", "pipeline_NSE"])
print(f"catchments {len(per)}, scored {len(d)}")
print("median            agent   pipeline")
for m in M:
    f = (lambda x: x) if m in ("NSE", "KGE") else (lambda x: x.abs())
    print(f"  {m:6s}{'' if m in ('NSE', 'KGE') else ' (abs)'}  {f(d[f'agent_{m}']).median():8.3f} {f(d[f'pipeline_{m}']).median():8.3f}")
diff = d.diff_NSE
print(f"NSE diff agent-pipeline: median {diff.median():.4f}, mean {diff.mean():.4f}; better >0.01: {(diff > 0.01).sum()},"
      f" within 0.01: {(diff.abs() <= 0.01).sum()}, worse <-0.01: {(diff < -0.01).sum()}")
print("Wilcoxon signed-rank (NSE): p =", round(wilcoxon(d.agent_NSE, d.pipeline_NSE).pvalue, 3))
inside = ((d.agent_NSE >= d.pipeline_NSE_min - 0.005) & (d.agent_NSE <= d.pipeline_NSE_max + 0.005)).mean()
print(f"agent inside pipeline seed range (+-0.005): {inside:.2f}")
print("largest gains:", diff.sort_values(ascending=False).head(5).round(3).to_dict())
print("largest losses:", diff.sort_values().head(5).round(3).to_dict())
print(f"agent NSE < 0.5: {(d.agent_NSE < 0.5).sum()}, pipeline NSE < 0.5: {(d.pipeline_NSE < 0.5).sum()}")
print(f"cost total ${per.cost_usd.sum():.2f}, median ${per.cost_usd.median():.3f}, max ${per.cost_usd.max():.3f};"
      f" calls median {per.tool_calls.median()}, wall median {per.wall_seconds.median() / 60:.1f} min")
print(f"ranges changed in {(per.ranges_changed.fillna('') != '').sum()} runs, initial states set in "
      f"{(per.set_initial > 0).sum()}, calibrations median {per.n_calibrate.median()}")
print("diff by BFI quartile (median):", d.groupby("bfi_q").diff_NSE.median().round(3).to_dict())
print("diff by aridity quartile (median):", d.groupby("aridity_q").diff_NSE.median().round(3).to_dict())
