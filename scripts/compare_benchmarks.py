"""Compare the pipeline (HBV_Lab DDS, no agent) with published conceptual models on the test period 1998-2008.

All metrics use the benchmark study's definitions (scripts/metrics_nh.py, NeuralHydrology): NSE, KGE (Gupta 2009),
FHV (top 2% flows), FLV (bottom 30%, log), FMS (mid-segment slope of the flow duration curve) and PBIAS.

Pipeline: median over its 3 seeds per catchment (results/runs_scored.csv, from evaluate_runs.py).
LC-HBV: median over the 5 restarts per catchment (../camels_uk_exp/results/hbv_local_r*/metrics_test_full.csv),
  computed there with the same metric code.
FUSE (TOPMODEL, ARNO/VIC, PRMS, Sacramento): Lane et al. (2019) best simulations, scored here on 1998-2008 against
  CAMELS-GB flow. They were selected on 1988-2008, so this is in-sample for them.

Writes results/benchmark_per_catchment.csv and results/benchmark_summary.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_nh  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--npz", default=os.path.join(ROOT, "..", "camels_uk_exp", "data", "processed", "camels_gb.npz"))
p.add_argument("--lc", default=os.path.join(ROOT, "..", "camels_uk_exp", "results"))
a = p.parse_args()
T0, T1 = "1998-01-01", "2008-12-31"
METRICS = ["NSE", "KGE", "FHV", "FLV", "FMS", "PBIAS"]
FUNCS = {**metrics_nh.ALL, **metrics_nh.EXTRA}

cols = {}
# pipeline
sc = pd.read_csv(os.path.join(ROOT, "results", "runs_scored.csv"))
pl = sc[(sc.kind == "baseline") & (sc.agent == "pipeline") & sc.NSE_test.notna()]
g = pl.groupby("gauge")
for m in METRICS:
    cols[f"pipeline_{m}"] = g[f"{m}_test"].median()
cols["pipeline_KGEp"] = g.KGEp_test.median()
cols["pipeline_NSE_seed_range"] = g.NSE_test.max() - g.NSE_test.min()
cols["pipeline_NSE_cal"] = g.NSE_cal.median()

# LC-HBV
lc = [pd.read_csv(f, index_col="basin") for f in sorted(glob.glob(os.path.join(a.lc, "hbv_local_r*", "metrics_test_full.csv")))]
for m in METRICS:
    cols[f"LC-HBV_{m}"] = pd.concat([d[m] for d in lc], axis=1).median(axis=1)

# Lane FUSE, scored on the test period against CAMELS-GB flow
with np.load(a.npz) as z:
    basins, dates, target = z["basins"], pd.to_datetime(z["dates"]), z["target"]
tmask = (dates >= T0) & (dates <= T1)
FUSE = {"TOPMODEL": "TOPMODEL", "ARNOVIC": "ARNO/VIC", "PRMS": "PRMS", "SACRAMENTO": "Sacramento"}
fuse = {k: {} for k in FUSE}
for f in glob.glob(os.path.join(ROOT, "data", "external", "lane2019", "Timeseries_SimQ_Best", "*_Best_Qsim.txt")):
    gauge = int(os.path.basename(f).split("_")[0])
    if gauge not in basins:
        continue
    lines = open(f).read().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("YYYY"))
    d = pd.read_csv(f, skiprows=h, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d.index = pd.to_datetime(dict(year=d.YYYY, month=d.MM, day=d.DD))
    obs = pd.Series(target[int(np.where(basins == gauge)[0][0])].astype(float), index=dates)[tmask]
    for k in FUSE:
        s = d[f"SimQ_{k}"].reindex(obs.index).to_numpy(float)
        fuse[k][gauge] = {m: FUNCS[m](s, obs.to_numpy()) for m in METRICS}
for k, name in FUSE.items():
    df = pd.DataFrame(fuse[k]).T
    for m in METRICS:
        cols[f"FUSE-{name}_{m}"] = df[m]

per = pd.DataFrame(cols)
per.index.name = "gauge"
per.to_csv(os.path.join(ROOT, "results", "benchmark_per_catchment.csv"))

models = ["LC-HBV"] + [f"FUSE-{n}" for n in FUSE.values()]
rows = []
for m in ["pipeline"] + models:
    own = per[f"{m}_NSE"].dropna()
    row = {"model": m, "n_own": len(own), "NSE_lt_0.5_own": (own < 0.5).mean()}
    for k in METRICS:
        row[f"{k}_median_own"] = per[f"{m}_{k}"].median()
        if k in ("FHV", "FLV", "FMS", "PBIAS"):
            row[f"{k}_absmedian_own"] = per[f"{m}_{k}"].abs().median()
    if m != "pipeline":
        both = per[["pipeline_NSE", f"{m}_NSE"]].dropna()
        diff = both["pipeline_NSE"] - both[f"{m}_NSE"]
        row.update(n_common=len(both), NSE_median_model_common=both[f"{m}_NSE"].median(),
                   NSE_median_pipeline_common=both["pipeline_NSE"].median(), median_diff_pipeline_minus_model=diff.median(),
                   share_pipeline_better=(diff > 0).mean())
        for k in METRICS:
            bk = per[[f"pipeline_{k}", f"{m}_{k}"]].dropna()
            f = (lambda x: x) if k in ("NSE", "KGE") else (lambda x: x.abs())
            row[f"{k}_common_model"], row[f"{k}_common_pipeline"] = f(bk[f"{m}_{k}"]).median(), f(bk[f"pipeline_{k}"]).median()
    rows.append(row)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(ROOT, "results", "benchmark_summary.csv"), index=False)
pd.set_option("display.width", 250)
show = ["model", "n_own", "n_common", "NSE_common_model", "NSE_common_pipeline", "share_pipeline_better", "KGE_common_model",
        "KGE_common_pipeline", "FHV_common_model", "FHV_common_pipeline", "FLV_common_model", "FLV_common_pipeline",
        "FMS_common_model", "FMS_common_pipeline", "PBIAS_common_model", "PBIAS_common_pipeline"]
print(summ[[c for c in show if c in summ]].round(3).T.to_string())
print("pipeline: NSE", round(per.pipeline_NSE.median(), 3), " KGE", round(per.pipeline_KGE.median(), 3),
      " KGE'", round(per.pipeline_KGEp.median(), 3), " seed NSE range median", round(per.pipeline_NSE_seed_range.median(), 4))
