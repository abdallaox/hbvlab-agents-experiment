"""Compare Nelder-Mead and DDS calibration of HBV_Lab on a few contrasting CAMELS-GB catchments.

Calibrates on the benchmark training period (1970-10-01 to 1997-12-31, first year warm-up) with NSE and evaluates on
the test period (1998-2008, with 1997 as warm-up). Picks the catchments with the lowest and highest aridity and the
highest snow fraction, plus any given with --basins.
Usage: python scripts/compare_optimisers.py [--basins 39001] [--dds-evals 2000] [--nm-iter 1000] [--seed 0]
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd

from HBV_Lab import HBVModel

p = argparse.ArgumentParser()
p.add_argument("--npz", default=os.path.expanduser("~/camels_uk_exp/data/processed/camels_gb.npz"))
p.add_argument("--basins", type=int, nargs="*", default=[39001])
p.add_argument("--dds-evals", type=int, default=2000)
p.add_argument("--nm-iter", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--methods", nargs="*", default=["Nelder-Mead", "DDS"])
p.add_argument("--ranges", choices=["default", "v1.4"], default="default",
               help="default: the installed HBV_Lab ranges. v1.4: the narrower FC, PERC and K2 ranges of HBV_Lab <= 1.4.2")
a = p.parse_args()

d = np.load(a.npz)
names = list(d["forcing_names"])
attrs = pd.DataFrame(d["attrs"], index=d["basins"], columns=d["attr_names"])
picks = list(a.basins) + [int(attrs["aridity"].idxmin()), int(attrs["aridity"].idxmax()),
                          int(attrs["frac_snow"].idxmax())]
picks = list(dict.fromkeys(picks))


def frame(basin):
    i = int(np.where(d["basins"] == basin)[0][0])
    f = d["forcing"][i]
    return pd.DataFrame({
        "Date": pd.to_datetime(d["dates"]),
        "Precipitation": f[:, names.index("precipitation")].astype(float),
        "Temperature": f[:, names.index("temperature")].astype(float),
        "PotentialET": f[:, names.index("pet")].astype(float),
        "Q": d["target"][i].astype(float),
    })


OLD = {"soil": {"FC": {"min": 50.0, "max": 300.0}},
       "response": {"PERC": {"min": 0.5, "max": 3.0}, "K2": {"min": 0.01, "max": 0.1}}}


def model(df, start, warm, end):
    m = HBVModel()
    if a.ranges == "v1.4":
        m.set_parameters(OLD)
    m.load_data(data=df.copy(), date_column="Date", obs_q_column="Q", date_format="%Y-%m-%d",
                start_date=start, warmup_end=warm, end_date=end)
    return m


def at_bound(params, tol=0.005):
    """Parameters within tol of their range, as 'NAME-' (lower) or 'NAME+' (upper)."""
    out = []
    for g in params.values():
        for k, v in g.items():
            span = v["max"] - v["min"]
            if span > 0 and v["default"] - v["min"] <= tol * span:
                out.append(k + "-")
            elif span > 0 and v["max"] - v["default"] <= tol * span:
                out.append(k + "+")
    return out


rows = []
for b in picks:
    df = frame(b)
    for method, iters in [(x, a.dds_evals if x == "DDS" else a.nm_iter) for x in a.methods]:
        m = model(df, "1970-10-01", "1971-09-30", "1997-12-31")
        runs = [0]
        orig = m.run
        m.run = lambda *x, **k: (runs.__setitem__(0, runs[0] + 1), orig(*x, **k))[1]
        t0 = time.perf_counter()
        out = m.calibrate(method=method, objective="NSE", iterations=iters, seed=a.seed,
                          verbose=False, plot_results=False)
        secs = time.perf_counter() - t0
        t = model(df, "1997-01-01", "1997-12-31", "2008-12-31")
        t.params = copy.deepcopy(m.params)
        t.run(verbose=False)
        ab = at_bound(m.params)
        rows.append(dict(basin=b, ranges=a.ranges, aridity=round(float(attrs.loc[b, "aridity"]), 2),
                         frac_snow=round(float(attrs.loc[b, "frac_snow"]), 3), method=method,
                         nse_cal=round(out["performance"]["NSE"], 3), nse_test=round(t.performance_metrics["NSE"], 3),
                         kge_test=round(t.performance_metrics["KGE"], 3), n_at_bound=len(ab), at_bound=" ".join(ab),
                         FC=round(m.params["soil"]["FC"]["default"], 1), PERC=round(m.params["response"]["PERC"]["default"], 2),
                         K2=round(m.params["response"]["K2"]["default"], 4),
                         model_runs=runs[0], seconds=round(secs, 1)))
        print(rows[-1], flush=True)

print()
print(pd.DataFrame(rows).to_string(index=False))
