"""Feasibility check: how long does HBV_Lab take on one CAMELS-GB catchment?

Times a single model run over the benchmark training period and a short Nelder-Mead calibration, then projects the
cost of a DDS baseline. Usage: python scripts/feasibility.py [--basin 39001] [--npz path] [--iterations 50]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from HBV_Lab import HBVModel

p = argparse.ArgumentParser()
p.add_argument("--npz", default=os.path.expanduser("~/camels_uk_exp/data/processed/camels_gb.npz"))
p.add_argument("--basin", type=int, default=39001)
p.add_argument("--iterations", type=int, default=50)
a = p.parse_args()

d = np.load(a.npz)
names = list(d["forcing_names"])
i = int(np.where(d["basins"] == a.basin)[0][0])
f = d["forcing"][i]
df = pd.DataFrame({
    "Date": pd.to_datetime(d["dates"]),
    "Precipitation": f[:, names.index("precipitation")].astype(float),
    "Temperature": f[:, names.index("temperature")].astype(float),
    "PotentialET": f[:, names.index("pet")].astype(float),
    "Q": d["target"][i].astype(float),
})

# Benchmark training period, first year as warm-up (config/benchmark_study.yml)
m = HBVModel()
m.load_data(data=df, date_column="Date", obs_q_column="Q", date_format="%Y-%m-%d",
            start_date="1970-10-01", warmup_end="1971-09-30", end_date="1997-12-31")
n = len(m.data)
print(f"basin {a.basin}  days {n}  obs missing {m.data['Q'].isna().mean():.1%}")

t = []
for _ in range(3):
    t0 = time.perf_counter()
    m.run(verbose=False)
    t.append(time.perf_counter() - t0)
run_s = min(t)
print(f"single run: {run_s:.2f} s  ({run_s / n * 1e6:.1f} us per day)  NSE default params {m.performance_metrics.get('NSE', np.nan):.3f}"
      if hasattr(m, "performance_metrics") else f"single run: {run_s:.2f} s")

calls = [0]
orig_run = m.run
def counted_run(*args, **kw):
    calls[0] += 1
    return orig_run(*args, **kw)
m.run = counted_run

t0 = time.perf_counter()
res = m.calibrate(method="Nelder-Mead", objective="NSE", iterations=a.iterations, verbose=False, plot_results=False)
cal_s = time.perf_counter() - t0
print(f"Nelder-Mead {a.iterations} iterations: {cal_s:.1f} s, {calls[0]} model runs, "
      f"NSE {res['performance'].get('NSE', np.nan):.3f}")

for evals in (1000, 2000, 5000):
    per = evals * run_s
    print(f"DDS {evals} evals: {per / 60:.1f} min per catchment, {per * 100 / 3600:.1f} core-h for 100 catchments, "
          f"{per * 669 / 3600:.1f} core-h for 669")
