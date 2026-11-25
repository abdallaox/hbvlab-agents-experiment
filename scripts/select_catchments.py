"""Select a stratified, nested subset of CAMELS-GB catchments for the agent experiment.

Strata: aridity quartile x baseflow index quartile (16 cells, quartile edges over all 669 benchmark catchments).
Eligible: flow records at least 80% complete in the training period and 90% in the test period
(config/flow_completeness.csv). Catchments are drawn in rounds of one per cell, in random order within each cell, so
the first 16 x k rows of the output are a balanced subset for any k. Any first 16 x k rows can be used as the subset.

Usage: python scripts/select_catchments.py [--rounds 8] [--seed 0] [--camels ../camels_uk_exp/data/camels_gb/data]
Writes config/subset_ordered.csv.
"""
import argparse

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--camels", default="../camels_uk_exp/data/camels_gb/data")
p.add_argument("--rounds", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--min-train", type=float, default=0.8)
p.add_argument("--min-test", type=float, default=0.9)
p.add_argument("--out", default="config/subset_ordered.csv")
a = p.parse_args()

basins = pd.read_csv("config/basins_all.txt", header=None)[0].astype(int)
spatial = set(pd.read_csv("config/basins_spatial_test.txt", header=None)[0].astype(int))
read = lambda f: pd.read_csv(f"{a.camels}/CAMELS_GB_{f}_attributes.csv", index_col=0)
df = pd.concat([
    read("climatic")[["aridity", "frac_snow", "p_mean"]],
    read("hydrologic")[["baseflow_index", "runoff_ratio"]],
    read("topographic")[["area", "gauge_lat", "gauge_lon"]],
    read("humaninfluence")[["benchmark_catch"]],
    pd.read_csv("config/flow_completeness.csv", index_col=0)[["train", "test"]].add_prefix("complete_"),
], axis=1).loc[basins]
df.index.name = "gauge_id"
df["benchmark_catch"] = df["benchmark_catch"].astype(str).str.upper().str.startswith("Y")  # near-natural (Y/N)

df["aridity_q"] = pd.qcut(df["aridity"], 4, labels=False)
df["bfi_q"] = pd.qcut(df["baseflow_index"], 4, labels=False)
df["benchmark_split"] = np.where(df.index.isin(spatial), "spatial_test", "train")
ok = (df["complete_train"] >= a.min_train) & (df["complete_test"] >= a.min_test)
pool = df[ok]

rng = np.random.default_rng(a.seed)
queues = {cell: list(rng.permutation(g.index.values)) for cell, g in pool.groupby(["aridity_q", "bfi_q"])}
rows = []
for r in range(a.rounds):
    cells = list(queues)
    rng.shuffle(cells)
    for cell in cells:
        if queues[cell]:
            rows.append((queues[cell].pop(0), r + 1))
short = [c for c, q in queues.items() if not q]

out = df.loc[[g for g, _ in rows]].copy()
out.insert(0, "round", [r for _, r in rows])
out.insert(0, "order", range(1, len(out) + 1))
out.round(4).to_csv(a.out)

print(f"eligible {ok.sum()} of {len(df)}; wrote {len(out)} catchments in {a.rounds} rounds to {a.out}")
if short:
    print("cells used up before the last round:", short)
print(pd.crosstab(out["aridity_q"], out["bfi_q"]))
for n in (32, 64):
    s, s_all = out.head(n), df
    print(f"first {n}: aridity median {s.aridity.median():.2f} (GB {s_all.aridity.median():.2f}), "
          f"BFI median {s.baseflow_index.median():.2f} (GB {s_all.baseflow_index.median():.2f}), "
          f"area median {s.area.median():.0f} km2 (GB {s_all.area.median():.0f}), "
          f"near-natural {s.benchmark_catch.mean():.0%} (GB {s_all.benchmark_catch.mean():.0%}), "
          f"spatial-test {(s.benchmark_split == 'spatial_test').mean():.0%}")
