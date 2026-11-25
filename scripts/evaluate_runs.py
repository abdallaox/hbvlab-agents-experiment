"""Score every submission (agents and baseline) on the calibration and test periods with one code path.

Each submission's parameters are run with HBV_Lab over
  calibration: 1970-10-01 to 1997-12-31, scored from 1971-10-01 (what the agent saw)
  test:        1970-10-01 to 2008-12-31 as one continuous run, scored from 1998-01-01 (never shown to agents),
               the same way as the benchmark's LC-HBV (metrics_test_full.csv)
starting from the initial states stored in the submission (the states the model was calibrated with; HBV_Lab's
defaults for the pipeline). For agent runs the test score from the default states is also kept (NSE_test_default),
to show how much the choice matters. Test data come from the processed CAMELS-GB array, not from the agent
input files. Parameters at a bound are counted against the submitted ranges, separately for bounds that are physical
limits (CFR = 0, CWH = 0: snow refreezing and water holding capacity, which may well be zero) and for the rest.

  python scripts/evaluate_runs.py [--npz ../camels_uk_exp/data/processed/camels_gb.npz]
Writes results/runs_scored.csv: one row per submission, or per run without one (status kept, metrics empty).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

from HBV_Lab import HBVModel

import metrics_nh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSICAL = {("CFR", "lower"), ("CWH", "lower")}
DEFAULT_RANGES = {k: (float(v["min"]), float(v["max"])) for grp in HBVModel().params.values() for k, v in grp.items()}
# agents start from the experiment ranges (config/agents.yml); changes are counted against those
EXPERIMENT_RANGES = {k: tuple(float(x) for x in v) for k, v in yaml.safe_load(
    open(os.path.join(ROOT, "config", "agents.yml"), encoding="utf-8"))["ranges"].items()}
p = argparse.ArgumentParser()
p.add_argument("--npz", default=os.path.join(ROOT, "..", "camels_uk_exp", "data", "processed", "camels_gb.npz"))
p.add_argument("--out", default=os.path.join(ROOT, "results", "runs_scored.csv"))
a = p.parse_args()

with np.load(a.npz) as z:
    D = {k: z[k] for k in ("forcing", "target", "basins", "dates", "forcing_names")}
names = list(D["forcing_names"])
frames = {}


def frame(g):
    if g not in frames:
        i = int(np.where(D["basins"] == g)[0][0])
        f = D["forcing"][i]
        frames[g] = pd.DataFrame({"Date": pd.to_datetime(D["dates"]),
                                  "P": f[:, names.index("precipitation")].astype(float),
                                  "T": f[:, names.index("temperature")].astype(float),
                                  "PET": f[:, names.index("pet")].astype(float),
                                  "Q": D["target"][i].astype(float)})
    return frames[g]


def score(g, params, start, warm, end, states=None):
    m = HBVModel()
    m.load_data(data=frame(g).copy(), date_column="Date", precip_column="P", temp_column="T", pet_column="PET",
                obs_q_column="Q", date_format="%Y-%m-%d", start_date=start, warmup_end=warm, end_date=end)
    m.set_parameters({grp: {k: {"default": params[k]} for k in m.params[grp]} for grp in m.params})
    if states:
        m.set_initial_conditions(**{k: float(v) for k, v in states.items()})
    m.run(verbose=False)
    dates = pd.to_datetime(np.asarray(m.results["dates"]))
    w = dates > pd.Timestamp(warm)
    sim, obs = np.asarray(m.results["discharge"], float)[w], np.asarray(m.results["observed_q"], float)[w]
    out = {"KGEp": float("nan")}
    pm = getattr(m, "performance_metrics", None)
    if pm is not None:
        out["KGEp"] = float(pm["KGE"])  # HBV_Lab's KGE' (Kling et al. 2012)
    # benchmark-study definitions (NeuralHydrology): NSE, KGE (Gupta 2009), FHV, FLV, FMS, r, alpha, beta, PBIAS
    for k, f in {**metrics_nh.ALL, **metrics_nh.EXTRA}.items():
        out[k] = float(f(sim, obs)) if np.isfinite(obs).sum() >= 2 else float("nan")
    return out


def bounds(sub, tol=0.005):
    out = []
    for k, v in sub["parameters"].items():
        lo, hi = sub["ranges"][k]
        if hi > lo and v - lo <= tol * (hi - lo):
            out.append((k, "lower"))
        elif hi > lo and hi - v <= tol * (hi - lo):
            out.append((k, "upper"))
    return out


rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "runs", "agents", "*", "*", "r*")) +
                   glob.glob(os.path.join(ROOT, "runs", "baseline", "*", "*", "s*"))):
    parts = path.replace("\\", "/").split("/")
    kind = "agent" if "agents" in parts else "baseline"
    agent, gauge, rep = parts[-3], int(parts[-2]), parts[-1]
    if agent == "mock":
        continue
    row = {"kind": kind, "agent": agent, "gauge": gauge, "repeat": rep}
    summ = os.path.join(path, "summary.json")
    if os.path.exists(summ):
        s = json.load(open(summ))
        row.update({k: s.get(k) for k in ("status", "tool_calls", "model_runs", "api_turns", "prompt_tokens",
                                          "completion_tokens", "reasoning_tokens", "cost_usd", "wall_seconds",
                                          "tool_errors", "rejected_calls", "reminders")})
    sub_path = os.path.join(path, "submission.json")
    if os.path.exists(sub_path):
        sub = json.load(open(sub_path))
        if kind == "baseline":
            row.update(status="submitted", model_runs=sub.get("model_runs_used"), wall_seconds=sub.get("wall_seconds"))
        st = sub.get("initial_states")
        cal = score(gauge, sub["parameters"], "1970-10-01", "1971-09-30", "1997-12-31", st)
        # continuous run over the record, scored on the test period, as for the benchmark's LC-HBV
        test = score(gauge, sub["parameters"], "1970-10-01", "1997-12-31", "2008-12-31", st)
        if kind == "agent":
            row["NSE_test_default"] = score(gauge, sub["parameters"], "1970-10-01", "1997-12-31", "2008-12-31")["NSE"]
        row.update({f"{k}_cal": v for k, v in cal.items()})
        row.update({f"{k}_test": v for k, v in test.items()})
        b = bounds(sub)
        row["at_bound_physical"] = sum(x in PHYSICAL for x in b)
        row["at_bound_other"] = sum(x not in PHYSICAL for x in b)
        row["at_bound_list"] = " ".join(f"{k}{'-' if s == 'lower' else '+'}" for k, s in b)
        start = EXPERIMENT_RANGES if kind == "agent" else DEFAULT_RANGES
        row["ranges_changed"] = " ".join(k for k, r in sub["ranges"].items() if tuple(r) != start[k])
        row["nse_reported"] = (sub.get("metrics_reported") or {}).get("NSE")
        row["diagnosis_words"] = len((sub.get("diagnosis") or "").split())
        row.update({f"p_{k}": v for k, v in sub["parameters"].items()})
    rows.append(row)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(f"scored {len(df)} runs -> {a.out}")
if len(df) and "NSE_test" in df:
    print(df.groupby(["kind", "agent"])[["NSE_cal", "NSE_test", "at_bound_other"]].median().round(3))
