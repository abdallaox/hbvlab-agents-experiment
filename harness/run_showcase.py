"""Run the Claude agent on the showcase catchments, a few runs at a time, with a hard stop on total spend.

  python harness/run_showcase.py --repeats 3 --parallel 3 --max-cost 60
  python harness/run_showcase.py --gauges 12001 --repeats 1          # one test run first

Catchments from config/showcase.txt. Each run is harness/run_agent_claude.py; finished runs (with summary.json) are
skipped, so the script can be restarted. Before starting each run it adds up cost_usd over all finished runs and
stops starting new runs once that total passes --max-cost (runs already going finish).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--gauges", type=int, nargs="*")
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--parallel", type=int, default=3)
p.add_argument("--max-cost", type=float, default=60.0, help="USD over all runs under runs/agents/<label>")
p.add_argument("--label", default="claude-api")
a = p.parse_args()

gauges = a.gauges or [int(x) for x in open(os.path.join(ROOT, "config", "showcase.txt"))
                      if x.strip() and not x.startswith("#")]
jobs = [(g, r) for r in range(1, a.repeats + 1) for g in gauges]  # repeat 1 of every catchment first


def spent():
    return sum(json.load(open(f)).get("cost_usd", 0) for f in
               glob.glob(os.path.join(ROOT, "runs", "agents", a.label, "*", "r*", "summary.json")))


def run(job):
    g, r = job
    if os.path.exists(os.path.join(ROOT, "runs", "agents", a.label, str(g), f"r{r}", "summary.json")):
        return g, r, "done already"
    total = spent()
    if total >= a.max_cost:
        return g, r, f"skipped: spent ${total:.2f} >= ${a.max_cost:.2f}"
    t = time.time()
    out = subprocess.run([sys.executable, os.path.join(ROOT, "harness", "run_agent_claude.py"), "--gauge", str(g),
                          "--repeat", str(r), "--label", a.label], capture_output=True, text=True)
    try:
        summ = json.loads(out.stdout.strip().splitlines()[-1])
        msg = (f"{summ['status']}, {summ['tool_calls']} calls, {summ.get('model_runs')} model runs, "
               f"${summ['cost_usd']:.3f}, {time.time() - t:.0f} s")
    except (IndexError, json.JSONDecodeError, KeyError):
        msg = "FAILED: " + (out.stderr.strip().splitlines() or ["no output"])[-1][:200]
    return g, r, msg


with ThreadPoolExecutor(a.parallel) as ex:
    for g, r, msg in ex.map(run, jobs):
        print(f"{g} r{r}: {msg}  (total spent ${spent():.2f})", flush=True)
