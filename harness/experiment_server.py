"""HBV_Lab MCP server for the agent experiment: the released server plus a run budget, a file sandbox and a submit tool.

HBV_Lab itself is not changed. This module imports the released MCP server and, inside this process only:

* counts every model run (HBVModel.run) and refuses runs beyond --max-runs. calibrate and evaluate_uncertainty are
  clamped to the runs left, and each clamp is logged so the harness can tell the agent.
* restricts every file read and write by pandas to the run folder, so the agent cannot reach any other data.
* adds submit_calibration (the agent's final answer, written to submission.json) and budget_status (for the
  harness only, never shown to the agent).
* removes every tool not listed in config/agents.yml (plus budget_status), so tools left out of the experiment, such
  as load_model or save_model, cannot be called at all, whatever the client sends.
* gives every new model the experiment's starting ranges (config/agents.yml, `ranges`) instead of HBV_Lab's narrower
  defaults. The agent can change them with set_parameter_ranges.

Usage (started by harness/run_agent.py over stdio): python harness/experiment_server.py --run-dir D --max-runs 10000
"""
import argparse
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
p.add_argument("--max-runs", type=int, default=10000)
p.add_argument("--http-port", type=int, help="serve over HTTP on 127.0.0.1:<port>/mcp instead of stdio "
                                              "(for harness/call_tool.py, the interactive subagent route)")
a = p.parse_args()
RUN_DIR = os.path.realpath(a.run_dir)
os.chdir(RUN_DIR)

# stdout carries the MCP messages over stdio. HBV_Lab prints progress text, which would corrupt them, so every
# print goes to stderr unless a file is given.
import builtins  # noqa: E402

_print = builtins.print
builtins.print = lambda *x, **k: _print(*x, **{**k, "file": k.get("file") or sys.stderr})

import pandas as pd  # noqa: E402

import HBV_Lab.mcp_server as srv  # noqa: E402
from HBV_Lab import HBVModel  # noqa: E402

STATE = {"runs": 0, "max": a.max_runs, "submitted": False}
EVENTS = os.path.join(RUN_DIR, "server_events.jsonl")


def log(kind, **kw):
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": round(time.time(), 3), "event": kind, **kw}) + "\n")


class BudgetExhausted(RuntimeError):
    pass


# --- run budget -----------------------------------------------------------------------------------------------------
_run = HBVModel.run


def run(self, *args, **kw):
    if STATE["runs"] >= STATE["max"]:
        raise BudgetExhausted(f"Model run budget exhausted ({STATE['max']} runs used). "
                              "No more simulations are possible; call submit_calibration with your best model.")
    STATE["runs"] += 1
    return _run(self, *args, **kw)


HBVModel.run = run

_calibrate = HBVModel.calibrate


def calibrate(self, method="Nelder-Mead", objective="NSE", iterations=100, *args, **kw):
    left = STATE["max"] - STATE["runs"]
    if left < 3:
        raise BudgetExhausted(f"Only {left} model runs left, too few to calibrate. Call submit_calibration.")
    # DDS: one run per iteration plus the final run. Nelder-Mead: about 2.4 runs per iteration (measured), use 3.
    cap = left - 1 if str(method).upper() == "DDS" else max(1, (left - 1) // 3)
    if iterations > cap:
        log("clamp", tool="calibrate", method=method, requested=iterations, allowed=cap)
        iterations = cap
    return _calibrate(self, method, objective, iterations, *args, **kw)


HBVModel.calibrate = calibrate

_uncertainty = HBVModel.evaluate_uncertainty


def evaluate_uncertainty(self, n_runs=10000, objective="NSE", save_best=10, *args, **kw):
    left = STATE["max"] - STATE["runs"]
    cap = left - save_best - 3  # the samples, plus one run per kept set and a few reference runs
    if cap < 10:
        raise BudgetExhausted(f"Only {left} model runs left, too few for an uncertainty analysis.")
    if n_runs > cap:
        log("clamp", tool="evaluate_uncertainty", requested=n_runs, allowed=cap)
        n_runs = cap
    return _uncertainty(self, n_runs, objective, save_best, *args, **kw)


HBVModel.evaluate_uncertainty = evaluate_uncertainty


# --- file sandbox ---------------------------------------------------------------------------------------------------
def _inside(path):
    if not isinstance(path, (str, os.PathLike)):
        return path
    full = os.path.realpath(os.path.join(RUN_DIR, path))
    if os.path.commonpath([full, RUN_DIR]) != RUN_DIR:
        log("sandbox_block", path=str(path))
        raise PermissionError(f"Access outside the run folder is not allowed: {path}")
    return full


for name in ("read_csv", "read_excel"):
    _f = getattr(pd, name)
    setattr(pd, name, (lambda f: lambda path, *x, **k: f(_inside(path), *x, **k))(_f))
_to_csv = pd.DataFrame.to_csv
pd.DataFrame.to_csv = lambda self, path=None, *x, **k: _to_csv(self, _inside(path) if path else path, *x, **k)


# --- extra tools ----------------------------------------------------------------------------------------------------
@srv.mcp.tool()
def submit_calibration(model_id: str, diagnosis: str) -> dict:
    """Submit your final calibrated model. This ends the task.

    ``model_id`` is the model whose current parameters are your answer. ``diagnosis`` is a short text (at most about
    200 words): how well the model performs, which processes or periods it simulates poorly, and how confident you
    are in the parameters. Only the first submission counts.
    """
    if STATE["submitted"]:
        return {"accepted": False, "reason": "A calibration was already submitted."}
    model = srv._get(model_id)
    sub = {
        "model_id": model_id,
        "diagnosis": diagnosis,
        "parameters": {k: float(v["default"]) for g in model.params.values() for k, v in g.items()},
        "ranges": {k: [float(v["min"]), float(v["max"])] for g in model.params.values() for k, v in g.items()},
        "initial_states": {k: float(v) for k, v in model.states.items()},
        "metrics_reported": srv._metrics(model),
        "data_window": {"start": str(getattr(model, "start_date", None)), "end": str(getattr(model, "end_date", None)),
                        "warmup_end": str(getattr(model, "warmup_end", None)),
                        "obs_column": (model.column_names or {}).get("obs_q") if hasattr(model, "column_names") else None},
        "model_runs_used": STATE["runs"],
    }
    with open(os.path.join(RUN_DIR, "submission.json"), "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=1)
    STATE["submitted"] = True
    log("submit", model_id=model_id)
    return {"accepted": True, "message": "Calibration submitted. The task is complete."}


@srv.mcp.tool()
def budget_status() -> dict:
    """Harness use only."""
    return {"model_runs_used": STATE["runs"], "model_runs_max": STATE["max"], "submitted": STATE["submitted"]}


import yaml  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "agents.yml"),
                          encoding="utf-8"))
ALLOWED = set(CFG["tools"]) | {"budget_status"}

# --- starting ranges ------------------------------------------------------------------------------------------------
# every model created in this server (create_model) starts with the experiment's ranges from config/agents.yml
_init = HBVModel.__init__


def __init__(self, *args, **kw):
    _init(self, *args, **kw)
    for grp in self.params.values():
        for k, v in grp.items():
            if k in CFG.get("ranges", {}):
                v["min"], v["max"] = (float(x) for x in CFG["ranges"][k])


HBVModel.__init__ = __init__
REMOVED = sorted(t for t in list(srv.mcp._tool_manager._tools) if t not in ALLOWED)
for t in REMOVED:
    srv.mcp._tool_manager.remove_tool(t)

if __name__ == "__main__":
    log("start", max_runs=a.max_runs, python=sys.version.split()[0], hbv_lab=srv.__version__,
        transport="http" if a.http_port else "stdio",
        tools=sorted(srv.mcp._tool_manager._tools), tools_removed=REMOVED)
    if a.http_port:
        srv.mcp.settings.host, srv.mcp.settings.port = "127.0.0.1", a.http_port
        srv.mcp.run(transport="streamable-http")
    else:
        srv.mcp.run()
