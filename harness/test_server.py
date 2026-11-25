"""Checks of the experiment server's guards: run budget, clamping and file sandbox. No LLM involved.

  python harness/test_server.py [--gauge 12001]
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", type=int, default=12001)
    a = ap.parse_args()
    run_dir = tempfile.mkdtemp(prefix="hbv_guard_")
    shutil.copy(os.path.join(ROOT, "data", "agent_inputs", str(a.gauge), "forcing.csv"), run_dir)
    outside = os.path.join(os.path.dirname(run_dir), "outside_test.csv")
    shutil.copy(os.path.join(run_dir, "forcing.csv"), outside)
    server = StdioServerParameters(command=sys.executable, args=[
        os.path.join(ROOT, "harness", "experiment_server.py"), "--run-dir", run_dir, "--max-runs", "120"])
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))

    async with stdio_client(server, errlog=sys.stderr) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            call = lambda n, **k: s.call_tool(n, k)
            text = lambda res: "\n".join(getattr(x, "text", "") for x in res.content)

            names = {t.name for t in (await s.list_tools()).tools}
            check("tools outside the experiment list are removed from the server",
                  not names & {"load_model", "save_model", "save_results", "plot_results"}, sorted(names))
            res = await call("load_model", model_path="forcing.csv")
            check("calling a removed tool fails", res.isError)

            await call("create_model")
            rng = json.loads(text(await call("get_parameter_ranges", model_id="model-1")))
            check("new models start with the experiment ranges",
                  rng["soil"]["FC"]["max"] == 1500.0 and rng["response"]["K2"]["min"] == 1e-6
                  and rng["soil"]["FC"]["default"] == 150.0, rng["soil"]["FC"])
            res = await call("load_data", model_id="model-1", file_path=outside, obs_q_column="Discharge")
            check("load_data outside the run folder is refused", res.isError, text(res)[:80])
            res = await call("load_data", model_id="model-1", file_path="../outside_test.csv", obs_q_column="Discharge")
            check("relative path escaping the run folder is refused", res.isError)
            res = await call("load_data", model_id="model-1", file_path="forcing.csv", obs_q_column="Discharge",
                             warmup_end="19710930")
            check("load_data inside the run folder works", not res.isError, text(res)[:60])

            res = await call("run_model", model_id="model-1")
            b = json.loads(text(await call("budget_status")))
            check("run_model counts one run", b["model_runs_used"] == 1, b)

            res = await call("calibrate", model_id="model-1", method="DDS", iterations=500, seed=0)
            b = json.loads(text(await call("budget_status")))
            check("DDS clamped to the runs left", not res.isError and b["model_runs_used"] == 120,
                  f"used {b['model_runs_used']} of 120")
            ev = open(os.path.join(run_dir, "server_events.jsonl")).read()
            check("clamp logged", '"clamp"' in ev and '"requested": 500' in ev)

            res = await call("run_model", model_id="model-1")
            check("run refused once the budget is used", res.isError and "budget" in text(res).lower(), text(res)[:80])

            res = await call("evaluate_uncertainty", model_id="model-1", n_runs=50, output_file="../band.csv")
            check("uncertainty refused with no runs left", res.isError)

            res = await call("submit_calibration", model_id="model-1", diagnosis="test")
            check("submit accepted", not res.isError and json.loads(text(res))["accepted"])
            res = await call("submit_calibration", model_id="model-1", diagnosis="again")
            check("second submit not accepted", not json.loads(text(res))["accepted"])
            sub = json.load(open(os.path.join(run_dir, "submission.json")))
            check("submission has 14 parameters", len(sub["parameters"]) == 14)

    os.remove(outside)
    shutil.rmtree(run_dir, ignore_errors=True)
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
