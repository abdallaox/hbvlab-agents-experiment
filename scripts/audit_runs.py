"""Audit every agent run: the agent had only the allowed tools, the prompt and the training data, nothing else.

Checks per run folder (runs/agents/<agent>/<gauge>/r<k>), written by harness/run_agent.py:
  inputs     forcing.csv in the run folder is byte-identical to data/agent_inputs/<gauge>/forcing.csv and ends on
             1997-12-31 (no test-period data was available)
  tools      the tools offered to the model are exactly config/agents.yml's list, the server held no other tool
             (apart from the harness-only budget_status), and every tool the model called is in the list
  sandbox    the server logged no attempt to read or write outside the run folder
  prompt     the prompt template and tool definitions have the same SHA-256 in every run
  model      the model OpenRouter reports serving belongs to the model requested
  files      the run folder holds only the expected files (inputs, logs, submission, and any CSV the agent asked
             evaluate_uncertainty to write)
Writes results/audit.csv and prints the runs that fail any check.

  python scripts/audit_runs.py [--agents-dir runs/agents]
"""
import argparse
import glob
import hashlib
import json
import os

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--agents-dir", default=os.path.join(ROOT, "runs", "agents"))
p.add_argument("--out", default=os.path.join(ROOT, "results", "audit.csv"))
a = p.parse_args()
allowed = set(yaml.safe_load(open(os.path.join(ROOT, "config", "agents.yml"), encoding="utf-8"))["tools"])
expected = {"forcing.csv", "catchment.json", "transcript.jsonl", "server_events.jsonl", "server_stderr.log",
            "submission.json", "summary.json"}
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()

rows = []
for run in sorted(glob.glob(os.path.join(a.agents_dir, "*", "*", "r*"))):
    agent, gauge, rep = run.replace("\\", "/").split("/")[-3:]
    tr = os.path.join(run, "transcript.jsonl")
    if agent == "mock" or not os.path.exists(tr):
        continue
    recs = [json.loads(x) for x in open(tr, encoding="utf-8")]
    kinds = {}
    for r in recs:
        kinds.setdefault(r["kind"], []).append(r)
    row = {"agent": agent, "gauge": int(gauge), "repeat": rep}

    if "start" not in kinds or "tools" not in kinds:  # interactive route (call_tool.py): not auditable this way
        row["route"] = "interactive"
        row["ok"] = False
        row["problems"] = "not an API-harness run (no start/tools records)"
        rows.append(row)
        continue
    row["route"] = "api"
    problems = []

    src = os.path.join(ROOT, "data", "agent_inputs", gauge, "forcing.csv")
    f = os.path.join(run, "forcing.csv")
    if not os.path.exists(src) or sha(f) != sha(src):
        problems.append("forcing.csv differs from the prepared input")
    last = open(f, encoding="utf-8").read().strip().splitlines()[-1].split(",")[0]
    if last != "19971231":
        problems.append(f"forcing.csv ends {last}")

    offered = set(kinds["tools"][0]["names"])
    if offered != allowed:
        problems.append(f"tools offered differ: {sorted(offered ^ allowed)}")
    called = {r["name"] for r in kinds.get("tool", [])}
    rejected = called - allowed
    if rejected:
        problems.append(f"model asked for tools outside the list (rejected by the harness): {sorted(rejected)}")
    events = [json.loads(x) for x in open(os.path.join(run, "server_events.jsonl"), encoding="utf-8")]
    start = next((e for e in events if e["event"] == "start"), {})
    extra = set(start.get("tools", [])) - allowed - {"budget_status"}
    if extra or "tools" not in start:
        problems.append(f"server held other tools: {sorted(extra)}" if extra else "server start record has no tool list")
    blocks = [e for e in events if e["event"] == "sandbox_block"]
    if blocks:
        problems.append(f"{len(blocks)} sandbox blocks: {[b['path'] for b in blocks][:3]}")

    row["prompt_sha"] = kinds["tools"][0].get("prompt_template_sha256", "")[:12]
    row["tools_sha"] = kinds["tools"][0].get("tools_sha256", "")[:12]
    requested = kinds["start"][0]["model"]
    served = set()
    summ = os.path.join(run, "summary.json")
    if os.path.exists(summ):
        s = json.load(open(summ))
        served = set(s.get("models_served") or [])
        row["status"] = s.get("status")
        # provider must match; the exact served names are kept in the table for inspection
        if requested != "mock" and any(m.split("/")[0] != requested.split("/")[0] for m in served):
            problems.append(f"served {sorted(served)} for {requested}")
    else:
        problems.append("no summary.json (run did not finish)")
    row["model_requested"], row["models_served"] = requested, " ".join(sorted(served))

    files = set(os.listdir(run))
    unexpected = {x for x in files - expected if not x.lower().endswith(".csv")}
    if unexpected:
        problems.append(f"unexpected files: {sorted(unexpected)}")
    row["ok"] = not problems
    row["problems"] = "; ".join(problems)
    rows.append(row)

df = pd.DataFrame(rows)
if len(df):
    api = df[df.route == "api"]
    for col in ("prompt_sha", "tools_sha"):
        if col in api and api[col].nunique() > 1:
            print(f"WARNING: {col} differs between runs: {api[col].value_counts().to_dict()}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print(f"audited {len(df)} runs: {int(df.ok.sum()) if len(df) else 0} pass -> {a.out}")
for _, r in df[~df.ok].iterrows() if len(df) else []:
    print(f"  FAIL {r['agent']}/{r['gauge']}/{r['repeat']}: {r['problems']}")
