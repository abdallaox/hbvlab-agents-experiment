"""Build the public experiment archive (code, configuration, run records, results) in a separate folder, so the
private working repository (drafts, notes) is never published.

  python scripts/build_archive.py [--out ../hbvlab-agents-experiment]

Excluded: CAMELS-GB data and the copies of it in each run folder (forcing.csv), third-party benchmark data, the paper
draft, notes, caches, pilot runs that are not part of the paper. Cluster usernames are replaced by Slurm's %u.
"""
import argparse
import re
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(ROOT, "..", "hbvlab-agents-experiment"))
a = p.parse_args()
OUT = os.path.abspath(a.out)
if os.path.exists(OUT):
    for name in os.listdir(OUT):
        if name != ".git":
            path = os.path.join(OUT, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
os.makedirs(OUT, exist_ok=True)
ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "forcing.csv", ".env", "*.log.tmp")

for d in ("harness", "scripts", "config", "cluster"):
    shutil.copytree(os.path.join(ROOT, d), os.path.join(OUT, d), ignore=ignore)
for d in (os.path.join("runs", "agents", "claude-api"), os.path.join("runs", "baseline", "pipeline")):
    shutil.copytree(os.path.join(ROOT, d), os.path.join(OUT, d), ignore=ignore)
os.makedirs(os.path.join(OUT, "results"))
for f in os.listdir(os.path.join(ROOT, "results")):
    if f.endswith(".csv"):
        shutil.copy(os.path.join(ROOT, "results", f), os.path.join(OUT, "results", f))

# cluster job scripts: no personal username
for f in os.listdir(os.path.join(OUT, "cluster")):
    path = os.path.join(OUT, "cluster", f)
    s = re.sub(r"/home/[^/%]+/", "/home/%u/", open(path, encoding="utf-8").read())
    open(path, "w", encoding="utf-8", newline="\n").write(s)

req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
req = req.replace("-e ../HBV_Lab[mcp]",
                  "HBV-Lab[mcp] @ git+https://github.com/abdallaox/HBV_python_implementation@v1.5.1")
open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8", newline="\n").write(req)

n_agent = sum(1 for r, _, fs in os.walk(os.path.join(OUT, "runs", "agents")) if "submission.json" in fs)
n_pipe = sum(1 for r, _, fs in os.walk(os.path.join(OUT, "runs", "baseline")) if "submission.json" in fs)

readme = f"""# HBV-Lab agents experiment: code, run records and results

Experiment archive for the paper "HBV-Lab: an open HBV model that AI agents can set up, calibrate and diagnose
through the Model Context Protocol" (submitted to Environmental Modelling & Software).

The study calibrates the HBV model of [HBV-Lab](https://github.com/abdallaox/HBV_python_implementation)
(version 1.5.1, the code used for all runs) on CAMELS-GB in two ways:

- **pipeline**: DDS, 10,000 model runs, three seeds, all 669 catchments (`runs/baseline/pipeline`, {n_pipe} submissions)
- **agent**: Claude Opus 5.5 through the HBV-Lab MCP server, with only the MCP tools, the training data (1970-1997)
  and a fixed prompt, on 95 representative catchments plus three repeats on 18 (`runs/agents/claude-api`,
  {n_agent} runs, all audited)

## Contents

| Folder | What it holds |
|---|---|
| `harness/` | agent harness (`run_agent_claude.py`), experiment MCP server with tool list, file sandbox and run budget (`experiment_server.py`), the prompt (`prompt.md`), server tests |
| `scripts/` | data preparation, pipeline, scoring (`evaluate_runs.py`, metrics in `metrics_nh.py`), audit (`audit_runs.py`), comparison with benchmarks, analyses, figures |
| `config/` | experiment settings (`agents.yml`), catchment lists, stratified sample (`subset_ordered.csv`) |
| `cluster/` | Slurm job scripts used on UCL Myriad |
| `runs/agents/claude-api/<gauge>/r<k>/` | per run: catchment description given to the agent, full transcript (every tool call, reply and the model's summarised reasoning), server event log, submission (parameters, ranges, initial states, diagnosis) and summary (tokens, cost, time) |
| `runs/baseline/pipeline/<gauge>/s<seed>/` | pipeline submissions |
| `results/` | scored runs, audit, benchmark comparison, agent analyses, diagnosis check |

## How to check a run

Each agent run folder `runs/agents/claude-api/<gauge>/r<k>/` holds the full record of that run:

- `catchment.json`: exactly what the agent was told about the catchment.
- `transcript.jsonl`: one JSON record per line. The `start` and `tools` records hold the settings, the prompt and
  the tool definitions with their SHA-256 hashes. Each `response` record is one model turn, including the model's
  summarised reasoning (`thinking` blocks) and its tool calls. Each `tool` record is one tool call with its
  arguments and the full reply the agent received. The `end` record is the run summary.
- `server_events.jsonl`: the experiment server's own log (tools offered and removed, budget limits applied,
  submission).
- `submission.json`: the submitted parameters, ranges, initial states, the agent's diagnosis and the metrics it
  reported. `summary.json`: tokens, cost, wall time, tool calls and errors.

To reproduce the checks and scores (CAMELS-GB needed for the scores):

```
python scripts/audit_runs.py        # results/audit.csv: inputs, tools, sandbox, prompt and tool hashes per run
python scripts/evaluate_runs.py     # results/runs_scored.csv: calibration and test scores of every submission
python scripts/analyse_agent.py     # results/agent_behaviour.csv: each run's decisions (ranges, states, calibrations)
python scripts/check_diagnoses.py   # results/diagnosis_check.csv: diagnoses against CAMELS-GB records
```

The figures of the paper are made by `scripts/figures.py`, `scripts/schematic.py` and `scripts/hbv_schematic.py`.

## Not included

CAMELS-GB (Coxon et al., 2020, doi:10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9) is not redistributed here,
including the copy of the training data in each run folder (`forcing.csv`). `scripts/prepare_agent_data.py`
rebuilds it. The FUSE simulations of Lane et al. (2019) are available from doi:10.5523/bris.3ma509dlakcf720aw8x82aq4tm.

## Licence

Code: MIT. Run records and results: CC BY 4.0.
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n").write(readme)
shutil.copy(os.path.join(ROOT, "..", "HBV_Lab", "LICENSE"), os.path.join(OUT, "LICENSE")) \
    if os.path.exists(os.path.join(ROOT, "..", "HBV_Lab", "LICENSE")) else None
zen = {"title": "HBV-Lab agents experiment: code, run records and results",
       "upload_type": "dataset",
       "creators": [{"name": "Mohammed, Abdalla", "affiliation": "University College London"}],
       "description": "Code, agent transcripts, submissions and results of the HBV-Lab agents experiment on CAMELS-GB: "
                      "automatic DDS calibration of HBV-Lab on 669 catchments and an AI agent (Claude Opus 5.5) "
                      "calibrating HBV through the HBV-Lab MCP server on 95 representative catchments, every run audited.",
       "license": "cc-by-4.0",
       "keywords": ["hydrology", "HBV", "calibration", "AI agents", "Model Context Protocol", "CAMELS-GB"],
       "related_identifiers": [{"identifier": "10.5281/zenodo.24514257", "relation": "isSupplementTo",
                                "scheme": "doi"}]}
json.dump(zen, open(os.path.join(OUT, ".zenodo.json"), "w", encoding="utf-8"), indent=2)
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) if ".git" not in r for f in fs)
print(f"archive at {OUT}: {n_agent} agent runs, {n_pipe} pipeline submissions, {size / 1e6:.1f} MB")
