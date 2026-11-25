# HBV-Lab agents experiment: code, run records and results

Experiment archive for the paper "HBV-Lab: an open HBV model that AI agents can set up, calibrate and diagnose
through the Model Context Protocol" (submitted to Environmental Modelling & Software).

The study calibrates the HBV model of [HBV-Lab](https://github.com/abdallaox/HBV_python_implementation)
(version 1.5.1, the code used for all runs) on CAMELS-GB in two ways:

- **pipeline**: DDS, 10,000 model runs, three seeds, all 669 catchments (`runs/baseline/pipeline`, 1989 submissions)
- **agent**: Claude Opus 5.5 through the HBV-Lab MCP server, with only the MCP tools, the training data (1970-1997)
  and a fixed prompt, on 95 representative catchments plus three repeats on 18 (`runs/agents/claude-api`,
  131 runs, all audited)

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
