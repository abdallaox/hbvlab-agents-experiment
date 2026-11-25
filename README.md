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

## Not included

CAMELS-GB (Coxon et al., 2020, doi:10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9) is not redistributed here,
including the copy of the training data in each run folder (`forcing.csv`). `scripts/prepare_agent_data.py`
rebuilds it. The FUSE simulations of Lane et al. (2019) are available from doi:10.5523/bris.3ma509dlakcf720aw8x82aq4tm.

## Licence

Code: MIT. Run records and results: CC BY 4.0.
