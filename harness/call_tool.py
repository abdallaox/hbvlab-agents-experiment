"""Command-line client for the experiment server over HTTP: one tool call per invocation.

This is the interactive route, used when an agent works through a shell rather than through harness/run_agent.py
(for example a Claude Code subagent). The server keeps the models in memory between calls. This script enforces the
same tool list and tool call budget as run_agent.py and logs every call to <run-dir>/transcript.jsonl.

  python harness/call_tool.py --list                                  # tool names, descriptions and arguments
  python harness/call_tool.py create_model '{"name": "dee"}'
  python harness/call_tool.py calibrate '{"model_id": "model-1", "method": "DDS", "iterations": 2000}'

Server: python harness/experiment_server.py --run-dir <run-dir> --http-port 8765. The run folder and port are read
from HBV_RUN_DIR and HBV_PORT, or from --run-dir and --port.
"""
import argparse
import asyncio
import json
import os
import time

import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", nargs="?")
    ap.add_argument("args", nargs="?", default="{}")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run-dir", default=os.environ.get("HBV_RUN_DIR"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("HBV_PORT", "8765")))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "agents.yml"), encoding="utf-8"))
    allowed, budget = cfg["tools"], cfg["budget"]
    state_file = os.path.join(a.run_dir, "client_state.json")
    st = json.load(open(state_file)) if os.path.exists(state_file) else {"tool_calls": 0, "by_name": {}, "submitted": False}

    async with streamablehttp_client(f"http://127.0.0.1:{a.port}/mcp", timeout=60,
                                     sse_read_timeout=1800) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if a.list:
                for t in (await s.list_tools()).tools:
                    if t.name in allowed:
                        props = {k: v.get("type", v.get("anyOf", "")) for k, v in
                                 (t.inputSchema.get("properties") or {}).items()}
                        print(f"## {t.name}  args: {json.dumps(props)}\n{(t.description or '').strip()}\n")
                return
            try:
                args = json.loads(a.args)
            except json.JSONDecodeError:
                print("Error: the arguments must be one JSON object, e.g. '{\"model_id\": \"model-1\"}'.")
                return
            submit = a.tool == "submit_calibration"
            if a.tool not in allowed:
                text = f"Error: unknown tool '{a.tool}'. Run with --list to see the tools."
            elif st["submitted"]:
                text = "The calibration was already submitted. The task is complete."
            elif not submit and st["tool_calls"] >= budget["max_tool_calls"]:
                text = "Tool call budget exhausted. Only submit_calibration is possible now."
            else:
                if not submit:
                    st["tool_calls"] += 1
                    st["by_name"][a.tool] = st["by_name"].get(a.tool, 0) + 1
                t0 = time.time()
                res = await s.call_tool(a.tool, args)
                text = "\n".join(getattr(x, "text", "") for x in res.content)
                if submit and not res.isError:
                    try:
                        st["submitted"] = bool(json.loads(text).get("accepted"))
                    except (json.JSONDecodeError, AttributeError):
                        pass
                with open(os.path.join(a.run_dir, "transcript.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(t0, 3), "kind": "tool", "name": a.tool, "args": args,
                                        "error": bool(res.isError), "seconds": round(time.time() - t0, 2),
                                        "result": text}) + "\n")
            b = json.loads((await s.call_tool("budget_status", {})).content[0].text)
            events = os.path.join(a.run_dir, "server_events.jsonl")
            clamps = [json.loads(x) for x in open(events, encoding="utf-8") if '"clamp"' in x] \
                if os.path.exists(events) else []
            for e in clamps[st.get("clamps_seen", 0):]:
                text += (f"\n[Note: {e['tool']} was limited from {e['requested']} to {e['allowed']} "
                         f"to fit the model run budget.]")
            st["clamps_seen"] = len(clamps)
            json.dump(st, open(state_file, "w"))
            print(text)
            print(f"[Budget left: {budget['max_tool_calls'] - st['tool_calls']} tool calls, "
                  f"{b['model_runs_max'] - b['model_runs_used']} model runs.]")


if __name__ == "__main__":
    asyncio.run(main())
