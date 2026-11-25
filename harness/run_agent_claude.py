"""Run Claude (Anthropic API, official SDK) on one catchment: it calibrates HBV_Lab through the MCP tools until it
submits or runs out of budget. Same prompt, tools, server, budget and file layout as harness/run_agent.py (the
OpenRouter route), so scripts/evaluate_runs.py and scripts/audit_runs.py treat both alike.

  python harness/run_agent_claude.py --gauge 12001 --repeat 1

Settings in config/agents.yml (`claude_api`, `budget`, `tools`). The key is read from ANTHROPIC_API_KEY or from .env
in the repository root (not committed). Details of the request:
  * adaptive thinking (always on for Opus 5.5) with summarised display, so the transcript keeps the agent's reasoning
  * prompt caching: an explicit breakpoint on the system prompt plus automatic caching of the growing conversation
  * the conversation is append-only: assistant turns go back unchanged (thinking blocks included), tool results and
    budget notes are appended, nothing is edited or removed
  * no refusal fallback to another model: the study uses one model, so a refusal is recorded as the run's outcome
"""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time

import anthropic
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_agent import ROOT, build_prompt  # noqa: E402  (shared prompt construction)

MAX_TOOL_RESULT_CHARS = 12000


def load_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No ANTHROPIC_API_KEY in the environment or in .env")


def cost_usd(u, price):
    return ((u.get("input_tokens") or 0) * price["input"]
            + (u.get("cache_creation_input_tokens") or 0) * price["cache_write_5m"]
            + (u.get("cache_read_input_tokens") or 0) * price["cache_read"]
            + (u.get("output_tokens") or 0) * price["output"]) / 1e6


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--label", default="claude-api", help="folder name under runs/agents")
    ap.add_argument("--inputs", default=os.path.join(ROOT, "data", "agent_inputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "agents"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "agents.yml"), encoding="utf-8"))
    budget, api = cfg["budget"], cfg["claude_api"]
    run_dir = os.path.join(a.out, a.label, str(a.gauge), f"r{a.repeat}")
    if os.path.exists(os.path.join(run_dir, "summary.json")) and not a.force:
        print(f"done already: {run_dir}")
        return
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    src = os.path.join(a.inputs, str(a.gauge))
    for f in ("forcing.csv", "catchment.json"):
        shutil.copy(os.path.join(src, f), run_dir)
    meta = json.load(open(os.path.join(src, "catchment.json"), encoding="utf-8"))
    system, task = build_prompt(meta, budget)

    transcript = open(os.path.join(run_dir, "transcript.jsonl"), "w", encoding="utf-8")

    def record(kind, **kw):
        transcript.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **kw}, default=str) + "\n")
        transcript.flush()

    record("start", agent=a.label, model=api["model"], gauge=a.gauge, repeat=a.repeat, budget=budget,
           api_settings=api, system=system, task=task, sdk=anthropic.__version__)
    server = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(ROOT, "harness", "experiment_server.py"), "--run-dir", run_dir,
              "--max-runs", str(budget["max_model_runs"])],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})

    s = {"agent": a.label, "model_requested": api["model"], "gauge": a.gauge, "repeat": a.repeat,
         "status": "running", "tool_calls": 0, "tool_calls_by_name": {}, "tool_errors": 0, "rejected_calls": 0,
         "api_turns": 0, "reminders": 0, "input_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
         "output_tokens": 0, "cost_usd": 0.0, "models_served": [], "stop_reasons": {}, "submitted": False}
    # summary fields shared with the OpenRouter route, filled at the end
    t0 = time.time()
    clamp_seen = 0
    client = anthropic.AsyncAnthropic(api_key=load_key(), max_retries=4, timeout=600)

    async with stdio_client(server, errlog=open(os.path.join(run_dir, "server_stderr.log"), "w")) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                     for t in listed if t.name in cfg["tools"]]
            template = open(os.path.join(ROOT, "harness", "prompt.md"), "rb").read()
            s["prompt_template_sha256"] = hashlib.sha256(template).hexdigest()
            s["tools_sha256"] = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
            record("tools", names=[t["name"] for t in tools], definitions=tools,
                   prompt_template_sha256=s["prompt_template_sha256"], tools_sha256=s["tools_sha256"])

            system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            messages = [{"role": "user", "content": task}]

            while True:
                if time.time() - t0 > 60 * budget["max_wall_minutes"]:
                    s["status"] = "wall_time"
                    break
                if s["api_turns"] >= budget["max_api_turns"]:
                    s["status"] = "max_api_turns"
                    break
                t1 = time.time()
                try:
                    resp = await client.messages.create(
                        model=api["model"], max_tokens=api["max_tokens"], system=system_blocks, tools=tools,
                        messages=messages, cache_control={"type": "ephemeral"},
                        thinking={"type": "adaptive", "display": api["thinking_display"]},
                        output_config={"effort": api["effort"]})
                except anthropic.BadRequestError as e:  # not retryable: record and stop
                    record("api_error", error=str(e), status=400)
                    s["status"] = "api_error"
                    break
                except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    record("api_error", error=str(e))  # the SDK already retried 4 times
                    s["status"] = "api_error"
                    break

                s["api_turns"] += 1
                u = resp.usage.to_dict()
                c = cost_usd(u, api["price_per_mtok"])
                s["input_tokens"] += u.get("input_tokens") or 0
                s["cache_write_tokens"] += u.get("cache_creation_input_tokens") or 0
                s["cache_read_tokens"] += u.get("cache_read_input_tokens") or 0
                s["output_tokens"] += u.get("output_tokens") or 0
                s["cost_usd"] += c
                s["stop_reasons"][resp.stop_reason] = s["stop_reasons"].get(resp.stop_reason, 0) + 1
                if resp.model not in s["models_served"]:
                    s["models_served"].append(resp.model)
                record("response", seconds=round(time.time() - t1, 2), stop_reason=resp.stop_reason,
                       usage=u, cost_usd=round(c, 5), request_id=resp._request_id,
                       message=resp.to_dict()["content"])
                messages.append({"role": "assistant", "content": resp.content})  # unchanged, thinking included

                if resp.stop_reason == "refusal":
                    s["status"] = "refusal"
                    record("refusal", details=resp.stop_details.to_dict() if resp.stop_details else None)
                    break

                calls = [b for b in resp.content if b.type == "tool_use"]
                if not calls:
                    if s["submitted"]:
                        s["status"] = "submitted"
                        break
                    if s["reminders"] < budget["reminders_without_submit"]:
                        s["reminders"] += 1
                        messages.append({"role": "user", "content": "You have not submitted yet. Continue the task "
                                         "and finish with submit_calibration."})
                        record("reminder")
                        continue
                    s["status"] = "stopped_without_submit"
                    break

                results = []
                for b in calls:
                    name, args = b.name, b.input
                    t2 = time.time()
                    is_submit = name == "submit_calibration"
                    if resp.stop_reason == "max_tokens":
                        text, err = "Error: your response was cut off before this call was complete. Call again.", True
                        s["rejected_calls"] += 1
                    elif not isinstance(args, dict):
                        text, err = "Error: the tool arguments were not a JSON object.", True
                        s["rejected_calls"] += 1
                    elif name not in cfg["tools"]:
                        text, err = f"Error: unknown tool '{name}'.", True
                        s["rejected_calls"] += 1
                    elif s["submitted"]:
                        text, err = "The calibration was already submitted. The task is complete.", True
                        s["rejected_calls"] += 1
                    elif not is_submit and s["tool_calls"] >= budget["max_tool_calls"]:
                        text, err = "Tool call budget exhausted. Only submit_calibration is possible now.", True
                        s["rejected_calls"] += 1
                    else:
                        if not is_submit:
                            s["tool_calls"] += 1
                            s["tool_calls_by_name"][name] = s["tool_calls_by_name"].get(name, 0) + 1
                        res = await session.call_tool(name, args)
                        text = "\n".join(getattr(x, "text", "") for x in res.content)
                        err = bool(res.isError)
                        s["tool_errors"] += err
                        if is_submit and not err:
                            try:
                                s["submitted"] = bool(json.loads(text).get("accepted"))
                            except (json.JSONDecodeError, AttributeError):
                                pass
                    events = os.path.join(run_dir, "server_events.jsonl")
                    clamps = [json.loads(x) for x in open(events, encoding="utf-8") if '"clamp"' in x] \
                        if os.path.exists(events) else []
                    note = "".join(f"\n[Note: {e['tool']} was limited from {e['requested']} to {e['allowed']} "
                                   f"to fit the model run budget.]" for e in clamps[clamp_seen:])
                    clamp_seen = len(clamps)
                    bs = json.loads((await session.call_tool("budget_status", {})).content[0].text)
                    left = (f"\n[Budget left: {budget['max_tool_calls'] - s['tool_calls']} tool calls, "
                            f"{bs['model_runs_max'] - bs['model_runs_used']} model runs.]")
                    if len(text) > MAX_TOOL_RESULT_CHARS:
                        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[... result truncated]"
                    record("tool", name=name, args=args, error=err, seconds=round(time.time() - t2, 2),
                           result=text, runs_used=bs["model_runs_used"])
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": text + note + left,
                                    **({"is_error": True} if err else {})})
                messages.append({"role": "user", "content": results})  # all results of the turn in one message
                if s["submitted"] and all(b.name == "submit_calibration" for b in calls):
                    s["status"] = "submitted"
                    break

            bs = json.loads((await session.call_tool("budget_status", {})).content[0].text)
            s["model_runs"] = bs["model_runs_used"]
            if s["submitted"] and s["status"] == "running":
                s["status"] = "submitted"

    s["wall_seconds"] = round(time.time() - t0, 1)
    s["cost_usd"] = round(s["cost_usd"], 4)
    # names shared with the OpenRouter summaries
    s["prompt_tokens"] = s["input_tokens"] + s["cache_write_tokens"] + s["cache_read_tokens"]
    s["completion_tokens"] = s["output_tokens"]
    s["cached_tokens"] = s["cache_read_tokens"]
    record("end", summary=s)
    transcript.close()
    json.dump(s, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    asyncio.run(main())
