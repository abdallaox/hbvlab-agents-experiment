"""Run one agent on one catchment: the agent calibrates HBV_Lab through the MCP tools until it submits or runs out.

  python harness/run_agent.py --agent claude --gauge 12001 --repeat 1
  python harness/run_agent.py --agent mock --gauge 12001        # scripted, no LLM: tests the pipeline for free

Each run gets its own folder, runs/agents/<agent>/<gauge>/r<repeat>/, holding the input files, transcript.jsonl (every
API response and tool call, with tokens, cost and timings), server_events.jsonl, submission.json (if the agent
submits) and summary.json. The agent talks to OpenRouter's chat completions API with tool calling. The key is read
from OPENROUTER_API_KEY or from .env in the repository root (not committed). An existing summary.json means the run
is done and it is skipped, unless --force.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import sys
import time

import httpx
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOOL_RESULT_CHARS = 12000


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No OPENROUTER_API_KEY in the environment or in .env")


def build_prompt(meta, budget):
    text = open(os.path.join(ROOT, "harness", "prompt.md"), encoding="utf-8").read()
    system = text.split("# SYSTEM", 1)[1].split("# TASK", 1)[0].strip()
    task = text.split("# TASK", 1)[1].strip()
    task = (task.replace("{catchment_json}", json.dumps(meta, indent=1, ensure_ascii=False))
                .replace("{max_tool_calls}", str(budget["max_tool_calls"]))
                .replace("{max_runs}", str(budget["max_model_runs"])))
    return system, task


def with_cache_marks(messages):
    """Anthropic models cache only at explicit marks: mark the system prompt and the last message."""
    msgs = copy.deepcopy(messages)
    for m in (msgs[0], msgs[-1]):
        if isinstance(m.get("content"), str) and m["content"]:
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
    return msgs


async def call_llm(client, key, model, messages, tools, max_tokens):
    # plugins [] and transforms []: no web search or other OpenRouter add-ons, and no context compression, so the
    # model sees exactly the prompt, its own messages and the tool results.
    body = {"model": model, "messages": with_cache_marks(messages) if model.startswith("anthropic/") else messages,
            "tools": tools, "max_tokens": max_tokens, "usage": {"include": True},
            "provider": {"require_parameters": True}, "plugins": [], "transforms": []}
    last = None
    for attempt in range(4):
        try:
            r = await client.post(API, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=600)
            data = r.json()
            if r.status_code == 200 and data.get("choices"):
                return data, attempt
            last = f"HTTP {r.status_code}: {json.dumps(data)[:500]}"
        except Exception as e:  # network errors, bad JSON
            last = repr(e)
        await asyncio.sleep(10 * 2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after retries: {last}")


class MockLLM:
    """Scripted stand-in for an LLM: create, load, calibrate twice with DDS, submit."""

    def __init__(self):
        self.step = 0

    def respond(self):
        plan = [
            ("create_model", {"name": "mock"}),
            ("load_data", {"model_id": "model-1", "file_path": "forcing.csv", "obs_q_column": "Discharge",
                           "warmup_end": "19710930"}),
            ("calibrate", {"model_id": "model-1", "method": "DDS", "iterations": 300, "seed": 0}),
            ("set_parameter_ranges", {"model_id": "model-1", "ranges": {"FC": {"max": 600}}}),
            ("calibrate", {"model_id": "model-1", "method": "DDS", "iterations": 300, "seed": 1}),
            ("submit_calibration", {"model_id": "model-1", "diagnosis": "Mock run, no diagnosis."}),
        ]
        if self.step >= len(plan):
            return {"choices": [{"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}]}
        name, args = plan[self.step]
        self.step += 1
        call = {"id": f"mock-{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]},
                             "finish_reason": "tool_calls"}], "usage": {}, "model": "mock"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="a key of config/agents.yml, or mock")
    ap.add_argument("--gauge", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--inputs", default=os.path.join(ROOT, "data", "agent_inputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "agents"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "agents.yml"), encoding="utf-8"))
    budget = cfg["budget"]
    model = "mock" if a.agent == "mock" else cfg["agents"][a.agent]["model"]
    run_dir = os.path.join(a.out, a.agent, str(a.gauge), f"r{a.repeat}")
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

    record("start", agent=a.agent, model=model, gauge=a.gauge, repeat=a.repeat, budget=budget,
           system=system, task=task)
    server = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(ROOT, "harness", "experiment_server.py"), "--run-dir", run_dir,
              "--max-runs", str(budget["max_model_runs"])],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})

    s = {"agent": a.agent, "model_requested": model, "gauge": a.gauge, "repeat": a.repeat, "status": "running",
         "tool_calls": 0, "tool_calls_by_name": {}, "tool_errors": 0, "rejected_calls": 0, "api_turns": 0,
         "api_retries": 0, "reminders": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
         "reasoning_tokens": 0, "cost_usd": 0.0, "models_served": [], "providers": [], "submitted": False}
    t0 = time.time()
    clamp_seen = 0

    async with stdio_client(server, errlog=open(os.path.join(run_dir, "server_stderr.log"), "w")) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                       "parameters": t.inputSchema}}
                     for t in listed if t.name in cfg["tools"]]
            # fingerprints: identical across runs if every agent got the same prompt template and tools
            template = open(os.path.join(ROOT, "harness", "prompt.md"), "rb").read()
            s["prompt_template_sha256"] = hashlib.sha256(template).hexdigest()
            s["tools_sha256"] = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
            record("tools", names=[t["function"]["name"] for t in tools], definitions=tools,
                   prompt_template_sha256=s["prompt_template_sha256"], tools_sha256=s["tools_sha256"])
            messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
            mock = MockLLM() if a.agent == "mock" else None
            key = None if mock else load_key()

            async with httpx.AsyncClient() as client:
                while True:
                    if time.time() - t0 > 60 * budget["max_wall_minutes"]:
                        s["status"] = "wall_time"
                        break
                    if s["api_turns"] >= budget["max_api_turns"]:
                        s["status"] = "max_api_turns"
                        break
                    t1 = time.time()
                    try:
                        if mock:
                            data, retries = mock.respond(), 0
                        else:
                            data, retries = await call_llm(client, key, model, messages, tools,
                                                           budget["max_output_tokens"])
                    except RuntimeError as e:
                        record("api_error", error=str(e))
                        s["status"] = "api_error"
                        break
                    s["api_turns"] += 1
                    s["api_retries"] += retries
                    u = data.get("usage") or {}
                    s["prompt_tokens"] += u.get("prompt_tokens") or 0
                    s["completion_tokens"] += u.get("completion_tokens") or 0
                    s["cached_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    s["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                    s["cost_usd"] += float(u.get("cost") or 0)
                    for k, v in (("models_served", data.get("model")), ("providers", data.get("provider"))):
                        if v and v not in s[k]:
                            s[k].append(v)
                    choice = data["choices"][0]
                    msg = choice["message"]
                    record("response", seconds=round(time.time() - t1, 2), finish=choice.get("finish_reason"),
                           usage=u, message=msg)
                    keep = {k: msg[k] for k in ("role", "content", "tool_calls", "reasoning_details") if msg.get(k)}
                    keep.setdefault("role", "assistant")
                    if "content" not in keep:
                        keep["content"] = ""
                    messages.append(keep)

                    calls = msg.get("tool_calls") or []
                    if not calls:
                        if s["submitted"]:
                            s["status"] = "submitted"
                            break
                        if s["reminders"] < budget["reminders_without_submit"]:
                            s["reminders"] += 1
                            messages.append({"role": "user", "content": "You have not submitted yet. Continue the "
                                             "task and finish with submit_calibration."})
                            record("reminder")
                            continue
                        s["status"] = "stopped_without_submit"
                        break

                    for c in calls:
                        name = c["function"]["name"]
                        try:
                            args = json.loads(c["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = None
                        t2 = time.time()
                        is_submit = name == "submit_calibration"
                        if args is None:
                            text, err = "Error: the tool arguments were not valid JSON.", True
                            s["rejected_calls"] += 1
                        elif name not in cfg["tools"]:
                            text, err = f"Error: unknown tool '{name}'.", True
                            s["rejected_calls"] += 1
                        elif s["submitted"]:
                            text, err = "The calibration was already submitted. The task is complete.", True
                            s["rejected_calls"] += 1
                        elif not is_submit and s["tool_calls"] >= budget["max_tool_calls"]:
                            text, err = ("Tool call budget exhausted. Only submit_calibration is possible now."), True
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
                        # tell the agent when calibrate or evaluate_uncertainty was cut to the runs left
                        events = os.path.join(run_dir, "server_events.jsonl")
                        clamps = [json.loads(x) for x in open(events, encoding="utf-8") if '"clamp"' in x] \
                            if os.path.exists(events) else []
                        note = ""
                        for e in clamps[clamp_seen:]:
                            note += (f"\n[Note: {e['tool']} was limited from {e['requested']} to {e['allowed']} "
                                     f"to fit the model run budget.]")
                        clamp_seen = len(clamps)
                        b = json.loads((await session.call_tool("budget_status", {})).content[0].text)
                        left = (f"\n[Budget left: {budget['max_tool_calls'] - s['tool_calls']} tool calls, "
                                f"{b['model_runs_max'] - b['model_runs_used']} model runs.]")
                        if len(text) > MAX_TOOL_RESULT_CHARS:
                            text = text[:MAX_TOOL_RESULT_CHARS] + "\n[... result truncated]"
                        record("tool", name=name, args=args, error=err, seconds=round(time.time() - t2, 2),
                               result=text, runs_used=b["model_runs_used"])
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": text + note + left})
                    if s["submitted"]:
                        s["status"] = "submitted"
                        break

            b = json.loads((await session.call_tool("budget_status", {})).content[0].text)
            s["model_runs"] = b["model_runs_used"]

    s["wall_seconds"] = round(time.time() - t0, 1)
    s["cost_usd"] = round(s["cost_usd"], 4)
    record("end", summary=s)
    transcript.close()
    json.dump(s, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    asyncio.run(main())
