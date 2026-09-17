"""One agent episode = fresh workspace copy + the MCP servers of one configuration + a GPT tool-calling loop +
the task's oracle.  `fill_cache` evaluates a set of (task, configuration) pairs concurrently and persists the
rewards, so the exact tree analysis and any later training read rewards from disk instead of re-running agents."""
import os, json, asyncio, shutil, time, contextlib, random
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from . import catalog, workspace

AGENT_SYSTEM = ("You are an autonomous assistant working inside a software project on behalf of an engineer. "
                "Use the available tools to establish facts; do not guess values you could look up, and do not claim to have "
                "done something (written a file, sent mail, opened a ticket) unless a tool confirmed it. If a tool you would need "
                "is not available, do your best with what you have. When the task asks you to reply with a value, finish with a "
                "final line of the form `ANSWER: <value>` containing only the value.")
MAX_ROUNDS = 8
_client = None


def client():
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


def _oa_tool(server, t):
    schema = dict(t.input_schema or {"type": "object", "properties": {}})
    schema.pop("title", None)
    for p in schema.get("properties", {}).values():
        p.pop("title", None)
    return {"type": "function", "function": {"name": f"{server}__{t.name}", "description": (t.description or "")[:1000], "parameters": schema}}


async def run_episode(task, config, model="gpt-4.1-mini", log=False, max_rounds=MAX_ROUNDS):
    """Returns dict(f, rounds, calls, usage, transcript, servers, rejected)."""
    servers = catalog.config_servers(config)
    rec = dict(task=task["id"], config=list(config), servers=servers, model=model, rounds=0, calls=[], usage=0, transcript=[], rejected=False)
    if not catalog.feasible(config):                       # the client refuses to attach more than BUDGET servers
        rec.update(f=0.0, rejected=True, transcript=[f"client: {len(servers)} servers requested, cap is {catalog.BUDGET}: configuration rejected"])
        return rec
    ws = workspace.fresh_copy()
    try:
        async with contextlib.AsyncExitStack() as stack:
            sessions, tools = {}, []
            for name in servers:
                r, w = await stack.enter_async_context(stdio_client(catalog.server_params(name, ws), errlog=open(os.devnull, "w")))
                s = await stack.enter_async_context(ClientSession(r, w)); await s.initialize()
                sessions[name] = s
                tools += [_oa_tool(name, t) for t in (await s.list_tools()).tools]
            msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": "Task: " + task["spec"]}]
            final = ""
            for rnd in range(max_rounds + 1):
                kw = dict(model=model, messages=msgs, temperature=0, max_tokens=700)
                if tools and rnd < max_rounds:
                    kw.update(tools=tools, tool_choice="auto")
                resp = await client().chat.completions.create(**kw)
                rec["usage"] += resp.usage.total_tokens
                m = resp.choices[0].message
                msgs.append({"role": "assistant", "content": m.content, "tool_calls": [tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None})
                if not m.tool_calls:
                    final = m.content or ""; rec["transcript"].append(f"assistant: {final}"); break
                rec["rounds"] += 1
                for tc in m.tool_calls:
                    server, _, tool = tc.function.name.partition("__")
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        res = await sessions[server].call_tool(tool, args)
                        text = "\n".join(getattr(c, "text", "") for c in res.content)
                        if res.is_error:
                            text = "error: " + text
                    except Exception as e:                # bad JSON, unknown tool, server crash: the agent sees an error
                        args, text = tc.function.arguments, f"error: {type(e).__name__}: {e}"
                    rec["calls"].append(tc.function.name)
                    rec["transcript"].append(f"tool {tc.function.name}({json.dumps(args) if not isinstance(args, str) else args}) -> {text[:300]}")
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": text[:6000]})
            rec["f"] = float(bool(task["oracle"](ws, final)))
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    if log:
        print(f"[{task['id']}] config {config} servers {servers}\n  " + "\n  ".join(rec["transcript"]) + f"\n  oracle -> f = {rec['f']}   ({rec['usage']} tokens)")
    return rec


def load_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


async def fill_cache(tasks, configs, path, model="gpt-4.1-mini", repeats=1, concurrency=8, save_every=20, transcripts_path=None):
    """cache[task_id][config_key] = list of rewards (one per repeat).  Resumable; infeasible configs are stored as
    [0.0] without an API call.  Returns the cache."""
    cache = load_cache(path); todo = []
    for t in tasks:
        cache.setdefault(t["id"], {})
        for c in configs:
            key = ",".join(map(str, c)); have = cache[t["id"]].get(key, [])
            if not catalog.feasible(c):
                cache[t["id"]][key] = [0.0]; continue
            todo += [(t, tuple(c))] * max(0, repeats - len(have))
    random.Random(0).shuffle(todo)
    sem = asyncio.Semaphore(concurrency); done = 0; t0 = time.time(); tokens = 0; tlog = open(transcripts_path, "a") if transcripts_path else None

    async def one(t, c):
        nonlocal done, tokens
        async with sem:
            for attempt in range(3):
                try:
                    rec = await run_episode(t, c, model=model); break
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))
        cache[t["id"]].setdefault(",".join(map(str, c)), []).append(rec["f"])
        done += 1; tokens += rec["usage"]
        if tlog:
            tlog.write(json.dumps(rec) + "\n"); tlog.flush()
        if done % save_every == 0 or done == len(todo):
            json.dump(cache, open(path, "w"), indent=0)
            print(f"  {done}/{len(todo)} episodes, {tokens / 1e6:.2f}M tokens, {time.time() - t0:.0f}s", flush=True)

    if todo:
        print(f"running {len(todo)} agent episodes ({model}, concurrency {concurrency})", flush=True)
        await asyncio.gather(*(one(t, c) for t, c in todo))
    json.dump(cache, open(path, "w"), indent=0)
    return cache


def reward_table(cache, task_id, configs):
    """Mean oracle reward per configuration, in the order of `configs`."""
    import numpy as np
    return np.array([float(np.mean(cache[task_id][",".join(map(str, c))])) for c in configs])
