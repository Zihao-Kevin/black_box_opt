"""The server catalog the policy chooses from: four capability slots, three real MCP servers each, and the client's
cap on how many servers may be attached at once."""
import os, sys, json, asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

HERE = os.path.dirname(os.path.abspath(__file__))
SERVERS_DIR = os.path.join(HERE, "servers")

SLOTS = [
    ("files",     ["filesystem", "grep", "git"]),
    ("knowledge", ["sqlite", "fetch", "notes"]),
    ("execution", ["python", "pytest", "lint"]),
    ("delivery",  ["write_file", "mail", "ticket"]),
]
BUDGET = 2           # the MCP client attaches at most this many servers; larger configurations are rejected
K = 4                # menu per slot: 0 = none, 1..3 = the slot's servers
T = len(SLOTS)


def server_params(name, workspace):
    return StdioServerParameters(command=sys.executable, args=[os.path.join(SERVERS_DIR, f"{name}_server.py")],
                                 env={"WORKSPACE": workspace, "PYTHONUNBUFFERED": "1"})


def config_servers(config):
    """config = (d_1..d_T) with d in 0..3 -> list of server names."""
    return [opts[d - 1] for (_, opts), d in zip(SLOTS, config) if d > 0]


def feasible(config):
    return sum(1 for d in config if d > 0) <= BUDGET


async def describe_servers(workspace, cache=os.path.join(HERE, "catalog.json")):
    """Spawn every server once and record what it announces over MCP (instructions + tools). Cached to disk."""
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}
    for _, opts in SLOTS:
        for name in opts:
            async with stdio_client(server_params(name, workspace)) as (r, w):
                async with ClientSession(r, w) as s:
                    init = await s.initialize()
                    tl = await s.list_tools()
                    out[name] = dict(instructions=init.instructions or "",
                                     tools=[dict(name=t.name, description=t.description or "", schema=t.input_schema) for t in tl.tools])
    json.dump(out, open(cache, "w"), indent=1)
    return out


def menu_line(slot_name, opts, catalog):
    """One legend line per slot for the policy prompt, built from what the servers themselves announce."""
    return f"{slot_name}: 0 = no server; " + "; ".join(
        f"{i + 1} = {n} ({catalog[n]['instructions'].rstrip('.')}; tools: {', '.join(t['name'] for t in catalog[n]['tools'])})"
        for i, n in enumerate(opts))
