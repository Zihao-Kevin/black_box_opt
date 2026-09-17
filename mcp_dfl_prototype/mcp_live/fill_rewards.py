"""Fill the oracle-reward cache for every configuration of every task (resumable).
    python -m mcp_live.fill_rewards --repeats 2 --concurrency 8 --model gpt-4.1-mini
"""
import os, sys, argparse, asyncio, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from mcp_live import workspace, catalog, tasks, agent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-4.1-mini"); ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--out", default="")
args = ap.parse_args()
out = args.out or f"_live_rewards_{args.model}.json"
facts = workspace.build_template(); url = workspace.start_intranet()
TASKS = tasks.make_tasks(facts, url)
configs = list(itertools.product(range(catalog.K), repeat=catalog.T))
cache = asyncio.run(agent.fill_cache(TASKS, configs, out, model=args.model, repeats=args.repeats, concurrency=args.concurrency,
                                     transcripts_path=out.replace(".json", "_transcripts.jsonl")))
for t in TASKS:
    tab = agent.reward_table(cache, t["id"], configs)
    feas = [c for c in configs if catalog.feasible(c)]
    wins = [c for c in feas if tab[configs.index(c)] > 0.5]
    print(f"{t['id']:<10} feasible {len(feas)}  succeed {len(wins)}: {[catalog.config_servers(c) for c in wins][:8]}")
