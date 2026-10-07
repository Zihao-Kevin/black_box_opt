"""A fixed offline dataset and one warm start for the paper's live-agent runs.
usage: python offline.py <gpu> [n_per_task=50] [target=0.35]
1. Samples n_per_task episodes per training task from the base model (one fixed seed), scores them with the cached strict reward,
   and writes _sgd_long/offline_data.json.  The worker pre-fills the fitted reward table of every control-variate method from
   these episodes (SGD_OFFLINE=_sgd_long/offline_data.json); under another reward (graded) the same configurations are re-scored.
2. Warm start: supervised steps on the successful episodes of that dataset (one per task per step, drawn by a fixed rng),
   Adam lr 5e-4, until the exact mean P(success) over the training tasks reaches the target, at most 60 steps.
   Writes _sgd_long/warm_offline/warm.pt: one starting point shared by every method and every seed (SGD_WARM=_sgd_long/warm_offline).
Nothing here reads the reward table beyond the sampled episodes, except the stopping rule of the warm start (as in warm.py)."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]; NPT = int(sys.argv[2]) if len(sys.argv) > 2 else 50; TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.dirname(HERE)); sys.path.insert(0, os.getcwd())
nb = json.load(open(os.path.join(HERE, "_snapshot.ipynb")))
cell = lambda i: "\n".join(l for l in "".join(nb["cells"][i]["source"]).split("\n") if "await " not in l)
job = "lint"
for i in (1, 2, 4, 7, 9, 12, 14): exec(cell(i))
exec(cell(17)[:cell(17).index("logs = {}")])
import numpy as np, torch

def sft_loss(task, digits):                                    # cross-entropy of the answer fill(TEMPLATE, digits) after the prompt
    prompt = prompt_ids(task); answer = fill(TEMPLATE, digits)
    ids = torch.tensor([prompt + answer], device="cuda"); labels = torch.tensor([[-100] * len(prompt) + answer], device="cuda")
    return lm(input_ids=ids, labels=labels).loss

def p_success():
    ps = []
    for t in TRAIN:
        tree = build_tree(t); ps.append(float(tree.pb @ leaf_rewards(tree, REWARD[t]))); del tree
    return ps

t0 = time.time(); reset()
# ---- 1. the offline dataset: the base model's own episodes ---------------------------------------------------------------
ps0 = p_success(); print(f"base model: P(success) per task {dict(zip(TRAIN, [round(p, 3) for p in ps0]))}, mean {np.mean(ps0):.3f}", flush=True)
rng = np.random.default_rng(2026); data = []
for t in TRAIN:
    tree = build_tree(t); f = leaf_rewards(tree, REWARD[t]); ys = sample(tree, NPT, rng)
    rows = [(t, int(tree.conf[y]), float(f[y])) for y in ys]; data += rows; del tree
    wins = [c for _, c, r in rows if r > 0]
    print(f"  {t:8s}: {len(wins)} of {NPT} succeed (first 25: {sum(r > 0 for _, _, r in rows[:25])}); successful configurations {sorted(set(wins))}; "
          f"{sum(c < 0 for _, c, _ in rows)} unparseable", flush=True)
json.dump({"n_per_task": NPT, "seed": 2026, "tasks": TRAIN, "episodes": [[t, c] for t, c, _ in data], "f_strict": [r for _, _, r in data]},
          open(os.path.join(HERE, "offline_data.json"), "w"))
print(f"wrote offline_data.json: {len(data)} episodes, {sum(r > 0 for _, _, r in data)} successes  ({time.time() - t0:.0f}s)", flush=True)

# ---- 2. one warm start from the dataset's successes ------------------------------------------------------------------------
good = {t: [c for tt, c, r in data if tt == t and r > 0] for t in TRAIN}; tasks = [t for t in TRAIN if good[t]]
rng = np.random.default_rng(1000); opt = torch.optim.Adam(params, lr=5e-4)
ps = ps0; path = [np.mean(ps)]
for step in range(60):
    if np.mean(ps) >= TARGET: break
    opt.zero_grad()
    loss = sum(sft_loss(t, CONFIGS[rng.choice(good[t])]) for t in tasks) / len(tasks); loss.backward(); opt.step()
    ps = p_success(); path.append(np.mean(ps))
os.makedirs(os.path.join(HERE, "warm_offline"), exist_ok=True)
torch.save([p.detach().cpu().clone() for p in params], os.path.join(HERE, "warm_offline", "warm.pt"))
print(f"warm start: {len(path) - 1} supervised steps on {sum(len(v) for v in good.values())} successful episodes of {len(tasks)} tasks, "
      f"P(success) {path[0]:.3f} -> {path[-1]:.3f}  per task {dict(zip(TRAIN, [round(p, 3) for p in ps]))}  path {[round(p, 2) for p in path]}  ({time.time() - t0:.0f}s)", flush=True)
json.dump({"steps": len(path) - 1, "path": path, "per_task": dict(zip(TRAIN, ps)), "target": TARGET}, open(os.path.join(HERE, "warm_offline", "warm.json"), "w"))
os._exit(0)
