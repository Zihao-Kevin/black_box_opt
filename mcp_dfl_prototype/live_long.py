"""Longer-horizon / more-jobs training on the strict live-agent reward, fast enough for 20 seeds per method.

The LoRA is on the last layer's down_proj, after the last attention.  Nothing before it depends on the LoRA, so the
transformer is run ONCE per job (`prep`): for every accepted prefix we cache x (the LoRA's input) and v0 (the residual
stream plus the frozen down_proj output).  Training then needs no transformer:
    h = RMSNorm(v0 + s * B A x),  logits = W h,
and the score of every (prefix, token) entry is rank one in each LoRA factor, so the score Gram is
    K = s^2 [ (U U') o (XA XA') + (BU BU') o (X X') ]          (no E x 90k score matrix is ever formed).

Answer formats (grammars), both sampled from the full 128k vocabulary with an aggregated `anything else` branch:
    slots      files: <d>\\nknowledge: <d>\\nexecution: <d>\\ndelivery: <d>          4 decisions, 4 options each
    manifest   filesystem: <yes|no>\\ngrep: <yes|no>\\n ... \\nticket: <yes|no>       12 decisions; the client stops reading
               (reward 0) at a second server of one slot or at a third server

    python live_long.py prep  --format manifest --gpu 1
    python live_long.py stats --format manifest --gpu 1
    python live_long.py train --format manifest --gpu 1 --jobs all --methods "Q,Q + Δ" --seeds 0,1 --steps 150
"""
import os, sys, json, time, itertools, collections, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mcp_live import catalog

SLOTS, BUDGET = catalog.SLOTS, catalog.BUDGET
CONFIGS = list(itertools.product(range(4), repeat=4)); CONF_IDX = {c: i for i, c in enumerate(CONFIGS)}
OVER_CAP = np.array([sum(d > 0 for d in c) > BUDGET for c in CONFIGS])
SERVER_LIST = [(j, i, n) for j, (_, names) in enumerate(SLOTS) for i, n in enumerate(names)]      # (slot, index in slot, name)
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
JOBS = ["port", "shipped", "retries", "digest", "passing", "author", "ratelimit", "defines", "lint", "oncall"]
EASY = ["port", "digest", "defines", "lint"]


def reward_tables(strict=True):
    runs = collections.defaultdict(list)
    for line in open(os.path.join(HERE, "_live_rewards_gpt-4.1-mini_transcripts.jsonl")):
        ep = json.loads(line); used = {c.split("__")[0] for c in ep["calls"]}
        runs[ep["task"], tuple(ep["config"])].append(float(ep["f"] and (not strict or set(ep["servers"]) <= used)))
    return {t: np.array([np.mean(runs.get((t, c), [0.0])) for c in CONFIGS]) for t in JOBS}


# ---------------------------------------------------------------- prompts and grammars
def make_format(fmt, tok, servers, tasks):
    """returns prefill, prompt_ids(job), start state, transitions(state) -> [(token, next state | ('leaf', conf), is_decision)]"""
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    desc = lambda n: f"{n} ({servers[n]['instructions'].rstrip('.')}; tools: {', '.join(t['name'] for t in servers[n]['tools'])})"
    head = "You configure the MCP servers of an autonomous agent that will carry out a small task in a project workspace. "
    tail = ("Attach exactly the servers the task needs: the agent can only use what you attach, and a missing or wrong server makes "
            "the task impossible. ")
    if fmt == "slots":
        system = (head + "There are four capability slots; each offers three servers, and 0 leaves the slot empty. The client attaches at most "
                  f"{BUDGET} servers in total: a configuration with more than {BUDGET} is rejected and the task fails. " + tail +
                  "Answer with one line per slot, in the order given, in exactly this form:\n" + "\n".join(f"{s}: <0|1|2|3>" for s, _ in SLOTS) + "\nEND")
        legend = "Slots:\n" + "\n".join(f"{s}: 0 = no server; " + "; ".join(f"{i + 1} = {desc(n)}" for i, n in enumerate(names)) for s, names in SLOTS)
        prefill = "files: "; digits = [enc(str(d))[0] for d in range(4)]
        template = []
        for j, (s, _) in enumerate(SLOTS):
            template += (enc(f"\n{s}: ") if j else []) + [None]
        assert enc(prefill + "1\nknowledge: 0\nexecution: 2\ndelivery: 3") == enc(prefill) + [digits[[1, 0, 2, 3][template[:i].count(None)]] if t is None else t for i, t in enumerate(template)]

        def transitions(state):
            k, ds = state
            if template[k] is not None:
                return [(template[k], (k + 1, ds), False)]
            return [(digits[d], ("leaf", CONF_IDX[ds + (d,)]) if k + 1 == len(template) else (k + 1, ds + (d,)), True) for d in range(4)]
        start, n_dec = (0, ()), lambda state: len(state[1])
    else:
        names = [n for _, _, n in SERVER_LIST]
        system = (head + "There are twelve servers in four capability slots. The client attaches at most "
                  f"{BUDGET} servers in total and at most one per slot: a configuration that breaks either rule is rejected and the task fails. " + tail +
                  "Answer with one line per server, in the order given, in exactly this form:\n" + "\n".join(f"{n}: <yes|no>" for n in names) + "\nEND")
        legend = "Servers:\n" + "\n".join(f"{s}: " + "; ".join(desc(n) for n in ns) for s, ns in SLOTS)
        prefill = f"{names[0]}:"; YES, NO = enc(" yes"), enc(" no"); assert len(YES) == len(NO) == 1
        scaffold = [[]] + [enc(f"\n{n}:") for n in names[1:]]
        ans = [" yes", " no", " no", " no", " yes"] + [" no"] * 7
        assert enc("".join(f"{n}:{a}\n" for n, a in zip(names, ans))[:-1]) == enc(prefill) + sum([scaffold[i] + enc(ans[i]) for i in range(12)], [])

        def conf_of(chosen):
            c = [0] * 4
            for x in chosen: c[SERVER_LIST[x][0]] = SERVER_LIST[x][1] + 1
            return CONF_IDX[tuple(c)]

        def transitions(state):
            i, k, chosen = state
            if k < len(scaffold[i]):
                return [(scaffold[i][k], (i, k + 1, chosen), False)]
            nxt = lambda ch: ("leaf", conf_of(ch)) if i + 1 == len(names) else (i + 1, 0, ch)
            full = len(chosen) == BUDGET or SERVER_LIST[i][0] in {SERVER_LIST[x][0] for x in chosen}
            return [(YES[0], ("leaf", -1) if full else nxt(chosen + (i,)), True), (NO[0], nxt(chosen), True)]
        start, n_dec = (0, 0, ()), lambda state: state[0]

    def prompt_ids(job):
        user = f"Task: {tasks[job]['spec']}\n" + legend
        chat = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return enc(chat + prefill)
    return prompt_ids, start, transitions, n_dec


def enumerate_tree(start, transitions, n_dec):
    """nodes (token path after the prompt), entries (node, token or -1 = anything else), leaves (entry path, conf or -1).
    Leaf order: a node's `anything else` leaf first, then its branches depth-first (the notebook's order)."""
    nodes, entries, leaves = [], [], []

    def walk(state, toks, path):
        n = len(nodes); tr = transitions(state); nodes.append(dict(toks=tuple(toks), t=n_dec(state), dec=tr[0][2])); base = None
        ents = []
        for tok_id, _, _ in tr:
            ents.append(len(entries)); entries.append((n, tok_id))
        other = len(entries); entries.append((n, -1)); leaves.append((path + [other], -1))
        for e, (tok_id, nxt, _) in zip(ents, tr):
            if nxt[0] == "leaf": leaves.append((path + [e], nxt[1]))
            else: walk(nxt, toks + [tok_id], path + [e])
    sys.setrecursionlimit(10000); walk(start, [], [])
    return nodes, entries, leaves


def prep(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from dotenv import load_dotenv; load_dotenv(os.path.join(HERE, ".env"))
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from mcp_live import workspace, tasks as task_lib
    tasks = {t["id"]: t for t in task_lib.make_tasks(workspace.build_template(), workspace.start_intranet())}
    servers = json.load(open(os.path.join(HERE, "mcp_live", "catalog.json")))
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().requires_grad_(False)
    torch.manual_seed(args.lora_seed)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["down_proj"], layers_to_transform=[27])).eval()
    lm = model.base_model.model; last = lm.model.layers[-1]; saved = {}
    last.post_attention_layernorm.register_forward_pre_hook(lambda m, a: saved.update(r=a[0]))
    last.mlp.down_proj.register_forward_pre_hook(lambda m, a: saved.update(x=a[0]))
    prompt_ids, start, transitions, n_dec = make_format(args.format, tok, servers, tasks)
    nodes, entries, leaves = enumerate_tree(start, transitions, n_dec)
    paths = [nd["toks"] for nd in nodes]; inner = {p[:k] for p in paths for k in range(len(p))}
    maximal = [p for p in paths if p not in inner]; where = {}
    for s, p in enumerate(maximal):
        for k in range(len(p) + 1): where.setdefault(p[:k], (s, k))
    Lmax = max(len(p) for p in maximal)
    print(f"[{args.format}] {len(nodes)} nodes, {len(entries)} entries, {len(leaves)} leaves, {sum(nd['dec'] for nd in nodes)} decision nodes; "
          f"one forward over {len(maximal)} sequences of <= {Lmax} tokens", flush=True)
    os.makedirs(args.cache, exist_ok=True); common = os.path.join(args.cache, "_long_common.pt")
    if not os.path.exists(common): torch.save(dict(W=lm.lm_head.weight.detach().cpu()), common)       # 1.5 GB, shared by all formats
    out = dict(format=args.format, nodes=nodes, entries=entries, leaves=leaves, jobs={},
               g=lm.model.norm.weight.detach().cpu(), eps=lm.model.norm.variance_epsilon,
               A0=last.mlp.down_proj.lora_A["default"].weight.detach().cpu(), scaling=float(last.mlp.down_proj.scaling["default"]))
    for job in JOBS:
        t0 = time.time(); prompt = prompt_ids(job); X = torch.zeros(len(nodes), last.mlp.down_proj.in_features); V0 = torch.zeros(len(nodes), base.config.hidden_size)
        with torch.no_grad():
            cache = lm.model(input_ids=torch.tensor([prompt[:-1]], device="cuda")).past_key_values
            for c0 in range(0, len(maximal), 64):
                chunk = maximal[c0:c0 + 64]; import copy
                kv = copy.deepcopy(cache); kv.batch_repeat_interleave(len(chunk))
                ids = torch.tensor([prompt[-1:] + list(p) + [0] * (Lmax - len(p)) for p in chunk], device="cuda")
                lm.model(input_ids=ids, past_key_values=kv)
                v0 = saved["r"] + last.mlp.down_proj.base_layer(saved["x"])
                for n, p in enumerate(paths):
                    s, k = where[p]
                    if c0 <= s < c0 + 64: X[n] = saved["x"][s - c0, k].cpu(); V0[n] = v0[s - c0, k].cpu()
        out["jobs"][job] = dict(X=X, V0=V0); print(f"  {job:<10} {time.time() - t0:.1f}s", flush=True)
    torch.save(out, os.path.join(args.cache, f"_long_{args.format}.pt")); print("saved")


# ---------------------------------------------------------------- the policy without the transformer
class World:
    def __init__(self, fmt, jobs, cache, strict=True):
        d = torch.load(os.path.join(cache, f"_long_{fmt}.pt"), weights_only=False); dev = "cuda"
        self.W = torch.load(os.path.join(cache, "_long_common.pt"))["W"].to(dev)
        self.g, self.eps, self.s = d["g"].to(dev), d["eps"], d["scaling"]
        self.A0 = d["A0"].to(dev); self.jobs = jobs; nodes, entries, leaves = d["nodes"], d["entries"], d["leaves"]
        self.N, self.E, self.L = len(nodes), len(entries), len(leaves)
        self.ent_node = torch.tensor([n for n, _ in entries], device=dev); tok_id = torch.tensor([t for _, t in entries], device=dev)
        self.is_other = tok_id < 0; self.ent_tok = tok_id.clamp_min(0)
        self.node_t = np.array([nd["t"] for nd in nodes]); self.node_dec = np.array([nd["dec"] for nd in nodes])
        acc = ~self.is_other; self.okmask = torch.zeros(self.N, self.W.shape[0], dtype=torch.bool, device=dev)
        self.okmask[self.ent_node[acc], self.ent_tok[acc]] = True
        taken = np.zeros((self.L, self.E)); passed = np.zeros((self.L, self.E)); en = np.array([n for n, _ in entries])
        of_node = collections.defaultdict(list)
        for e, n in enumerate(en): of_node[n].append(e)
        for y, (path, _) in enumerate(leaves):
            taken[y, path] = 1
            for e in path: passed[y, of_node[en[e]]] = 1
        T = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)
        self.taken, self.passed = T(taken), T(passed); self.conf = np.array([c for _, c in leaves])
        R = reward_tables(strict)
        self.f = {j: torch.tensor(np.where(self.conf >= 0, R[j][np.maximum(self.conf, 0)], 0.0), device=dev) for j in jobs}
        self.X = {j: d["jobs"][j]["X"].to(dev) for j in jobs}; self.V0 = {j: d["jobs"][j]["V0"].to(dev) for j in jobs}
        self.XX = {j: (self.X[j] @ self.X[j].T) for j in jobs}                        # constant

    def tree(self, job, A, B):
        """branch probabilities P (E,), leaf probabilities pb (L,), score Gram K (E, E) and the factors for gradients."""
        X, en = self.X[job], self.ent_node
        XA = X @ A.T; v = self.V0[job] + self.s * XA @ B.T
        rms = (v.pow(2).mean(-1, keepdim=True) + self.eps).sqrt(); h = self.g * v / rms
        logp = torch.log_softmax(h @ self.W.T, -1); p = logp.exp()
        lp_other = logp.masked_fill(self.okmask, -torch.inf).logsumexp(-1)
        lp = torch.where(self.is_other, lp_other[en], logp[en, self.ent_tok]).double()
        q = p.masked_fill(self.okmask, 0.0); q = q / q.sum(-1, keepdim=True).clamp_min(1e-30)
        pW, qW = p @ self.W, q @ self.W; del logp, p, q
        a = torch.where(self.is_other[:, None], qW[en], self.W[self.ent_tok]) - pW[en]      # d log p(entry) / d h
        ga = self.g * a; ve = v[en]; u = ga / rms[en] - ve * (ve * ga).sum(-1, keepdim=True) / (v.shape[1] * rms[en] ** 3)   # through the RMSNorm
        P = lp.exp(); tot = torch.zeros(self.N, dtype=torch.float64, device=P.device).index_add_(0, en, P); P = P / tot[en]
        BU = u @ B; XAe = XA[en]
        K = self.s ** 2 * ((u @ u.T) * (XAe @ XAe.T) + (BU @ BU.T) * self.XX[job][en][:, en])
        return dict(P=P, pb=torch.exp(self.taken @ torch.log(P.clamp_min(1e-300))), K=K.double(), u=u, BU=BU, XAe=XAe, Xe=X[en])

    def grad(self, tr, w):
        """(sum_e w_e * score_e) as (dA, dB) for entry weights w (E,)."""
        w = w.float()[:, None]
        return self.s * (tr["BU"] * w).T @ tr["Xe"], self.s * (tr["u"] * w).T @ tr["XAe"]


# ---------------------------------------------------------------- estimators (as in tutorial_live_strict.ipynb)
def weights(W, tr, c, f): return W.taken * (f[:, None] - c) + W.passed * (tr["P"] * c)

def noise(W, tr, c, f):
    A = weights(W, tr, c, f); D = A - tr["pb"] @ A
    return float(tr["pb"] @ ((D @ tr["K"]) * D).sum(1))

def Q_table(W, tr, f):
    n = tr["pb"] @ W.taken
    return torch.where(n > 0, (tr["pb"] * f) @ W.taken / n.clamp_min(1e-300), 0.0)

def Delta_table(W, tr, f, Q, solver="pinv"):
    A = weights(W, tr, Q, f); R = A - tr["pb"] @ A; dA = W.passed * tr["P"] - W.taken
    M = tr["K"] * (dA.T @ (tr["pb"][:, None] * dA)); b = -(((R @ tr["K"]) * dA).T @ tr["pb"])
    if solver == "pinv":
        return torch.linalg.pinv(M, hermitian=True, rtol=1e-8) @ b
    lam = 1e-9 * M.diagonal().mean() + 1e-300                          # M is singular (one free constant per node): tiny ridge
    try:
        x = torch.linalg.solve(M + lam * torch.eye(len(M), dtype=M.dtype, device=M.device), b)
        return x if torch.isfinite(x).all() else torch.zeros_like(b)
    except Exception:                                                      # a deterministic policy: nothing left to cancel
        return torch.zeros_like(b)

class Seen:
    def __init__(self): self.n, self.sum, self.all = np.zeros(len(CONFIGS)), np.zeros(len(CONFIGS)), []
    def add(self, conf, f):
        self.all.append(f)
        if conf >= 0: self.n[conf] += 1; self.sum[conf] += f
    def f_hat(self, prior=1.0, zero=False):
        tried = (self.n > 0) & ~OVER_CAP; m = (self.sum[tried] / self.n[tried]).mean() if tried.any() and not zero else 0.0
        return np.where(OVER_CAP, 0.0, (self.sum + prior * m) / (self.n + prior))

def control_variate(method, W, tr, seen, f_true, solver):
    if method == "REINFORCE":
        return torch.full_like(tr["P"], np.mean(seen.all) if seen.all else 0.0)
    if "exact" in method: f = f_true
    else:
        fh = seen.f_hat(zero="0" in method); f = torch.tensor(np.where(W.conf >= 0, fh[np.maximum(W.conf, 0)], 0.0), device="cuda")
    Q = Q_table(W, tr, f)
    return Q + Delta_table(W, tr, f, Q, solver) if "Δ" in method else Q


def alignment(W, tr, pairs=2000, seed=0):
    rng = np.random.default_rng(seed); en = W.ent_node.cpu().numpy(); oth = W.is_other.cpu().numpy(); K = tr["K"].cpu().numpy()
    dec = np.flatnonzero(W.node_dec); ents = {n: [e for e in np.flatnonzero(en == n) if not oth[e]] for n in dec}; fr = []
    while len(fr) < pairs:
        n, m = rng.choice(dec, 2)
        if W.node_t[n] == W.node_t[m]: continue
        Kp = np.linalg.pinv(K[np.ix_(ents[m], ents[m])], rcond=1e-8)
        for e in ents[n]:
            kv = K[e, ents[m]]; fr.append(float(kv @ Kp @ kv) / max(K[e, e], 1e-30))
    return float(np.mean(fr))


def stats(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    W = World(args.format, JOBS, args.cache, strict=not args.lenient); A, B = W.A0.clone(), torch.zeros(W.V0[JOBS[0]].shape[1], W.A0.shape[0], device="cuda")
    print(f"[{args.format}] nodes {W.N} entries {W.E} leaves {W.L}"); tot = collections.Counter(); g_all = {}
    for j in JOBS:
        t0 = time.time(); tr = W.tree(j, A, B); f = W.f[j]; Ef = float(tr["pb"] @ f); t1 = time.time()
        Q = Q_table(W, tr, f); D = Delta_table(W, tr, f, Q, "pinv"); t2 = time.time(); D2 = Delta_table(W, tr, f, Q, "ridge"); t3 = time.time()
        r = dict(rf=noise(W, tr, torch.full_like(tr["P"], Ef), f), Q=noise(W, tr, Q, f), QD=noise(W, tr, Q + D, f), QD_ridge=noise(W, tr, Q + D2, f))
        for k, v in r.items(): tot[k] += v
        parse = float(tr["pb"][W.conf >= 0].sum())
        print(f"{j:<10} P(valid) {parse:.3f}  P(success) {Ef:.4f}  align {alignment(W, tr):.2f}   rf 10  Q {10*r['Q']/r['rf']:5.2f}  Q+Δ {10*r['QD']/r['rf']:5.2f}  (Δ/Q {r['QD']/r['Q']:.2f}; ridge {r['QD_ridge']/r['Q']:.2f})"
              f"   tree {t1-t0:.2f}s pinv {t2-t1:.2f}s ridge {t3-t2:.2f}s", flush=True)
    print(f"POOLED rf 10  Q {10*tot['Q']/tot['rf']:.2f}  Q+Δ {10*tot['QD']/tot['rf']:.2f}  Δ/Q {tot['QD']/tot['Q']:.3f}  (ridge {tot['QD_ridge']/tot['Q']:.3f})")


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cuda.matmul.allow_tf32 = True          # the 128k-vocabulary matmuls: same results to 4 digits, 5x faster
    jobs = JOBS if args.jobs == "all" else EASY if args.jobs == "easy" else args.jobs.split(",")
    W = World(args.format, jobs, args.cache); os.makedirs(args.out, exist_ok=True)
    for method in args.methods.split(","):
        for seed in map(int, args.seeds.split(",")):
            path = os.path.join(args.out, f"{args.format}_{len(jobs)}jobs_lr{args.lr:g}_{method.replace(' ', '').replace('+', 'p').replace('(', '_').replace(')', '')}_s{seed}.npz")
            if os.path.exists(path): continue
            t0 = time.time(); rng = np.random.default_rng(seed)
            A = W.A0.clone(); B = torch.zeros(W.V0[jobs[0]].shape[1], A.shape[0], device="cuda"); opt = torch.optim.Adam([A, B], lr=args.lr)
            seen = {j: Seen() for j in jobs}; succ, diag = [], []
            for step in range(args.steps + 1):
                gA, gB, s_now = 0.0, 0.0, []; d = collections.Counter(); log_now = step % args.log_every == 0
                for j in jobs:
                    tr = W.tree(j, A, B); f = W.f[j]; s_now.append(float(tr["pb"] @ f))
                    c = control_variate(method, W, tr, seen[j], f, args.solver)
                    if log_now:                                          # exact diagnostics at the current policy
                        Ef = s_now[-1]; Qx = Q_table(W, tr, f)
                        d["rf"] += noise(W, tr, torch.full_like(tr["P"], Ef), f); d["own"] += noise(W, tr, c, f)
                        d["Q"] += noise(W, tr, Qx, f); d["QD"] += noise(W, tr, Qx + Delta_table(W, tr, f, Qx, args.solver), f)
                        dA, dB = W.grad(tr, tr["pb"] @ (W.taken * f[:, None])); d["gA"] = d["gA"] + dA / len(jobs); d["gB"] = d["gB"] + dB / len(jobs)
                    if step < args.steps:
                        ys = rng.choice(W.L, size=args.episodes, p=(tr["pb"] / tr["pb"].sum()).cpu().numpy())
                        dA, dB = W.grad(tr, weights(W, tr, c, f)[ys].mean(0)); gA = gA + dA / len(jobs); gB = gB + dB / len(jobs)
                        for y in ys: seen[j].add(W.conf[y], float(f[y]))
                    del tr
                succ.append(s_now)
                if log_now:
                    g2 = float(d["gA"].pow(2).sum() + d["gB"].pow(2).sum()); n2 = len(jobs) ** 2 * args.episodes
                    diag.append([step, d["rf"], d["own"], d["Q"], d["QD"], g2, n2])
                if step < args.steps:
                    opt.zero_grad(); A.grad, B.grad = -gA, -gB; opt.step()
            np.savez(path, success=np.array(succ), diag=np.array(diag), jobs=np.array(jobs))
            print(f"{method:<14} seed {seed}: mean P(success) {np.mean(succ[0]):.3f} -> {np.mean(succ[-1]):.3f}   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("prep", "stats", "train"))
    ap.add_argument("--format", default="slots", choices=("slots", "manifest")); ap.add_argument("--gpu", default="1"); ap.add_argument("--lora_seed", type=int, default=0)
    ap.add_argument("--lenient", action="store_true"); ap.add_argument("--jobs", default="all"); ap.add_argument("--methods", default="REINFORCE,Q,Q + Δ,Q (exact),Q + Δ (exact)")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--steps", type=int, default=150); ap.add_argument("--episodes", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log_every", type=int, default=10); ap.add_argument("--solver", default="ridge", choices=("pinv", "ridge")); ap.add_argument("--out", default=os.path.join(HERE, "_long_runs"))
    ap.add_argument("--cache", default=HERE, help="where the cached forward passes live (about 2 GB; use a local disk)")
    args = ap.parse_args(); {"prep": prep, "stats": stats, "train": train}[args.cmd](args)
