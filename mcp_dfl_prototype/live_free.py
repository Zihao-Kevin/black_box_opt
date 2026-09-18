"""Free-text answers on the strict live-agent reward, with exact accounting on trees too large for a dense Gram.

Answer formats (every token is sampled from the full 128k vocabulary; `anything else` is one aggregated failing branch):
    list     servers: grep, filesystem              names in any order, no repeats, `,` continues, newline / EOS stops
    plan     calls: notes, notes, notes, mail       the plan of server calls, one name per call, repeats allowed, up to --L
             calls; the client attaches the servers that appear in the plan
    queue    1: grep\n2: filesystem python\n3: lint   a queue of --queue tasks answered in ONE generation, one line per task,
             names separated by spaces (one menu at every decision: a name, or the end of the line).  The agent works
             through the queue in order and the client stops at the first task that fails; reward = share of the queue
             that got done (--reward progress) or 1 if all of it did (--reward product)
    slots / manifest                                the two formats of live_long.py (used here to validate the accounting)
The client's rules are the cached experiment's: at most 2 different servers, at most one per slot; the parse stops with
reward 0 at the first name that breaks one.  Rewards are the cached strict live-agent rewards, so nothing is re-run.

Finding (LIVE_FREE.md): a scalar Delta(prefix, token) can only cancel noise along that token's own score, so on multi-way name
menus it adds almost nothing (0.83-0.97 of Q's noise).  Methods "Q + rowΔ (exact)" / "Q0 + rowΔ" use a ROW-WISE residual instead:
inside one weight row every token's score is a multiple of the layer input, so the component of the Q estimator's increment
xi_v = A_v S_n + D_v - Dbar_n along that input can be subtracted exactly (Tree.rowwise, Tree.row_correction).
--init B --r 1 is the collinearity diagnostic (LoRA started with B random, A = 0).

Exact accounting without the E x E Gram.  Control-variate terms of different prefixes are uncorrelated, so the
variance-minimizing residual is one small solve per prefix n (k = number of branches of n):
    C_n = K_n o (diag P - P P'),   b_n[v] = -P_v ( A_v <s_v, S_n> + <s_v, D_v - Dbar_n> ),   Delta_n = -C_n^+ b_n
with S_n the summed score of the path to n (top-down), D_v the expected sum of the local exact gradients below branch v
(bottom-up), and A_v = Q_v - V_n.  The exact noise of any table is one more top-down pass.  Cost O(E x 90k), not O(E^3).

    python live_free.py prep  --format plan --L 6 --gpu 0
    python live_free.py stats --format plan --L 6 --gpu 0
    python live_free.py train --format plan --L 6 --gpu 0 --methods "Q0,Q0 + Δ" --seeds 0,1 --steps 200
"""
import os, sys, json, time, copy, collections, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import live_long as LL
from live_long import SLOTS, BUDGET, CONF_IDX, SERVER_LIST, MODEL, JOBS, EASY, reward_tables, Seen


# ---------------------------------------------------------------- prompts and grammars
def make_queue(tok, servers, tasks, queue, R):
    """One generation answers the whole queue.  state = ("A", k, seq) a name or the end of line k | ("N", k, seq, x, i) inside a
    name | ("H", k, i) the header of line k.  A finished line is a closing entry ("close", stage, conf, next state or None):
    the client runs that task, and reads on only if it succeeded."""
    enc = lambda s: tok.encode(s, add_special_tokens=False); K = len(queue)
    names = [n for _, _, n in SERVER_LIST]; slot = [j for j, _, _ in SERVER_LIST]; groups = ", ".join("at most one of " + "/".join(ns) for _, ns in SLOTS)
    system = ("You configure the MCP servers of an autonomous agent that works through a queue of small tasks in a project workspace, one task "
              "at a time. You do not solve the tasks. For every task, name the servers the client should attach while the agent works on it. "
              f"The client attaches at most {BUDGET} servers per task ({groups}); a line that breaks a rule is rejected. Attach exactly the servers "
              "each task needs, often a single one: the agent can only use what you attach, and a missing or unneeded server makes the task fail. "
              "The agent stops at the first task it cannot complete. Answer with one line per task, in the order given, server names separated by "
              "spaces, and nothing else.")
    shot = ("Tasks:\n1. Summarize what the team notes say about the release process. Reply with the summary.\n2. Compute 2**64 in Python and open a ticket "
            "whose title contains the result.\n", "servers for task 1: notes\nservers for task 2: python ticket")
    legend = "Servers:\n" + "\n".join(f"{n} - {servers[n]['instructions']}" for n in names)
    user = "Tasks:\n" + "\n".join(f"{k + 1}. {tasks[j]['spec']}" for k, j in enumerate(queue)) + "\n" + legend
    ntok = [enc(" " + n) for n in names]; assert len({t[0] for t in ntok}) == len(ntok)
    NL = enc("\n"); assert len(NL) == 1; NL = NL[0]; END = [tok.convert_tokens_to_ids("<|eot_id|>")] + enc("\n\n"); assert len(END) == 2; head = [enc(f"servers for task {k + 1}:") for k in range(max(K, 2))]
    assert enc("servers for task 1: grep write_file\nservers for task 2: lint") == head[0] + ntok[1] + ntok[9] + [NL] + head[1] + ntok[8]

    def conf_of(seq):
        c = [0] * 4
        for x in seq: c[slot[x]] = SERVER_LIST[x][1] + 1
        return CONF_IDX[tuple(c)]

    def transitions(state):
        kind, k = state[0], state[1]
        if kind == "H": return [(head[k][state[2]], ("H", k, state[2] + 1) if state[2] + 1 < len(head[k]) else ("A", k, ()), False)]
        seq = state[2]
        if kind == "N":
            x, i = state[3], state[4]; return [(ntok[x][i], ("N", k, seq, x, i + 1) if i + 1 < len(ntok[x]) else ("A", k, seq + (x,)), False)]
        out = []
        for x, toks in enumerate(ntok):
            bad = x in seq or len(seq) == BUDGET or slot[x] in {slot[y] for y in seq}
            out.append((toks[0], ("leaf", -1) if bad else ("N", k, seq, x, 1) if len(toks) > 1 else ("A", k, seq + (x,)), True))
        c = conf_of(seq); go = k + 1 < K and R[queue[k]][c] > 0
        return out + [(NL, ("close", k, c, ("H", k + 1, 0) if go else None), True)] + [(t, ("close", k, c, None), True) for t in END]   # EOS or a blank line ends the answer

    def prompt_ids(_):
        chat = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": shot[0] + legend}, {"role": "assistant", "content": shot[1]},
                                        {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return enc(chat + "servers for task 1:")
    return prompt_ids, ("A", 0, ()), transitions, lambda state: state[1]


def make_format(fmt, tok, servers, tasks, L):
    if fmt in ("slots", "manifest"): return LL.make_format(fmt, tok, servers, tasks)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    desc = lambda n: f"{n} ({servers[n]['instructions'].rstrip('.')}; tools: {', '.join(t['name'] for t in servers[n]['tools'])})"
    names = [n for _, _, n in SERVER_LIST]; slot = [j for j, _, _ in SERVER_LIST]
    head = "You configure the MCP servers of an autonomous agent that will carry out a small task in a project workspace. "
    rules = (f"The client attaches at most {BUDGET} different servers and at most one per slot: an answer that breaks either rule is rejected and "
             "the task fails. Attach exactly the servers the task needs: the agent can only use what you attach, and a missing or wrong "
             "server makes the task impossible. ")
    if fmt == "plan":
        system = (head + "There are twelve servers in four capability slots. Write the plan of server calls the agent will make: one server name "
                  "per call, in the order of the calls, repeating a server when it is called again. The client attaches the servers that appear "
                  f"in your plan. " + rules + f"Answer with one line of at most {L} calls, in exactly this form:\ncalls: <server>, <server>, ...")
        prefill = "calls:"
    else:
        system = (head + "There are twelve servers in four capability slots. " + rules +
                  "Answer with one line, in exactly this form:\nservers: <server>, <server>")
        prefill = "servers:"; L = BUDGET
    legend = "Servers:\n" + "\n".join(f"{s}: " + "; ".join(desc(n) for n in ns) for s, ns in SLOTS)
    ntok = [enc(" " + n) for n in names]; assert len({t[0] for t in ntok}) == len(ntok), "first tokens of the names must be distinct"
    SEP = enc(","); assert len(SEP) == 1; SEP = SEP[0]; STOP = [tok.convert_tokens_to_ids("<|eot_id|>"), enc("\n")[0]]
    assert enc(prefill + " grep, write_file, grep") == enc(prefill) + ntok[1] + [SEP] + ntok[9] + [SEP] + ntok[1]

    def conf_of(seq):
        c = [0] * 4
        for x in set(seq): c[slot[x]] = SERVER_LIST[x][1] + 1
        return CONF_IDX[tuple(c)]

    def transitions(state):
        kind, seq = state[0], state[1]
        if kind == "A":                                                  # a server name is expected
            out = []
            for x, toks in enumerate(ntok):
                bad = (x in seq and fmt == "list") or (x not in seq and (len(set(seq)) == BUDGET or slot[x] in {slot[y] for y in seq}))
                out.append((toks[0], ("leaf", -1) if bad else ("N", seq, x, 1) if len(toks) > 1 else ("S", seq + (x,)), True))
            return out
        if kind == "N":                                                  # inside a multi-token name
            x, k = state[2], state[3]; toks = ntok[x]
            return [(toks[k], ("N", seq, x, k + 1) if k + 1 < len(toks) else ("S", seq + (x,)), False)]
        return [(SEP, ("A", seq) if len(seq) < L else ("leaf", -1), True)] + [(s, ("leaf", conf_of(seq)), True) for s in STOP]

    def prompt_ids(job):
        user = f"Task: {tasks[job]['spec']}\n" + legend
        chat = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return enc(chat + prefill)
    return prompt_ids, ("A", ()), transitions, lambda state: len(state[1])


def enumerate_tree(start, transitions, n_dec):
    """Flat arrays.  Entries of a node are contiguous and end with its `anything else` entry (token -1, a failing leaf)."""
    N = dict(toks=[], t=[], dec=[], pe=[], depth=[]); E = dict(node=[], tok=[], child=[], conf=[], stage=[]); stack = [(start, (), -1, 0)]
    while stack:
        state, toks, pe, depth = stack.pop(); n = len(N["toks"]); tr = transitions(state)
        N["toks"].append(toks); N["t"].append(n_dec(state)); N["dec"].append(bool(tr[0][2])); N["pe"].append(pe); N["depth"].append(depth)
        if pe >= 0: E["child"][pe] = n
        for tok_id, nxt, _ in tr:
            e = len(E["node"]); E["node"].append(n); E["tok"].append(tok_id); E["child"].append(-1)
            if nxt[0] == "leaf": nxt = ("close", 0, nxt[1], None)
            if nxt[0] == "close":                                         # a finished answer (or line): stage, configuration, what follows
                E["stage"].append(nxt[1] if nxt[2] >= 0 else -1); E["conf"].append(nxt[2]); nxt = nxt[3]
            else: E["stage"].append(-1); E["conf"].append(-2)
            if nxt is not None: stack.append((nxt, toks + (tok_id,), e, depth + 1))
        E["node"].append(n); E["tok"].append(-1); E["child"].append(-1); E["conf"].append(-1); E["stage"].append(-1)
    return N, E


def run_name(args): return args.format + (str(args.L) if args.format == "plan" else "_" + "-".join(args.queue.split(",")) if args.format == "queue" else "")
def cache_name(args, job=None): return os.path.join(args.cache, f"_free_{run_name(args)}" + (f"_{job}.pt" if job else ".pt"))
def job_list(args): return ["queue"] if args.format == "queue" else JOBS if args.jobs == "all" else EASY if args.jobs == "easy" else args.jobs.split(",")
def units(args, strict=True):
    """(tree, job, label) for every prompt of a training step; --queue "a,b;c,d" is two queues, each with its own tree."""
    if args.format != "queue": W = Tree(args, job_list(args), strict); return [(W, j, j) for j in W.jobs]
    out = []
    for q in args.queue.split(";"):
        a = copy.copy(args); a.queue = q; out.append((Tree(a, ["queue"], strict), "queue", q))
    return out


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
    if args.format == "queue": prompt_ids, start, transitions, n_dec = make_queue(tok, servers, tasks, args.queue.split(","), reward_tables(True))
    else: prompt_ids, start, transitions, n_dec = make_format(args.format, tok, servers, tasks, args.L)
    N, E = enumerate_tree(start, transitions, n_dec); paths = N["toks"]
    inner = {p[:k] for p in paths for k in range(len(p))}; maximal = [p for p in paths if p not in inner]; where = {}
    for s, p in enumerate(maximal):
        for k in range(len(p) + 1): where.setdefault(p[:k], (s, k))
    node_s = np.array([where[p][0] for p in paths]); node_k = np.array([where[p][1] for p in paths]); Lmax = max(len(p) for p in maximal)
    print(f"[{args.format}] {len(paths)} nodes, {len(E['node'])} entries, {sum(N['dec'])} decision nodes, depth {max(N['depth'])}; "
          f"one forward over {len(maximal)} sequences of <= {Lmax} tokens", flush=True)
    os.makedirs(args.cache, exist_ok=True); common = os.path.join(args.cache, "_long_common.pt")
    if not os.path.exists(common): torch.save(dict(W=lm.lm_head.weight.detach().cpu()), common)
    torch.save(dict(N={k: v for k, v in N.items() if k != "toks"}, E=E, g=lm.model.norm.weight.detach().cpu(), eps=lm.model.norm.variance_epsilon,
                    A0=last.mlp.down_proj.lora_A["default"].weight.detach().cpu(), scaling=float(last.mlp.down_proj.scaling["default"])), cache_name(args))
    for job in (["queue"] if args.format == "queue" else JOBS):
        t0 = time.time(); prompt = prompt_ids(job); X = torch.zeros(len(paths), last.mlp.down_proj.in_features); V0 = torch.zeros(len(paths), base.config.hidden_size)
        with torch.no_grad():
            cache = lm.model(input_ids=torch.tensor([prompt[:-1]], device="cuda")).past_key_values
            for c0 in range(0, len(maximal), args.batch):
                chunk = maximal[c0:c0 + args.batch]; kv = copy.deepcopy(cache); kv.batch_repeat_interleave(len(chunk))
                ids = torch.tensor([prompt[-1:] + list(p) + [0] * (Lmax - len(p)) for p in chunk], device="cuda")
                lm.model(input_ids=ids, past_key_values=kv)
                v0 = saved["r"] + last.mlp.down_proj.base_layer(saved["x"])
                idx = np.flatnonzero((node_s >= c0) & (node_s < c0 + args.batch)); s_, k_ = torch.tensor(node_s[idx] - c0), torch.tensor(node_k[idx])
                X[idx] = saved["x"][s_, k_].cpu(); V0[idx] = v0[s_, k_].cpu()
        torch.save(dict(X=X, V0=V0), cache_name(args, job)); print(f"  {job:<10} {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- the policy and the exact accounting, level by level
class tf32:
    def __enter__(self): self.old = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = FAST
    def __exit__(self, *a): torch.backends.cuda.matmul.allow_tf32 = self.old
FAST = False


class Tree:
    shared = None
    def __init__(self, args, jobs, strict=True):
        dev = "cuda"; d = torch.load(cache_name(args), weights_only=False); N, E = d["N"], d["E"]; T = lambda a, dt=torch.long: torch.tensor(a, dtype=dt, device=dev)
        if Tree.shared is None: Tree.shared = torch.load(os.path.join(args.cache, "_long_common.pt"))["W"].to(dev)     # the 128k x 3072 output embedding, once
        self.W = Tree.shared; self.g, self.eps, self.s, self.A0 = d["g"].to(dev), d["eps"], d["scaling"], d["A0"].to(dev)
        self.N, self.E = len(N["pe"]), len(E["node"]); self.ent_node, tok_id, self.ent_child = T(E["node"]), T(E["tok"]), T(E["child"])
        self.is_other = tok_id < 0; self.ent_tok = tok_id.clamp_min(0); self.node_pe = T(N["pe"]); self.node_t = np.array(N["t"]); self.node_dec = np.array(N["dec"]); self.depth = np.array(N["depth"])
        self.conf = np.array(E["conf"]); self.stage = np.array(E.get("stage", [0 if c >= 0 else -1 for c in E["conf"]])); self.leaf_np = np.array(E["child"]) < 0; self.leaf = T(self.leaf_np, torch.bool); self.leaf_idx = np.flatnonzero(self.leaf_np)
        self.pe_np, self.en_np = np.array(N["pe"]), np.array(E["node"])
        cnt = np.bincount(E["node"], minlength=self.N); self.start_np = np.concatenate([[0], np.cumsum(cnt)]); self.start = T(self.start_np)
        depth = np.array(N["depth"]); self.levels = [T(np.flatnonzero(depth == k)) for k in range(depth.max() + 1)]
        self.pos = T(np.zeros(self.N, int));                                     # position of a node inside its level
        for lv in self.levels: self.pos[lv] = torch.arange(len(lv), device=dev)
        edepth = depth[np.array(E["node"])]; self.lev_ent = [T(np.flatnonzero(edepth == k)) for k in range(depth.max() + 1)]
        self.by_k = {int(k): T(np.flatnonzero(cnt == k)) for k in np.unique(cnt)}
        self.train = getattr(args, "train", "AB"); R = reward_tables(strict); self.jobs = jobs; self.X, self.V0, self.xx, self.f, self.phi = {}, {}, {}, {}, {}
        queue = args.queue.split(",") if args.format == "queue" else None; self.K = len(queue) if queue else 1
        self.w = np.eye(self.K)[-1] if getattr(args, "reward", "progress") == "product" else np.full(self.K, 1.0 / self.K)
        for j in jobs:
            dj = torch.load(cache_name(args, j)); self.X[j], self.V0[j] = dj["X"].to(dev), dj["V0"].to(dev); self.xx[j] = self.X[j].double().pow(2).sum(-1)
            self.phi[j] = np.stack([R[q] for q in queue]) if queue else R[j][None]; self.f[j] = self.totals(self.phi[j])

    def totals(self, phi, w=None):
        """reward of every leaf, sum_k w_k prod_{j<=k} phi_j, for a success table phi (K, configurations); w = None is the run's reward."""
        w = self.w if w is None else w; F, Rn = np.ones(self.N), np.zeros(self.N); out = np.zeros(self.E); order = np.argsort(self.depth, kind="stable")[1:]
        for lv in np.split(order, np.cumsum(np.bincount(self.depth)[1:])[:-1]):
            pe = self.pe_np[lv]; par = self.en_np[pe]; close = self.stage[pe] >= 0; ph = np.where(close, phi[np.maximum(self.stage[pe], 0), np.maximum(self.conf[pe], 0)], 1.0)
            F[lv] = F[par] * ph; Rn[lv] = Rn[par] + np.where(close, F[lv] * w[np.maximum(self.stage[pe], 0)], 0.0)
        e = self.leaf_idx; n = self.en_np[e]; close = self.stage[e] >= 0; k, c = np.maximum(self.stage[e], 0), np.maximum(self.conf[e], 0)
        out[e] = Rn[n] + np.where(close, F[n] * phi[k, c] * w[k], 0.0)
        return torch.tensor(out, dtype=torch.float64, device="cuda")

    def forward(self, job, A, B, chunk=2048):
        X, V0, en = self.X[job], self.V0[job], self.ent_node
        XA = X @ A.T; v = V0 + self.s * XA @ B.T; rms = (v.pow(2).mean(-1, keepdim=True) + self.eps).sqrt(); h = self.g * v / rms
        lp = torch.empty(self.E, dtype=torch.float64, device=X.device); a = torch.empty(self.E, v.shape[1], device=X.device)
        for n0 in range(0, self.N, chunk):
            n1 = min(self.N, n0 + chunk); e0, e1 = int(self.start_np[n0]), int(self.start_np[n1]); loc = en[e0:e1] - n0; oth = self.is_other[e0:e1]; tk = self.ent_tok[e0:e1]
            with tf32(): logp = torch.log_softmax(h[n0:n1] @ self.W.T, -1)
            ok = torch.zeros_like(logp, dtype=torch.bool); ok[loc[~oth], tk[~oth]] = True
            p = logp.exp(); lpo = logp.masked_fill(ok, -torch.inf).logsumexp(-1); q = p.masked_fill(ok, 0.0); q = q / q.sum(-1, keepdim=True).clamp_min(1e-30)
            with tf32(): pW, qW = p @ self.W, q @ self.W
            lp[e0:e1] = torch.where(oth, lpo[loc], logp[loc, tk]).double(); a[e0:e1] = torch.where(oth[:, None], qW[loc], self.W[tk]) - pW[loc]
            del logp, p, q, ok
        ga = self.g * a; ve = v[en]; u = ga / rms[en] - ve * (ve * ga).sum(-1, keepdim=True) / (v.shape[1] * rms[en] ** 3)      # through the RMSNorm
        P = lp.exp(); tot = torch.zeros(self.N, dtype=torch.float64, device=P.device).index_add_(0, en, P); P = P / tot[en]
        lpn = torch.zeros(self.N, dtype=torch.float64, device=P.device); logP = torch.log(P.clamp_min(1e-300))
        for lv in self.levels[1:]: pe = self.node_pe[lv]; lpn[lv] = lpn[en[pe]] + logP[pe]
        if self.train == "A": XA = torch.zeros_like(XA)                     # B is frozen: only A's half of every score counts
        return dict(job=job, P=P, pn=lpn.exp(), u=u, BU=u @ B, XA=XA, useA=bool(B.abs().sum() > 0))

    def values(self, st, f):
        """Q (E,) = expected reward through each branch, V (N,) = expected reward at each prefix."""
        Q = f.clone(); V = torch.zeros(self.N, dtype=torch.float64, device=f.device)
        for lv, le in zip(self.levels[::-1], self.lev_ent[::-1]):
            V.index_add_(0, self.ent_node[le], st["P"][le] * Q[le])
            if (self.node_pe[lv] >= 0).all(): Q[self.node_pe[lv]] = V[lv]
        return Q, V

    def _local(self, st, c):
        w = (st["P"] * c).float()[:, None]; z = lambda k: torch.zeros(self.N, k, device=w.device)
        return z(st["u"].shape[1]).index_add_(0, self.ent_node, w * st["u"]), z(st["BU"].shape[1]).index_add_(0, self.ent_node, w * st["BU"])

    def _outer(self, st, rows_u, rows_b, nodes):
        X = self.X[st["job"]]; DB = self.s * rows_u[:, :, None] * st["XA"][nodes][:, None, :]
        return DB, (self.s * rows_b[:, :, None] * X[nodes][:, None, :] if st["useA"] else None)

    def _dot(self, st, ents, DB, DA, rows, nodes):
        """<score of entry, D-vector>: the D-vector of entry i is row rows[i], and its node is nodes[rows[i]]."""
        y = torch.einsum("nij,nj->ni", DB, st["XA"][nodes]); out = (st["u"][ents] * y[rows]).sum(-1)
        if DA is not None: out = out + (st["BU"][ents] * torch.einsum("nij,nj->ni", DA, self.X[st["job"]][nodes])[rows]).sum(-1)
        return self.s * out.double()

    def topdown(self, st, f, tables=()):
        """sS (E,) = <s_e, S_node(e)>, and for every table c the exact E|g_hat|^2 of its estimator."""
        dev = f.device; sS = torch.zeros(self.E, dtype=torch.float64, device=dev); P, pn, en = st["P"], st["pn"], self.ent_node
        ss = self.s ** 2 * ((st["u"].double().pow(2).sum(-1)) * st["XA"].double().pow(2).sum(-1)[en] + st["BU"].double().pow(2).sum(-1) * self.xx[st["job"]][en])
        loc = [self._local(st, c) for c in tables]; second = [0.0 for _ in tables]; S = R = None
        for d, (lv, le) in enumerate(zip(self.levels, self.lev_ent)):
            rows = self.pos[en[le]]
            if d == 0:
                S = self._outer(st, torch.zeros(1, st["u"].shape[1], device=dev), torch.zeros(1, st["BU"].shape[1], device=dev), lv); R = [tuple(None if x is None else x.clone() for x in S) for _ in tables]
            sS[le] = self._dot(st, le, S[0], S[1], rows, lv)
            for k, c in enumerate(tables):
                m = self._outer(st, loc[k][0][lv], loc[k][1][lv], lv); R[k] = tuple(None if x is None else x + y for x, y in zip(R[k], m))     # R' = R + m_n(c)
                sR = self._dot(st, le, R[k][0], R[k][1], rows, lv); ip = lambda a, b: sum((x.double() * y.double()).flatten(1).sum(-1) for x, y in zip(a, b) if x is not None)
                SS, SR, RR = ip(S, S)[rows], ip(S, R[k])[rows], ip(R[k], R[k])[rows]; fl, cl = f[le], c[le]
                sq = fl ** 2 * SS + 2 * fl * SR + RR + 2 * (fl - cl) * (fl * sS[le] + sR) + (fl - cl) ** 2 * ss[le]
                second[k] = second[k] + float((pn[en[le]] * P[le] * sq)[self.leaf[le]].sum())
            if d + 1 < len(self.levels):
                ch = self.levels[d + 1]; pe = self.node_pe[ch]; par = en[pe]; prow = self.pos[par]; step = self._outer(st, st["u"][pe], st["BU"][pe], par)
                for k, c in enumerate(tables): R[k] = tuple(None if x is None else x[prow] - c[pe].float()[:, None, None] * y for x, y in zip(R[k], step))
                S = tuple(None if x is None else x[prow] + y for x, y in zip(S, step))
        return sS, second

    def delta(self, st, f, solver="ridge", sS=None):
        """The variance-minimizing residual for reward table f on top of Q: Delta (E,), and the noise it removes."""
        Q, V = self.values(st, f); P, en = st["P"], self.ent_node; A = Q - V[en]
        if sS is None: sS, _ = self.topdown(st, f)
        mu, mb = self._local(st, Q); t1 = torch.zeros(self.E, dtype=torch.float64, device=f.device); t2 = torch.zeros_like(t1); Dc = None
        for d in range(len(self.levels) - 1, -1, -1):
            lv, le = self.levels[d], self.lev_ent[d]; Dbar = self._outer(st, torch.zeros(len(lv), mu.shape[1], device=mu.device), torch.zeros(len(lv), mb.shape[1], device=mu.device), lv)
            if Dc is not None:
                ch = self.levels[d + 1]; pe = self.node_pe[ch]; prow = self.pos[en[pe]]
                t1[pe] = self._dot(st, pe, Dc[0], Dc[1], torch.arange(len(ch), device=mu.device), en[pe])
                for x, y in zip(Dbar, Dc):
                    if x is not None: x.index_add_(0, prow, y * P[pe].float()[:, None, None])
            t2[le] = self._dot(st, le, Dbar[0], Dbar[1], self.pos[en[le]], lv)
            m = self._outer(st, mu[lv], mb[lv], lv); Dc = tuple(None if x is None else x + y for x, y in zip(Dbar, m))
        b = -P * (A * sS + t1 - t2); D = torch.zeros_like(b); gain = 0.0; X2 = self.xx[st["job"]]; xa2 = st["XA"].double().pow(2).sum(-1)
        for k, nodes in self.by_k.items():
            for c0 in range(0, len(nodes), 1024):
                nd = nodes[c0:c0 + 1024]; idx = self.start[nd][:, None] + torch.arange(k, device=nd.device); U = st["u"][idx].double(); BU = st["BU"][idx].double(); Pk = P[idx]
                K = self.s ** 2 * (U @ U.transpose(1, 2) * xa2[nd][:, None, None] + BU @ BU.transpose(1, 2) * X2[nd][:, None, None])
                C = K * (torch.diag_embed(Pk) - Pk[:, :, None] * Pk[:, None, :]); bk = b[idx]
                if solver == "pinv": x = -(torch.linalg.pinv(C, hermitian=True, rtol=1e-8) @ bk[:, :, None])[:, :, 0]
                else:
                    lam = 1e-9 * C.diagonal(dim1=1, dim2=2).mean(-1) + 1e-14 * K.diagonal(dim1=1, dim2=2).mean(-1) + 1e-300
                    try: x = -torch.linalg.solve(C + lam[:, None, None] * torch.eye(k, dtype=C.dtype, device=C.device), bk)
                    except Exception: x = torch.zeros_like(bk)
                x = torch.where(torch.isfinite(x).all(-1, keepdim=True), x, 0.0); D[idx] = x; gain = gain + float((st["pn"][nd] * (-(bk * x).sum(-1))).sum())
        return Q, D, gain

    def rowwise(self, st, f):
        """Noise of the Q estimator split by prefix, and the part a ROW-WISE residual removes.  The martingale increment of
        the Q estimator at prefix n, branch v, is xi_v = A_v S_n + D_v - Dbar_n (a matrix shaped like the weights).  The score of
        every token in one weight row is a multiple of the layer input (z_n = A x_n for B's rows, x_n for A's rows), so a
        residual with one scalar per (prefix, token, row) cancels exactly the component of xi_v along that input."""
        Q, V = self.values(st, f); P, en, pn = st["P"], self.ent_node, st["pn"]; Adv = (Q - V[en]).float(); mu, mb = self._local(st, Q); X = self.X[st["job"]]
        S = [None] * len(self.levels); dev = mu.device
        for d, lv in enumerate(self.levels):
            if d == 0: S[0] = self._outer(st, torch.zeros(1, mu.shape[1], device=dev), torch.zeros(1, mb.shape[1], device=dev), lv)
            if d + 1 < len(self.levels):
                ch = self.levels[d + 1]; pe = self.node_pe[ch]; par = en[pe]; step = self._outer(st, st["u"][pe], st["BU"][pe], par)
                S[d + 1] = tuple(None if x is None else x[self.pos[par]] + y for x, y in zip(S[d], step))
        total = gain = 0.0; Dc = None
        for d in range(len(self.levels) - 1, -1, -1):
            lv, le = self.levels[d], self.lev_ent[d]; rows = self.pos[en[le]]
            Dbar = self._outer(st, torch.zeros(len(lv), mu.shape[1], device=dev), torch.zeros(len(lv), mb.shape[1], device=dev), lv)
            xi = [None if x is None else Adv[le][:, None, None] * x[rows] for x in S[d]]                       # A_v S_n
            if Dc is not None:
                ch = self.levels[d + 1]; pe = self.node_pe[ch]; prow = self.pos[en[pe]]; where = torch.full((self.E,), -1, device=dev, dtype=torch.long); where[le] = torch.arange(len(le), device=dev)
                for k, (x, y) in enumerate(zip(Dbar, Dc)):
                    if x is not None: x.index_add_(0, prow, y * P[pe].float()[:, None, None]); xi[k].index_add_(0, where[pe], y)       # + D_v
            xi = [None if x is None else x - y[rows] for x, y in zip(xi, Dbar)]                                # - Dbar_n
            w = (pn[en[le]] * P[le]); zn = torch.nn.functional.normalize(st["XA"][lv], dim=-1)[rows]; xn = torch.nn.functional.normalize(X[lv], dim=-1)[rows]
            total += float((w * xi[0].double().pow(2).flatten(1).sum(-1)).sum()); gain += float((w * torch.einsum("eij,ej->ei", xi[0], zn).double().pow(2).sum(-1)).sum())
            if xi[1] is not None:
                total += float((w * xi[1].double().pow(2).flatten(1).sum(-1)).sum()); gain += float((w * torch.einsum("eij,ej->ei", xi[1], xn).double().pow(2).sum(-1)).sum())
            m = self._outer(st, mu[lv], mb[lv], lv); Dc = tuple(None if x is None else x + y for x, y in zip(Dbar, m))
        return total, gain

    def row_correction(self, st, f, path):
        """The row-wise residual of one episode, as (dA, dB) to subtract from the Q estimator: at every prefix n of the path, the
        component along the layer input of xi_v = A_v S_n + D_v - Dbar_n for the branch v that was taken (f: the reward table used)."""
        Q, V = self.values(st, f); P, en = st["P"], self.ent_node; mu, mb = self._local(st, Q); X = self.X[st["job"]]; dev = mu.device
        need = {int(self.en_np[e]): e for e in path}; got = {}; Dc = None                        # node on the path -> entry taken there
        for d in range(len(self.levels) - 1, -1, -1):
            lv = self.levels[d]; Dbar = self._outer(st, torch.zeros(len(lv), mu.shape[1], device=dev), torch.zeros(len(lv), mb.shape[1], device=dev), lv)
            if Dc is not None:
                ch = self.levels[d + 1]; pe = self.node_pe[ch]; prow = self.pos[en[pe]]
                for x, y in zip(Dbar, Dc):
                    if x is not None: x.index_add_(0, prow, y * P[pe].float()[:, None, None])
                for n, e in need.items():
                    c = int(self.ent_child[e])
                    if self.depth[n] == d and c >= 0: got[n] = [None if y is None else y[int(self.pos[c])].clone() for y in Dc]      # D_v of the taken branch
            for n, e in need.items():
                if self.depth[n] == d:
                    r = int(self.pos[n]); got[n] = [None if x is None else (got[n][k] if n in got else 0.0) - x[r] for k, x in enumerate(Dbar)]   # D_v - Dbar_n
            m = self._outer(st, mu[lv], mb[lv], lv); Dc = tuple(None if x is None else x + y for x, y in zip(Dbar, m))
        cA = torch.zeros(st["BU"].shape[1], X.shape[1], device=dev); cB = torch.zeros(st["u"].shape[1], st["XA"].shape[1], device=dev)
        SB = torch.zeros_like(cB); SA = torch.zeros_like(cA)
        for e in path[::-1]:                                                                  # from the root down
            n = int(self.en_np[e]); adv = float(Q[e] - V[n]); zn = torch.nn.functional.normalize(st["XA"][n], dim=0); xn = torch.nn.functional.normalize(X[n], dim=0)
            xiB = adv * SB + got[n][0]; cB += torch.outer(xiB @ zn, zn)
            if got[n][1] is not None: xiA = adv * SA + got[n][1]; cA += torch.outer(xiA @ xn, xn)
            SB += self.s * torch.outer(st["u"][e], st["XA"][n]); SA += self.s * torch.outer(st["BU"][e], X[n])
        return cA, cB

    def true_grad(self, st, f):
        Q, _ = self.values(st, f); mu, mb = self._local(st, Q); w = st["pn"].float()[:, None]
        return self.s * (mb * w).T @ self.X[st["job"]], self.s * (mu * w).T @ st["XA"]

    def noise(self, st, f, tables):
        _, second = self.topdown(st, f, tables); gA, gB = self.true_grad(st, f); g2 = float(gA.double().pow(2).sum() + gB.double().pow(2).sum())
        return [s2 - g2 for s2 in second], g2

    def sample(self, st, rng):
        pl = (st["pn"][self.ent_node] * st["P"])[self.leaf].cpu().numpy(); e = int(self.leaf_idx[rng.choice(len(pl), p=pl / pl.sum())]); path = [e]
        while self.pe_np[self.en_np[path[-1]]] >= 0: path.append(int(self.pe_np[self.en_np[path[-1]]]))
        return e, path

    def episode_grad(self, st, c, f_leaf, path):
        """one-sample estimator: sum over the path of (f - c_e) s_e + sum_v P_v c_v s_v, as (dA, dB)."""
        ents = torch.cat([torch.arange(int(self.start_np[n]), int(self.start_np[n + 1]), device="cuda") for n in self.ent_node[path].tolist()])
        w = st["P"][ents] * c[ents]; taken = torch.isin(ents, torch.tensor(path, device="cuda")); w = torch.where(taken, w + f_leaf - c[ents], w).float()[:, None]
        n = self.ent_node[ents]
        return self.s * (st["BU"][ents] * w).T @ self.X[st["job"]][n], self.s * (st["u"][ents] * w).T @ st["XA"][n]

    def grad_w(self, st, idx, w):
        """(dA, dB) = sum_e w_e s_e over the entries idx; linear in w (so differentiable through it)."""
        n = self.ent_node[idx]; w = w.float()[:, None]
        return self.s * (st["BU"][idx] * w).T @ self.X[st["job"]][n], self.s * (st["u"][idx] * w).T @ st["XA"][n]

    def score_norms(self, st):
        """|s_e|^2 for every entry (E,)."""
        en = self.ent_node
        return self.s ** 2 * (st["u"].double().pow(2).sum(-1) * st["XA"].double().pow(2).sum(-1)[en] + st["BU"].double().pow(2).sum(-1) * self.xx[st["job"]][en])

    def psb_exact(self, st, f):
        """the per-parameter optimal constant baseline b_i = E[f S_i^2] / E[S_i^2] (Greensmith et al. 2004; Peters & Schaal 2008),
        S = the whole episode's score, computed exactly at the current policy; returned as (b_A, b_B)."""
        en, P, pn = self.ent_node, st["P"], st["pn"]; X = self.X[st["job"]]; dev = X.device; acc = [None, None, None, None]; S = None
        for d, (lv, le) in enumerate(zip(self.levels, self.lev_ent)):
            if d == 0: S = self._outer(st, torch.zeros(1, st["u"].shape[1], device=dev), torch.zeros(1, st["BU"].shape[1], device=dev), lv)
            leaf = le[self.leaf[le]]
            if len(leaf):
                rows = self.pos[en[leaf]]; own = self._outer(st, st["u"][leaf], st["BU"][leaf], en[leaf]); w = (pn[en[leaf]] * P[leaf]).float()[:, None, None]; fw = w * f[leaf].float()[:, None, None]
                for k in range(2):
                    if S[k] is None: continue
                    sq = (S[k][rows] + own[k]).pow(2); acc[k] = (0 if acc[k] is None else acc[k]) + (fw * sq).sum(0); acc[2 + k] = (0 if acc[2 + k] is None else acc[2 + k]) + (w * sq).sum(0)
            if d + 1 < len(self.levels):
                ch = self.levels[d + 1]; pe = self.node_pe[ch]; par = en[pe]; step = self._outer(st, st["u"][pe], st["BU"][pe], par)
                S = tuple(None if x is None else x[self.pos[par]] + y for x, y in zip(S, step))
        b = [None if acc[k] is None else acc[k] / acc[2 + k].clamp_min(1e-30) for k in range(2)]      # (B-shaped, A-shaped)
        return (torch.zeros(st["BU"].shape[1], X.shape[1], device=dev) if b[1] is None else b[1]), b[0]

    def relax_batch(self, st, episodes, name_first):
        """what RELAX needs for a group of episodes: the entries of every node on the way (K, T, A), the branch taken (K, T),
        their log-probabilities (K, T, A) and a state per node (depth, line, servers named so far)."""
        K = len(episodes); T = max(len(p) for _, p in episodes); A = max(self.by_k); dev = st["P"].device
        ent = torch.full((K, T, A), -1, dtype=torch.long); b = torch.zeros(K, T, dtype=torch.long); state = torch.zeros(K, T, 64 + 8 + 12)
        for i, (_, path) in enumerate(episodes):
            named = torch.zeros(12)
            for t, e in enumerate(path[::-1]):
                n = int(self.en_np[e]); e0, e1 = int(self.start_np[n]), int(self.start_np[n + 1]); ent[i, t, :e1 - e0] = torch.arange(e0, e1); b[i, t] = e - e0
                state[i, t, min(int(self.depth[n]), 63)] = 1; state[i, t, 64 + min(int(self.node_t[n]), 7)] = 1; state[i, t, 72:] = named
                tk = int(self.ent_tok[e]) if not bool(self.is_other[e]) else -1
                if tk in name_first: named = named.clone(); named[name_first[tk]] = 1
        ent, b, state = ent.to(dev), b.to(dev), state.to(dev)
        logp = torch.log(st["P"].clamp_min(1e-300))[ent.clamp_min(0)].float()
        return ent, b, logp, state

    def alignment(self, st, rng, pairs=400, weighted=True):
        """fraction of |s_n(v)|^2 inside the span of another decision prefix's scores (pairs at different decision index)."""
        dec = np.flatnonzero(self.node_dec); w = st["pn"][dec].cpu().numpy() if weighted else np.ones(len(dec)); w = w / w.sum(); fr, wt = [], []; X = self.X[st["job"]]
        for _ in range(pairs):
            n, m = rng.choice(dec, 2, p=w)
            if self.depth[n] == self.depth[m]: continue
            en_ = torch.arange(int(self.start_np[n]), int(self.start_np[n + 1]) - 1, device="cuda"); em = torch.arange(int(self.start_np[m]), int(self.start_np[m + 1]) - 1, device="cuda")
            G = lambda a, b, na, nb: self.s ** 2 * ((st["u"][a].double() @ st["u"][b].double().T) * float(st["XA"][na].double() @ st["XA"][nb].double())
                                                    + (st["BU"][a].double() @ st["BU"][b].double().T) * float(X[na].double() @ X[nb].double()))
            Knm, Kmm, Knn = G(en_, em, n, m), G(em, em, m, m), G(en_, en_, n, n).diagonal()
            frac = ((Knm @ torch.linalg.pinv(Kmm, hermitian=True, rtol=1e-8)) * Knm).sum(-1) / Knn.clamp_min(1e-30); pw = st["P"][en_] if weighted else torch.ones_like(frac)
            fr.append(float((frac * pw).sum() / pw.sum())); wt.append(1.0)
        return float(np.average(fr, weights=wt))


# ---------------------------------------------------------------- stats and training
def stats(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu; U = units(args, strict=not args.lenient); A, B = init_lora(U[0][0], args)
    tot = collections.Counter(); rng = np.random.default_rng(0)
    for W, j, label in U:
        t0 = time.time(); st = W.forward(j, A, B); f = W.f[j]; t1 = time.time(); Q, D, gain = W.delta(st, f, args.solver); t2 = time.time()
        Ef = float(W.values(st, f)[1][0]); (rf, nq, nqd), g2 = W.noise(st, f, [torch.full_like(f, Ef), Q, Q + D]); t3 = time.time()
        pleaf = (st["pn"][W.ent_node] * st["P"]); fail = float(pleaf[torch.tensor(W.leaf_np & (W.conf == -1), device="cuda")].sum())
        ndec = float((st["pn"] * torch.tensor(W.node_dec, device="cuda")).sum())
        done = [float(W.values(st, W.totals(W.phi[j], np.eye(W.K)[k]))[1][0]) for k in range(W.K)]
        rw_total, rw_gain = W.rowwise(st, f) if args.rowwise else (nq, 0.0); tot["RW"] += nq * (1 - rw_gain / rw_total)
        for k, v in dict(rf=rf, Q=nq, QD=nqd).items(): tot[k] += v
        if args.rowwise: print(f"   row-wise residual: increments sum to {rw_total / nq:.4f} of the Q noise (should be 1); Q + row-wise Δ = {10 * (nq / rf) * (1 - rw_gain / rw_total):.2f}, ratio to Q {1 - rw_gain / rw_total:.3f}")
        print(f"{label:<10} [{W.N} nodes {W.E} entries] P(valid) {1 - fail:.3f} E[reward] {Ef:.4f} decisions/episode {ndec:.1f} align {W.alignment(st, rng):.2f}/{W.alignment(st, rng, weighted=False):.2f}  rf 10  "
              f"Q {10*nq/rf:5.2f}  Q+Δ {10*nqd/rf:5.2f}  Δ/Q {nqd/nq:.3f}  (check {(nq-gain)/nq:.3f})  SNR {np.sqrt(g2/nq):.2f}->{np.sqrt(g2/max(nqd,1e-300)):.2f}  "
              f"P(done>=k) " + " ".join(f"{x:.4f}" for x in done) + f"   [{t1-t0:.1f}s {t2-t1:.1f}s {t3-t2:.1f}s]", flush=True)
    print(f"POOLED rf 10  Q {10*tot['Q']/tot['rf']:.2f}  Q+Δ {10*tot['QD']/tot['rf']:.2f}  Δ/Q {tot['QD']/tot['Q']:.3f}" + (f"   row-wise Δ/Q {tot['RW']/tot['Q']:.3f}" if args.rowwise else ""))


def init_lora(W, args):
    """--init A: the usual LoRA start (A random, B = 0: only B has a gradient, scores = d log p/dh (x) A x).
       --init B: the mirrored start (B random, A = 0: only A has a gradient, scores = B' d log p/dh (x) x, rank --r on the token side)."""
    d = W.V0[W.jobs[0]].shape[1]; g = torch.Generator(device="cuda").manual_seed(args.lora_seed)
    if args.init == "A": return W.A0.clone(), torch.zeros(d, W.A0.shape[0], device="cuda")
    return torch.zeros(args.r, W.A0.shape[1], device="cuda"), torch.randn(d, args.r, device="cuda", generator=g) / d ** 0.5


sys.path.insert(0, os.path.dirname(HERE)); from baselines import group_weights, relax_weights, Surrogate
GROUP = ("RLOO", "GRPO", "OTB"); NO_TABLE = GROUP + ("RELAX", "DR", "DM", "PSB", "PSB (exact)", "exact gradient")


def control_variate(method, W, st, seen, hist, f_true, solver):
    """the per-token table c(prefix, token) a method uses, or None for the methods that are not of that form.
    Fitted tables come from the run's own episodes with the prior untried = fails (the "0" of Q0)."""
    if method in NO_TABLE: return None
    if method == "REINFORCE": return torch.full_like(f_true, np.mean(hist) if hist else 0.0)
    f = f_true if "exact" in method else W.totals(np.stack([sn.f_hat(zero=True) for sn in seen]))
    if method.startswith("V"): return W.values(st, f)[1][W.ent_node]                     # the state-value baseline: c independent of the token
    if "rowΔ" in method: return W.values(st, f)[0], f                                  # the Q table, and the reward table the row-wise residual is built from
    if "Δ" in method: Q, D, _ = W.delta(st, f, solver); return Q + D
    return W.values(st, f)[0]


def step_grad(method, W, st, f, c, seen, episodes, extra, key, args):
    """the gradient estimate (dA, dB) of one step from a group of episodes [(leaf entry, path)]; c = the method's table."""
    dev = st["P"].device; K = len(episodes); Z = lambda: (torch.zeros(st["BU"].shape[1], W.X[st["job"]].shape[1], device=dev), torch.zeros(st["u"].shape[1], st["XA"].shape[1], device=dev))
    add = lambda a, b, w=1.0: (a[0] + w * b[0], a[1] + w * b[1]); g = Z()
    if c is not None:
        f_row = None
        if isinstance(c, tuple): c, f_row = c
        for e, pth in episodes:
            dA, dB = W.episode_grad(st, c, f[e], pth)
            if f_row is not None: cA, cB = W.row_correction(st, f_row, pth); dA, dB = dA - cA, dB - cB
            g = add(g, (dA, dB), 1.0 / K)
        return g
    if method in ("DR", "DM", "exact gradient"):                                          # model-based: the exact gradient of the reward table ...
        fh = f if method == "exact gradient" else W.totals(np.stack([sn.f_hat(zero=True) for sn in seen])); g = W.true_grad(st, fh)
        if method == "DR":                                                                # ... plus the doubly-robust correction (f - f_hat) grad log p(y)
            for e, pth in episodes: g = add(g, W.grad_w(st, torch.tensor(pth, device=dev), torch.full((len(pth),), float(f[e] - fh[e]), device=dev)), 1.0 / K)
        return g
    if method.startswith("PSB"):                                                         # per-parameter constant baseline, (f - b_i) S_i
        if method == "PSB (exact)": bA, bB = W.psb_exact(st, f)
        else:
            r = extra.psb.get(key); bA, bB = (Z() if r is None else (r[0] / r[2].clamp_min(1e-30), r[1] / r[3].clamp_min(1e-30)))
        for e, pth in episodes:
            SA, SB = W.grad_w(st, torch.tensor(pth, device=dev), torch.ones(len(pth), device=dev)); g = add(g, (float(f[e]) * SA - bA * SA, float(f[e]) * SB - bB * SB), 1.0 / K)
            if method == "PSB":                                                           # running sums, updated after use so that b never sees the episode it corrects
                r = extra.psb.setdefault(key, [torch.zeros_like(SA), torch.zeros_like(SB), torch.zeros_like(SA), torch.zeros_like(SB)])
                r[0] += float(f[e]) * SA ** 2; r[1] += float(f[e]) * SB ** 2; r[2] += SA ** 2; r[3] += SB ** 2
        return g
    if method in GROUP:
        T = max(len(p) for _, p in episodes); pos = torch.full((K, T), -1, dtype=torch.long, device=dev)
        for i, (_, pth) in enumerate(episodes): pos[i, :len(pth)] = torch.tensor(pth[::-1], device=dev)
        fk = torch.stack([f[e] for e, _ in episodes]); w = group_weights(method, fk, (pos >= 0).double(), pos, W.score_norms(st)); nz = w.nonzero().squeeze(1)
        return W.grad_w(st, nz, w[nz])
    if method == "RELAX":
        ent, b, logp, state = W.relax_batch(st, episodes, extra.name_first)
        if key not in extra.surr: extra.surr[key] = Surrogate(state.shape[-1] + ent.shape[-1]).to(dev).double(); extra.surr[key].opt = torch.optim.Adam(extra.surr[key].parameters(), lr=1e-2)
        cphi = extra.surr[key]; fk = torch.stack([f[e] for e, _ in episodes]).double()         # float64: the double backward through exp(-logp)^2 overflows float32 once a branch is below e^-44
        w = relax_weights(cphi, logp.double(), ent, b, fk, W.E, state.double()); nz = (w.detach() != 0).nonzero().squeeze(1)
        gA, gB = W.grad_w(st, nz, w[nz]); loss = gA.pow(2).sum() + gB.pow(2).sum()          # fit phi on |g_hat|^2, through the factored score norm
        cphi.opt.zero_grad(); loss.backward(); cphi.opt.step()
        return gA.detach(), gB.detach()
    raise ValueError(method)


class Extra:
    def __init__(self, methods):
        self.surr, self.psb, self.name_first = {}, {}, {}
        if "RELAX" in methods:
            from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(MODEL)
            self.name_first = {tok.encode(" " + n, add_special_tokens=False)[0]: i for i, (_, _, n) in enumerate(SERVER_LIST)}


def train(args):
    global FAST
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu; FAST = True; U = units(args); nU = len(U); os.makedirs(args.out, exist_ok=True)
    tag = args.name or (run_name(args).replace(";", "+") + ("_product" if args.reward == "product" else "") + (f"_B{args.r}{args.train}" if args.init == "B" else ""))
    for method in args.methods.split(","):
        for seed in map(int, args.seeds.split(",")):
            path = os.path.join(args.out, f"{tag}_{nU}jobs_lr{args.lr:g}_{method.replace(' ', '').replace('+', 'p').replace('(', '_').replace(')', '')}_s{seed}.npz")
            if os.path.exists(path): continue
            t0 = time.time(); rng = np.random.default_rng(seed); torch.manual_seed(seed); A, B = init_lora(U[0][0], args); opt = torch.optim.Adam([A, B] if args.train == "AB" else [A], lr=args.lr)
            seen = [[Seen() for _ in range(W.K)] for W, _, _ in U]; hist = [[] for _ in U]; succ, done, diag = [], [], []; extra = Extra(args.methods)
            for step in range(args.steps + 1):
                gA, gB, s_now, d_now = 0.0, 0.0, [], []; d = collections.Counter(); log_now = step % args.log_every == 0
                for i, (W, j, _) in enumerate(U):
                    st = W.forward(j, A, B); f = W.f[j]; s_now.append(float(W.values(st, f)[1][0]))
                    c = control_variate(method, W, st, seen[i], hist[i], f, args.solver); c_tab = c[0] if isinstance(c, tuple) else c
                    if log_now:                                          # exact diagnostics at the current policy
                        sS, _ = W.topdown(st, f); Qx, Dx, _ = W.delta(st, f, args.solver, sS); rw_total, rw_gain = W.rowwise(st, f)
                        (rf, own, nq, nqd), _ = W.noise(st, f, [torch.full_like(f, s_now[-1]), Qx if c_tab is None else c_tab, Qx, Qx + Dx])
                        d["rf"] += rf; d["own"] += own if (c_tab is not None and not isinstance(c, tuple)) else float("nan"); d["Q"] += nq; d["QD"] += nqd; d["QRW"] += nq * (1 - rw_gain / max(rw_total, 1e-300)); tA, tB = W.true_grad(st, f); d["gA"] = d["gA"] + tA / nU; d["gB"] = d["gB"] + tB / nU
                        d_now.append([float(W.values(st, W.totals(W.phi[j], np.eye(W.K)[k]))[1][0]) for k in range(W.K)])
                    if step < args.steps:
                        episodes = [W.sample(st, rng) for _ in range(args.episodes)]
                        dA, dB = step_grad(method, W, st, f, c, seen[i], episodes, extra, i, args); gA = gA + dA / nU; gB = gB + dB / nU
                        for e, pth in episodes:                              # the reward table and the REINFORCE mean see the episodes only after the step
                            hist[i].append(float(f[e]))
                            for x in pth:
                                if W.stage[x] >= 0: seen[i][W.stage[x]].add(W.conf[x], float(W.phi[j][W.stage[x], W.conf[x]]))
                    del st
                succ.append(s_now)
                if log_now:
                    g2 = float(d["gA"].pow(2).sum() + (d["gB"].pow(2).sum() if args.train == "AB" else 0.0)); diag.append([step, d["rf"], d["own"], d["Q"], d["QD"], g2, nU ** 2 * args.episodes, d["QRW"]]); done.append(d_now)
                if step < args.steps:
                    opt.zero_grad(); A.grad = -gA
                    if args.train == "AB": B.grad = -gB
                    opt.step()
            np.savez(path, success=np.array(succ), diag=np.array(diag), jobs=np.array([l for _, _, l in U]), done=np.array(done, dtype=object) if len({len(x) for y in done for x in y}) > 1 else np.array(done))
            print(f"{method:<14} seed {seed}: mean reward {np.mean(succ[0]):.3f} -> {np.mean(succ[-1]):.3f}   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("prep", "stats", "train"))
    ap.add_argument("--format", default="queue", choices=("queue", "plan", "list", "slots", "manifest")); ap.add_argument("--L", type=int, default=6); ap.add_argument("--gpu", default="0")
    ap.add_argument("--queue", default="defines,lint,port,digest", help="the tasks of the queue, in order"); ap.add_argument("--reward", default="progress", choices=("progress", "product"))
    ap.add_argument("--init", default="A", choices=("A", "B")); ap.add_argument("--r", type=int, default=8); ap.add_argument("--train", default="AB", choices=("AB", "A")); ap.add_argument("--name", default=""); ap.add_argument("--rowwise", action="store_true"); ap.add_argument("--lora_seed", type=int, default=0); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lenient", action="store_true"); ap.add_argument("--jobs", default="all")
    ap.add_argument("--methods", default="REINFORCE,Q0,Q0 + Δ,Q0 + rowΔ,Q (exact),Q + Δ (exact),Q + rowΔ (exact)",
                    help="also: V0, V (exact), DR, DM, exact gradient, PSB, PSB (exact), RLOO, GRPO, OTB (need --episodes >= 2), RELAX"); ap.add_argument("--seeds", default="0"); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--solver", default="ridge", choices=("pinv", "ridge")); ap.add_argument("--out", default=os.path.join(HERE, "_free_runs")); ap.add_argument("--cache", default="/tmp/zzhao628_free")
    args = ap.parse_args(); {"prep": prep, "stats": stats, "train": train}[args.cmd](args)
