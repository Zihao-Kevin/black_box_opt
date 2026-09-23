"""Exact algebra extracted from Zihao-Kevin/black_box_opt.
Source: 33ffc0ab4d82c86e84dac722f5c71662d8a06dfe, mcp_dfl_prototype/live_free.py Tree.
JobShop policy/cache/tree construction live in row_tree.py. Device literals generalized for CPU algebra tests; formulas unchanged.
"""
import numpy as np
import torch

class TreeMath:
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
        ents = torch.cat([torch.arange(int(self.start_np[n]), int(self.start_np[n + 1]), device=self.device) for n in self.ent_node[path].tolist()])
        w = st["P"][ents] * c[ents]; taken = torch.isin(ents, torch.tensor(path, device=self.device)); w = torch.where(taken, w + f_leaf - c[ents], w).float()[:, None]
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

