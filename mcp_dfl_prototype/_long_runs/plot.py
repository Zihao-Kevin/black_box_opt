import numpy as np, os, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import R, HERE
style = {"REINFORCE": ("C3", "-"), "Q": ("C1", ":"), "Q + Δ": ("C0", ":"), "Q0": ("C1", "-"), "Q0 + Δ": ("C0", "-"), "Q (exact)": ("C1", "--"), "Q + Δ (exact)": ("C0", "--")}
label = {"Q": "Q (fitted, prior = tried mean)", "Q + Δ": "Q + Δ (fitted, prior = tried mean)", "Q0": "Q (fitted, untried = fails)", "Q0 + Δ": "Q + Δ (fitted, untried = fails)"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2)); eps = np.arange(201) * 10
for m, (c, ls) in style.items():
    A = np.stack([R[m][s][0].mean(1) for s in range(40)]); mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(40)
    ax[0].plot(eps, mu, color=c, ls=ls, lw=2, label=label.get(m, m)); ax[0].fill_between(eps, mu - se, mu + se, color=c, alpha=0.10)
ax[0].set(xlabel="agent episodes", ylabel="mean P(success) over the 10 jobs (exact)", ylim=(0, 0.62), title="12 decisions per episode, 10 jobs, 40 seeds (band = ±1 s.e.)"); ax[0].legend(fontsize=7.5, loc="upper right", ncol=2)
def ratio_curve(pattern, n):
    out = []
    for s in range(n):
        d = np.load(pattern.format(s))["diag"]; live = d[:, 1] > 1e-9 * d[0, 1]; out.append(np.where(live & (d[:, 3] > 0), d[:, 4] / np.maximum(d[:, 3], 1e-300), np.nan))
    out = np.array(out); return np.load(pattern.format(0))["diag"][:, 0] * 10, np.exp(np.nanmean(np.log(out), 0)), (~np.isnan(out)).sum(0)
for pat, n, c, lab in [(os.path.join(HERE, "manifest_10jobs_lr0.001_QpΔ_exact_s{}.npz"), 40, "C0", "12 decisions (manifest), lr 1e-3"), (os.path.join(HERE, "manifest_10jobs_lr0.0003_QpΔ_exact_s{}.npz"), 20, "C2", "12 decisions (manifest), lr 3e-4"),
                       (os.path.join(HERE, "slots_10jobs_lr0.001_QpΔ_exact_s{}.npz"), 20, "C1", "4 decisions (slots), lr 1e-3")]:
    x, y, k = ratio_curve(pat, n); ok = k >= 5; ax[1].plot(x[ok], y[ok], color=c, lw=2, marker="o", ms=3, label=lab)
ax[1].axhline(1, color="k", lw=0.5); ax[1].set(xlabel="agent episodes", ylabel="exact noise of Q + Δ*  /  exact noise of Q", ylim=(0, 1.05), xlim=(0, 1200), title="Δ's share of the noise at the policies visited in training"); ax[1].legend(fontsize=8)
names = ["REINFORCE", "Q0", "Q0 + Δ", "Q (exact)", "Q + Δ (exact)"]; coll = [np.mean([R[m][s][0][-1].mean() < 0.02 for s in range(40)]) for m in names]
ax[2].bar([label.get(m, m).replace(" (fitted, untried = fails)", "\n(fitted)") for m in names], coll, color=[style[m][0] for m in names]); ax[2].set(ylabel="fraction of runs that collapse to 0", title="collapsed runs (40 seeds)")
for i, v in enumerate(coll): ax[2].text(i, v + 0.01, f"{int(round(v * 40))}/40", ha="center")
plt.setp(ax[2].get_xticklabels(), fontsize=8); plt.tight_layout(); plt.savefig(sys.argv[1], dpi=110)
