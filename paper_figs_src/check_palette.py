"""Adjacent-pair colour check (OKLab distance x100, normal vision and Machado-2009 protan/deutan/tritan at full severity)."""
import numpy as np, sys
def hex2lin(h):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def oklab(lin):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ np.cbrt(M1 @ lin)
CVD = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
       "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
       "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}
def dist(a, b, kind=None):
    la, lb = hex2lin(a), hex2lin(b)
    if kind: M = np.array(CVD[kind]); la, lb = np.clip(M @ la, 0, 1), np.clip(M @ lb, 0, 1)
    return 100 * np.linalg.norm(oklab(la) - oklab(lb))
def lum(h):
    l = hex2lin(h); return 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]
def report(name, cols):
    print(f"\n{name}")
    for h, n in cols: print(f"  {n:18s} {h}  contrast on white {(1.05) / (lum(h) + 0.05):.2f}:1")
    worst = 99
    for (h1, n1), (h2, n2) in zip(cols[:-1], cols[1:]):
        d = {k: dist(h1, h2, k) for k in (None, "protan", "deutan", "tritan")}
        worst = min(worst, min(d["protan"], d["deutan"], d["tritan"]))
        flag = "  <-- CVD < 8" if min(d["protan"], d["deutan"], d["tritan"]) < 8 else ""
        print(f"  {n1:>16s} | {n2:<16s} normal {d[None]:5.1f}  protan {d['protan']:5.1f}  deutan {d['deutan']:5.1f}  tritan {d['tritan']:5.1f}{flag}")
    print(f"  worst adjacent CVD distance {worst:.1f}")
if __name__ == "__main__":
    from make_paper_figs import COL, ORDER
    for name, seq in ORDER.items(): report(name, [(COL[m], m) for m in seq])
