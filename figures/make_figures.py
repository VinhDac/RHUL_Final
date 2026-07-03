"""make_figures.py - regenerate every figure in Core.md from the canonical numbers.

The numbers below are the ones reported in Core.md, produced by the Phase-1 runs
(see results/*.txt). Run `python figures/make_figures.py` from the repo root.
The 2-D case picture is generated from data; the rest plot the measured tables.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 100, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
FIG = "figures/"
def save(name): plt.tight_layout(); plt.savefig(FIG + name, format="svg", bbox_inches="tight"); plt.close()

# 1. headline: apparent vs true vs N (Case 1)
N = [1, 2, 5, 10, 20, 50, 100, 200]
app = [.509, .524, .531, .551, .556, .564, .577, .581]
tru = [.501, .501, .500, .500, .500, .500, .500, .501]
plt.figure(figsize=(5, 3.2))
plt.plot(N, app, "o-", label="apparent (best validation)")
plt.plot(N, tru, "s--", label="true (sealed test)", color="#666")
plt.fill_between(N, tru, app, alpha=0.12, label="gap")
plt.xscale("log"); plt.xlabel("N (configurations searched)"); plt.ylabel("accuracy")
plt.title("Headline: the gap grows with the search (Case 1)"); plt.legend(); save("headline_gap_vs_N.svg")

# 2. optimal budget (Case 2 + 20% noise)
N2 = [1, 5, 20, 50, 100, 200]
app2 = [.700, .762, .779, .789, .798, .800]
tru2 = [.694, .742, .759, .766, .759, .760]
plt.figure(figsize=(5, 3.2))
plt.plot(N2, app2, "o-", label="apparent (keeps climbing)")
plt.plot(N2, tru2, "s--", label="true (peaks ~N=50)", color="#c0392b")
plt.axvline(50, color="#aaa", ls=":", lw=1)
plt.xscale("log"); plt.xlabel("N"); plt.ylabel("accuracy")
plt.title("Optimal search budget (Case 2 + noise)"); plt.legend(); save("optimal_budget.svg")

# 3. gap vs noise
flip = [0, .1, .2, .3, .4, .5]; gnoise = [.008, .005, .011, .026, .041, .057]
plt.figure(figsize=(5, 3.2))
plt.plot(flip, gnoise, "o-", color="#2c7fb8")
plt.xlabel("label noise (fraction flipped)"); plt.ylabel("gap"); plt.ylim(bottom=0)
plt.title("More noise widens the gap (H2)"); save("gap_vs_noise.svg")

# 4. gap vs capacity (Case 2 and Case 4)
w = [4, 8, 16, 32, 64, 128, 256]
gc2 = [.028, .026, .022, .029, .009, .040, .029]
gc4 = [.027, .019, .012, .030, .024, .047, .034]
plt.figure(figsize=(5, 3.2))
plt.plot(w, gc2, "o-", label="Case 2 (linear)")
plt.plot(w, gc4, "s-", label="Case 4 (XOR)")
plt.axhspan(0, 0.03, color="#eee", alpha=0.6, zorder=0)
plt.xscale("log", base=2); plt.xlabel("hidden width"); plt.ylabel("gap"); plt.ylim(bottom=0)
plt.title("Capacity does not widen the gap (H3 refuted)"); plt.legend(); save("gap_vs_capacity.svg")

# 5. gap vs protocol
Np = [1, 5, 20, 50]; gs = [-.004, .039, .067, .071]; gk = [-.003, .015, .022, .024]
plt.figure(figsize=(5, 3.2))
plt.plot(Np, gs, "o-", label="single split")
plt.plot(Np, gk, "s-", label="5-fold CV", color="#27ae60")
plt.xscale("log"); plt.xlabel("N"); plt.ylabel("gap")
plt.title("An honest protocol shrinks the gap (H4)"); plt.legend(); save("gap_vs_protocol.svg")

# 6. isometry (appendix): Case 2 vs Case 3, four methods
methods = ["kNN", "logreg", "SVM", "tree"]; c2 = [.805, .989, .986, 1.000]; c3 = [.805, .989, .986, .709]
x = np.arange(len(methods)); wbar = 0.38
plt.figure(figsize=(5, 3.2))
plt.bar(x - wbar/2, c2, wbar, label="Case 2 (axis-aligned)")
plt.bar(x + wbar/2, c3, wbar, label="Case 3 (rotated)", color="#e67e22")
plt.xticks(x, methods); plt.ylabel("test accuracy"); plt.ylim(0.4, 1.05)
plt.title("Isometry: only the axis-aligned tree drops"); plt.legend(); save("isometry.svg")

# 7. cases_2d (from data): the three labelling rules in 2-D
rng = np.random.default_rng(0); X = rng.standard_normal((600, 2))
labels = [("Case 1: random", rng.integers(0, 2, 600)),
          ("Case 2: sign(x1)", (X[:, 0] > 0).astype(int)),
          ("Case 4: XOR", ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int))]
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, (title, y) in zip(axes, labels):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.7)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
plt.tight_layout(); plt.savefig(FIG + "cases_2d.svg", format="svg", bbox_inches="tight"); plt.close()

print("wrote: headline_gap_vs_N, optimal_budget, gap_vs_noise, gap_vs_capacity, gap_vs_protocol, isometry, cases_2d (.svg)")
