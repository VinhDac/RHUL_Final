"""test_pipeline.py - sanity check for the gap machine, PRINT style (exits 1 on FAIL).

Run from the repo root:   python -m tests.test_pipeline

Sweeps N on Case 1 (random labels, true performance = 0.5) and shows the winner's
curse: apparent (best-validation) climbs, true (sealed test) stays at 0.5, so the
gap grows with N. Small/fast settings here - the trend is the point; the full
headline (larger N, more repeats) lives in notebooks/01_core_snooping.ipynb.
Also covers run_once and gap_kfold (the integrity-critical calls).
"""
import sys
import numpy as np
from snooping_backend.pipeline import sweep, run_once, gap_kfold, synthetic_splits

_fail = 0
def result(passed):
    global _fail
    _fail += 0 if passed else 1
    return "OK" if passed else "FAIL"

sizes = [1000, 200, 10000]          # train, SMALL val (the snoop), large sealed test (truth)
N_values = [1, 5, 20]
d, R, epochs = 20, 6, 60

print("=== HEADLINE SWEEP on Case 1 (random labels) ===")
print(f"  n_val={sizes[1]} (small)  n_test={sizes[2]} (large)  R={R}  N_max={max(N_values)}\n")
print(f"  {'N':>4}  {'apparent':>9}  {'true':>6}  {'gap':>8}")
rng = np.random.default_rng(0)
res = sweep(synthetic_splits(1, d, 0.0, sizes), N_values, rng, R=R, epochs=epochs)
for N in N_values:
    r = res[N]
    print(f"  {N:>4}  {r['apparent']:>9.3f}  {r['true']:>6.3f}  {r['gap']:>+8.3f}")
trues = [res[N]["true"] for N in N_values]
gaps = [res[N]["gap"] for N in N_values]
print("\n  -> true ~0.5 at every N (no generalisation) ->", result(all(abs(t - 0.5) < 0.05 for t in trues)))
print("  -> gap grows with N (cumulative)             ->", result(gaps[-1] > gaps[0]))

print("\n=== run_once returns a coherent gap = apparent - true ===")
one = run_once(synthetic_splits(1, d, 0.0, sizes), 10, np.random.default_rng(1), epochs=epochs)
print(f"  apparent={one['apparent']:.3f}  true={one['true']:.3f}  gap={one['gap']:+.3f}")
print("  -> gap == apparent - true ->", result(abs(one["gap"] - (one["apparent"] - one["true"])) < 1e-9))
print("  -> true ~0.5 (Case 1)     ->", result(abs(one["true"] - 0.5) < 0.06))

print("\n=== gap_kfold (H4) runs and returns a coherent gap ===")
kf = gap_kfold(synthetic_splits(1, d, 0.0, sizes), 10, np.random.default_rng(2), epochs=epochs)
print(f"  apparent={kf['apparent']:.3f}  true={kf['true']:.3f}  gap={kf['gap']:+.3f}")
print("  -> gap == apparent - true ->", result(abs(kf["gap"] - (kf["apparent"] - kf["true"])) < 1e-9))
print("  -> true ~0.5 (Case 1)     ->", result(abs(kf["true"] - 0.5) < 0.06))

print("\n" + ("ALL OK" if _fail == 0 else f"{_fail} CHECK(S) FAILED"))
sys.exit(1 if _fail else 0)
