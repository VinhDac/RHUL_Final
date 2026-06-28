"""test_pipeline.py - sanity check for the gap machine, PRINT style.

Run from the repo root:   python -m tests.test_pipeline

Sweeps N on Case 1 (random labels, true performance = 0.5) and shows the
winner's curse: apparent (best-validation) climbs, true (sealed test) stays at
0.5, so the gap grows with N. Small/fast settings here - the trend is the point;
the real headline (larger N, more repeats) lives in the notebook (Phan 5).
"""
import numpy as np
from snooping_backend.pipeline import sweep, synthetic_splits

sizes = [1000, 200, 10000]      # train, SMALL val (the snoop), large sealed test (truth)
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
print("\n  -> true ~0.5 at every N (no generalisation) ->",
      "OK" if all(abs(t - 0.5) < 0.05 for t in trues) else "FAIL")
print("  -> gap grows with N (cumulative) ->",
      "OK" if gaps[-1] > gaps[0] else "FAIL")
