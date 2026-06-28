"""test_pipeline.py - sanity check for the gap machine, PRINT style.

Run from the repo root:   python -m tests.test_pipeline

Shows the winner's curse on Case 1 (random labels, true performance = 0.5):
the gap grows with N. Settings are small/fast here - the trend is the point;
the real headline uses larger settings (Phan 5).
"""
import numpy as np
from snooping_backend.pipeline import run_once

sizes = [1000, 200, 10000]      # train, SMALL val (the snoop), large sealed test (truth)
d, R, epochs = 20, 8, 80

print("=== GAP MACHINE on Case 1 (random labels) ===")
print(f"  n_val={sizes[1]} (small)  n_test={sizes[2]} (large)  mean over R={R} runs\n")
print(f"  {'N':>4}  {'apparent':>9}  {'true':>6}  {'gap':>7}")

rng = np.random.default_rng(0)
gap, true = {}, {}
for N in [1, 5, 20]:
    runs = [run_once(1, N, sizes, d, 0.0, rng, epochs=epochs) for _ in range(R)]
    a = np.mean([r["apparent"] for r in runs])
    t = np.mean([r["true"] for r in runs])
    g = np.mean([r["gap"] for r in runs])
    gap[N], true[N] = g, t
    print(f"  {N:>4}  {a:>9.3f}  {t:>6.3f}  {g:>+7.3f}")

print("\n  -> true ~0.5 at every N (no generalisation) ->",
      "OK" if all(abs(t - 0.5) < 0.05 for t in true.values()) else "FAIL")
print("  -> gap grows with N (gap@20 > gap@1) ->",
      "OK" if gap[20] > gap[1] else "FAIL")
