"""test_mlp.py - sanity checks for the MLP, PRINT style.

Run from the repo root:   python -m tests.test_mlp

Trains the MLP at two widths on Case 2 (learnable) and Case 1 (random labels),
and prints train / test accuracy. You read the numbers.

Expected:
  Case 2 -> train & test ~1.0  (it learns sign(feature 0))
  Case 1 -> test ~0.5          (no signal to generalise)
            train > 0.5 (it fits some of the noise - the supervisor's
            "how well can you fit?"; the clear capacity effect belongs to SS6.2)
"""
import numpy as np
from snooping_backend.lab import make_dataset
from snooping_backend.mlp import train, accuracy

d = 20
sizes = [2000, 500, 20000]      # train, val (unused here), large sealed test = truth
rng = np.random.default_rng(0)


def run(case, width, lr=0.5):
    (Xtr, ytr), _, (Xte, yte) = make_dataset(case, d, 0.0, sizes, rng)
    model = train(Xtr, ytr, width=width, lr=lr)
    return accuracy(model, Xtr, ytr), accuracy(model, Xte, yte)


print("=== MLP CHECKS (read the numbers; lines end OK / FAIL) ===\n")

print("CASE 2  (y = sign(feature 0), learnable):")
tr_a, te_a = run(2, width=16)
tr_b, te_b = run(2, width=128)
print(f"  width 16  : train {tr_a:.3f}   test {te_a:.3f}")
print(f"  width 128 : train {tr_b:.3f}   test {te_b:.3f}")
print("  -> expect test ~1.0  ->", "OK" if te_a > 0.9 and te_b > 0.9 else "FAIL")

print("\nCASE 1  (random labels, no signal):")
tr_a, te_a = run(1, width=16)
tr_b, te_b = run(1, width=128)
print(f"  width 16  : train {tr_a:.3f}   test {te_a:.3f}")
print(f"  width 128 : train {tr_b:.3f}   test {te_b:.3f}")
print("  -> expect test ~0.5 (no generalisation) ->",
      "OK" if 0.45 < te_a < 0.55 and 0.45 < te_b < 0.55 else "FAIL")
print("  -> train > 0.5 = fits some of the noise (clear capacity effect: SS6.2) ->",
      "OK" if tr_a > 0.5 and tr_b > 0.5 else "FAIL")

print("\n(read the numbers above; this is E0 - the network trains)")
