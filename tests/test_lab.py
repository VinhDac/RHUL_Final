"""test_lab.py - sanity checks for the lab, PRINT style (exits 1 on any FAIL).

Run from the repo root:   python -m tests.test_lab

Each line prints the real number next to what we expect and ends in OK / FAIL.
You read the numbers; a FAIL jumps out AND makes the script exit non-zero (so CI
catches a regression instead of silently passing). The "rotate keeps distance"
line IS the supervisor's isometry insight, as a number (Appendix C).
"""
import sys
import numpy as np
from snooping_backend.lab import (make_X, labels_random, labels_sign, labels_xor,
                                  random_isometry, rotate, inject_noise, make_dataset)

_fail = 0
def result(passed):
    global _fail
    _fail += 0 if passed else 1
    return "OK" if passed else "FAIL"

rng = np.random.default_rng(0)
print("=== LAB CHECKS (read the numbers; each line ends OK / FAIL) ===\n")

X = make_X(1000, 10, rng)
print("make_X shape          :", X.shape, "  expect (1000, 10)        ->", result(X.shape == (1000, 10)))
mean_err = float(np.abs(X.mean(axis=0)).max())
print("make_X column mean    :", round(mean_err, 3), "          expect ~0  (< 0.1)       ->", result(mean_err < 0.1))
std_err = float(np.abs(X.std(axis=0) - 1).max())
print("make_X column std-1   :", round(std_err, 3), "          expect ~0  (< 0.1)       ->", result(std_err < 0.1))

yr = labels_random(1000, rng)
print("labels_random balance :", round(float(yr.mean()), 3), "          expect ~0.5 (0.4-0.6)    ->", result(0.4 < yr.mean() < 0.6))
ys = labels_sign(X)
print("labels_sign  balance  :", round(float(ys.mean()), 3), "          expect ~0.5 (0.4-0.6)    ->", result(0.4 < ys.mean() < 0.6))
y4 = labels_xor(X)
xor_ok = 0.4 < y4.mean() < 0.6 and (y4 != ys).mean() > 0.2   # balanced AND differs from the linear rule
print("labels_xor  balance   :", round(float(y4.mean()), 3), "          expect ~0.5, nonlinear   ->", result(xor_ok))

R = random_isometry(10, rng)
iso_err = float(np.abs(R @ R.T - np.eye(10)).max())
print("isometry R @ R.T = I  :", f"{iso_err:.2e}", "       expect ~0               ->", result(iso_err < 1e-6))
Xr = rotate(X, R)
dist_err = float(np.abs(X @ X.T - Xr @ Xr.T).max())
print("rotate keeps distance :", f"{dist_err:.2e}", "       expect ~0  (INSIGHT)    ->", result(dist_err < 1e-6))

yn = inject_noise(ys, 0.1, rng)
frac = float(np.mean(yn != ys))
print("inject_noise flipped  :", round(frac, 3), "          expect ~0.10 (0.08-0.12) ->", result(0.08 < frac < 0.12))

(Xtr, ytr), (Xval, yval), (Xte, yte) = make_dataset(1, 10, 0.1, [600, 200, 200], rng)
shapes_ok = Xtr.shape == (600, 10) and Xval.shape == (200, 10) and Xte.shape == (200, 10)
print("make_dataset splits   :", (Xtr.shape, Xval.shape, Xte.shape), "expect (600,10)(200,10)(200,10) ->", result(shapes_ok))

print("\n" + ("ALL OK" if _fail == 0 else f"{_fail} CHECK(S) FAILED"))
sys.exit(1 if _fail else 0)
