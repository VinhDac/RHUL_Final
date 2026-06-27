"""test_lab.py - sanity checks for the lab, PRINT style.

Run from the repo root:   python -m tests.test_lab

Each line prints the real number next to what we expect, and ends in OK / FAIL.
You read the numbers; a FAIL jumps out if a change ever breaks something.
The "rotate keeps distance" line IS the supervisor's isometry insight, as a number.
"""
import numpy as np
from snooping_backend.lab import (make_X, labels_random, labels_sign,
                                  random_isometry, rotate, inject_noise, make_dataset)

rng = np.random.default_rng(0)

print("=== LAB CHECKS (read the numbers; each line ends OK / FAIL) ===\n")

# make_X: Gaussian samples, shape (n, d), each column ~ N(0, 1)
X = make_X(1000, 10, rng)
ok = X.shape == (1000, 10)
print("make_X shape          :", X.shape, "  expect (1000, 10)        ->", "OK" if ok else "FAIL")

mean_err = float(np.abs(X.mean(axis=0)).max())
print("make_X column mean    :", round(mean_err, 3), "          expect ~0  (< 0.1)       ->", "OK" if mean_err < 0.1 else "FAIL")

std_err = float(np.abs(X.std(axis=0) - 1).max())
print("make_X column std-1   :", round(std_err, 3), "          expect ~0  (< 0.1)       ->", "OK" if std_err < 0.1 else "FAIL")

# labels_random (Case 1): fair coin, ~50/50, independent of X
yr = labels_random(1000, rng)
print("labels_random balance :", round(float(yr.mean()), 3), "          expect ~0.5 (0.4-0.6)    ->", "OK" if 0.4 < yr.mean() < 0.6 else "FAIL")

# labels_sign (Case 2): sign of feature 0, also ~50/50 (Gaussian is symmetric)
ys = labels_sign(X)
print("labels_sign  balance  :", round(float(ys.mean()), 3), "          expect ~0.5 (0.4-0.6)    ->", "OK" if 0.4 < ys.mean() < 0.6 else "FAIL")

# random_isometry: R is orthogonal  ->  R @ R.T = identity
R = random_isometry(10, rng)
iso_err = float(np.abs(R @ R.T - np.eye(10)).max())
print("isometry R @ R.T = I  :", f"{iso_err:.2e}", "       expect ~0               ->", "OK" if iso_err < 1e-6 else "FAIL")

# rotate (Case 3): an isometry preserves every pairwise inner product / distance.
# This single number IS the insight: why kNN / SVM / logreg are unchanged by the rotation.
Xr = rotate(X, R)
dist_err = float(np.abs(X @ X.T - Xr @ Xr.T).max())
print("rotate keeps distance :", f"{dist_err:.2e}", "       expect ~0  (INSIGHT)    ->", "OK" if dist_err < 1e-6 else "FAIL")

# inject_noise: flips ~ flip_y fraction of the labels
yn = inject_noise(ys, 0.1, rng)
frac = float(np.mean(yn != ys))
print("inject_noise flipped  :", round(frac, 3), "          expect ~0.10 (0.08-0.12) ->", "OK" if 0.08 < frac < 0.12 else "FAIL")

# make_dataset: builds a case and splits it; shapes must match the requested sizes
(Xtr, ytr), (Xval, yval), (Xte, yte) = make_dataset(1, 10, 0.1, [600, 200, 200], rng)
shapes_ok = Xtr.shape == (600, 10) and Xval.shape == (200, 10) and Xte.shape == (200, 10)
print("make_dataset splits   :", (Xtr.shape, Xval.shape, Xte.shape), "expect (600,10)(200,10)(200,10) ->", "OK" if shapes_ok else "FAIL")

print("\n(if every line says OK, the lab is good)")
