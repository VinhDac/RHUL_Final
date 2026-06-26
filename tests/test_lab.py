"""test_lab.py - sanity checks for the lab.

These are not just tests: the rotate() check IS the supervisor's isometry
insight, proven by numbers. Keep them as plain asserts - no framework needed.

Run from the repo root:   python -m tests.test_lab

------------------------------------------------------------------------------
Checks to implement (bodies are yours):

  make_X            -> shape == (n, d); column mean ~ 0; column std ~ 1
  labels_random     -> class balance ~ 50/50
  labels_sign       -> logistic regression ~ 100% accuracy when flip_y = 0
  random_isometry   -> R @ R.T ~ I (allclose with a tolerance)
  rotate            -> kNN accuracy on Case 3 == kNN accuracy on Case 2
                       (the distances are preserved by the rotation)
  inject_noise      -> fraction of labels actually flipped ~ flip_y
------------------------------------------------------------------------------
"""
import numpy as np
from snooping_backend.lab import make_X, labels_random, labels_sign, \
                                 random_isometry, rotate, inject_noise, make_dataset

rng = np.random.default_rng(0)

assert make_X(1000, 10, rng).shape == (1000, 10), "make_X"
assert np.abs(make_X(1000, 10, rng).mean(axis=0)).max() < 0.1, "make_X mean"
assert np.abs(make_X(1000, 10, rng).std(axis=0) - 1).max() < 0.1, "make_X std"
y_random = labels_random(1000, rng)
assert np.mean(y_random == 0) > 0.4 and np.mean(y_random == 1) > 0.4, "labels_random"
X = make_X(1000, 10, rng)
y_sign = labels_sign(X)
assert np.mean(y_sign == 0) > 0.4 and np.mean(y_sign == 1) > 0.4, "labels_sign"
R = random_isometry(10, rng)
assert np.allclose(R @ R.T, np.eye(10), atol=1e-6), "random_isometry"
X_rotated = rotate(X, R)
assert np.allclose(X @ X.T, X_rotated @ X_rotated.T) "Rotate"
y_noisy = inject_noise(y_sign, 0.1, rng)
assert np.mean(y_noisy == y_sign) > 0.8 and np.mean(y_noisy != y_sign) > 0.08, "inject_noise" 
assert make_dataset(1, 1000, 10, 0.1, [600, 200, 200], rng)[0][0].shape == (600, 10), "make_dataset"  



print("ALL CHECKS PASSED")   