"""lab.py - the synthetic laboratory (known-truth data).

Builds a Gaussian dataset and the three labelling regimes, so the
true performance is known by construction and the gap can be measured exactly.

Design note (locked): EVERYTHING is a variable. Each repeat re-samples fresh
train / validation / sealed-test from the generator. The sealed test stays LARGE
and is opened only once, after selection - re-drawing it is fine, leaking it is not.

"""

import numpy as np

""" make_X(n, d, rng) -> X : (n, d) array of i.i.d. N(0, 1)
      tool : rng.standard_normal((n, d))   (rng passed in, never seeded inside)
      check: X.shape == (n, d); column mean ~ 0; column std ~ 1"""
def make_X(n, d, rng):   
    """Make a Gaussian dataset of shape (n,d) with i.i.d. N(0,1) entries"""
    X = rng.standard_normal((n, d))
    return X

"""  labels_random(n, rng) -> y : random labels, independent of X      [Case 1]
      meaning: no signal -> true generalisation = chance (~50%)
      check  : class balance ~ 50/50"""
def labels_random(n, rng):
    """Random labels, independent of X"""
    y = rng.integers(0, 2, size=n)
    return y   

"""  labels_sign(X) -> y = sign(X[:, 0])                                [Case 2]
      meaning: a clean, axis-aligned linear rule on feature 0
      check  : logistic regression ~ 100% when flip_y = 0 (linearly separable)"""
def labels_sign(X):
    """Labels according to the sign of the first feature"""
    y = (X[:, 0] > 0).astype(int)
    return y

"""  random_isometry(d, rng) -> R : (d, d) orthogonal (rotation +/- reflection)
      tool : QR factorisation of a random Gaussian matrix
      check: R @ R.T ~ I"""
def random_isometry(d, rng):
    """Generate a random orthogonal matrix (isometry) of shape (d,d)"""
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q

"""  rotate(X, R) -> X @ R                                              [Case 3]
      meaning: same rule as Case 2, now hidden along an oblique direction
      check  : kNN accuracy on Case 3 == kNN accuracy on Case 2
               (an isometry preserves distances - this is the supervisor's
                insight, proven by numbers)
"""
def rotate(X, R):
    """Rotate the dataset X using the orthogonal matrix R"""
    return X @ R

"""  inject_noise(y, flip_y, rng) -> y with a fraction flip_y of labels flipped
      check: fraction actually flipped ~ flip_y"""
def inject_noise(y, flip_y, rng):
    """Inject label noise by flipping a fraction of the labels"""
    n = len(y)
    n_flip = int(flip_y * n)
    flip_indices = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]  # Flip labels
    return y_noisy

"""  make_dataset(case, n, d, flip_y, sizes, rng)
        -> (X_train, y_train), (X_val, y_val), (X_test, y_test)
      val is SMALL (snooping engine); test is LARGE (truth ~ no sampling error).
      re-sample fresh on every call."""
def make_dataset(case, n, d, flip_y, sizes, rng):
    X = make_X(n, d, rng)
    if case == 1:
        y = labels_random(n, rng)
    elif case == 2:
        y = labels_sign(X)
    elif case == 3:
        R = random_isometry(d, rng)
        y = labels_sign(X)
        X = rotate(X, R)
        
    else:
        raise ValueError("Invalid case")
    y_noisy = inject_noise(y, flip_y, rng)
    X_train, X_val, X_test = np.split(X, np.cumsum(sizes[:-1]))
    y_train, y_val, y_test = np.split(y_noisy, np.cumsum(sizes[:-1]))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

