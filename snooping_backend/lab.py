"""lab.py - the synthetic laboratory (known-truth data).

Builds a Gaussian dataset and the three labelling regimes, so the true
performance is known by construction and the gap can be measured exactly.

Design note (locked): everything is re-sampled fresh on each call. The sealed
test set stays LARGE and is opened only once, after selection - re-drawing it is
fine, leaking it is not.

Detailed provenance (source -> code -> check) lives in Core.md, Appendix A.
"""
import numpy as np


def make_X(n, d, rng):
    """(n, d) matrix of i.i.d. standard-Gaussian entries. rng is passed in, never seeded here."""
    return rng.standard_normal((n, d))


def labels_random(n, rng):
    """Case 1 labels: a fair coin, independent of X -> no signal (true accuracy = 0.5)."""
    return rng.integers(0, 2, size=n)


def labels_sign(X):
    """Case 2 labels: sign of feature 0 -> a clean, axis-aligned linear rule.

    Uses (X[:, 0] > 0) to give {0, 1} and dodge the measure-zero sign(0)=0 tie.
    """
    return (X[:, 0] > 0).astype(int)


def random_isometry(d, rng):
    """A random (d, d) orthogonal matrix R: a rotation, possibly with a reflection.

    The Q factor of the QR factorisation of a random Gaussian matrix is orthogonal.
    """
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)      # Q = orthogonal factor (our isometry); the triangular factor is unused
    return Q


def rotate(X, R):
    """Apply the isometry: X @ R. An orthogonal R preserves all distances and inner products."""
    return X @ R


def inject_noise(y, flip_y, rng):
    """Flip a flip_y fraction of the labels (exactly-known noise). flip_y = 0 -> no-op."""
    n = len(y)
    n_flip = int(flip_y * n)
    flip_indices = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]   # flip {0,1} labels at those rows
    return y_noisy


def make_dataset(case, d, flip_y, sizes, rng):
    """Build one case and split it into train / (small) val / (large) sealed test.

    sizes = (n_train, n_val, n_test). Re-samples fresh on every call.
    """
    n_train, n_val, n_test = sizes          # sizes is the single source of truth ...
    n = n_train + n_val + n_test            # ... so n can never disagree with the splits below

    X = make_X(n, d, rng)

    if case == 1:
        y = labels_random(n, rng)           # no signal -> true accuracy = 0.5
    elif case == 2:
        y = labels_sign(X)                  # axis-aligned linear rule on feature 0
    elif case == 3:
        y = labels_sign(X)                  # label FIRST, from the original X ...
        R = random_isometry(d, rng)
        X = rotate(X, R)                    # ... THEN rotate. Swap these two lines and the boundary
                                            # re-aligns with an axis -> the tree-drop artifact vanishes.
    else:
        raise ValueError(f"case must be 1, 2 or 3, got {case}")

    y = inject_noise(y, flip_y, rng)        # no-op when flip_y == 0

    # Split by explicit row ranges (no cumsum / np.split index math to decode):
    X_train = X[:n_train]
    X_val   = X[n_train:n_train + n_val]
    X_test  = X[n_train + n_val:]
    y_train = y[:n_train]
    y_val   = y[n_train:n_train + n_val]
    y_test  = y[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
