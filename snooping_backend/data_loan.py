"""data_loan.py - the UCI loan-default dataset as a make_splits provider.

Real, messy data with stakes (bad loans wrongly approved). The SAME gap machine
(pipeline.run_once / sweep) runs on it unchanged - this module only supplies data.

    load_loan()                -> X (n, d) float, y (n,) int {0, 1}   (fetched once)
    loan_provider(X, y, sizes) -> make_splits(rng)                    (re-shuffled splits)
"""
import numpy as np


def load_loan():
    """Fetch UCI 'default of credit card clients' (id 350); return (X, y).

    X = all features as floats; y = default next month (1) / not (0).
    """
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=350)
    X = ds.data.features.to_numpy(dtype=float)
    y = ds.data.targets.to_numpy().ravel().astype(int)
    return X, y


def loan_provider(X, y, sizes):
    """A make_splits provider for the loan data (parallels synthetic_splits).

    Each call re-shuffles into train / (small) val / (large) test and standardises
    features using the TRAIN split's mean and std ONLY (no leak from val/test).
    sizes = (n_train, n_val, n_test), summing to <= len(X).
    """
    n_train, n_val, n_test = sizes
    n = n_train + n_val + n_test

    def make_splits(rng):
        idx = rng.permutation(len(X))[:n]
        Xs, ys = X[idx].copy(), y[idx]
        mu = Xs[:n_train].mean(axis=0)
        sd = Xs[:n_train].std(axis=0) + 1e-8          # train-only stats -> no leak
        Xs = (Xs - mu) / sd
        a, b = n_train, n_train + n_val
        return ((Xs[:a], ys[:a]), (Xs[a:b], ys[a:b]), (Xs[b:], ys[b:]))

    return make_splits
