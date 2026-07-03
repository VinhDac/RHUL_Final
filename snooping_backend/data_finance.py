"""data_finance.py - financial prices (^GSPC) as a make_splits provider.

The warning case: signal ~ 0, so the "edge" found by searching is almost all luck
and collapses out-of-sample. The SAME gap machine runs on it - this module only
supplies the data, with a CHRONOLOGICAL (walk-forward) split so no future leaks
into training.

    load_finance(ticker, start, k)     -> X (n, k) float, y (n,) int {0, 1}, r (n,) returns
    finance_provider(X, y, sizes)      -> make_splits(rng)   (FIXED walk-forward split)
"""
import numpy as np


def load_finance(ticker="^GSPC", start="2000-01-01", k=5, cache="data/gspc_2026-07-03.csv"):
    """Load the daily close series (FROZEN to a dated CSV on first download), then build
    lagged-return features + next-day direction.

    First run downloads via yfinance and writes `cache`; every later run reads `cache`,
    so features/labels are deterministic and reproducible OFFLINE (no live drift - which
    was the cause of report-vs-notebook number mismatch).
    X[i] = the k daily returns ending at day i+k-1; y[i] = 1 if day (i+k)'s return > 0;
    r[i] = day (i+k)'s actual (signed) return, for the out-of-sample equity curve.
    Rows stay in chronological order (oldest first) - never shuffled.
    """
    import os
    import pandas as pd
    if cache and os.path.exists(cache):
        close = pd.read_csv(cache)["close"].to_numpy().ravel()          # frozen snapshot
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        close = df["Close"].squeeze().to_numpy().ravel()
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            pd.DataFrame({"close": close}).to_csv(cache, index=False, float_format="%.17g")  # lossless freeze
    ret = close[1:] / close[:-1] - 1.0                  # daily returns (== pct_change)
    L = len(ret)
    X = np.array([ret[i:i + k] for i in range(L - k)])  # k past returns
    r = ret[k:]                                         # the next-day (signed) return
    y = (r > 0).astype(int)                             # its direction (the label)
    return X, y, r


def finance_provider(X, y, sizes):
    """A make_splits provider for the finance data - WALK-FORWARD (no shuffle).

    Uses the most recent (n_train + n_val + n_test) rows, split chronologically:
    train = oldest, val = small middle window (the snoop), test = newest (the future).
    Features standardised on the TRAIN window only (no leak). The split is FIXED,
    so make_splits ignores rng (time cannot be re-shuffled); repeats vary only the
    searched configurations.
    """
    n_train, n_val, n_test = sizes
    n = n_train + n_val + n_test
    Xc, yc = X[-n:].copy(), y[-n:]
    mu = Xc[:n_train].mean(axis=0)
    sd = Xc[:n_train].std(axis=0) + 1e-8
    Xc = (Xc - mu) / sd
    a, b = n_train, n_train + n_val
    splits = ((Xc[:a], yc[:a]), (Xc[a:b], yc[a:b]), (Xc[b:], yc[b:]))
    return lambda rng: splits
