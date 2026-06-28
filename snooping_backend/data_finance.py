"""data_finance.py - financial prices (^GSPC) as a make_splits provider.

The warning case: signal ~ 0, so the "edge" found by searching is almost all luck
and collapses out-of-sample. The SAME gap machine runs on it - this module only
supplies the data, with a CHRONOLOGICAL (walk-forward) split so no future leaks
into training.

    load_finance(ticker, start, k)     -> X (n, k) float, y (n,) int {0, 1}, r (n,) returns
    finance_provider(X, y, sizes)      -> make_splits(rng)   (FIXED walk-forward split)
"""
import numpy as np


def load_finance(ticker="^GSPC", start="2000-01-01", k=5):
    """Download daily prices (yfinance); build lagged-return features + next-day direction.

    X[i] = the k daily returns ending at day i+k-1; y[i] = 1 if day (i+k)'s return > 0;
    r[i] = day (i+k)'s actual return (magnitude, for the out-of-sample equity curve, R4).
    Rows stay in chronological order (oldest first) - never shuffled.
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    close = df["Close"].squeeze()                       # Series of daily closes
    ret = close.pct_change().dropna().to_numpy().ravel()
    L = len(ret)
    X = np.array([ret[i:i + k] for i in range(L - k)])  # k past returns
    r = ret[k:]                                         # the next-day return for each row
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
