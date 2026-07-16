import numpy as np
from lab_demo import train, accuracy   # the SAME small MLP again

# The task is section 2.3-A's: a busy day, the next move larger than the median move.
# The label never changes, and the split stays honest (chronological) throughout.
# Only two things move: whether the FEATURE drifts, and how it is SCALED.
#   percent size |r| : stationary, the sensible feature
#   dollar size      : the same information in points, which drifts with the index level
L = 5

def load(kind="percent"):
    close = np.loadtxt("data/gspc_2026-07-03.csv", skiprows=1)
    r = np.diff(close) / close[:-1]
    a = np.abs(r)                                            # percent size: stationary
    src = a if kind == "percent" else np.abs(np.diff(close))  # dollar size: drifts
    X = np.column_stack([src[i:len(src)-L+i] for i in range(L)])
    fut = a[L:]; X = X[:len(fut)]
    y = (fut > np.median(a)).astype(int)                      # label always from the percent size
    return X, y

def frozen(Xtr, Xte):          # the textbook rule: fit on train, freeze, apply to test
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s

def rolling_z(X, W=250):       # drift-aware: every row is scaled by its OWN trailing window
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        c = X[:, j]
        cs = np.concatenate([[0.0], np.cumsum(c)])
        cs2 = np.concatenate([[0.0], np.cumsum(c * c)])
        for t in range(len(c)):
            lo = max(0, t - W); n = t - lo
            if n < 30:
                continue                                      # not enough past yet: leave at 0
            mu = (cs[t] - cs[lo]) / n
            var = (cs2[t] - cs2[lo]) / n - mu * mu
            Z[t, j] = (c[t] - mu) / (np.sqrt(max(var, 1e-18)) + 1e-9)
    return Z                   # causal: row t uses only the days before it, so nothing leaks

def split(n):                  # chronological throughout: A's problem is off the table
    ntest = n // 5
    return np.arange(n - ntest), np.arange(n - ntest, n)

def score(X, y, scaling, seed, width=16, lr=0.5, epochs=300):
    tr, te = split(len(y))
    if scaling == "frozen":
        Xtr, Xte = frozen(X[tr], X[te])
    else:
        Z = rolling_z(X); Xtr, Xte = Z[tr], Z[te]
    p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed + 1))
    return accuracy(p, Xte, y[te])

if __name__ == "__main__":
    Xp, y = load("percent")
    Xd, _ = load("dollars")

    print("### under the frozen rule, how far outside the training range does the test land?")
    tr, te = split(len(y))
    for name, X in (("percent (stationary)", Xp), ("dollars (drifts)", Xd)):
        _, Zte = frozen(X[tr], X[te])
        print(f"  {name:22s} test |z|: mean {np.abs(Zte).mean():5.2f}   max {np.abs(Zte).max():6.2f}")

    print("\n### same MLP, same chronological split, only the feature and the scaling change")
    print(f"{'feature':22s} | {'scaling':8s} | accuracy over 10 seeds")
    for name, X, mode in (("percent (stationary)", Xp, "frozen"),
                          ("dollars (drifts)", Xd, "frozen"),
                          ("dollars (drifts)", Xd, "rolling")):
        accs = [score(X, y, mode, s) for s in range(10)]
        print(f"{name:22s} | {mode:8s} | mean {np.mean(accs):.3f}   sd {np.std(accs):.3f}")
