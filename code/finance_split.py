import numpy as np
from lab_demo import train, accuracy   # the SAME small MLP from the section 2.2 gap lab

# ================= the task =================
# S&P 500 daily closes -> predict a HIGH-VOLATILITY day (|return| above its median)
# from the last L days of |returns|. Volatility clusters (busy days follow busy days),
# so this target has real, persistent signal -- unlike next-day direction, which has none.
def load_finance(target="volatility", L=5):
    close = np.loadtxt("data/gspc_2026-07-03.csv", skiprows=1)
    ret = np.diff(close) / close[:-1]                # daily returns
    if target == "volatility":
        s = np.abs(ret)                              # volatility proxy = |return|
        X = np.column_stack([s[i:len(s)-L+i] for i in range(L)])
        fut = s[L:]; X = X[:len(fut)]
        y = (fut > np.median(s)).astype(int)         # busy day? (|next return| above median)
    else:                                            # direction: the no-signal contrast
        X = np.column_stack([ret[i:len(ret)-L+i] for i in range(L)])
        fut = ret[L:]; X = X[:len(fut)]
        y = (fut > 0).astype(int)                    # up day?
    return X, y

def standardize(Xtr, Xte):                            # fit on TRAIN only
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s

# ================= same model, two splits =================
def score(X, y, how, seed, width=16, lr=0.5, epochs=300):
    n = len(y); ntest = n // 5
    if how == "chronological":                        # RIGHT: past -> train, future -> test
        tr, te = np.arange(n - ntest), np.arange(n - ntest, n)
    else:                                             # WRONG: shuffle -> future leaks into train
        idx = np.random.default_rng(seed).permutation(n)
        te, tr = idx[:ntest], idx[ntest:]
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed + 1))   # same MLP init
    return accuracy(p, Xte, y[te])

# ================= walk-forward (rolling origin) =================
def walk_forward(X, y, folds=5, width=16, lr=0.5, epochs=300, seed=1):
    n = len(y); bs = n // (folds + 1); out = []
    for k in range(1, folds + 1):
        tr, te = np.arange(0, k*bs), np.arange(k*bs, (k+1)*bs)   # expanding past -> next block
        Xtr, Xte = standardize(X[tr], X[te])
        p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed))
        out.append(accuracy(p, Xte, y[te]))
    return np.mean(out)

if __name__ == "__main__":
    X, y = load_finance("volatility")
    print("### FINANCE (volatility): predict a busy day from lagged |returns|")
    print(f"days {len(y)} | high-vol rate {y.mean():.3f} | features {X.shape[1]} lagged |returns|")

    print("\n### same MLP, shuffle (WRONG) vs chronological (RIGHT), 10 seeds")
    print(f"{'seed':>4} | {'shuffle':>7} | {'chrono':>7} | {'diff':>6}")
    sh, ch = [], []
    for s in range(10):
        a = score(X, y, "shuffle", s); b = score(X, y, "chronological", s)
        sh.append(a); ch.append(b)
        print(f"{s:>4} | {a:>7.3f} | {b:>7.3f} | {a-b:>+6.3f}")
    print(f"MEAN | {np.mean(sh):>7.3f} | {np.mean(ch):>7.3f} | {np.mean(sh)-np.mean(ch):>+6.3f}")
    print(f"(shuffle > chrono in {sum(a>b for a,b in zip(sh,ch))}/10 seeds)")

    print(f"\n### walk-forward (rolling origin, 5 folds): {walk_forward(X, y):.3f}")

    Xd, yd = load_finance("direction")
    sh2 = [score(Xd, yd, "shuffle", s, lr=0.3, epochs=200) for s in range(10)]
    ch2 = [score(Xd, yd, "chronological", s, lr=0.3, epochs=200) for s in range(10)]
    print("\n### CONTRAST -- next-day DIRECTION (no signal): nothing to leak")
    print(f"MEAN shuffle {np.mean(sh2):.3f} | chrono {np.mean(ch2):.3f} | diff {np.mean(sh2)-np.mean(ch2):+.3f}  (up-rate {yd.mean():.3f})")
