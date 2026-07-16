import numpy as np, time

# =========================================================
# PIECE 1 - the no-signal data
# features are pure noise; labels are coin flips that never look at the features.
# so the truth is known: no model can beat 0.5.
# =========================================================
def make_data(n, d, rng):
    X = rng.standard_normal((n, d))        # features: pure noise
    y = rng.integers(0, 2, size=n)         # labels: coin flips, drawn WITHOUT X
    return X, y

def split(X, y, n_train, n_val):
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, None)
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])

# =========================================================
# PIECE 2 - the small MLP.  a configuration = (width, learning rate).
# one hidden layer, ReLU, 2 outputs, softmax + cross-entropy, full-batch GD.
# =========================================================
def init(d, width, rng):
    return [rng.standard_normal((d, width)) * np.sqrt(2.0/d), np.zeros(width),
            rng.standard_normal((width, 2)) * np.sqrt(2.0/width), np.zeros(2)]

def forward(X, p):
    W1, b1, W2, b2 = p
    a = X @ W1 + b1
    h = np.maximum(a, 0.0)                  # ReLU
    z = h @ W2 + b2                         # logits
    return a, h, z

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def train(Xtr, ytr, width, lr, epochs, rng):
    p = init(Xtr.shape[1], width, rng)
    n = len(ytr)
    Y = np.zeros((n, 2)); Y[np.arange(n), ytr] = 1.0
    for _ in range(epochs):
        a, h, z = forward(Xtr, p)
        dz = (softmax(z) - Y) / n           # d(cross-entropy)/d(logits)
        W1, b1, W2, b2 = p
        dW2 = h.T @ dz; db2 = dz.sum(0)
        da = (dz @ W2.T) * (a > 0)          # back through ReLU
        dW1 = Xtr.T @ da; db1 = da.sum(0)
        p = [W1-lr*dW1, b1-lr*db1, W2-lr*dW2, b2-lr*db2]   # gradient descent step
    return p

def accuracy(p, X, y):
    return (forward(X, p)[2].argmax(1) == y).mean()

def sample_config(rng):
    return int(rng.choice([4, 8, 16, 32, 64])), float(10**rng.uniform(-2, -0.5))

# =========================================================
# PIECE 3 - the gap machine: draw N configs, keep the best on validation,
# open the sealed test EXACTLY ONCE on the winner.
# =========================================================
def run_once(N, data, epochs, rng):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    best_val, best_p = -1.0, None
    for _ in range(N):
        w, lr = sample_config(rng)
        p = train(Xtr, ytr, w, lr, epochs, rng)
        v = accuracy(p, Xva, yva)           # score on validation
        if v > best_val:
            best_val, best_p = v, p          # keep best on validation
    true = accuracy(best_p, Xte, yte)        # open sealed test ONCE, on winner
    return best_val, true

# =========================================================
# PIECE 4 & 6 - one sweep, two readings of the SAME configs:
#   select on VALIDATION (correct)  -> validation lies, sealed test is honest
#   select on the TEST   (reuse!)   -> the test lies too; truth is known = 0.5
# =========================================================
def sweep(N_values, R, epochs, n_train, n_val, n_test, d, seed):
    rng = np.random.default_rng(seed)
    Nmax = max(N_values)
    # va = apparent selecting on val;  vt = its sealed-test score (true)
    # ta = apparent REUSING the test;  tt = its validation score (honest)
    rec = {N: {k: [] for k in "va vt ta tt".split()} for N in N_values}
    for r in range(R):
        X, y = make_data(n_train + n_val + n_test, d, rng)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, n_train, n_val)
        vals, tests = [], []
        for _ in range(Nmax):
            w, lr = sample_config(rng)
            p = train(Xtr, ytr, w, lr, epochs, rng)
            vals.append(accuracy(p, Xva, yva))
            tests.append(accuracy(p, Xte, yte))
        vals, tests = np.array(vals), np.array(tests)
        for N in N_values:
            iv = vals[:N].argmax()          # winner if we select on validation
            it = tests[:N].argmax()         # winner if we REUSE the test to select
            rec[N]["va"].append(vals[iv]);  rec[N]["vt"].append(tests[iv])
            rec[N]["ta"].append(tests[it]); rec[N]["tt"].append(vals[it])
    return {N: (np.mean(rec[N]["va"]), np.mean(rec[N]["vt"]),
                np.mean(rec[N]["ta"]), np.mean(rec[N]["tt"])) for N in N_values}

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    D, NTR, NVA, NTE, EPOCHS = 20, 1000, 200, 10000, 60

    print("### PIECE 1 - the no-signal data")
    rng = np.random.default_rng(0)
    X, y = make_data(NTR+NVA+NTE, D, rng)
    (Xtr,ytr),(Xva,yva),(Xte,yte) = split(X, y, NTR, NVA)
    print(f"shapes  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    print(f"label balance  train {ytr.mean():.3f}  val {yva.mean():.3f}  test {yte.mean():.3f}")
    print(f"corr(feature0, y) on train = {np.corrcoef(Xtr[:,0], ytr)[0,1]:+.4f}  (~0: no signal)")

    print("\n### PIECE 2 - one configuration (train one MLP)")
    t0 = time.time()
    w, lr = 32, 0.1
    p = train(Xtr, ytr, w, lr, EPOCHS, np.random.default_rng(1))
    print(f"config width={w} lr={lr}: train {accuracy(p,Xtr,ytr):.3f} | val {accuracy(p,Xva,yva):.3f} | test {accuracy(p,Xte,yte):.3f}")
    print(f"(one train took {time.time()-t0:.3f}s)")

    print("\n### PIECE 3 - the gap machine for N=20")
    data = ((Xtr,ytr),(Xva,yva),(Xte,yte))
    app, tru = run_once(20, data, EPOCHS, np.random.default_rng(2))
    print(f"N=20:  apparent(best val) {app:.3f} | true(test of winner) {tru:.3f} | gap {app-tru:+.3f}")

    print("\n### PIECE 4 - the sweep over N (R=15): select on VALIDATION")
    t0 = time.time()
    Ns = [1,2,5,10,20,50,100]
    res = sweep(Ns, R=15, epochs=EPOCHS, n_train=NTR, n_val=NVA, n_test=NTE, d=D, seed=0)
    print(f"{'N':>4} | {'apparent':>8} | {'true':>6} | {'gap':>6}")
    for N in Ns:
        va, vt, ta, tt = res[N]; print(f"{N:>4} | {va:>8.3f} | {vt:>6.3f} | {va-vt:>+6.3f}")
    print(f"(sweep took {time.time()-t0:.1f}s)")

    print("\n### PIECE 5 - compare the gap with the formula 0.5 + sigma*sqrt(2 ln N)")
    sigma = 0.5/np.sqrt(NVA)
    print(f"sigma = 0.5/sqrt({NVA}) = {sigma:.4f}")
    print(f"{'N':>4} | {'measured app':>12} | {'formula':>8}")
    for N in Ns:
        va = res[N][0]; f = 0.5 + sigma*np.sqrt(2*np.log(N)) if N>1 else 0.5
        print(f"{N:>4} | {va:>12.3f} | {f:>8.3f}")

    print("\n### PIECE 6 - the test is not special: REUSE the test (select on it)")
    print("truth is 0.5 (known), so no second test is needed to expose the lie.")
    print(f"{'N':>4} | {'select on VAL':>13} | {'reuse the TEST':>14} | {'truth':>6}")
    for N in Ns:
        va, vt, ta, tt = res[N]
        print(f"{N:>4} | {va:>13.3f} | {ta:>14.3f} | {0.5:>6.3f}")
    print(f"(sigma_val = {0.5/np.sqrt(NVA):.4f} on n={NVA};  sigma_test = {0.5/np.sqrt(NTE):.4f} on n={NTE})")
