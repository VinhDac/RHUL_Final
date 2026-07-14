import numpy as np, os

# ---- the SAME one-hidden-layer MLP as section 2.2, now with K outputs (one per activity) ----
def init(d, width, K, rng):
    return [rng.standard_normal((d, width))*np.sqrt(2.0/d), np.zeros(width),
            rng.standard_normal((width, K))*np.sqrt(2.0/width), np.zeros(K)]
def forward(X, p):
    W1,b1,W2,b2 = p; a = X@W1+b1; h = np.maximum(a, 0.0); z = h@W2+b2; return a,h,z
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e/e.sum(1, keepdims=True)
def train(Xtr, ytr, K, width, lr, epochs, rng):
    p = init(Xtr.shape[1], width, K, rng); n = len(ytr)
    Y = np.zeros((n, K)); Y[np.arange(n), ytr] = 1.0
    for _ in range(epochs):
        a,h,z = forward(Xtr, p); dz = (softmax(z) - Y)/n
        W1,b1,W2,b2 = p; dW2 = h.T@dz; db2 = dz.sum(0)
        da = (dz@W2.T)*(a > 0); dW1 = Xtr.T@da; db1 = da.sum(0)
        p = [W1-lr*dW1, b1-lr*db1, W2-lr*dW2, b2-lr*db2]
    return p
def accuracy(p, X, y): return (forward(X, p)[2].argmax(1) == y).mean()

# ---- HAR: 561 features, activity 1-6, subject 1-30 (many windows per subject) ----
def load_har():
    if os.path.exists("data/har.npz"):
        d = np.load("data/har.npz"); return d["X"], d["y"], d["subj"]
    b = "data/UCI HAR Dataset/"
    X = np.vstack([np.loadtxt(b+"train/X_train.txt"), np.loadtxt(b+"test/X_test.txt")])
    y = np.concatenate([np.loadtxt(b+"train/y_train.txt"), np.loadtxt(b+"test/y_test.txt")]).astype(int) - 1
    subj = np.concatenate([np.loadtxt(b+"train/subject_train.txt"), np.loadtxt(b+"test/subject_test.txt")]).astype(int)
    np.savez("data/har.npz", X=X, y=y, subj=subj)
    return X, y, subj

def standardize(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m)/s, (Xte - m)/s

def score(X, y, subj, how, seed, K=6, width=64, lr=0.1, epochs=400):
    n = len(y); rng = np.random.default_rng(seed)
    if how == "subject-wise":                 # RIGHT: hold out whole SUBJECTS -> test on new people
        subs = rng.permutation(np.unique(subj)); test_subs = subs[:max(1, len(subs)//5)]
        te = np.isin(subj, test_subs)
    else:                                     # WRONG: shuffle windows -> same person on both sides
        te = np.zeros(n, bool); te[rng.permutation(n)[:n//5]] = True
    tr = ~te
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], K, width, lr, epochs, np.random.default_rng(seed + 1))
    return accuracy(p, Xte, y[te])

if __name__ == "__main__":
    X, y, subj = load_har()
    print("### HAR (activity recognition): 561 features, 6 activities, 30 subjects")
    print(f"windows {len(y)} | subjects {len(np.unique(subj))} | classes {len(np.unique(y))}")

    print("\n### same MLP, record-wise (WRONG) vs subject-wise (RIGHT), 10 seeds")
    print(f"{'seed':>4} | {'record':>7} | {'subject':>7} | {'diff':>6}")
    rw, sw = [], []
    for s in range(10):
        a = score(X, y, subj, "record-wise", s); b = score(X, y, subj, "subject-wise", s)
        rw.append(a); sw.append(b)
        print(f"{s:>4} | {a:>7.3f} | {b:>7.3f} | {a-b:>+6.3f}")
    print(f"MEAN | {np.mean(rw):>7.3f} | {np.mean(sw):>7.3f} | {np.mean(rw)-np.mean(sw):>+6.3f}")
    print(f"(record > subject in {sum(a>b for a,b in zip(rw,sw))}/10 seeds)")
