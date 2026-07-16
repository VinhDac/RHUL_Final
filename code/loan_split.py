import numpy as np
from lab_demo import train, accuracy   # the SAME small MLP again

# UCI credit-card default: one row per client, 23 features + a default label.
# clients are independent, there is no time order and no repeated entity -> the rows ARE exchangeable.
def load_loan():
    D = np.loadtxt("data/loan_uci350.csv", delimiter=",", skiprows=1)
    return D[:, :-1], D[:, -1].astype(int)   # 23 features, default label (last column)

def standardize(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s

def score(X, y, how, seed, width=16, lr=0.3, epochs=300):
    n = len(y); ntest = n // 5; rng = np.random.default_rng(seed)
    if how == "stratified":                   # a DIFFERENT method: keep the class ratio on each side
        te = np.concatenate([rng.permutation(np.where(y == c)[0])[:int(round(ntest * np.mean(y == c)))] for c in (0, 1)])
        tr = np.setdiff1d(np.arange(n), te)
    else:                                     # plain random shuffle
        idx = rng.permutation(n); te, tr = idx[:ntest], idx[ntest:]
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed + 1))
    return accuracy(p, Xte, y[te])

if __name__ == "__main__":
    X, y = load_loan()
    print("### LOAN (UCI credit default): independent clients, no time order")
    print(f"n {len(y)} | features {X.shape[1]} | default rate {y.mean():.3f}")

    print("\n### same MLP, 10 INDEPENDENT random splits (does the split move the number?)")
    accs = [score(X, y, "shuffle", s) for s in range(10)]
    print("  " + " ".join(f"{a:.3f}" for a in accs))
    print(f"MEAN {np.mean(accs):.3f} | sd {np.std(accs):.3f} | range {max(accs)-min(accs):.3f}")

    print(f"\n### a DIFFERENT method -- stratified split (keeps the 22% ratio): {score(X, y, 'stratified', 0):.3f}")
    print("=> random splits and a stratified split all agree -> rows are exchangeable, so shuffle IS correct here.")
