import numpy as np
from lab_demo import train, forward   # the SAME small MLP again

# UCI credit-card default: one row per client, 23 features + a default label (1 = defaulted).
# the classes are IMBALANCED (~22% default). the split is honest and there is no drift here,
# so nothing leaks: the number is measured perfectly. the point is that accuracy measures
# the WRONG thing when one class is rare.
def load_loan():
    D = np.loadtxt("data/loan_uci350.csv", delimiter=",", skiprows=1)
    return D[:, :-1], D[:, -1].astype(int)   # 23 features, default label (last column)

def standardize(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s

def predict(p, X):
    return forward(X, p)[2].argmax(1)

def run(seed, width=16, lr=0.3, epochs=300):
    X, y = load_loan()
    n = len(y); ntest = n // 5
    idx = np.random.default_rng(seed).permutation(n)     # honest random split (loan is exchangeable)
    te, tr = idx[:ntest], idx[ntest:]
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed + 1))
    yhat, ytrue = predict(p, Xte), y[te]
    TP = int(((yhat == 1) & (ytrue == 1)).sum())         # default(1) = the positive class
    FP = int(((yhat == 1) & (ytrue == 0)).sum())
    TN = int(((yhat == 0) & (ytrue == 0)).sum())
    FN = int(((yhat == 0) & (ytrue == 1)).sum())
    acc = (yhat == ytrue).mean()
    recall = TP / (TP + FN) if (TP + FN) else 0.0         # of real defaulters, fraction caught
    spec = TN / (TN + FP) if (TN + FP) else 0.0           # of non-defaulters, fraction correct
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    bal = 0.5 * (recall + spec)                           # treats both classes equally
    return acc, recall, spec, prec, bal, (TN, FP, FN, TP)

if __name__ == "__main__":
    X, y = load_loan()
    rate = y.mean()
    print("### LOAN metric: the number is measured perfectly, and still measures the wrong thing")
    print(f"n {len(y)} | features {X.shape[1]} | default rate {rate:.3f}")
    print(f"baseline 'always predict NO default': accuracy {1-rate:.3f}, defaulters caught 0 of {int(y.sum())}")

    res = np.array([run(s)[:5] for s in range(10)])
    acc, recall, spec, prec, bal = res.mean(0)
    print("\n### same MLP, honest random split, mean over 10 seeds (no leakage, no drift)")
    print(f"  accuracy            {acc:.3f}   (baseline {1-rate:.3f}, so only +{acc-(1-rate):.3f})")
    print(f"  recall (defaulters) {recall:.3f}   <- of the real defaulters, the fraction actually caught")
    print(f"  precision           {prec:.3f}")
    print(f"  specificity         {spec:.3f}")
    print(f"  balanced accuracy   {bal:.3f}   <- treats both classes equally")

    TN, FP, FN, TP = run(0)[5]
    print(f"\n### one split's confusion (seed 0): TN {TN}  FP {FP}  FN {FN}  TP {TP}")
    print(f"  of {FN+TP} real defaulters in this test set, the model caught {TP} and missed {FN}")
