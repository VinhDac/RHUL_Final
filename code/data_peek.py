import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=115)

print("="*72); print("LOAN   data/loan_uci350.csv"); print("="*72)
D = np.loadtxt("data/loan_uci350.csv", delimiter=",", skiprows=1)
print("shape:", D.shape, "  (23 features + 1 label)")
print("first 5 rows:"); print(D[:5])
print("columns: 0 LIMIT_BAL | 1 SEX | 2 EDUCATION | 3 MARRIAGE | 4 AGE |")
print("         5-10 PAY_0..PAY_6 | 11-16 BILL_AMT1..6 | 17-22 PAY_AMT1..6 | 23 default")
y = D[:, -1]
print(f"label: default rate {y.mean():.4f}  ({int(y.sum())} of {len(y)})")
print("exact duplicate rows:", len(D) - len(np.unique(D, axis=0)))
print("no id column, no date column -> nothing to order the rows by")

print(); print("="*72); print("MARKET   data/gspc_2026-07-03.csv"); print("="*72)
close = np.loadtxt("data/gspc_2026-07-03.csv", skiprows=1)
print("raw file: one column of closes, rows:", len(close))
print("close[:5] :", close[:5])
print("close[-5:]:", close[-5:])
print(f"drift: first {close[0]:.0f} -> last {close[-1]:.0f}   ({close[-1]/close[0]:.1f}x over the series)")
r = np.diff(close)/close[:-1]
print("returns  r[:5]:", r[:5])
a = np.abs(r)
print("size    |r|[:5]:", a[:5])
print(f"median(|r|) = {np.median(a):.5f}   <- the label threshold")
print(f"lag-1 autocorrelation of |r| = {np.corrcoef(a[:-1], a[1:])[0,1]:+.3f}   (busy days follow busy days)")
L = 5
X = np.column_stack([a[i:len(a)-L+i] for i in range(L)])
fut = a[L:]; X = X[:len(fut)]
yv = (fut > np.median(a)).astype(int)
print("X (5 lags of |r|) shape:", X.shape, "  y shape:", yv.shape)
print("X[:3] :"); print(X[:3])
print("y[:12]:", yv[:12], f"  balance {yv.mean():.3f}")

print(); print("="*72); print("ACTIVITY   data/UCI HAR Dataset  (cached data/har.npz)"); print("="*72)
d = np.load("data/har.npz"); Xh, yh, subj = d["X"], d["y"], d["subj"]
print("X shape:", Xh.shape, "  y shape:", yh.shape, "  subject shape:", subj.shape)
print("X[:3, :6]   (first 6 of the 561 features):"); print(Xh[:3, :6])
print("y[:12]   :", yh[:12], "  (0 WALKING, 1 UPSTAIRS, 2 DOWNSTAIRS, 3 SITTING, 4 STANDING, 5 LAYING)")
print("subj[:12]:", subj[:12])
u, c = np.unique(subj, return_counts=True)
print(f"subjects: {len(u)}   windows per subject: min {c.min()}, max {c.max()}, mean {c.mean():.0f}")
print("subject ids:", u)
print("-> every subject supplies hundreds of rows: the rows are NOT independent draws")
