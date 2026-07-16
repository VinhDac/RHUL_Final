# Data Snooping in Deep Learning

A Master's dissertation on why a model's reported score cannot be trusted on its own,
and what to trust instead.

It is written as a journey rather than a proof. We start where everyone starts, by the
book: follow every textbook rule for an honest number, and believe what it gives us.
Then we watch that trust come apart. First on pure noise, where trying enough
configurations inflates the score all by itself. Then on three real datasets, where the
same by-the-book recipe betrays us three separate ways: the split leaks the future or
the person, the frozen scaler quietly kills a working model when the world drifts, and
accuracy hides a model that catches almost none of the defaulters it was built to find.
Each safeguard we reach for turns in our hand.

The way out is not a better number or a cleverer method. It is to stop trusting the
score and trust the way it was made: a procedure whose assumptions are stated, checked,
and matched to the data in front of us. Underneath that, the real thing we trust is
understanding; the careless practitioner and the honest one run the same steps, and only
understanding tells them apart.

Every claim is grounded twice: derived in place from first-year probability, and
measured by a small experiment you can rerun. Nothing depends on a result you cannot
check.

## Layout

| Path | Purpose |
|------|---------|
| `Core.md` | the dissertation, read top to bottom; it stands on its own |
| `KEY_CORE.html` | the flowchart the dissertation follows (its source of structure) |
| `code/` | the experiments, numpy only, each one short and self-contained |
| `data/` | the frozen datasets |
| `Plan/` | the approved project plan |
| `Docs/` | course materials |

Inside `code/` (run each from the repo root, e.g. `python code/lab_demo.py`):

| File | Section | What it does |
|------|---------|--------------|
| `code/lab_demo.py` | §2.2, Appendix B | the synthetic gap lab; the hand-written MLP lives here and the other scripts import it |
| `code/loan_split.py` | §2.3-A | the i.i.d. control: every split agrees, so shuffle is correct |
| `code/finance_split.py` | §2.3-A | time order: shuffle leaks the future, walk-forward is honest |
| `code/har_split.py` | §2.3-A | grouped rows: record-wise recognises people, subject-wise is honest |
| `code/scaling_split.py` | §2.3-B | standardising a drifting feature quietly kills a real model |
| `code/metric_loan.py` | §2.3-C | the metric: accuracy looks fine while the model misses most defaulters; balanced accuracy is honest |
| `code/data_peek.py` | Appendix C | prints the raw head of each of the three datasets |

## Run

Only numpy is needed. From the repo root:

```
pip install -r requirements.txt
python code/lab_demo.py
python code/loan_split.py
python code/finance_split.py
python code/har_split.py
python code/scaling_split.py
python code/metric_loan.py
```

Everything is deterministic (`seed = 0`) and runs offline. The loan and market data are
frozen CSVs in `data/`. The activity data (UCI Human Activity Recognition, archive
dataset 240) is cached to `data/har.npz`; if that cache is missing, download and unzip
the dataset into `data/UCI HAR Dataset/` and `code/har_split.py` will rebuild it.
