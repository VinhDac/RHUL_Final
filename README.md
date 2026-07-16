# Data Snooping in Deep Learning

A Master's dissertation on why a model's reported score cannot be trusted on its own,
and what to trust instead.

The argument runs in two branches that meet at one conclusion. Snooping: chasing a
better number by trying too many configurations, and by using methods whose hidden
assumptions do not fit the problem. Deep learning: the black box where that search is
largest and least counted. Both land in the same place. We cannot trust the GOAL (the
number), only the PROCEDURE that produced it, with its assumptions made clear and
matched to the situation.

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
```

Everything is deterministic (`seed = 0`) and runs offline. The loan and market data are
frozen CSVs in `data/`. The activity data (UCI Human Activity Recognition, archive
dataset 240) is cached to `data/har.npz`; if that cache is missing, download and unzip
the dataset into `data/UCI HAR Dataset/` and `code/har_split.py` will rebuild it.
