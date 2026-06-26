# Data Snooping in Deep Learning

Measure how much of a model's apparent improvement from searching configurations
is real, and how much is luck.

A validation score is only a noisy guess at true performance
(`validation = true + luck`). Searching N configurations and keeping the best
validation score keeps the *luckiest*, not the *best*, so that score is inflated
— and the inflation grows with the search. We **measure** the inflation with a
sealed test set, opened once:

```
gap = best_validation_score − sealed_test_score
```

## Layout

| Path | Purpose |
|------|---------|
| `Core.md` | the core report (reasoning first → PDF for the supervisor) |
| `snooping_backend/lab.py` | synthetic lab: Gaussian X, 3 labelling cases, isometry, noise, splits |
| `snooping_backend/models.py` | the model zoo (sklearn; from-scratch MLP later) |
| `snooping_backend/pipeline.py` | the machine: search → keep best on validation → reveal on sealed test → gap |
| `tests/test_lab.py` | sanity checks (incl. the isometry insight, proven by numbers) |
| `notebooks/` | runnable experiments + figures |
| `results/` | logged run records (CSV) |
| `figures/` | output plots |

## Run

From the repo root:

```
pip install -r requirements.txt
python -m tests.test_lab      # once the lab is implemented
jupyter notebook              # then open notebooks/01_core_snooping.ipynb
```

Notebooks add the repo root to `sys.path` so they can `import snooping_backend`.
