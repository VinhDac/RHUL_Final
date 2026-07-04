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

A *configuration* is a model plus its settings (architecture, width, learning rate).
Picture ten students who know nothing sitting a short quiz: by luck the best scores
80%, but a long final reveals the truth, 50%. Configurations are the students, the
small validation set is the short quiz, the sealed test is the long final, and the
`gap` is the drop — a *larger* gap is *worse* (more of the reported score was luck).

## The argument at a glance

Read top to bottom — each arrow is one reasoning step; the two amber links
(deep-learning amplification and the cure) are what make the effect matter.

![Logic flow of the thesis — score = truth + luck; searching keeps the luckiest (winner's curse); deep learning amplifies it through hidden knobs (epochs, seed); measured exactly in a synthetic lab with a sealed test; confirmed on real data (loan, finance); cured by an honest, restrained procedure. Trust is a property of the process, not the number.](figures/logic_flow.svg)

## How to read this

`Core.md` is the report — read it top to bottom; it stands on its own. For more
depth, the appendix derives the maths in full; to reproduce any number, the
notebooks and `snooping_backend/` code are linked from the point that uses them.
Core.md is the spine — the code supports it, not the other way round.

## Layout

| Path | Purpose |
|------|---------|
| `Core.md` | the core report (reasoning first → PDF for the supervisor) |
| `snooping_backend/config.py` | canonical split sizes — the single source of truth |
| `snooping_backend/lab.py` | synthetic lab: Gaussian X, the labelling cases, noise, splits |
| `snooping_backend/mlp.py` | the searched instrument — a small PyTorch MLP (knobs: width, lr) |
| `snooping_backend/models.py` | the sklearn garden (fixed-setting kNN/tree/logreg/SVM) — Case-4 demo + isometry appendix |
| `snooping_backend/pipeline.py` | the machine: search → keep best on validation → reveal on sealed test → gap |
| `snooping_backend/experiments.py` | the E-2 (hidden search) and H5 (remedy) experiments |
| `snooping_backend/data_loan.py`, `data_finance.py` | real-data providers (loan default, finance) |
| `tests/` | fail-loud sanity checks (lab, mlp, pipeline) — exit non-zero on any failure |
| `data/` | input data frozen to CSV — offline, reproducible runs |
| `notebooks/` | runnable experiments + figures |
| `figures/` | output plots |

## Run

From the repo root:

```
pip install -r requirements.txt
python -m tests.test_lab && python -m tests.test_mlp && python -m tests.test_pipeline
jupyter notebook              # then open notebooks/01_core_snooping.ipynb
```

The input data is frozen to `data/*.csv` (committed), so everything runs **offline**
and reproducibly (`seed=0`); only regenerating that frozen data needs the internet.
The parameter sweeps in the notebooks take a few minutes each on CPU. Notebooks add
the repo root to `sys.path` so they can `import snooping_backend`.
