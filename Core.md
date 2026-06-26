# Data Snooping in Deep Learning — core report

*Working report for the ~6 July go/no-go gate. Audience: supervisor.*
*__Order of work: reasoning first (why → hypothesis → method), code last.__ Every claim names a source you can verify.*

## Sources (verified Jun 2026)

| Tag | Source | Link | Used for |
|-----|--------|------|----------|
| `ISL` | James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning* | https://www.statlearning.com/ | the learning problem, train/test, model selection |
| `CT2010` | Cawley & Talbot (2010), JMLR 11:2079–2107 | https://www.jmlr.org/papers/v11/cawley10a.html | selection bias from model selection (this project) |
| `Nielsen` | M. Nielsen — *Neural Networks and Deep Learning* | http://neuralnetworksanddeeplearning.com/ | from-scratch network (later phase) |
| `Course` | `Docs/Lesson materials/` (Week1–3) | local | what you were taught |
| `Supervisor` | feedback in `Plan/Plan_2.docx` | local | the synthetic Gaussian design + the three cases |

---

## 1. Introduction — the problem

**The trusted practice.** To "improve" a model, we try many configurations (architectures, hyper-parameters), keep the one with the best validation score, and call that "the improvement".

**The hidden flaw.** A validation score is not true performance — it is *true performance + luck* (noise, because the validation set is finite, and here deliberately small). Keeping the best of *N* such scores means taking the **maximum of N noisy draws**, which usually lands on the *luckiest* configuration, not the *best*. So the reported score is **systematically inflated**, and the inflation grows with N.

**Why it matters.** Practitioners trust that number. If it is inflated, they deploy a *worse* model with *false confidence* — and the more they search, the more they fool themselves. (winner's curse = selection bias = data snooping: three names, one phenomenon.)

**Central hypothesis.** `gap = best-validation score − true performance` is **positive** and **grows with N**. We **measure** it — we do not predict it with a formula — using a sealed test set opened exactly once.

> Sources: core principle (`Plan/PROJECT_SPEC.md`); selection bias — `CT2010`.

## 2. Why a synthetic laboratory
*To write — only generated data gives exact true labels, so the gap is measured exactly and known noise can be injected. On real data the "true" performance is itself only an estimate.*

## 3. Hypotheses
*To write — the four questions: gap vs N / label noise / capacity / selection protocol.*

## 4. Method — the one machine
*To write — split → search N configs → keep best on validation → reveal once on sealed test → record the gap. One knob varies per experiment.*

## 5. The synthetic lab — design
*To write — the supervisor's Gaussian design: the matrix X, the three label cases (random / sign(X₁) / rotated), and why the isometry separates the methods. Code comes after the reasoning.*

## 6. Results
*Figures from `notebooks/01_core_snooping.ipynb` — later.*

## 7. Discussion
*Later.*

---

## Appendix A — code provenance (build log)
*Per piece, filled only when we implement (last): exact source → code → check.*

### `lab.py` · `make_X(n, d, rng)` — the Gaussian samples
**Why.** Supervisor: *"construct a matrix of Gaussian samples. This is your X — each row is one synthetic 'sample'."* (`Plan/Plan_2.docx`)
**Source.** numpy `Generator.standard_normal` — *"Draw samples from a standard Normal distribution (mean=0, stdev=1)"*; with `size=(m,n)` it draws `m*n` samples into that shape. <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html>
**Check (when coded).** `X.shape == (n, d)`; each column mean ≈ 0; each column std ≈ 1.
