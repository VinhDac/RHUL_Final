# Data Snooping in Deep Learning — Project Spec (handoff to Claude Code)

> **How to use this file:** This is the working specification for an MSc dissertation
> project. Read it fully before writing any code. The **Guardrails** section is binding —
> it encodes the author's working style. Do not add complexity, methods, or formulas that
> are not derived from the Core Principle. When in doubt, do less.

---

## 0. Status (what is locked, what is pending)

**Locked:**
- Core principle, research question, four hypotheses.
- Three datasets (sources verified).
- Shared experiment pipeline + search space.
- Experiments E0–E5 (designs).
- Basic-math-only rule.
- σ·√(2 ln N) extreme-value formula is **removed** (see Guardrails).

**Pending supervisor confirmation (do NOT block E0/E1 on these, but flag in the dissertation):**
1. The "clean/controlled" dataset = **general synthetic** (not loan-flavored). This is a
   change from the approved plan, which had both datasets as loan default. Awaiting sign-off.
2. Network implementation: **from-scratch NumPy MLP** vs framework. Working assumption below.
3. Overall scope vs the 14 Aug draft deadline.
4. AI-tool usage policy (handbook §6.1).

**First milestone (build this first):** E0 + E1 on the synthetic lab → produce the headline
figure *apparent-vs-true accuracy as N grows*. Everything else waits on this result.

---

## 1. Core Principle (everything derives from this)

> **A validation score is only a noisy estimate of true performance.**
> validation_score = true_performance + luck_noise

Every part of this project is a direct consequence of this one fact. If a proposed feature
does not follow from it, it does not belong in the project.

**The trap (winner's curse / selection bias / data snooping — same thing, three field names):**
When we train **N** configurations and keep the one with the best validation score, we are
taking the **maximum of N noisy draws**. The winner is usually the *luckiest*, not the *best*.
So the best validation score is **systematically inflated** above true performance, and the
inflation grows with N.

**How we measure it:** a **sealed test set**, opened exactly once at the very end, after the
winning configuration is chosen. It is the unfakeable arbiter of true performance.

> **gap = best_validation_score − sealed_test_score**

---

## 2. Research Question & Hypotheses

**Question:** When we search many configurations and keep the best validation score, how much
of the apparent performance is real, and how much is luck — and what makes the gap bigger or
smaller?

Each hypothesis is tested by **direct measurement** against the sealed test set. We never
*predict* the gap with a formula we do not fully understand; we *measure* it.

- **H1 — a gap exists and grows with N (search size).**
  Test: run the pipeline at rising N, measure gap, plot. Watch for a turning point
  (true performance rises → plateaus → falls as N grows).
- **H2 — more label noise → bigger gap.** (lab only, where noise is injectable)
  Test: inject known noise levels, measure gap at each.
- **H3 — bigger capacity → bigger gap.**
  Test: vary capacity (depth × width), measure gap at each.
- **H4 — honest selection protocols shrink the gap.**
  Test: compare single split / k-fold / nested CV (+ correct early stopping) by measured gap.
  (Possible honest outcome: a protocol changes only confidence, not the selection — that is
  also a result.)

Each experiment ends with a clear verdict: **confirmed or refuted, reported honestly.**

---

## 3. Guardrails (BINDING — the author's working style)

1. **Essence-first.** Every element must be a consequence of the Core Principle. Reject
   complexity added for appearance — it breeds bugs and fails in practice.
2. **Hypothesis-first.** Every experiment has the shape: *hypothesis → why → how to test →
   verdict from data*. Formulas appear only as a hypothesis's prediction, then confirmed or
   refuted by data. Report refutations honestly; never rationalize them away.
3. **Basic math only.** Allowed operations, and nothing else:
   - subtraction (the gap),
   - mean across repeated runs,
   - spread (standard deviation, or min–max),
   - reading trends from plots,
   - counts on a confusion matrix (for the loan consequence),
   - cumulative return + a simple Sharpe ratio = mean(returns) / std(returns) (for finance).
   **Forbidden:** extreme-value formulas, order-statistic approximations, or any statistic the
   author has not fully understood. Specifically, **do NOT introduce σ·√(2 ln N)** or similar.
   The whole point is that we *measure* the gap, not approximate it with something we cannot
   defend to an examiner.
4. **Sealed-test discipline.** The test set is touched **once**, only to read true performance
   of the already-chosen configuration. It is **never** used to select, tune, or early-stop.
   Any leak invalidates the measurement.
5. **Plain language.** Code and comments should be simple and read-once. Technical terms are
   fine but kept fast to read.
6. **Honest reporting.** If the gap is small, or does not grow with N, that is a real finding,
   not a failure to hide. The dissertation reports what the data says.
7. **The dissertation is what is marked, not the tooling.** Do not over-invest in
   infrastructure or UI. One reproducible harness + clear figures is enough.

---

## 4. Datasets (3 roles, sources verified)

The three datasets sit on **one axis — true signal strength**. The synthetic lab is the dial;
loan and finance are two real points on that dial.

### 4.1 Lab — synthetic (we generate it; truth is known)
- **Why synthetic:** to measure the gap *exactly* and inject *known* noise, we must know the
  true labels. Only generated data gives this. (Even the UCI loan authors note the true
  default probability is unknown and must be estimated — real data cannot give exact truth.)
- **How:** `sklearn.datasets.make_classification`, with explicit control of:
  - `n_informative` (number of useful features),
  - `class_sep` (**signal strength** — turn this from high → near 0 to reproduce loan vs finance),
  - `flip_y` (**label noise** — known injected noise for H2),
  - `n_samples`.
- **Splits:**
  - train: a few thousand rows.
  - **validation: small (a few hundred)** — deliberately small so noise is high and snooping
    is strong (matches the supervisor's "small validation set").
  - **sealed test: very large (50k–100k)** — so the test score ≈ true performance (sampling
    error ≈ 0). Opened once, only to measure.
- **Role:** measure the gap precisely for E1–E4; dial signal to connect all three datasets.
  This is the **heart of the project** — where the marks are earned.

### 4.2 Loan default — UCI "Default of Credit Card Clients"
- **Verified:** 30,000 rows, 23 features, binary target (default next month); Taiwan,
  Apr–Sep 2005; license **CC BY 4.0** (free use with attribution).
- **Load:** `from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=350)`.
- **Citation:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI ML Repository.
  https://doi.org/10.24432/C55S3H
- **Character:** real, moderately messy, imbalanced (~1 in 5 default). Has genuine signal
  (payment-history features) → the gap should appear but not be catastrophic.
- **Role:** the **stakes** case. Translate the gap into a consequence: **how many bad loans
  get wrongly approved** (counted at the decision threshold from a confusion matrix).

### 4.3 Finance — daily prices via `yfinance`
- **Verified:** `yfinance` is the common free way to pull historical daily prices since Yahoo
  retired its data API; good for research/backtesting; long histories available. Tool is
  Apache-licensed; the *data* follows Yahoo's terms (research/personal use). It is a scraper —
  heavy call volume can get the IP rate-limited, and data needs basic cleaning.
- **Start:** one liquid index, **^GSPC (S&P 500)**. Extension: a basket of stocks (more
  configurations to search → stronger snooping).
- **Role:** the **extreme** case. Predictable signal ≈ 0, so the "edge" found by searching is
  almost entirely luck → the best-on-validation strategy **collapses out-of-sample**. This is
  the supervisor's "shocking warning to practitioners." Consequence: **money lost** (equity
  curve soars then crashes).

---

## 5. The Shared Machine (one pipeline, used for all datasets)

```
1. Split data into train / validation / SEALED test.
   (Finance: split CHRONOLOGICALLY — walk-forward. Never shuffle time.)
2. Define the configuration search space (Section 6).
3. Search: sample N configurations at random from the space.
   For each: train on train, score on validation.
4. Select: keep the configuration with the best validation score. (= the snoop)
5. Reveal: score that configuration on the SEALED test set. (= true performance)
6. Record (best_validation_score, sealed_test_score).
   gap = best_validation_score − sealed_test_score
```

The experiments differ only in **which single knob they vary** around this machine.

---

## 6. Search Space (6 knobs, ~1,944 configurations)

| Knob            | Values                                  |
|-----------------|-----------------------------------------|
| depth           | 1, 2, 3 hidden layers                   |
| width           | 16, 32, 64, 128                         |
| activation      | ReLU, tanh                              |
| L2              | 0, 1e-4, 1e-3                           |
| dropout         | 0, 0.2, 0.5                            |
| optimizer       | SGD, SGD+momentum, Adam                 |
| learning rate   | 0.1, 0.01, 0.001                       |

Total ≈ **1,944** configurations. **Search = sample N at random** (random search — simple and
standard). N is the variable swept in E1.

Note the **dual role of regularization** (L2, dropout): it is both a search knob *and* one of
the few genuine gap-*reducers* (it limits capacity → less memorizing of validation noise).
**Early stopping** is itself a form of selection-on-validation, so it can *cause* snooping if
done carelessly — handle it explicitly in E4.

---

## 7. Experiments

For each: **hypothesis → method → what varies → outputs → math used.**

### E0 — sanity + convergence (do first; satisfies Early Deliverable 1)
- **Goal:** confirm the network learns at all; show the effect of hyperparameters on convergence.
- **Method:** train a few small configs on the lab data; plot loss vs epoch for several
  learning rates.
- **Outputs:** loss-vs-epoch curves.
- **Math:** none beyond reading curves.

### E1 — gap vs N (the backbone)
- **Hypothesis:** H1.
- **Method (pseudocode):**
  ```
  for N in [1, 2, 5, 10, 20, 50, 100, 200]:
      repeat R = 30 times:
          sample N configurations at random from the search space
          train each on train, score each on validation (small)
          keep the configuration with the best validation score
          score it on the SEALED test set  ->  true performance
          record (best_validation_score, sealed_test_score)
      gap(N) = mean(best_validation_score - sealed_test_score) over R
      also record spread (std or min-max) over R
  ```
- **Outputs (the headline figure):** apparent (best-validation) vs N, true (sealed-test) vs N,
  on one axis, with the gap band. Watch for the turning point.
- **Math:** subtraction, mean over R, spread, trend.

### E2 — gap vs label noise (lab only)
- **Hypothesis:** H2.
- **Method:** at fixed N, set `flip_y` ∈ {0, 0.05, 0.10, 0.20}; measure gap at each level.
- **Outputs:** gap vs noise level.
- **Math:** subtraction, mean, spread.

### E3 — gap vs capacity
- **Hypothesis:** H3.
- **Method:** at fixed N, vary capacity (depth × width) across a few levels; measure gap at each.
- **Outputs:** gap vs capacity.
- **Math:** subtraction, mean, spread.

### E4 — selection protocol vs gap
- **Hypothesis:** H4.
- **Method:** repeat the select-then-reveal loop under: single split / k-fold / nested CV
  (and correct vs careless early stopping); compare the true performance of the chosen config.
- **Outputs:** gap by protocol.
- **Math:** subtraction, mean, spread. (Honest possible result: protocol changes confidence,
  not the choice.)

### E5 — reveal in the wild (re-run E1 on real data)
- **5a Loan (UCI id 350):** same pipeline; does the gap still appear? Translate to consequence:
  count bad loans wrongly approved at the decision threshold (confusion matrix).
- **5b Finance (^GSPC):** chronological / walk-forward split; search many configurations
  (models predicting next-period direction, or simple strategies); the best-on-validation
  configuration's **equity curve collapses** on the untouched test period; gap vs N.
  Translate to consequence: **money** (cumulative return; simple Sharpe = mean/std of returns).

---

## 8. Implementation Notes

- **Network (working assumption, pending supervisor sign-off):** build a **from-scratch MLP
  with backprop in NumPy** for the lab. Reasons: earns "deep understanding" marks and satisfies
  "derive the formulae for the methods you use." A framework is acceptable for heavy real-data
  runs if from-scratch is too slow — but the lab should be from-scratch unless the supervisor
  says otherwise.
- **Formulas to derive in the dissertation** (because we *use* them): backprop; the optimizer
  update rules (SGD, SGD+momentum, Adam); L2 regularization. These — **not** any snooping
  bound — fulfil the formula-derivation deliverable.
- **Reproducibility:** fix random seeds; log every run's full configuration + both scores so
  results are reproducible and auditable. The sealed test set must be created once and never
  read during search.
- **Keep it simple:** plain NumPy + a little pandas/sklearn for data loading and splitting.
  No heavyweight infrastructure. One reproducible harness, clear figures.

**Suggested repo structure:**
```
/data            # loaders for synthetic, UCI loan, yfinance prices
/model           # from-scratch MLP + backprop + optimizers
/search          # config space, random sampling, the shared pipeline
/experiments     # e0_sanity, e1_gap_vs_n, e2_noise, e3_capacity, e4_protocols, e5_real
/figures         # output plots
/results         # logged run records (config + scores)
/notebooks       # thin story notebooks per experiment / dataset
README.md
```

---

## 9. Deliverable mapping (against the project brief)

- **Early D1** (POC on simple data + hyperparameter→convergence): **E0**.
- **Early D2** (overview of related DL methods): dissertation Background chapter.
- **Early D3** (assess initial implementation): E0 + E1 write-up.
- **Final D1** (real dataset): **E5a (loan), E5b (finance)**.
- **Final D2** (compare multiple **architectures and** optimisation methods): the search space
  varies depth/width/activation (architectures) **and** SGD/momentum/Adam (optimizers);
  surfaced in E1/E3.
- **Final D3** (visualise + assess performance): all figures, esp. the E1 headline figure.
- **Final D4** (derive formulae for methods used + discuss implementation): backprop +
  optimizer update rules + L2, in the Method chapter.

---

## 10. Dissertation Structure (the writing target — 7 chapters)

Standard academic scaffold; the *drama* comes from the order of revelation and from
translating the gap into consequences, not from an unusual structure.

1. **Introduction** — state the trusted practice (search configs → keep best validation →
   call it "the improvement"); state the Core Principle; the research question; aims,
   objectives, **career relevance** (handbook-required).
2. **Background** — the same phenomenon across fields: winner's curse, selection bias
   (Cawley & Talbot 2010), generalisation gap (Recht et al. 2019), data snooping (White 2000;
   verify each citation before use). Overview of related DL methods; selection protocols.
3. **Method — the instrument** — from-scratch MLP (derive backprop); the search space;
   optimizers (derive their formulas); convergence (E0); the sealed test set; selection
   protocols. Basic-math-only statement.
4. **Controlled lab — measure the gap** (THE CORE) — E1 (gap vs N, headline figure), E2
   (noise), E3 (capacity), E4 (protocols). Confirm/refute each hypothesis.
5. **Real-world demonstrations** — E5a loan (gap + bad loans approved), E5b finance (chrono
   split, equity-curve collapse, money lost — the shocking warning).
6. **Discussion / critical review** — how much is real vs luck, what drives it, what reduces
   it, where the picture breaks; report refutations. (Weighted heavily by the second marker.)
7. **Conclusion + self-assessment** (handbook-required) — restate the answer; appraise what
   went right/wrong, lessons on planning & execution, where next.

Appendices: code + reproducibility, full results, dataset sources & licences.

---

## 11. Definition of Done — first milestone

Build E0 + E1 on the synthetic lab and produce:
- a from-scratch MLP that trains (E0 loss curves), and
- the **headline figure**: apparent-vs-true accuracy as N grows, with the gap band, over
  R = 30 repeats.

This tests the central bet of the whole project — *does a gap appear and grow with N?* — at
the lowest cost. Do not expand to E2–E5 or harden the harness until this figure exists and the
gap behaviour is understood.
