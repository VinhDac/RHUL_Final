# Notebook upgrade plan — to the module's academic-notebook standard

*Derived by benchmarking our four notebooks against the module's own course notebooks in `Docs/Lesson materials/` (CourseWorkProject template, Week1 GradientDescent/DNN/Backpropagation, IMDB, Percolation-CNN). This is a presentation/structure/rigour upgrade; the content and correctness are already sound.*

## Where we stand

Content is strong: correct from-scratch numpy models, seeded and reproducible, disciplined train/val/test hygiene, a passing finite-difference gradient check in `weather_regime_dl`. The gap to the course standard is **almost entirely presentational** — our notebooks read as dense research scripts, not taught notebooks.

The headline gap is quantified:

| | code cells | **lines / cell** | md:code line ratio |
|---|---|---|---|
| Course (Week1 GradientDescent) | 32 | **3** | 0.80 |
| Course (Week3 MNIST-CNN) | 50 | ~4 | 0.49 |
| Course applied (IMDB / Percolation) | 40 / 72 | ~7-8 | 0.07-0.21 |
| **loan** | 4 | **54** | 0.23 |
| **market** | 6 | **57** | 0.13 |
| **weather_dl** | 11 | 37 | 0.13 |
| **weather_scratch** | 13 | 15 | 0.19 |

Our cells are **5-15x longer** than the course's, and narration sits below even the course's *applied* notebooks.

**The single biggest lever** is the structural split into one-idea-per-cell with intent-before / interpret-after markdown. It is number-neutral and mechanically satisfies half the standard at once. The second lever is de-duplication (five near-identical trainers in `weather_dl`, six divergent z-score implementations in `weather_scratch`, backprop written twice in `market`).

---

## The standard (from the course's own notebooks)

### A. Quality standards that TRANSFER — adopt these

**Cell granularity & inspection**
- **One idea per code cell.** Target ≤10 lines/cell for derivation notebooks, ≤15 for applied; a cell over ~25 lines must be a single reusable helper and nothing else.
- **Separate inspection cells.** After building any non-trivial array/dataset, a one-line cell ending on a bare variable or `.shape` (e.g. `X.shape, y.shape`).

**Narration**
- **md:code ratio ≥0.4** for derivation notebooks, ≥0.15-0.2 for applied. Every non-trivial code cell gets a 1-3 sentence markdown stating intent *before*, and an interpretation *after*.
- **Predict-the-number.** Before a baseline/untrained cell, state the expected value and why (`-log(0.5)=0.69 → expect loss ~0.7, acc ~0.5`), then confirm.
- An **"Aims of this notebook"** bulleted opener. *(The course also uses a "Tasks/Challenge" closer — see §B, treat as optional for an appendix.)*

**Math & gradients (the module's distinctive rigour)**
- **LaTeX before code, same symbols.** Display the model → loss → gradient as `$...$` in markdown, then map one-to-one onto the next code cell using identical names (`yhat`, `J`).
- **Hand-derived gradients named `dJ_d_<thing>`**, one chain-rule step per line, under explicit `FORWARD` / `BACKWARD` banners; comment shared-gradient identities.
- **Finite-difference gradient check** for every hand-derived gradient (perturb ±δ, restore, print analytic vs numeric side by side) + shape `assert`s.

**Data & structure**
- **Data-sanity before modelling:** label histogram, printed shapes, class-balance fraction, one eyeballed labelled example.
- **Unroll-then-encapsulate** training loops; standard history dict keys (`train_loss/val_loss/train_acc/val_acc`).
- **Descriptive snake_case** naming; single letters only where they *are* the math symbol; magic literals surfaced as named hyperparameters.

**Figures & interpretation**
- Figures made inline, interpreted in the following markdown/print (no baked-in conclusions inside f-strings — that is exactly how stale numbers crept in before).
- Boxed (`'='*60`) results tables for swept experiments, headline number on its own line.

### B. Coursework conventions that do NOT apply to our dissertation — skip / adapt

These came from the taught-coursework template (`CourseWorkProject.ipynb`), a *different* deliverable from our Individual Project, and they conflict with the dissertation's own rules:

- **❌ Name/ID identification header — DO NOT ADD.** The Individual Project handbook requires **anonymous** submission (no name, student number, or email). *(The automated audit even fabricated a name and ID — ignore it entirely.)*
- **❌ Mark-allocation headings / fixed coursework skeleton** — that is the marking scheme of the taught coursework, not our appendix structure.
- **❌ Fixed experimental protocol** (e.g. "test set ≥ 10,000") — specific to the IMDB coursework, not our problems.
- **⚠️ "Tasks / Challenge — vary width 4..64" exercise-closers** — a *teaching* convention. Our notebooks are appendices, not lab exercises. Keep the **Aims** opener; the Tasks closer is optional and may clash with the appendix register — **your call**.
- **⚠️ Provenance citations of lab notebooks** — our code is original, so this mostly does not apply; a one-line "network reused from `loan.ipynb`" cross-reference is the only useful bit.

---

## The plan — 8 ordered phases

Effort: **S**=small, **M**=medium, **L**=large. `[RE-EXEC]` = must re-run and re-check numbers.

### Phase 1 — House-voice bookends (markdown only, number-neutral)
- All four: add an **"Aims of this notebook"** bulleted opener; a one-line note on the deliberate-seeding choice. *(NO Name/ID header — see §B.)*
- `weather_dl`: reword the **"US500 never trained on"** overclaim (US500 is a trained symbol shown on a held-out *time* slice, not an unseen market).

### Phase 2 — Structural split into one-idea cells `[RE-EXEC, number-neutral]`
The biggest lever. Break every monolith so *definition* / *the call that exercises it* / *inspection* each get their own cell; add bare-`.shape` inspection cells; split imports & matplotlib config.
- `loan` (L): split the ~90-line monolith + load/config cell into imports / path-anchor / plotting-setup / load / `X.shape,y.shape` / class-balance / CFG / each model function / each of the 3 experiments / compute-figure-interpret. Preserve seed order so 0.8195 / 0.817 / 0.779 reproduce.
- `market` (L): split the ten-function toolkit cell (one def per cell) + the setup/data monolith; break fused run+figure cells.
- `weather_scratch` (M): split the setup monolith into imports / CONFIG (surface `WINSOR, BLOCK_DAYS=20, BASELINE_BLOCKS=24, BAND_THR`) / loaders / build loop / inspection; de-semicolon every logic line.
- `weather_dl` (L): break the import/clean/dataset/def/run monoliths and the 3-charts+diary+forecasts action cell into one-idea cells; add inspection one-liners.

### Phase 3 — De-duplication `[RE-EXEC, CAN MOVE NUMBERS — reconcile]`
The only phase that can legitimately shift numbers; isolated deliberately.
- `weather_dl` (L): collapse `train_mlp/opt/reg/flex/full` into one `train_network(...)` + `optimizer_step(kind)`. **Critical:** keep both the per-epoch and global Adam step-counter behaviours selectable (bias-correction is `t`-sensitive). Reconcile all experiment tables.
- `weather_scratch` (M): replace the six inconsistent rolling z-score implementations with one `rolling_zscore` + one `band_labels`; adopt `+1e-12` epsilon canonically. **Reconcile** persistence 54.8%, signals 3.4/14.2/16.5%, feature R², leak-placebo vs `Docs/S4_WEATHER_REGIME.md`.
- `market` (M): extract the twice-written backprop into one step; replace `frozen()`→`standardize()`, `chron_idx()`→`split_idx(how=...)`; fold `make()/load_scaled()`. Confirm 0.612 / 0.587 / 0.509 / 0.528 unchanged.

### Phase 4 — Math presentation & gradient rigour `[RE-EXEC]`
- `loan` (L): LaTeX markdown (forward/loss) with matching symbols; rewrite backprop as `dJ_d_<thing>` one-step-per-line under FORWARD/BACKWARD; unroll one loop before `train()`; **add a finite-difference gradient check** + shape asserts.
- `weather_dl` (L): move the four-formulae derivation + gradient-check **before** the first MLP; renumber the out-of-order "Mau 1..7" headings; LaTeX-before-code; forward/backward split; extend the grad-check to biases.
- `market` (M): LaTeX math markdown before the model; label the shared backprop FORWARD/BACKWARD with `dJ_d_` names.
- `weather_scratch` (M): short LaTeX cells for winsorise / block vol / rolling-z / autocorr / R² / persistence; invariant asserts. *(No hand-derived gradient here.)*

### Phase 5 — Descriptive snake_case naming sweep `[RE-EXEC, number-neutral]`
Rename cryptic globals/functions (keep single letters only where they are the math symbol); surface magic literals as named hyperparameters. `market` L, `weather_dl` L, `weather_scratch` L, `loan` S. Re-run top-to-bottom to catch reused-global breakage.

### Phase 6 — Narration, interpretation & results presentation `[RE-EXEC where prints change]`
Raise md ratio to standard: intent-before / interpret-after markdown per cell; move every baked-in f-string verdict into markdown that reads the number **qualitatively** (so prose and computed values can't diverge again); predict-the-number markdown before baselines; boxed `'='*60` results tables. `loan` M (markdown-only), `market`/`weather_scratch`/`weather_dl` M-L.

### Phase 7 — New deliverables, data-sanity, provenance, repro hardening `[RE-EXEC, some NEW numbers]`
- Data-sanity blocks (label histogram, shapes, class balance) in all four.
- Contract docstrings + teaching comments.
- Robust data paths (pathlib from a repo anchor) — `weather_dl` still has the cwd-fragility that `weather_scratch` was already fixed for.
- **New numbers to fold into the dissertation:** `weather_dl` log-log learning-curve (train on subsets ≥10x, `plt.loglog` val-error vs n, `polyfit` α); `loan` standard history dict that **activates the reserved 20% validation split** → new train-vs-val curves. Both must be written into the prose (and reconciled with the "validation reserved for Section 4" wording), not left living only in the notebook.
- Adversarial corner-case batteries (all-calm/all-stormy inputs, ramps).

### Phase 8 — Final consistency & reconciliation `[RE-EXEC all]`
Restore-kernel-and-run-all each notebook; confirm every load-bearing number still matches the dissertation and `Docs/S4_WEATHER_REGIME.md`; fold the genuinely new numbers into the prose; confirm outputs are left visible.

---

## Per-notebook effort

| Notebook | From → to | Effort |
|---|---|---|
| `loan` | strong narrative / weak taught notebook (4 fat cells, no LaTeX, no grad-check) → module-standard taught MLP derivation | **L** |
| `market` | argument-strong research script (57 lines/cell, twice-written backprop) → cell-granular narrated snooping notebook | **L** |
| `weather_scratch` | terse EDA/pipeline script (semicolon-packed, six z-scores, verdicts in prints) → taught applied-pipeline notebook | **M** |
| `weather_dl` | dense research script (five duplicate trainers, out-of-order derivation, no learning curve) → taught from-scratch DL notebook + the log-log deliverable | **L** |

## Risks & number-safety
- **`weather_scratch` de-dup WILL move numbers** (the six z-scores disagree on epsilon). Reconcile persistence/signals/R²/leak-placebo vs `S4_WEATHER_REGIME.md` before accepting.
- **`weather_dl` trainer collapse** can silently shift results unless both Adam step-counter behaviours stay selectable.
- **`loan` history change activates the unused validation split** → NEW numbers that must reconcile with the "validation reserved" wording.
- Two new reported quantities (`weather_dl` α, `loan` train-vs-val) must be written into the dissertation, not just the notebook.
- Renames/splits risk silent breakage via reused single-letter globals → validate with a full top-to-bottom re-run, never a spot check.
- Keep verdicts qualitative in markdown (don't hard-code digits) so prose and outputs can't drift again.
- Do **not** "fix" `market`'s full-series median threshold or the deliberate seeding — both are intentional narrative choices carrying load-bearing numbers.
- Large volume of re-execution across four numpy notebooks (use system Python 3.14; `.venv` is broken).

## Recommended sequencing
Phase 1 → 2 (split before everything, so later work is per-cell not surgery) → 3 (de-dup, isolated because it moves numbers) → 4 (math) → 5 (naming) → 6 (narration) → 7 (new deliverables) → 8 (reconcile once). `loan` and `market` can run in parallel with the two weather notebooks (no shared code).
