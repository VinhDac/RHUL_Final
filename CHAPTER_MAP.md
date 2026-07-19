# Chapter map (build spec for the dataset-journey rebuild)

*Working blueprint. Structure = the pipeline map (in §1) plus one deep journey per dataset. Every chapter has a KEY QUESTION and a decision-tree trail of small tries. Voice and rules live in `NORTH_STAR.md`.*

**Order:** §0 Trailhead → §1 What DL really is (+ the map) → Loan → Market → Phone → The Lab → Valley → Summit.

---

## §0 — The number we are about to trust  [KEEP as is]
- **Key question:** Can we trust the number?
- We judge a model by a number; we try many and keep the best; can we trust it? Promise to do everything by the book.
- Figures: none.

## §1 — What deep learning really is, and the plan  [ENRICH]
- **Key question:** This tool can fit almost anything. If it can fit anything, how do we know it learned the truth and not just the noise?
- Trail (essence, kept light):
  1. the model: a stack of simple parts (the MLP), a forward pass to a probability.
  2. training: gradient descent nudges the knobs until it fits the data.
  3. the catch: with enough knobs it can fit ANYTHING, including pure noise. [LOSS-CURVE figure]
  4. so the whole game is generalisation, not fitting. Which is exactly why we hold data out, search the knobs, and measure.
  5. that is the recipe (the pipeline). It looks airtight. We will walk it on real problems, asking at each step: is it as safe as it looks?
- Seeds every trap: loss = proxy (metric), capacity (overfit/search), hold-out + scale (split/drift).
- Figures: MLP (have), **LOSS-CURVE overfitting (NEW)**, pipeline map (have).

## Chapter LOAN — the well-behaved one that still lies
- **Key question:** The accuracy looks fine. But did the model catch the very people we built it for?
- Trail:
  1. Context: 30 000 credit clients, predict who defaults. One row = one person. [raw row]
  2. Split by the book. Worry: does the split matter? Check 11 ways → all agree, 0.813 ± 0.003. Relief: the book works, the split is a formality.
  3. Report accuracy 0.813, beats the 0.779 do-nothing baseline. Feels fine.
  4. KEY DOUBT: but what is it doing? The point was defaulters. → of 1 353 defaulters, caught 438, missed 915. [confusion figure]
  5. Realise: accuracy counts the easy majority; the rare defaulters vanish inside it. A do-nothing model scores 0.779; accuracy cannot tell them apart.
  6. Fix: balanced accuracy 0.633, or just read the confusion boxes.
- Lesson: measured perfectly, answered the wrong question. The split was fine here, so the trap was not in the data; it was in what we chose to measure.
- Traps: METRIC (and SPLIT vindicated = the honest control).
- Figures: confusion (have); optional loan trail-tree.

## Chapter MARKET — the edge that was too easy  (two traps)
- **Key question:** A real edge on the market? That is too easy. What is leaking in?
- Trail A, the split leak:
  1. Context: S&P closes. What can we even predict?
  2. Try direction (up/down) → 0.53, a coin flip. Dead end.
  3. Try size (a busy day) → shuffle 0.615. An edge? (suspicion: too easy)
  4. Check honest (past → future) → 0.585. They disagree! (this stopped us; loan's cuts all agreed)
  5. Look at the rows → near-twins, sharing 4 of 5. The shuffle leaks the twin. (self-critique: we never looked at the data)
  6. Confirm with the direction task → gap vanishes → a real leak. [market_tree figure]
- Trail B, the drift (same market):
  7. New doubt: we scaled by the book (freeze the training stats). Safe here?
  8. Build the feature in points not percent (innocent) → 0.512, the model is dead. (horror)
  9. Why? look again → the index climbs 1 455 → 7 483; points drift; the frozen scaler is lost. [drift figure]
  10. Fix: a rolling scaler → 0.549. Deeper: never build a drifting feature.
- Lesson: two traps in one dataset. The split leaked the future; the scaler froze a moving world. Both from not looking at the data.
- Traps: SPLIT + SCALE.
- Figures: market_tree (have), drift (have); optional split_leak.

## Chapter PHONE — recognising the person, not the task
- **Key question:** 0.973, near-perfect. But is it reading activities, or just recognising these thirty people?
- Trail:
  1. Context: 30 people, phone motion → activity. One row = a moment of one person. [raw rows: 3 in a row, same person]
  2. Shuffle by the book → 0.973. Amazing. (but we are wary now)
  3. KEY DOUBT: how would we use it? On a NEW person. Shuffle scatters each person to both sides.
  4. Hold out whole people (subject-wise) → 0.946, lower.
  5. Realise: part of 0.973 was recognising people, not activities.
  6. Real stumble to keep: at learning rate 0.5 the training diverged (scores 0.0–0.35, the model was not learning at all); we had to check it learns before trusting any comparison.
- Lesson: a third structure (people), the same leak. And: check the model actually learns before you trust any number it prints.
- Traps: SPLIT (entities).
- Figures: optional phone leak/tree.

## Chapter THE LAB — the trap that needs no data
- **Key question:** We keep tuning until the number looks good. Does trying harder make a better model, or just a luckier number?
- Trail:
  1. In every chapter we tuned: tried settings, kept the best. Is THAT safe?
  2. Cannot tell on real data (truth unknown). So build no-signal data: pure noise, label a coin flip. Truth = 0.5, known.
  3. Search: draw N settings, keep the best → winner's curse: best climbs 0.51 → 0.58, truth stays 0.50. [winner's curse figure]
  4. Realise: searching inflates any number; the gap = how hard we searched.
  5. The DL twist: DL does this by itself. Early stopping peeks at validation every epoch; seeds add tries; 5 settings become 150. It is a machine for it. The name: data snooping.
- Lesson: this trap is not in the data at all. It is in us, in the searching. And the tool we trust most snoops automatically.
- Traps: SEARCH.
- Figures: winner's curse (have).

## §Valley — so what is left?
- **Key question:** If a number can lie through the measure, through the data's structure, and through our own searching, what is left to trust?
- Short. Every path reached the same place: the number cannot carry our trust.

## §Summit — what we can trust
- Trust understanding made into a procedure. The careless practitioner and the honest one use the same steps; the only difference is understanding. Recap the four traps on the map.
- Figures: pipeline_traps (have).
- Ending: we cannot trust the goal; we can trust the understanding that earns it.

---

## New figures needed
- **LOSS-CURVE** (train down / validation up = overfitting + early stopping) — §1. Essential.
- (optional) overfitting toy (a wiggly fit of noise vs a smooth fit) — §1.
- (optional) small trail-trees for loan / phone (market already has one).

## Code plan (per dataset, replaces the by-experiment layout)
`mlp.py` (shared model) · `loan.py` · `market.py` · `phone.py` · `search_lab.py` · `data_peek.py`. Each script runs its dataset's whole trail and prints the tries in order, mirroring the chapter's decision tree. Appendices reorganised per dataset, each with a line-by-line data-transformation breakdown (print what one row is, and what it becomes after each step).
