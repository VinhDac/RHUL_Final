# Project Plan v2 — Data Snooping in Deep Learning

**Topic:** measure how much of a model's "improvement" from searching architectures + hyperparameters is real, and how much is just luck.

## The one fact everything grows from
A validation score is only a noisy guess at true performance ⇒ when I try many configurations and keep the best validation score, I keep the *luckiest*, not the *best* ⇒ that score is inflated, and the inflation grows the more I search.

I **measure** this, I don't predict it: a sealed test set, opened **once** at the very end ⇒ `gap = best-validation score − sealed-test score`.

## Research question
When I improve a network by trying many configurations and keeping the best validation score, how much of that improvement is real, and how much is the illusion of having picked the luckiest result?

## What drives the gap (my four questions)
Each: state what I expect ⇒ let the data confirm or refute ⇒ report honestly, even against me.
- more configurations searched (N) ⇒ bigger gap?
- more label noise ⇒ bigger gap?
- more model capacity ⇒ bigger gap?
- which honest protocol (single split / k-fold / nested CV) shrinks it?

## Datasets — three points on one axis (signal strength)
1. **Synthetic lab** (I generate it; truth known). Why: only generated data gives *exact* true labels ⇒ I can measure the gap exactly and inject *known* noise. This is where I earn the core results.
2. **UCI loan default** (real, messy, stakes). Does the gap appear on its own? Consequence: how many bad loans get wrongly approved.
3. **Financial prices (^GSPC)** — *your suggestion*. Signal ≈ 0 ⇒ the "edge" found by searching is almost all luck ⇒ the best-on-validation strategy collapses out-of-sample. Chronological (walk-forward) split. Consequence: money lost. The shocking warning.

**One change I'd like to confirm: the clean set → synthetic.**
To measure the gap I must compare the selected model against its *true* performance. On real data (loan included) the "true" performance is itself only an estimate from a finite test set ⇒ the gap I measure is blurred by the very noise I am trying to study. With a synthetic set I control the data-generating process, which buys three things real data cannot give:
- **exact truth** — I can make the test set as large as I like (e.g. 100k rows) ⇒ the test score ≈ true performance with almost no sampling error, so the gap is measured exactly.
- **a signal dial** — I can set the signal strength from easy (loan-like) down to near-random (finance-like) ⇒ all three datasets sit on one axis.
- **known noise** — I can inject exactly-known label noise ⇒ this is what the noise question (does more noise widen the gap?) requires.

So synthetic is the only place I can measure the gap *precisely* and run controlled experiments; the loan and finance sets then test whether the same effect appears in the wild. The approved plan had the clean set as loan — this is the one change I'd like to confirm. Loan stays, as the real-stakes case. Net: three datasets on one signal axis.

## The instrument
A from-scratch MLP with backprop in NumPy (I derive backprop + the optimizer updates + L2 in the dissertation). Small validation set on purpose ⇒ noise high ⇒ snooping strong, as you noted. Test set untouched all project ⇒ reveals true performance.

## Workflow (one machine, all three datasets)
split → search N configs → keep best on validation → reveal once on sealed test → record the gap. Each experiment varies **one** knob.

## Headline figure
True performance of the selected config vs N, drawn next to its apparent (best-validation) score ⇒ expected: apparent keeps rising; true rises, plateaus, then turns down ⇒ search has an optimal budget, and past it more search buys a worse model and more false confidence.

## How I'll work — core first, one milestone
I'll build the core first and fast.

**Near-term milestone — by ~6 July: the core.** A from-scratch network that trains, plus the headline figure: true vs apparent accuracy as the number of searched configurations grows. This is a **go/no-go gate** — if the core proves *technically* unworkable, I'd rather find out now and adjust direction with you than in August. (A small or absent gap is not a failure — it's an honest finding I'll report; the gate is about technical viability, not dramatic results.)

After the gate, the same machine extends — with no new parts — to the remaining questions (label noise, model capacity, selection protocol) on the lab, then to the real loan and finance data. I'll agree those dates with you once the core stands.

## What I'm asking you to approve
The direction — specifically the switch to a synthetic clean lab — and the ~6 July core milestone (the headline figure) as the go/no-go gate.
