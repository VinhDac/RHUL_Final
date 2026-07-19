# North Star

*The compass for the whole dissertation. When any paragraph feels wrong, come back here. `KEY_CORE.html` holds the underlying argument.*

## The one line

An ordinary person builds a deep-learning model the normal way, by the book, step by step. At each step the by-the-book move looks bulletproof, and then hides an assumption that, on the wrong data, quietly breaks. Walking that failure together teaches the reader what to actually trust: not the number, but understanding, turned into a procedure.

## Who is telling it

A fellow traveller, not a professor. Someone very ordinary, doing ordinary things, who gets confused, gets it wrong, and works it out beside the reader. Our authority is "I fell for this too, and here is how I climbed out," never "I know better." This is a journey of discovery, not a lecture. We never make the reader memorise anything; we help them see.

## The structure: the pipeline is the map

Section 1 lays out the normal way to build a DL model as a pipeline, with a diagram, so the reader always knows where they are. Then we walk it, in pipeline order, station by station. Four stations hold a trap:

1. **SPLIT** the data. Loan: it works, the book is vindicated. Market: it breaks (time). Phone: it breaks (people). Hidden assumption: the rows are interchangeable.
2. **SCALE** the features. The drift collapse. Hidden assumption: the world stands still.
3. **BUILD and SEARCH** (architecture, loss, regularise, train, keep the best). Even pure noise inflates when we keep the best of many tries, and deep learning snoops by itself through early stopping. Hidden assumption: a higher validation score means a better model.
4. **MEASURE** (report a metric). Accuracy hides a useless model on rare cases. Hidden assumption: the classes are balanced and the two errors cost the same.

Then a short bottom (every step hid an assumption, so the number cannot be trusted) and the summit (trust understanding made into a procedure). Note: the no-signal search lab lives at station 3, not at the opening. We open on real data.

## The rhythm at each station: keep it light

Every station is the same small beat: do the by-the-book move, watch it look fine, find the trap on the wrong data, understand why, reach the matched fix. State each beat once, plainly, and move on. Do not dwell, do not dramatise, do not keep saying "feel how strange this is." The map makes the walk feel like progress; heaviness makes it feel like a slog.

## Every section carries a key question

Each section must surface at least one KEY QUESTION out loud, and let the reader feel it: a genuine doubt that interrogates the essence of what we are doing, not a rhetorical flourish. It is the emotional and intellectual spine of that section, the thing that makes the reader stop and think, and the section is the search for its answer. Examples: "An edge on the market? That is too easy, what is leaking in?" "A high accuracy, but did it catch the very people we built the model for?" "Does trying harder make a better model, or just a luckier number?" No section should read as a smooth statement of results; each should be driven by its own doubt.

## The depth: each context is its own journey

Do not march one pipeline step across several datasets in a row (that flattens each dataset into a bullet and reads like a report). Instead, give each dataset its own full journey: its own opening question, its own doubts, its own dead ends and mistakes, its own trail of small tries (a decision tree, branch by branch) until its own answer falls out. The pipeline stays as the shared map; the datasets are where we actually live and think.

## The voice, and the one rule above all: LIGHTNESS

Section 0 is the standard for the whole piece: short, plain, familiar, curious, easy, never challenging the reader. The gravest failure mode of this project is the opposite, and we have already fallen into it once: long-winded, heavy, over-dramatic, hard to read. So:

- When in doubt, CUT. The fix is almost always less, not more.
- Short sentences. Plain words. No jargon before it is earned.
- Let a picture carry the logic instead of a wall of prose.
- Bring the concrete data into the story; do not exile it to an appendix and make the reader imagine it.
- Never use the em dash character ( — ). Use a comma, a colon, parentheses, or a full stop.

## Visuals carry the logic

The maths and the mechanisms lean on pictures, not prose walls. The key diagrams: the MLP itself; the build pipeline; the winner's curse (a cloud of scores around 0.5, its maximum creeping right as we try more); a shuffle dropping a near-twin across the split; a feature drifting out from under a frozen scaler; a confusion matrix; a train-versus-validation loss curve for early stopping.

## Datasets tell their own story

Each dataset's lesson emerges FROM its raw data, never imposed. Show the actual rows first; let the reader notice the thing (two neighbouring days share four of five numbers; every window belongs to a person), and let that observation drive the question and the fix. Never "here is the trick, now here are the numbers."

## Appendices show the transformation

Code appendices show the data changing, step by step: print what one row is, and what it becomes after each line, so a learner sees the transformation instead of imagining it.

## What never changes / what changes completely

- **Never changes:** the underlying argument (each pipeline step hides an assumption, so the number cannot be trusted, so we trust understanding made into a procedure); the labs; the datasets; the formulas; the numbers.
- **Changes:** the telling. Lighter, shorter, more visual, ordered along the pipeline.

## How we work together

- Every time a choice comes up, offer three concrete options. The user decides.
- Tiny, sure steps. One small piece at a time, checked against this compass, locked before the next.

## The one test for any paragraph

Ask two things. Does it sound like a curious ordinary person discovering this beside the reader, or a professor who already knows? And: is it as short and light as Section 0, or has it gone heavy and long? If professor, or if heavy, rewrite it shorter.
