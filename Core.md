# Data Snooping in Deep Learning — dissertation (working draft)

*This draft follows KEY_CORE. Two branches — Snooping and Deep Learning — run in parallel and meet at one conclusion: we cannot trust the GOAL, so we trust the PROCEDURE, with clear assumptions for each situation. Reasoning first, code last.*

*Constraint (handbook): final submission ≤ 50 pages, including bibliography, tables and figures, excluding appendices.*

---

## 0. What we talk about, and what we claim

Here we only talk about deep learning for forecasting and classification. Not vision, not text recognition, not the rest.

The usual way to make a model better is simple: we try many configurations, we keep the one with the best validation score, and we call that score the improvement. So in the end we trust a number.

But can we trust that number? If we cannot, then everything we build on top of it is also not safe. This is the whole question of the project.

=> Our claim: we cannot trust the GOAL — the number we chase. We can only trust the PROCEDURE — the way we build the model, with clear assumptions for each situation.

Two things make the GOAL not trustable, and we follow them as two branches:

- **Snooping** — we try too much to make the number look good.
- **Deep learning** — the black box where snooping is worst.

Both meet at the same place: do not trust the number, trust the procedure.

---

## 1. Snooping — trying too much to make the number look good

> *To write. Evidence to fold in: the winner's-curse gap measured on synthetic data (best-of-N inflation).*

### 1.1 What snooping is

We want a better model. So what do we actually do to get one?

We try things. We pick an architecture, a width, a learning rate — call one such choice a configuration. We train it, we score it on held-out data, and we write the score down. Then we change something and try again. In the end we keep the configuration with the best score and throw the rest away.

This is the normal work of machine learning. Nobody trains one model and stops. We try ten, a hundred, and we keep the best one.

Snooping is when we do too much of this. We try many configurations, and many methods, not because we have a reason for each one, but because trying more makes the score go up. We keep searching until the number looks good, and then we report that number as the result.

=> Snooping = trying too many configurations and methods to make the result better.

The problem is not that we try. The problem is what "better" even means once we search this hard — and that is the next question.

### 1.2 The goal is not clear, and it defeats itself

So we search for a "better" score. Better on what?

When we hold data out, we hold out two kinds. The validation set is the one we look at again and again — once for every configuration we try — and use to pick the winner. The test set is the one we are supposed to look at only once, at the very end, to get the honest score of the model we picked.

So which one is the goal — the best validation score, or the best test score?

It cannot be the validation score. We looked at that set once for every configuration, and we kept the best. The best of a hundred tries is high by luck, not by skill, so a high validation score does not mean a good model.

So the honest goal must be the test score. But the test is used only once. That is what makes it honest: the model never touched it. So what do we do when we open it and the score is bad?

We do the natural thing. We go back, change the model, and train again to make the score better. But now we have used the test to steer us. => The test is not true anymore. We can push its score up, but the moment we do, it stops being the honest number we wanted.

So how do we believe the number? Only if we looked once and stopped. The moment we are unhappy and try again, we break the one thing that made the test trustable.

=> The goal defeats itself: the score we can trust is the one we are not allowed to chase. And once we start chasing, there is no honest number left to reach.

But there is a problem even deeper than looking too many times. Before we count our looks, the way we split the data already hides an assumption — and that is next.

### 1.3 The method hides assumptions that do not fit the situation

The problem in §1.2 was that we look too many times. But suppose we are disciplined. Suppose we look only once. Is the number honest then?

Not always. The method we use to make the split already carries an assumption, and we never check it.

Here is the standard recipe. Take all the data, shuffle it, and cut it into train, validation, and test. Why shuffle? So that each set looks like the whole — same mix, same distribution. But shuffling only makes sense if the rows are interchangeable: if the order does not matter, if any row could sit in any set. That is the hidden assumption — the data is random enough to shuffle.

Is it? For forecasting, and for a lot of classification, no. The data has its own structure. In a time series, today is close to yesterday; the rows are correlated, not independent. If we shuffle, we drop tomorrow into the training set and yesterday into the test set. The model learns from the future to predict the past. Train and test are no longer really separate, so the test score is not honest — even on the first and only look.

Why does the recipe shuffle anyway? Because it is built for all problems at once. It is normalised — a general rule that knows nothing about our specific data. On average, across generic problems, shuffling is fine. On one specific problem with real structure, it quietly measures the wrong thing.

We can watch this happen. Take daily stock direction — will the market go up or down tomorrow. Shuffle the days into train, validation, and test, search some models, and one of them looks like it found an edge: an accuracy above a coin flip. Now split the same data by time instead — train on the old days, test on the newest days, no shuffle — and the edge is gone. The score drops back to a coin flip. The shuffled split did not find a real edge; it leaked structure across the cut and reported a number that was never true.

=> So the GOAL can be wrong from the very first honest look, before we search at all. Not because we cheated — because the method assumed the data was random to shuffle, and here it was not.

Snooping is dangerous everywhere. It is worst in one place — deep learning — and that is the next branch.

---

## 2. Deep learning — the black box where snooping is worst

Why is deep learning the worst place for snooping? Because it is where we search the hardest, and where most of the search is hidden.

Start with the knobs. To train a deep model we set a lot of things: how many layers, how wide each one is, which activation, the learning rate, how many epochs, how much regularisation, L1, L2, and more. Turning any of them gives a different model. It is a wall of a million buttons, and we keep pressing until the score looks good. Every press is another configuration — another try in the sense of §1.

So far this is just a big search, and §1 already told us that a big search inflates the number. But deep learning has a second problem that is worse: we do not even count most of the buttons we press.

Take two of them. The first is the epoch. Training runs for many epochs, and the model changes at every one. We do not keep a fixed model — we keep the epoch where the validation score looked best. That is early stopping, and it is normal, sensible practice. But it is also a choice made by looking at the validation set, and every epoch we checked was another try. The second is the random seed. The same configuration, trained from a different seed, gives a different model and a different score. Retry a few seeds, keep the best, and again we have looked more times than we admit.

Now count honestly. We say we tried five configurations. But inside each one we kept the best of five seeds, and inside each of those we kept the best of six epochs. Five times five times six is a hundred and fifty. So the real number of models we chose between is not five — it is about a hundred and fifty. And by §1.2, the best of a hundred and fifty noisy scores is high by luck. The gap between the number we report and the truth is the gap of a hundred and fifty tries, not five.

This is why the danger is worst here. A simple model — a logistic regression — has almost nothing to turn, so it can barely fool us. A deep model hides a whole search behind every "configuration": a seed, a training curve, an architecture. We think we made one careful choice; we made a hundred blind ones.

And there is a last twist. The deep model does not just allow this hidden search — it runs on it. Early stopping keeps the epoch that scores best on validation, so the machine, by its own design, is tuned toward the exact number we already said we cannot trust. => A deep model is a machine for making the validation score the best — which is the same as a machine for snooping.

Now we have both branches: snooping in general (§1), and deep learning where it is worst (§2). Put them together, and from both sides we reach the same place — the next section.

---

## 3. The two branches meet — we cannot trust the GOAL

Put the two branches side by side.

From §1 and §2: the number goes up when we search, and deep learning lets us search almost without limit, most of it uncounted. So a high score can be nothing but the best of many tries — luck, dressed as skill. From §1.3: even one honest look can be wrong, if the way we split the data assumed something false about it. So the number can be inflated, and the number can be wrong at the root. Often both at once.

So here is the question we cannot avoid. If the score is the thing we push up by searching, and the score is also the thing that can be false from the first look — then how do we believe the model at all? The number we chase is the number we cannot trust.

=> We cannot trust the GOAL.

And deep learning does not soften this — it sharpens it. A deep model works by making the validation score the best: early stopping, seed picking, the whole search all pull toward that one number. So the tool we reach for is, by its own design, the tool that inflates the thing we are not allowed to believe. The harder it works, the less its number means.

There is a deeper way to say the same thing. The trouble is not only that we searched too much. It is that we never made our assumption clear. We shuffled without asking whether the data was random to shuffle. Data that is not random to be shuffled, but split as if it were, gives a goal that measures the wrong thing — and a goal that measures the wrong thing is not really a goal at all.

=> Unclear goal ⇔ train model wrong.

So both branches land in the same place. The number cannot carry our trust — not because we were careless, but because searching inflates it and a wrong assumption corrupts it. If trust does not live in the number, then where does it live? That is the last question.

---

## 4. What we trust instead — the PROCEDURE

If trust does not live in the number, then it lives in the way the number was made. We trust the PROCEDURE, not the score.

What makes a procedure worth trusting? Two things, and they are exactly the two failures turned around.

First, its assumptions are clear. We say out loud what we are assuming about the data, and we check it. If the data has an order, we do not pretend it does not. This is the opposite of the shuffle in §1.3, which assumed the data was random and never said so.

Second, its assumptions fit the situation. There is no single recipe that is honest for every problem. A split that is fine for independent rows is a lie for a time series. So we match the method to the data in front of us. For a time series that means a walk-forward split: train on the old days, test on the newest, and never let the future leak back. The assumption — the past comes before the future — is now built into the method instead of broken by it.

From there the rest is discipline against §1 and §2. Search less, and search honestly: do not fish with hidden seeds and epochs, and choose the winner with a less noisy estimate than a single small validation set. Keep one sealed test, and open it exactly once, on the single model the procedure already chose. A number we looked at once, produced by a method whose assumptions were clear and right, is a number we can believe. A number we chased is not.

This changes what "the best model" means. Put two procedures on the same problem. One searches hard, fishes seeds and epochs, and reports the prettier number. The other searches less, chooses honestly, and reports a plainer one. The prettier number is the inflated one — its real performance is worse, and its report is off by the size of the search. The plainer number is close to the truth. The honest procedure wins twice: a model that is actually as good or better, and a number we are allowed to believe.

So we do not accept a model because its score is high. We accept it only when that score came from a procedure that could not have inflated it — an honest method, assumptions clear and fit to the problem, a sealed test opened once — and only when it beats the baseline by more than the measurement noise. The number to believe is the one the procedure earned the right to report.

=> We trust the PROCEDURE of making the model, with clear assumptions for each situation.

---
