# Data Snooping in Deep Learning — dissertation (working draft)

*This draft follows KEY_CORE. Two branches — Deep Learning and Snooping — run in parallel and meet at one conclusion: we cannot trust the GOAL, so we trust the PROCEDURE, with clear assumptions for each situation. Reasoning first, code last.*

*Constraint (handbook): final submission ≤ 50 pages, including bibliography, tables and figures, excluding appendices.*

---

## 0. What we talk about, and what we claim

Here we only talk about deep learning for forecasting and classification. Not vision, not text recognition, not the rest.

The usual way to make a model better is simple: we try many configurations, we keep the one with the best validation score, and we call that score the improvement. So in the end we trust a number.

But can we trust that number? If we cannot, then everything we build on top of it is also not safe. This is the whole question of the project.

=> Our claim: we cannot trust the GOAL — the number we chase. We can only trust the PROCEDURE — the way we build the model, with clear assumptions for each situation.

Two things make the GOAL not trustable, and we follow them as two branches:

- **Deep learning** — the black box where snooping is worst.
- **Snooping** — we try too much to make the number look good.

Both meet at the same place: do not trust the number, trust the procedure.

---

## 1. Deep learning — the black box where snooping is worst

Why is deep learning the worst place for snooping? Because it is where we search the hardest, and where most of the search is hidden.

Start with the knobs. To train a deep model we set a lot of things: how many layers, how wide each one is, which activation, the learning rate, how many epochs, how much regularisation, L1, L2, and more. Turning any of them gives a different model. It is a wall of a million buttons, and we keep pressing until the score looks good. Every press is another configuration — another try.

So far this is just a big search, and a big search inflates the number. But deep learning has a second problem that is worse: we do not even count most of the buttons we press.

Take two of them. The first is the epoch. Training runs for many epochs, and the model changes at every one. We do not keep a fixed model — we keep the epoch where the validation score looked best. That is early stopping, and it is normal, sensible practice. But it is also a choice made by looking at the validation set, and every epoch we checked was another try. The second is the random seed. The same configuration, trained from a different seed, gives a different model and a different score. Retry a few seeds, keep the best, and again we have looked more times than we admit.

Now count honestly. We say we tried five configurations. But inside each one we kept the best of five seeds, and inside each of those we kept the best of six epochs. Five times five times six is a hundred and fifty. So the real number of models we chose between is not five — it is about a hundred and fifty. And so the best of a hundred and fifty noisy scores is high by luck. The gap between the number we report and the truth is the gap of a hundred and fifty tries, not five.

This is why the danger is worst here. A simple model — a logistic regression — has almost nothing to turn, so it can barely fool us. A deep model hides a whole search behind every "configuration": a seed, a training curve, an architecture. We think we made one careful choice; we made a hundred blind ones.

And there is a last twist. The deep model does not just allow this hidden search — it runs on it. Early stopping keeps the epoch that scores best on validation, so the machine, by its own design, is tuned toward the exact number we already said we cannot trust. => A deep model is a machine for making the validation score the best — which is the same as a machine for snooping.

That is the deep-learning branch — the worst place to snoop. But what is the snooping it makes worst? That is the other branch.

---

## 2. Snooping — trying too much to make the number look good

### 2.1 What snooping is

We want a better model. So what do we actually do to get one?

We try things. We pick an architecture, a width, a learning rate — call one such choice a configuration. We train it, we score it on held-out data, and we write the score down. Then we change something and try again. In the end we keep the configuration with the best score and throw the rest away.

This is the normal work of machine learning. Nobody trains one model and stops. We try ten, a hundred, and we keep the best one.

Snooping is when we do too much of this. We try many configurations, and many methods, not because we have a reason for each one, but because trying more makes the score go up. We keep searching until the number looks good, and then we report that number as the result.

=> Snooping = trying too many configurations and methods to make the result better.

The problem is not that we try. The problem is what "better" even means once we search this hard — and that is the next question.

### 2.2 The goal is not clear, and it defeats itself

So we search for a "better" score. Better on what?

When we hold data out, we hold out two kinds. The validation set is the one we look at again and again — once for every configuration we try — and use to pick the winner. The test set is the one we are supposed to look at only once, at the very end, to get the honest score of the model we picked.

So which one is the goal — the best validation score, or the best test score?

It cannot be the validation score. We looked at that set once for every configuration, and we kept the best. The best of a hundred tries is high by luck, not by skill, so a high validation score does not mean a good model.

In §2.1 we called snooping "trying too many configurations to make the score go up." That sounds harmless. Let us make it happen, in the cleanest case, and watch what it does to the number.

We build the data ourselves, with no signal. The labels are coin flips: 0 or 1 at random, and nothing in the features points to the answer. There is nothing to learn. Because we made it, we know the truth exactly — no model can beat 0.5. Every model's real accuracy is 0.5.

We cut this data three ways, by position — the labels are already random, so there is nothing to shuffle for: a training part to fit each model, a small validation set of 200 points to pick the winner, and a large sealed test of 10 000 points, kept shut until the very end. Two of the sizes are on purpose. The validation is small, so its score is noisy — and that noise is the whole mechanism. The test is large, so its average barely wobbles from the truth, which is what lets it stand in for the 0.5 we know.

A configuration here is one setting of the small network: a width and a learning rate, drawn at random. Adding a configuration means drawing another pair, training it, and scoring it on a validation set of n points. On random labels the network learns nothing, so no configuration is really better than another — its validation score is not skill, only the fraction of correct guesses over n coin flips.

That is the key. A validation score on n points, from a model that is really at chance, is the fraction of heads in n fair flips. Its average is 0.5, but it wobbles, with a spread of

> σ = 0.5 / √n.

With n = 200 that spread is about 0.035. So one configuration already lands near 0.5 ± 0.035 by luck alone, before we have done anything wrong.

Now we snoop. We draw N configurations and keep the best validation score. We are no longer looking at one draw; we are taking the highest of N noisy draws around 0.5. And the highest of many draws sits above their average — further above, the more we take. For spread σ, the best of N is about

> 0.5 + σ · √(2 ln N).

That is the whole engine: keep the best of more tries, and the best runs further above 0.5. So we run it. We build the no-signal data, sweep N, and each time keep the best on validation and open the sealed test once on that winner (n = 200, averaged over a few repeats):

| N   | apparent (best val) | true (sealed test) | gap    |
| --- | ------------------- | ------------------ | ------ |
| 1   | 0.510               | 0.500              | +0.010 |
| 2   | 0.521               | 0.500              | +0.021 |
| 5   | 0.548               | 0.500              | +0.048 |
| 10  | 0.553               | 0.500              | +0.053 |
| 20  | 0.558               | 0.499              | +0.059 |
| 50  | 0.568               | 0.500              | +0.068 |
| 100 | 0.577               | 0.500              | +0.077 |

The reported number climbs from 0.51 to 0.58 as N grows, while the truth never moves off 0.50 — every time, the sealed test drops the winner straight back to chance. The gap grows from +0.01 to +0.08, climbing with N just as √(ln N) says it should. It sits a little under the bare formula 0.5 + σ√(2 ln N), because the configurations share one training set, so their validation scores are not fully independent draws. The derivation is in Appendix A; the code that produced this table is in Appendix B.

Every point above 0.5 was luck, picked out by keeping the best of many tries.

So this is §2.1's "harmless" trying, measured. Trying more did push the number up — that part is real. But it did not make the model better; that was never possible here. => The rising number is exactly the gap between the reported score and the truth, and the distance between them is how hard we searched. The goal did not fail quietly — it told us we were improving when we were not.

And the formula says where the lie lives: σ = 0.5/√n. Make the validation set smaller and σ grows, so the gap grows. The small validation set is not a side detail — it is what feeds the gap.

So the honest goal must be the test score. But the test is used only once. That is what makes it honest: the model never touched it. So what do we do when we open it and the score is bad?

We do the natural thing. We go back, change the model, and train again to make the score better. But now we have used the test to steer us. => The test is not true anymore. We can push its score up, but the moment we do, it stops being the honest number we wanted.

We can watch this too, with the same machine — and we need no new data, because we already know the truth is 0.5. Instead of looking at the test once, we reuse it: we keep the configuration that scores best on the test, exactly as retraining until the test looks good would. Side by side with the validation search:

| N   | select on validation | reuse the test | truth |
| --- | -------------------- | -------------- | ----- |
| 1   | 0.510                | 0.500          | 0.500 |
| 10  | 0.553                | 0.508          | 0.500 |
| 100 | 0.577                | 0.512          | 0.500 |

Reusing the test does inflate it — 0.500 creeps to 0.512 — so the test is not special: selected on, it lies like any other set. It creeps slowly only because it is large; its noise σ = 0.5/√10 000 is about seven times smaller than the validation's, so it resists longer. But it gives way. The test's honesty was never in the test — it was in the discipline of looking once, on a set large enough to be quiet.

So how do we believe the number? Only if we looked once and stopped. The moment we are unhappy and try again, we break the one thing that made the test trustable.

=> The goal defeats itself: the score we can trust is the one we are not allowed to chase. And once we start chasing, there is no honest number left to reach.

But there is a problem even deeper than looking too many times. Before we count our looks, the way we split the data already hides an assumption — and that is next.

### 2.3 The same trap, hidden in the procedures we never question

§2.2 was about looking too many times. But suppose we are disciplined and look only once. Is the number honest then? Not always — and this time the fault is not in how hard we searched, but in the recipe itself.

The standard recipe is plain: shuffle the data, cut it into train, validation, and test, and cross-validate. We treat it as neutral, the obviously correct thing to do. It is not neutral. It quietly assumes something about the data — and that something is a property of the *problem*, not of the recipe, so one recipe cannot be right for every problem. Below we walk the recipe step by step — (A) how we split, (B) how we standardise, (C) what we measure — and on **real** data we show each step hiding an assumption that, on the wrong problem, ruins the result.

We do it the same way every time: the same small network from §2.2, on a real dataset, with everything held fixed and **only the split changed**. Whatever moves the number is the split, and nothing else.

**(A) The split — is the data safe to shuffle?**

Why shuffle before we cut? So that each part looks like the whole. But shuffling only makes sense if the rows are interchangeable — if the order does not matter, if any row could sit in any set. That is the hidden assumption: the rows are independent, drawn from the same pot. The textbook calls it a random split, or k-fold cross-validation, and hands it to us as the default.

Sometimes that is true, and then shuffle is exactly right. We start there, so we are not accusing shuffle of a crime it did not commit.

*Loan default — independent clients.* Thirty thousand credit-card clients, twenty-three features each, and one label: did the client default? A row is a client; there is no time order between clients, and no client appears twice. The rows really are interchangeable. So the split should not matter — and it does not. We run the same MLP under ten different random splits, and under a stratified split that holds the 22% default rate on both sides:

| split              | test accuracy |
| ------------------ | ------------- |
| ten random splits  | 0.813 ± 0.003 |
| stratified split   | 0.814         |

Every split lands in the same place, inside the run-to-run wobble. The assumption holds, so shuffle is the correct, honest method — exactly as the textbook promises. => When the rows are exchangeable, the split is immaterial: changing it moved nothing.

But that is the easy case. Now two problems where the rows are *not* interchangeable — and the same shuffle turns into a lie.

*Time order — the stock market.*

The S&P 500, one row per trading day. We do not try to call the market's direction — that is a coin flip, and we will come back to why that matters. We predict something the market really does carry: whether tomorrow is a **busy** day — its move larger than usual — from the sizes of the last five days' moves. Busy days cluster: a stormy week tends to stay stormy, so this target has real, persistent signal.

The rows are days now, and days are not interchangeable — today looks like yesterday. Shuffle them into train and test and tomorrow's near-twin lands in the training set; the model is not predicting the future, it is recalling a neighbour it already saw. Split by time instead — train on the old days, test on the newest — and there is no neighbour to lean on. Same MLP, same everything, only the split changes:

| split                                   | test accuracy |
| --------------------------------------- | ------------- |
| shuffle (random)                        | 0.615         |
| chronological (train past, test future) | 0.585         |

Over ten seeds the shuffle score is higher every single time, by +0.030 on average. That extra 0.030 is not skill — it is the future leaking backward across the cut. The honest split is the one that respects the order of time.

Which honest number, though? Test on the last stretch alone and we read 0.585; roll the cut forward through the series and average many future windows — this is **walk-forward**, or rolling-origin, the standard honest protocol for a time series — and we read 0.603. The honest number wobbles between the two, because the market drifts: a later year is not the same market as an earlier one. But every honest reading sits below the shuffled 0.615. So we do not claim the whole gap is "leakage": shuffling is over-optimistic for two reasons at once — it lets near-twin days leak across the cut, *and* it pretends the future is the same market as the past. Walk-forward is what shows the deficit is real and persistent, not one unlucky window.

One last check, and it is the honest one. Leakage can only inflate an edge that is really there. So take a target with no edge — next-day direction, up or down, a coin flip on this series — and run the same test. The gap vanishes: shuffle and chronological land within a whisker of each other, both near the 0.537 always-up rate. No signal, nothing to leak. That the volatility task shows a gap and the direction task shows none is the surest sign we are measuring a real effect, not a lucky wiring.

*Repeated people — activity recognition.*

Thirty people carried a phone while walking, climbing stairs, sitting, standing, lying down. Each window of movement becomes one row — 561 numbers describing the motion — labelled with the activity. Ten thousand rows, but only thirty people, so each person fills hundreds of rows.

That is the structure the recipe cannot see. A row is not an independent draw; it belongs to a person, and one person's rows are near-copies of each other — the same gait, the same way of sitting. Shuffle the rows and almost everyone lands on both sides of the cut: the model sees a given person walking in training, then is tested on that same person walking again. It scores well not because it learned *walking*, but because it learned *the person*. The real question — can it read someone it has never met? — is never asked.

So ask it. Instead of shuffling rows, hold out whole people: train on some, test on others the model has never seen. That is the matched method here — group the rows by subject and leave whole subjects out (GroupKFold, or leave-one-subject-out; the mistake and its fix are named in Saeb et al., 2017). Same small network — now with one output per activity — only the split changes:

| split                          | test accuracy |
| ------------------------------ | ------------- |
| shuffle rows (record-wise)     | 0.973         |
| hold out people (subject-wise) | 0.946         |

Record-wise reads 0.973 — but part of that is recognising people it already met. Test on strangers and it falls to 0.946, lower in nine of ten runs. The honest question is harder, and the honest number is lower. The 0.973 was never the model's skill at reading activities; it was partly its skill at reading *these thirty people*.

**(B) Standardising the features.**

> *To write. Textbook rule: fit the scaler (mean, std) on train only, then transform valid and test — the "leakage-safe" way taught in every course. Hidden assumption: the training mean and std are permanent — the data does not drift (stationarity). How it fails: in forecasting a feature drifts (a price level rises over the years), so later inputs land at +4 or +5 sigma, a region the model never saw in training → the real predictions are ruined, even on a perfectly honest walk-forward split with no leakage. Distinct from A: no leakage — A breaks the number's honesty, B breaks the real model even when the number is honest.*

**(C) Accuracy as the metric.**

> *To write. Textbook rule: report accuracy — the fraction of predictions we get right — as "how good the model is". Hidden assumption: the classes are balanced and the two kinds of error cost the same. How it fails: on rare events (~1% positive — fraud, default, disease) "always predict no" scores 99% and catches zero of the cases we built the model for; a genuinely useful model scores lower. Distinct from A and B: no leakage, no drift — the number is measured perfectly honestly and still measures the wrong thing.*

Snooping is dangerous everywhere — and worst of all in deep learning, the branch we just saw. Now put the two branches together.

---

## 3. The two branches meet — we cannot trust the GOAL

Put the two branches side by side.

From §1 and §2: the number goes up when we search, and deep learning lets us search almost without limit, most of it uncounted. So a high score can be nothing but the best of many tries — luck, dressed as skill. From §2.3: even one honest look can be wrong, if the way we split the data assumed something false about it. So the number can be inflated, and the number can be wrong at the root. Often both at once.

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

First, its assumptions are clear. We say out loud what we are assuming about the data, and we check it. If the data has an order, we do not pretend it does not. This is the opposite of the shuffle in §2.3, which assumed the data was random and never said so.

Second, its assumptions fit the situation. There is no single recipe that is honest for every problem. A split that is fine for independent rows is a lie for a time series. So we match the method to the data in front of us. For a time series that means a walk-forward split: train on the old days, test on the newest, and never let the future leak back. The assumption — the past comes before the future — is now built into the method instead of broken by it.

From there the rest is discipline against §1 and §2. Search less, and search honestly: do not fish with hidden seeds and epochs, and choose the winner with a less noisy estimate than a single small validation set. Keep one sealed test, and open it exactly once, on the single model the procedure already chose. A number we looked at once, produced by a method whose assumptions were clear and right, is a number we can believe. A number we chased is not.

This changes what "the best model" means. Put two procedures on the same problem. One searches hard, fishes seeds and epochs, and reports the prettier number. The other searches less, chooses honestly, and reports a plainer one. The prettier number is the inflated one — its real performance is worse, and its report is off by the size of the search. The plainer number is close to the truth. The honest procedure wins twice: a model that is actually as good or better, and a number we are allowed to believe.

So we do not accept a model because its score is high. We accept it only when that score came from a procedure that could not have inflated it — an honest method, assumptions clear and fit to the problem, a sealed test opened once — and only when it beats the baseline by more than the measurement noise. The number to believe is the one the procedure earned the right to report.

=> We trust the PROCEDURE of making the model, with clear assumptions for each situation.

---

## Appendix A — Where the two formulas come from

*Outside the page limit; here so the two formulas the lab leans on can be checked, not taken on trust. Nothing below goes past first-year probability — we derive it in place.*

The lab in §2.2 used two formulas: the spread of a chance score, σ = 0.5/√n, and the best of N scores, about 0.5 + σ√(2 ln N). Here is where each one comes from.

### Part 1 — The spread of a chance score: σ = 0.5/√n

A validation score is the fraction of the n validation points the model gets right. Take a model that is really at chance — on random labels, every model is. On each point it is either right or wrong, and being right is a fair coin: probability 0.5, independent of the other points.

So the number it gets right, call it k, is the number of heads in n fair flips — a Binomial(n, 0.5). Two standard facts about it:

- its mean is n·0.5, so the score k/n has mean 0.5;
- its variance is n·0.5·0.5 = n/4, so the score k/n has variance (n/4)/n² = 1/(4n).

The spread is the square root of the variance:

> σ = √(1/(4n)) = 0.5/√n.

That is the first formula. A chance model does not sit at exactly 0.5 — it scatters around 0.5, and the smaller the validation set n, the wider it scatters. With n = 200, σ = 0.5/√200 ≈ 0.035.

We need one more thing. For n even moderately large, the Binomial looks like a bell curve (the Central Limit Theorem), so we may write each score as 0.5 + σ·Z, where Z is a standard normal — mean 0, spread 1. For n = 200 this is a close fit.

### Part 2 — The best of N: 0.5 + σ√(2 ln N)

Now we draw N configurations. Each gives a score 0.5 + σ·Z, with Z₁, …, Z_N standard normals, and for now independent. We keep the best, so the best score is 0.5 + σ · (the biggest of the N normals). Everything comes down to one question: how big is the biggest of N standard normals?

The answer is about √(2 ln N). Here is why. The biggest of N draws is roughly the value t that only about one draw in N gets past — the level where P(Z > t) ≈ 1/N. So we need how far out the normal's tail sits.

The tail of the bell curve falls off fast. Beyond t, the tail area is close to

> P(Z > t) ≈ e^(−t²/2) / (t·√(2π)),

and the e^(−t²/2) is what dominates. Set this equal to 1/N and take logs:

> −t²/2 − ln(t√(2π)) ≈ −ln N.

The ln N term is the big one; the ln(t√(2π)) grows only like ln t, far slower, so we drop it and keep the leading term:

> t²/2 ≈ ln N  =>  t ≈ √(2 ln N).

So the biggest of N standard normals is about √(2 ln N), and the best score is

> 0.5 + σ · √(2 ln N).

That is the second formula.

### Part 3 — Reading the formula, and its honest limits

The formula is worth more than its number, because it explains the shape of the gap:

- **It grows with N.** More tries, bigger best. Searching harder inflates the number — the winner's curse in one line.
- **But it grows only like √(ln N).** ln N barely moves when we double N, and its square root moves less. So the gap jumps early and then crawls. This is why we try N at 1, 2, 5, 10, 20, … and not 1, 2, 3, 4 — the action is in the first few tries.
- **It scales with σ = 0.5/√n.** A smaller validation set means a larger σ means a larger gap. The small validation set is the fuel, exactly as §2.2 said.

Two honest limits, both pushing the real gap a little below the formula:

- **Independence.** We assumed the N draws were independent. Configurations that differ only by a seed, or by one more epoch, are near-copies, not fresh draws — so N of them count as fewer than N independent tries, and the real best sits a touch below σ√(2 ln N). (This is the hidden-search point of §1: seeds and epochs are extra draws, but correlated ones.)
- **The Gaussian approximation.** The Binomial is only approximately a bell curve, and √(2 ln N) is the leading term of a longer expression. Both are close for the n and N we use, not exact.

So we do not lean on the formula as truth. We use it to see the mechanism, and we measure the real gap in the lab (§2.2). The two agree in shape — and that agreement is the point: the gap is not a quirk of one dataset, it is the biggest of many noisy draws, behaving as the biggest of many noisy draws must.

---

## Appendix B — the code that ran the lab

*Outside the page limit; the measurement half of the grounding. Appendix A shows why the formulas hold; this shows the code that measured the real gap, and that running it reproduces the numbers in §2.2.*

The whole lab is one short file, `lab_demo.py` (repo root), that runs on numpy alone — `python lab_demo.py`, with `seed = 0`, so every number below reproduces exactly. Here it is in the five pieces of §2.2. The helper functions (`init`, `forward`, `softmax`, `accuracy`, `sample_config`) are in the file; `forward` is just `h = ReLU(X·W1 + b1)` then `z = h·W2 + b2`, and `sample_config` draws a width from {4, 8, 16, 32, 64} and a learning rate around 0.01–0.3.

**Piece 1 — the no-signal data.** Random features; labels that are coin flips, drawn without ever looking at the features. Split into train, a small validation set (200), and a large sealed test (10 000).

```python
def make_data(n, d, rng):
    X = rng.standard_normal((n, d))    # features: pure noise
    y = rng.integers(0, 2, size=n)     # labels: coin flips, drawn WITHOUT X
    return X, y
```

```
shapes  train (1000, 20)  val (200, 20)  test (10000, 20)
label balance  train 0.533  val 0.485  test 0.509
corr(feature0, y) on train = +0.0601  (~0: no signal)
```

The labels are near 50/50 and barely correlated with any feature — there is nothing to learn, so the true accuracy is 0.5 by construction.

**Piece 2 — the small MLP; a configuration is (width, learning rate).** One hidden layer, ReLU, two outputs, softmax with cross-entropy, trained by full-batch gradient descent. The gradient is the chain rule, written out by hand.

```python
def train(Xtr, ytr, width, lr, epochs, rng):
    p = init(Xtr.shape[1], width, rng); n = len(ytr)
    Y = np.zeros((n, 2)); Y[np.arange(n), ytr] = 1.0
    for _ in range(epochs):
        a, h, z = forward(Xtr, p)
        dz = (softmax(z) - Y) / n                        # d(cross-entropy)/d(logits)
        W1, b1, W2, b2 = p
        dW2 = h.T @ dz;  db2 = dz.sum(0)
        da  = (dz @ W2.T) * (a > 0)                       # back through ReLU
        dW1 = Xtr.T @ da; db1 = da.sum(0)
        p = [W1-lr*dW1, b1-lr*db1, W2-lr*dW2, b2-lr*db2]  # gradient descent
    return p
```

```
config width=32 lr=0.1: train 0.546 | val 0.520 | test 0.498   (one train 0.025s)
```

One model: its test accuracy is 0.498 — chance, as it must be. Its training accuracy is a little higher (0.546: it memorises a few points), and its validation score, 0.520, is one noisy draw around 0.5.

**Piece 3 — the gap machine.** Draw N configurations, keep the one with the best validation score, then open the sealed test exactly once, on that winner.

```python
def run_once(N, data, epochs, rng):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    best_val, best_p = -1.0, None
    for _ in range(N):
        w, lr = sample_config(rng)
        p = train(Xtr, ytr, w, lr, epochs, rng)
        v = accuracy(p, Xva, yva)                     # score on validation
        if v > best_val: best_val, best_p = v, p      # keep best on validation
    return best_val, accuracy(best_p, Xte, yte)       # open sealed test ONCE
```

```
N=20:  apparent(best val) 0.565 | true(test of winner) 0.508 | gap +0.057
```

The best of twenty on validation reads 0.565, but that same model on the sealed test is 0.508 — chance. The 0.057 between them is pure luck.

**Piece 4 — the sweep.** Run the gap machine over a grid of N, averaging over fifteen repeats. This is the table in §2.2.

```
   N | apparent |   true |    gap
   1 |    0.510 |  0.500 | +0.010
   2 |    0.521 |  0.500 | +0.021
   5 |    0.548 |  0.500 | +0.048
  10 |    0.553 |  0.500 | +0.053
  20 |    0.558 |  0.499 | +0.059
  50 |    0.568 |  0.500 | +0.068
 100 |    0.577 |  0.500 | +0.077
```

Apparent climbs from 0.510 to 0.577; true holds at 0.500; the gap grows from +0.010 to +0.077.

**Piece 5 — the formula check.** Compare the measured apparent with 0.5 + σ√(2 ln N), where σ = 0.5/√200 = 0.0354.

```
   N | measured app |  formula
   1 |        0.510 |    0.500
  10 |        0.553 |    0.576
 100 |        0.577 |    0.607
```

Same shape, but the measured value sits a little under the formula. The formula assumed N independent draws; here the N configurations share one training set, so their validation scores are correlated — and the best of correlated draws runs less far than the best of independent ones (Appendix A, the honest limits).

**Piece 6 — the test is not special.** The gap machine kept the sealed test honest by opening it once. What if we reuse it — select on it, as retraining until the test looks good would? The same sweep records both readings of the same configurations; since the truth is known (0.5), no second test is needed to expose the lie.

```python
        for N in N_values:
            iv = vals[:N].argmax()      # winner if we select on validation
            it = tests[:N].argmax()     # winner if we REUSE the test to select
            rec[N]["va"].append(vals[iv]);  rec[N]["vt"].append(tests[iv])
            rec[N]["ta"].append(tests[it]); rec[N]["tt"].append(vals[it])
```

```
   N | select on VAL | reuse the TEST |  truth
   1 |         0.510 |          0.500 |  0.500
  10 |         0.553 |          0.508 |  0.500
 100 |         0.577 |          0.512 |  0.500
(sigma_val = 0.0354 on n=200;  sigma_test = 0.0050 on n=10000)
```

Reusing the test inflates it too — 0.500 to 0.512 — so selecting on the test poisons it like any other set. It moves less than the validation only because the test is larger (its σ is seven times smaller). The lesson: selection is the poison, not the set; the test's honesty lives in looking once, on a set large enough to be quiet.

So the whole result is one line: a gap of +0.077 at N = 100, out of data with no signal at all, growing with the search — and the same curse hits any set we select on, the sealed test included. Rerun `python lab_demo.py` to reproduce every number here.

---
