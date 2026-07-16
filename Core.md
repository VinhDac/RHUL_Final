# Data Snooping in Deep Learning: dissertation (working draft)

*This draft follows KEY_CORE. Two branches, Deep Learning and Snooping, run in parallel and meet at one conclusion: we cannot trust the GOAL, so we trust the PROCEDURE, with clear assumptions for each situation. Reasoning first, code last.*

*Constraint (handbook): final submission ≤ 50 pages, including bibliography, tables and figures, excluding appendices.*

---

## 0. What we talk about, and what we claim

Here we only talk about deep learning for forecasting and classification. Not vision, not text recognition, not the rest.

The usual way to make a model better is simple: we try many configurations, we keep the one with the best validation score, and we call that score the improvement. So in the end we trust a number.

But can we trust that number? If we cannot, then everything we build on top of it is also not safe. This is the whole question of the project.

=> Our claim: we cannot trust the GOAL, the number we chase. We can only trust the PROCEDURE, the way we build the model, with clear assumptions for each situation.

Two things make the GOAL not trustable, and we follow them as two branches:

- **Deep learning**: the black box where snooping is worst.
- **Snooping**: we try too much to make the number look good.

Both meet at the same place: do not trust the number, trust the procedure.

---

## 1. Deep learning: the black box where snooping is worst

Why is deep learning the worst place for snooping? Because it is where we search the hardest, and where most of the search is hidden.

Start with the knobs. To train a deep model we set a lot of things: how many layers, how wide each one is, which activation, the learning rate, how many epochs, how much regularisation, L1, L2, and more. Turning any of them gives a different model. It is a wall of a million buttons, and we keep pressing until the score looks good. Every press is another configuration, another try.

So far this is just a big search, and a big search inflates the number. But deep learning has a second problem that is worse: we do not even count most of the buttons we press.

Take two of them. The first is the epoch. Training runs for many epochs, and the model changes at every one. We do not keep a fixed model, we keep the epoch where the validation score looked best. That is early stopping, and it is normal, sensible practice. But it is also a choice made by looking at the validation set, and every epoch we checked was another try. The second is the random seed. The same configuration, trained from a different seed, gives a different model and a different score. Retry a few seeds, keep the best, and again we have looked more times than we admit.

Now count honestly. We say we tried five configurations. But inside each one we kept the best of five seeds, and inside each of those we kept the best of six epochs. Five times five times six is a hundred and fifty. So the real number of models we chose between is not five, it is about a hundred and fifty. And so the best of a hundred and fifty noisy scores is high by luck. The gap between the number we report and the truth is the gap of a hundred and fifty tries, not five.

This is why the danger is worst here. A simple model, a logistic regression, has almost nothing to turn, so it can barely fool us. A deep model hides a whole search behind every "configuration": a seed, a training curve, an architecture. We think we made one careful choice; we made a hundred blind ones.

And there is a last twist. The deep model does not just allow this hidden search, it runs on it. Early stopping keeps the epoch that scores best on validation, so the machine, by its own design, is tuned toward the exact number we already said we cannot trust. => A deep model is a machine for making the validation score the best, which is the same as a machine for snooping.

That is the deep-learning branch, the worst place to snoop. But what is the snooping it makes worst? That is the other branch.

---

## 2. Snooping: trying too much to make the number look good

### 2.1 What snooping is

We want a better model. So what do we actually do to get one?

We try things. We pick an architecture, a width, a learning rate. Call one such choice a configuration. We train it, we score it on held-out data, and we write the score down. Then we change something and try again. In the end we keep the configuration with the best score and throw the rest away.

This is the normal work of machine learning. Nobody trains one model and stops. We try ten, a hundred, and we keep the best one.

Snooping is when we do too much of this. We try many configurations, and many methods, not because we have a reason for each one, but because trying more makes the score go up. We keep searching until the number looks good, and then we report that number as the result.

=> Snooping = trying too many configurations and methods to make the result better.

The problem is not that we try. The problem is what "better" even means once we search this hard, and that is the next question.

### 2.2 The goal is not clear, and it defeats itself

So we search for a "better" score. Better on what?

When we hold data out, we hold out two kinds. The validation set is the one we look at again and again, once for every configuration we try, and use to pick the winner. The test set is the one we are supposed to look at only once, at the very end, to get the honest score of the model we picked.

So which one is the goal, the best validation score, or the best test score?

It cannot be the validation score. We looked at that set once for every configuration, and we kept the best. The best of a hundred tries is high by luck, not by skill, so a high validation score does not mean a good model.

In §2.1 we called snooping "trying too many configurations to make the score go up." That sounds harmless. Let us make it happen, in the cleanest case, and watch what it does to the number.

We build the data ourselves, with no signal. The labels are coin flips: 0 or 1 at random, and nothing in the features points to the answer. There is nothing to learn. Because we made it, we know the truth exactly: no model can beat 0.5. Every model's real accuracy is 0.5.

We cut this data three ways, by position (the labels are already random, so there is nothing to shuffle for): a training part to fit each model, a small validation set of 200 points to pick the winner, and a large sealed test of 10 000 points, kept shut until the very end. Two of the sizes are on purpose. The validation is small, so its score is noisy, and that noise is the whole mechanism. The test is large, so its average barely wobbles from the truth, which is what lets it stand in for the 0.5 we know.

A configuration here is one setting of the small network: a width and a learning rate, drawn at random. Adding a configuration means drawing another pair, training it, and scoring it on a validation set of n points. On random labels the network learns nothing, so no configuration is really better than another, its validation score is not skill, only the fraction of correct guesses over n coin flips.

That is the key. A validation score on n points, from a model that is really at chance, is the fraction of heads in n fair flips. Its average is 0.5, but it wobbles, with a spread of

> σ = 0.5 / √n.

With n = 200 that spread is about 0.035. So one configuration already lands near 0.5 ± 0.035 by luck alone, before we have done anything wrong.

Now we snoop. We draw N configurations and keep the best validation score. We are no longer looking at one draw; we are taking the highest of N noisy draws around 0.5. And the highest of many draws sits above their average, further above, the more we take. For spread σ, the best of N is about

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

The reported number climbs from 0.51 to 0.58 as N grows, while the truth never moves off 0.50, every time, the sealed test drops the winner straight back to chance. The gap grows from +0.01 to +0.08, climbing with N just as √(ln N) says it should. It sits a little under the bare formula 0.5 + σ√(2 ln N), because the configurations share one training set, so their validation scores are not fully independent draws. The derivation is in Appendix A; the code that produced this table is in Appendix B.

Every point above 0.5 was luck, picked out by keeping the best of many tries.

So this is §2.1's "harmless" trying, measured. Trying more did push the number up, that part is real. But it did not make the model better; that was never possible here. => The rising number is exactly the gap between the reported score and the truth, and the distance between them is how hard we searched. The goal did not fail quietly, it told us we were improving when we were not.

And the formula says where the lie lives: σ = 0.5/√n. Make the validation set smaller and σ grows, so the gap grows. The small validation set is not a side detail, it is what feeds the gap.

So the honest goal must be the test score. But the test is used only once. That is what makes it honest: the model never touched it. So what do we do when we open it and the score is bad?

We do the natural thing. We go back, change the model, and train again to make the score better. But now we have used the test to steer us. => The test is not true anymore. We can push its score up, but the moment we do, it stops being the honest number we wanted.

We can watch this too, with the same machine, and we need no new data, because we already know the truth is 0.5. Instead of looking at the test once, we reuse it: we keep the configuration that scores best on the test, exactly as retraining until the test looks good would. Side by side with the validation search:

| N   | select on validation | reuse the test | truth |
| --- | -------------------- | -------------- | ----- |
| 1   | 0.510                | 0.500          | 0.500 |
| 10  | 0.553                | 0.508          | 0.500 |
| 100 | 0.577                | 0.512          | 0.500 |

Reusing the test does inflate it, 0.500 creeps to 0.512, so the test is not special: selected on, it lies like any other set. It creeps slowly only because it is large; its noise σ = 0.5/√10 000 is about seven times smaller than the validation's, so it resists longer. But it gives way. The test's honesty was never in the test, it was in the discipline of looking once, on a set large enough to be quiet.

So how do we believe the number? Only if we looked once and stopped. The moment we are unhappy and try again, we break the one thing that made the test trustable.

=> The goal defeats itself: the score we can trust is the one we are not allowed to chase. And once we start chasing, there is no honest number left to reach.

But there is a problem even deeper than looking too many times. Before we count our looks, the way we split the data already hides an assumption, and that is next.

### 2.3 The same trap, hidden in the procedures we never question

§2.2 was about looking too many times. But suppose we are disciplined and look only once. Is the number honest then? Not always, and this time the fault is not in how hard we searched, but in the recipe itself.

The standard recipe is plain: shuffle the data, cut it into train, validation, and test, and cross-validate. We treat it as neutral, the obviously correct thing to do. It is not neutral. It quietly assumes something about the data, and that something is a property of the *problem*, not of the recipe, so one recipe cannot be right for every problem. Below we walk the recipe step by step, (A) how we split, (B) how we standardise, (C) what we measure, and on **real** data we show each step hiding an assumption that, on the wrong problem, ruins the result.

We do it the same way every time: the same small network from §2.2, on a real dataset, with everything held fixed and **only the split changed**. Whatever moves the number is the split, and nothing else.

*The three problems.*

The rest of this report leans on three real datasets, so it is worth meeting them properly before we put them to work. They were not chosen for variety. They were chosen because their structures differ, and the structure is exactly what the recipe assumes away. One of them satisfies the recipe's assumption and so acts as a control: if we only ever showed problems where shuffling fails, we would be proving that we can pick our examples, not that structure is what matters. The other two break the assumption in two different ways.

| problem | what one row is | its structure | source and size |
| ------- | --------------- | ------------- | --------------- |
| **Loan default** | one credit-card client: 23 features, and whether they defaulted | independent rows, so the assumption **holds**: no client appears twice, and all of them come from one cross-section, so there is no order between rows | UCI credit-card default (Yeh and Lien, 2009); 30 000 rows, 22% default |
| **The market** | one trading day of the S&P 500 | **time order**: today's move looks like yesterday's, and the series drifts over the years | ^GSPC daily closes, frozen at 2026-07-03; 6 658 days |
| **Activity recognition** | one window of phone motion: 561 features, and which activity it was | **people**: every row carries a subject id, and 30 people supply hundreds of windows each | UCI HAR (Anguita et al., 2013); 10 299 rows, 6 activities |

Loan is the control. The market breaks the assumption through time; the activity data breaks it through people. Two different violations and one honest control are enough to say the failure comes from the structure of the problem, not from a lucky choice of example.

These same three carry the rest of §2.3: the split (A), the scaling (B), and the metric (C). Appendix C shows each one up close, the raw rows as they arrive, the evidence for the structure claimed above, and how each raw file becomes the task we run; Appendix D holds the code that runs every experiment below.

**(A) The split: is the data safe to shuffle?**

Why shuffle before we cut? So that each part looks like the whole. But shuffling only makes sense if the rows are interchangeable, if the order does not matter, if any row could sit in any set. That is the hidden assumption: the rows are independent, drawn from the same pot. The textbook calls it a random split, or k-fold cross-validation, and hands it to us as the default (Stone, 1974; Kohavi, 1995).

Sometimes that is true, and then shuffle is exactly right. We start there, so we are not accusing shuffle of a crime it did not commit.

*Loan: where the assumption holds.*

The rows here are interchangeable, so the split should not matter. That is a claim we can test rather than assert. We do not change the split once, we change it eleven times and hold everything else still. Ten of them are independent random 80/20 draws, one per seed. The eleventh is a different method rather than a different seed: a stratified draw that forces the 22% default rate onto both sides of the cut. In all eleven runs the model is rebuilt from scratch with the same recipe (one hidden layer of 16 units, learning rate 0.3, 300 full-batch steps), and the features are standardised using the training side of that particular split only, never the test side. The logic of the check is this: if the rows really are exchangeable, then no split can see anything the others missed, so all eleven readings must agree. If some structure were hiding in the data, at least one split would break ranks.

| split              | test accuracy |
| ------------------ | ------------- |
| ten random splits  | 0.813 ± 0.003 |
| stratified split   | 0.814         |

Nothing breaks ranks. The ten random draws run from 0.807 to 0.816, a spread of 0.009, and the stratified draw lands in the middle of them at 0.814. That spread is just the sampling wobble of a 6 000-row test set, not a difference between methods. The assumption holds, so shuffle is the correct, honest method, exactly as the textbook promises. => When the rows are exchangeable, the split is immaterial: changing it moved nothing.

But that is the easy case. Now two problems where the rows are *not* interchangeable, and the same shuffle turns into a lie.

*The market: where time breaks the assumption.*

First the task, because it has to be one the market really answers. We do not try to call direction, up or down, which on this series is a coin flip; we will use that fact in a moment. We predict whether tomorrow is a **busy** day, its move larger than the median move, from the sizes of the last five days' moves. Busy days cluster: a stormy week tends to stay stormy (volatility clustering, Engle, 1982), so the target carries real, persistent signal. That matters more than it looks, because leakage can only inflate an edge that is actually there.

Why days are not interchangeable: today looks like yesterday. Shuffle them and tomorrow's near-twin lands in the training set, so the model is not predicting the future, it is recalling a neighbour it already saw.

Now the two splits, and they differ by one line. The honest one takes the first 80% of the days in date order as training and the last 20%, the newest days, as test: the model only ever sees the past and is asked about the future. The wrong one takes the same 6 658 days, shuffles them, and cuts 80/20 at random, so a day from one year can sit in training while the day beside it sits in test. Everything else is pinned: the same MLP recipe (16 hidden units, learning rate 0.5, 300 full-batch steps), the same starting weights for both arms of a given seed, and standardisation fitted on whichever side is training. We run the pair ten times and read them paired, seed against seed, so nothing rests on one lucky draw.

| split                                   | test accuracy |
| --------------------------------------- | ------------- |
| shuffle (random)                        | 0.615         |
| chronological (train past, test future) | 0.585         |

Over ten seeds the shuffle score is higher every single time, by +0.030 on average. That extra 0.030 is not skill, it is the future leaking backward across the cut (leakage: Kaufman, Rosset and Perlich, 2012). The honest split is the one that respects the order of time.

Which honest number, though? Test on the last stretch alone and we read 0.585; roll the cut forward through the series and average many future windows, this is **walk-forward**, or rolling-origin, the standard honest protocol for a time series (Tashman, 2000; Bergmeir and Benitez, 2012), and we read 0.603. The honest number wobbles between the two, because the market drifts: a later year is not the same market as an earlier one. But every honest reading sits below the shuffled 0.615. So we do not claim the whole gap is "leakage": shuffling is over-optimistic for two reasons at once, it lets near-twin days leak across the cut, *and* it pretends the future is the same market as the past. Walk-forward is what shows the deficit is real and persistent, not one unlucky window.

One last check, and it is the honest one. Leakage can only inflate an edge that is really there. So take a target with no edge, next-day direction, up or down, a coin flip on this series, and run the same test. The gap vanishes: shuffle and chronological land within a whisker of each other, both near the 0.537 always-up rate. No signal, nothing to leak. That the volatility task shows a gap and the direction task shows none is the surest sign we are measuring a real effect, not a lucky wiring.

*The activity data: where people break the assumption.*

A row here is not an independent draw; it belongs to a person, and one person's rows are near-copies of each other, the same gait, the same way of sitting. Shuffle the rows and almost everyone lands on both sides of the cut: the model sees a given person walking in training, then is tested on that same person walking again. It scores well not because it learned *walking*, but because it learned *the person*. The real question is never asked: can it read someone it has never met?

So ask it, and again the two splits differ by one line. The wrong one, called record-wise, shuffles all 10 299 windows and takes a random fifth as test, which leaves almost every one of the thirty people sitting on both sides. The right one, subject-wise, draws six of the thirty people at random and gives every window they ever produced to the test set, so the model meets those six for the first time at test. That is the matched method: group the rows by subject and leave whole subjects out (GroupKFold, or leave-one-subject-out; the mistake and its fix are named in Saeb et al., 2017). Everything else is pinned again: the same network with one output per activity instead of two (64 hidden units, learning rate 0.1, 400 steps), the same standardisation rule, ten paired seeds.

| split                          | test accuracy |
| ------------------------------ | ------------- |
| shuffle rows (record-wise)     | 0.973         |
| hold out people (subject-wise) | 0.946         |

Record-wise reads 0.973, but part of that is recognising people it already met. Test on strangers and it falls to 0.946, lower in nine of ten runs. Notice how the two rows scatter, too. The record-wise readings barely move off 0.973, while the subject-wise ones run from 0.881 to 0.973 depending on which six strangers we happened to draw. That scatter is information: some people are harder to read than others, and the record-wise split hides that fact completely. The honest question is harder, and the honest number is both lower and less certain. The 0.973 was never the model's skill at reading activities; it was partly its skill at reading *these thirty people*.

*What the three say together.*

Three problems, one model, one thing changed. On loan the split was immaterial. On the market and on the phone data it moved the number by about three points, and always in the same direction: the shuffled split reads higher than the honest one.

So the fault was never shuffling. Shuffle is right for loan and wrong for the other two, and the recipe cannot tell the difference, because the difference is not in the recipe. It is in the data. Loan's rows are independent clients; the market's rows are days in a row; the phone's rows belong to people. Exchangeability is a property of the problem, and the recipe assumes it for free.

That is why there is no universal split. Each structure has its own matched method, and each one is standard, not something we invented (the structures and their matched splits are surveyed in Roberts et al., 2017): rows in time order want a chronological split, or walk-forward (rolling-origin) cross-validation; rows that belong to entities want the entities held out, by GroupKFold or leave-one-subject-out; rows that really are independent want the plain random split the textbook teaches. The work is not finding a clever method. The work is knowing which situation we are in.

=> The recipe is normalised for all problems, so it fits no problem in particular. Apply it to a structure it violates and the number it hands us is wrong from the first honest look.

And notice what this is. In §2.2 the number lied because we looked too many times. Here we looked once, and it lied anyway, because the procedure that produced it assumed something false. Same gap, same direction, different cause: there it was too much searching, here it is an assumption nobody stated.

**(B) Standardising: does the world stand still?**

The next step in the recipe is scaling. Features arrive on wildly different scales, so we standardise them: subtract the mean, divide by the standard deviation. And every course teaches the careful version, the one that avoids exactly the leak we spent (A) on: fit the scaler on the training side only, then apply that frozen transform to validation and test. Never fit it on the test. That rule is right, and we have followed it everywhere in this report.

But look at what "frozen" is doing. The training mean and standard deviation get baked into the model as though they were constants of the world. That is this step's hidden assumption: **the world stands still**. Whatever we measured on the training years will still describe the years we deploy in.

For loan that is true: one cross-section, no time, nothing that can drift. For the market it is not, and Appendix C has already measured how badly. The index runs from 1 455 to 7 483 across the series, five times larger, so the same one-percent day that moved fifteen points at the start moves seventy-five at the end.

So we test this step the way we tested the split. The split stays honest and never moves: chronological throughout, train on the past, test on the future, nothing leaked. The task does not move either, it is the same busy-day label. The model does not move. We change only two things: whether the feature drifts, and how it is scaled.

The drifting feature is the same information in a different unit. Instead of the size of a move in percent, we take the size in points, which is what anyone gets by subtracting two closes and forgetting to divide. Same market, same days, same label. Only the unit drifts.

| feature                      | scaling                    | test accuracy |
| ---------------------------- | -------------------------- | ------------- |
| size in percent (stationary) | frozen, the textbook rule  | 0.585         |
| size in points (drifts)      | frozen, the textbook rule  | 0.512         |
| size in points (drifts)      | rolling, a trailing window | 0.549         |

Read the middle row slowly. The rule that is taught as the correct one, applied to a drifting feature, on a perfectly honest split, with nothing leaked anywhere, took a model that scores 0.585 and left it at 0.512. That is chance. And this is the damage: the practitioner who did everything the book says would read 0.512, conclude that the market has no signal here, and walk away. The signal is there. The rule destroyed it.

The mechanism is visible once we ask what the model is actually being shown. Under the frozen scaler the test days arrive this far from the training mean:

| feature         | test distance, average | worst  |
| --------------- | ---------------------- | ------ |
| size in percent | 0.57σ                  | 9.19σ  |
| size in points  | 1.67σ                  | 27.69σ |

The drifting feature puts the test days three times further out, with extremes near twenty-eight standard deviations. The model is being questioned about a region it never saw. It learned a world where a busy day was fifteen points; it is deployed in a world where a busy day is seventy-five, and the frozen scaler still insists that fifteen is average.

The last row is the fix matched to the structure. Instead of freezing one mean and one standard deviation for all time, scale every day by its own recent past, a trailing window of the days before it. It is causal, no future is used, so nothing leaks, and it recovers most of the loss: 0.512 climbs back to 0.549.

Most, but not all. 0.549 is still short of the 0.585 that the stationary feature reached, and that gap is worth saying out loud, because it is the honest lesson of this step. Patching the scaler is second best. The real cure was to never build a drifting feature: divide by the price, and the drift is gone before the scaler ever meets it.

=> The hidden assumption of this step is stationarity, that the training statistics still describe the future (Shimodaira, 2000; Quiñonero-Candela et al., 2009; Gama et al., 2014). Where it holds, the textbook rule is exactly right. Where it fails, the same rule quietly kills a working model, and no honest split will warn us, because nothing was leaked. The number is truthful. It truthfully reports a model that the recipe broke.

And notice the shape of this against (A). A hidden assumption in the split made the number too **high**, an edge that was never there. A hidden assumption in the scaling made the number too **low**, an edge that was there and got destroyed. The same disease with opposite symptoms: one recipe, normalised for all problems, meeting a problem it does not fit.

**(C) Accuracy as the metric.**

> *To write. Textbook rule: report accuracy, the fraction of predictions we get right, as "how good the model is". Hidden assumption: the classes are balanced and the two kinds of error cost the same. How it fails: on rare events (~1% positive, fraud, default, disease) "always predict no" scores 99% and catches zero of the cases we built the model for; a genuinely useful model scores lower. Distinct from A and B: no leakage, no drift, the number is measured perfectly honestly and still measures the wrong thing.*

Snooping is dangerous everywhere, and worst of all in deep learning, the branch we just saw. Now put the two branches together.

---

## 3. The two branches meet: we cannot trust the GOAL

Put the two branches side by side.

From §1 and §2: the number goes up when we search, and deep learning lets us search almost without limit, most of it uncounted. So a high score can be nothing but the best of many tries, luck, dressed as skill. From §2.3: even one honest look can be wrong, if the way we split the data assumed something false about it. So the number can be inflated, and the number can be wrong at the root. Often both at once.

So here is the question we cannot avoid. If the score is the thing we push up by searching, and the score is also the thing that can be false from the first look, then how do we believe the model at all? The number we chase is the number we cannot trust.

=> We cannot trust the GOAL.

And deep learning does not soften this, it sharpens it. A deep model works by making the validation score the best: early stopping, seed picking, the whole search all pull toward that one number. So the tool we reach for is, by its own design, the tool that inflates the thing we are not allowed to believe. The harder it works, the less its number means.

There is a deeper way to say the same thing. The trouble is not only that we searched too much. It is that we never made our assumption clear. We shuffled without asking whether the data was random to shuffle. Data that is not random to be shuffled, but split as if it were, gives a goal that measures the wrong thing, and a goal that measures the wrong thing is not really a goal at all.

=> Unclear goal ⇔ train model wrong.

So both branches land in the same place. The number cannot carry our trust, not because we were careless, but because searching inflates it and a wrong assumption corrupts it. If trust does not live in the number, then where does it live? That is the last question.

---

## 4. What we trust instead: the PROCEDURE

If trust does not live in the number, then it lives in the way the number was made. We trust the PROCEDURE, not the score.

What makes a procedure worth trusting? Two things, and they are exactly the two failures turned around.

First, its assumptions are clear. We say out loud what we are assuming about the data, and we check it. If the data has an order, we do not pretend it does not. This is the opposite of the shuffle in §2.3, which assumed the data was random and never said so.

Second, its assumptions fit the situation. There is no single recipe that is honest for every problem. A split that is fine for independent rows is a lie for a time series. So we match the method to the data in front of us. For a time series that means a walk-forward split: train on the old days, test on the newest, and never let the future leak back. The assumption, the past comes before the future, is now built into the method instead of broken by it.

From there the rest is discipline against §1 and §2. Search less, and search honestly: do not fish with hidden seeds and epochs, and choose the winner with a less noisy estimate than a single small validation set. Keep one sealed test, and open it exactly once, on the single model the procedure already chose. A number we looked at once, produced by a method whose assumptions were clear and right, is a number we can believe. A number we chased is not.

This changes what "the best model" means. Put two procedures on the same problem. One searches hard, fishes seeds and epochs, and reports the prettier number. The other searches less, chooses honestly, and reports a plainer one. The prettier number is the inflated one, its real performance is worse, and its report is off by the size of the search. The plainer number is close to the truth. The honest procedure wins twice: a model that is actually as good or better, and a number we are allowed to believe.

So we do not accept a model because its score is high. We accept it only when that score came from a procedure that could not have inflated it, an honest method, assumptions clear and fit to the problem, a sealed test opened once, and only when it beats the baseline by more than the measurement noise. The number to believe is the one the procedure earned the right to report.

=> We trust the PROCEDURE of making the model, with clear assumptions for each situation.

---

## References

*These are the sources for §2.3-A and §2.3-B. The list grows as the other sections are written; the formatting is to be fixed to the handbook style at the end.*

**The standard recipe we put on trial.**

- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society, Series B*, 36(2).
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., ch. 7.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12. (The utilities named in this section: `train_test_split`, `KFold`, `TimeSeriesSplit`, `GroupKFold`, `LeaveOneGroupOut`.)

**Splits matched to a structure.**

- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4). (Rolling-origin evaluation: the academic name for walk-forward.)
- Bergmeir, C. and Benitez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191.
- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*. (Rolling origin, time-series cross-validation.)
- Saeb, S. et al. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5). (Subject-wise versus record-wise cross-validation: the mistake and its fix.)
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8). (One survey covering every structure in this section.)

**Drift, and the assumption that the world stands still.**

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2). (Covariate shift.)
- Quinonero-Candela, J., Sugiyama, M., Schwaighofer, A. and Lawrence, N. (eds.) (2009). *Dataset Shift in Machine Learning*. MIT Press.
- Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M. and Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4).
- Box, G. E. P. and Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. (Differencing a series to stationarity.)

**Leakage.**

- Kaufman, S., Rosset, S. and Perlich, C. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4).

**The data, and the effect the market task leans on.**

- Yeh, I.-C. and Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2). (The UCI loan-default set.)
- Anguita, D. et al. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN*. (The UCI HAR set.)
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4). (Volatility clustering.)

---

## Appendix A: Where the two formulas come from

*Outside the page limit; here so the two formulas the lab leans on can be checked, not taken on trust. Nothing below goes past first-year probability, we derive it in place.*

The lab in §2.2 used two formulas: the spread of a chance score, σ = 0.5/√n, and the best of N scores, about 0.5 + σ√(2 ln N). Here is where each one comes from.

### Part 1: The spread of a chance score: σ = 0.5/√n

A validation score is the fraction of the n validation points the model gets right. Take a model that is really at chance, on random labels, every model is. On each point it is either right or wrong, and being right is a fair coin: probability 0.5, independent of the other points.

So the number it gets right, call it k, is the number of heads in n fair flips, a Binomial(n, 0.5). Two standard facts about it:

- its mean is n·0.5, so the score k/n has mean 0.5;
- its variance is n·0.5·0.5 = n/4, so the score k/n has variance (n/4)/n² = 1/(4n).

The spread is the square root of the variance:

> σ = √(1/(4n)) = 0.5/√n.

That is the first formula. A chance model does not sit at exactly 0.5, it scatters around 0.5, and the smaller the validation set n, the wider it scatters. With n = 200, σ = 0.5/√200 ≈ 0.035.

We need one more thing. For n even moderately large, the Binomial looks like a bell curve (the Central Limit Theorem), so we may write each score as 0.5 + σ·Z, where Z is a standard normal, mean 0, spread 1. For n = 200 this is a close fit.

### Part 2: The best of N: 0.5 + σ√(2 ln N)

Now we draw N configurations. Each gives a score 0.5 + σ·Z, with Z₁, …, Z_N standard normals, and for now independent. We keep the best, so the best score is 0.5 + σ · (the biggest of the N normals). Everything comes down to one question: how big is the biggest of N standard normals?

The answer is about √(2 ln N). Here is why. The biggest of N draws is roughly the value t that only about one draw in N gets past, the level where P(Z > t) ≈ 1/N. So we need how far out the normal's tail sits.

The tail of the bell curve falls off fast. Beyond t, the tail area is close to

> P(Z > t) ≈ e^(−t²/2) / (t·√(2π)),

and the e^(−t²/2) is what dominates. Set this equal to 1/N and take logs:

> −t²/2 − ln(t√(2π)) ≈ −ln N.

The ln N term is the big one; the ln(t√(2π)) grows only like ln t, far slower, so we drop it and keep the leading term:

> t²/2 ≈ ln N  =>  t ≈ √(2 ln N).

So the biggest of N standard normals is about √(2 ln N), and the best score is

> 0.5 + σ · √(2 ln N).

That is the second formula.

### Part 3: Reading the formula, and its honest limits

The formula is worth more than its number, because it explains the shape of the gap:

- **It grows with N.** More tries, bigger best. Searching harder inflates the number, the winner's curse in one line.
- **But it grows only like √(ln N).** ln N barely moves when we double N, and its square root moves less. So the gap jumps early and then crawls. This is why we try N at 1, 2, 5, 10, 20, … and not 1, 2, 3, 4, the action is in the first few tries.
- **It scales with σ = 0.5/√n.** A smaller validation set means a larger σ means a larger gap. The small validation set is the fuel, exactly as §2.2 said.

Two honest limits, both pushing the real gap a little below the formula:

- **Independence.** We assumed the N draws were independent. Configurations that differ only by a seed, or by one more epoch, are near-copies, not fresh draws, so N of them count as fewer than N independent tries, and the real best sits a touch below σ√(2 ln N). (This is the hidden-search point of §1: seeds and epochs are extra draws, but correlated ones.)
- **The Gaussian approximation.** The Binomial is only approximately a bell curve, and √(2 ln N) is the leading term of a longer expression. Both are close for the n and N we use, not exact.

So we do not lean on the formula as truth. We use it to see the mechanism, and we measure the real gap in the lab (§2.2). The two agree in shape, and that agreement is the point: the gap is not a quirk of one dataset, it is the biggest of many noisy draws, behaving as the biggest of many noisy draws must.

---

## Appendix B: the code that ran the lab

*Outside the page limit; the measurement half of the grounding. Appendix A shows why the formulas hold; this shows the code that measured the real gap, and that running it reproduces the numbers in §2.2.*

The whole lab is one short file, `code/lab_demo.py`, that runs on numpy alone, `python code/lab_demo.py`, with `seed = 0`, so every number below reproduces exactly. Here it is in the five pieces of §2.2. The helper functions (`init`, `forward`, `softmax`, `accuracy`, `sample_config`) are in the file; `forward` is just `h = ReLU(X·W1 + b1)` then `z = h·W2 + b2`, and `sample_config` draws a width from {4, 8, 16, 32, 64} and a learning rate around 0.01–0.3.

**Piece 1: the no-signal data.** Random features; labels that are coin flips, drawn without ever looking at the features. Split into train, a small validation set (200), and a large sealed test (10 000).

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

The labels are near 50/50 and barely correlated with any feature, there is nothing to learn, so the true accuracy is 0.5 by construction.

**Piece 2: the small MLP; a configuration is (width, learning rate).** One hidden layer, ReLU, two outputs, softmax with cross-entropy, trained by full-batch gradient descent. The gradient is the chain rule, written out by hand.

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

One model: its test accuracy is 0.498, chance, as it must be. Its training accuracy is a little higher (0.546: it memorises a few points), and its validation score, 0.520, is one noisy draw around 0.5.

**Piece 3: the gap machine.** Draw N configurations, keep the one with the best validation score, then open the sealed test exactly once, on that winner.

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

The best of twenty on validation reads 0.565, but that same model on the sealed test is 0.508, chance. The 0.057 between them is pure luck.

**Piece 4: the sweep.** Run the gap machine over a grid of N, averaging over fifteen repeats. This is the table in §2.2.

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

**Piece 5: the formula check.** Compare the measured apparent with 0.5 + σ√(2 ln N), where σ = 0.5/√200 = 0.0354.

```
   N | measured app |  formula
   1 |        0.510 |    0.500
  10 |        0.553 |    0.576
 100 |        0.577 |    0.607
```

Same shape, but the measured value sits a little under the formula. The formula assumed N independent draws; here the N configurations share one training set, so their validation scores are correlated, and the best of correlated draws runs less far than the best of independent ones (Appendix A, the honest limits).

**Piece 6: the test is not special.** The gap machine kept the sealed test honest by opening it once. What if we reuse it, select on it, as retraining until the test looks good would? The same sweep records both readings of the same configurations; since the truth is known (0.5), no second test is needed to expose the lie.

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

Reusing the test inflates it too, 0.500 to 0.512, so selecting on the test poisons it like any other set. It moves less than the validation only because the test is larger (its σ is seven times smaller). The lesson: selection is the poison, not the set; the test's honesty lives in looking once, on a set large enough to be quiet.

So the whole result is one line: a gap of +0.077 at N = 100, out of data with no signal at all, growing with the search, and the same curse hits any set we select on, the sealed test included. Rerun `python code/lab_demo.py` to reproduce every number here.

---

## Appendix C: the three datasets, up close

*Outside the page limit. §2.3 leans on these three for the rest of the report, so here they are as they actually arrive: where each came from, what the raw rows hold, what makes its structure the structure we claim it is, and how each becomes the task we run. Every number below is read straight off `python code/data_peek.py`.*

### Loan: UCI credit-card default

Fetched once and frozen to `data/loan_uci350.csv` (Yeh and Lien, 2009). Thirty thousand rows and twenty-four columns: twenty-three features and the label. Here are the first three clients, one column-group per line so a whole row fits on the page:

| column   | what it holds                                | client 1                   | client 2                                 | client 3                                       |
| -------- | -------------------------------------------- | -------------------------- | ---------------------------------------- | ---------------------------------------------- |
| 0        | LIMIT_BAL, the credit limit                  | 20 000                     | 120 000                                  | 90 000                                         |
| 1        | SEX                                          | 2                          | 2                                        | 2                                              |
| 2        | EDUCATION                                    | 2                          | 2                                        | 2                                              |
| 3        | MARRIAGE                                     | 1                          | 2                                        | 2                                              |
| 4        | AGE                                          | 24                         | 26                                       | 34                                             |
| 5 to 10  | PAY_0 to PAY_6, repayment status, six months | 2, 2, -1, -1, -2, -2       | -1, 2, 0, 0, 0, 2                        | 0, 0, 0, 0, 0, 0                               |
| 11 to 16 | BILL_AMT1 to 6, the bill, six months         | 3 913, 3 102, 689, 0, 0, 0 | 2 682, 1 725, 2 682, 3 272, 3 455, 3 261 | 29 239, 14 027, 13 559, 14 331, 14 948, 15 549 |
| 17 to 22 | PAY_AMT1 to 6, what they paid, six months    | 0, 689, 0, 0, 0, 0         | 0, 1 000, 1 000, 1 000, 0, 2 000         | 1 518, 1 500, 1 000, 1 000, 1 000, 5 000       |
| 23       | **default**                                  | **1**                      | **1**                                    | **0**                                          |

Client 1 reads as a person: twenty-four years old, a 20 000 limit, small bills, almost nothing paid back, and a 1 at the end. They defaulted. We use the file as it comes: nothing dropped, nothing engineered.

The rest of the file in words: 22.12% of clients default (6 636 of 30 000), so always saying "no default" already scores 0.779. Thirty-five rows are exact copies of another row. There is no id column and no date column.

Why we call the rows exchangeable, from the file itself: with no identifier and no date there is literally nothing to order the rows by, and the source is a single cross-section, every client watched over the same six months. The thirty-five duplicates are one in a thousand, far too few to move any split; with a feature set this coarse, two different people can simply land on the same values.

Honest limits: one bank, one country, one six-month window (Taiwan, 2005). That is a caution about carrying the model to another period, but it cannot create an order between rows, which is why the set still serves as the control.

### The market: S&P 500 daily closes

Downloaded once and frozen to `data/gspc_2026-07-03.csv`. The file is one column of numbers and nothing else: 6 664 closing prices in date order. The task is built from that column in four steps, and the first days show every one of them:

| day | close    | the move into it, r | its size, abs(r) |
| --- | -------- | ------------------- | ---------------- |
| 1   | 1 455.22 |                     |                  |
| 2   | 1 399.42 | -0.0383             | 0.0383           |
| 3   | 1 402.11 | +0.0019             | 0.0019           |
| 4   | 1 403.45 | +0.0010             | 0.0010           |
| 5   | 1 441.47 | +0.0271             | 0.0271           |

A day's features are then the five previous sizes, and its label is whether the next size beats the median size, 0.00544:

| row  | lag 5  | lag 4  | lag 3  | lag 2  | lag 1  | label |
| ---- | ------ | ------ | ------ | ------ | ------ | ----- |
| X[0] | 0.0383 | 0.0019 | 0.0010 | 0.0271 | 0.0112 | 1     |
| X[1] | 0.0019 | 0.0010 | 0.0271 | 0.0112 | 0.0131 | 0     |
| X[2] | 0.0010 | 0.0271 | 0.0112 | 0.0131 | 0.0044 | 1     |

Look at what that table shows on its own. `X[1]` is `X[0]` shifted one step left with one new number added on the end: consecutive rows share four of their five columns. They are near-twins by construction, before we say a single word about markets.

Two more numbers are the structure. The **drift**: the index runs from 1 455 to 7 483, five times larger, so the same one-percent day is worth five times the points at the end that it was at the start. The **lag-1 autocorrelation of abs(r) is +0.287**: the size of today's move really does predict the size of tomorrow's. That is volatility clustering (Engle, 1982). It is the real signal the task learns, and it is also precisely what a shuffle leaks, because it is what makes neighbouring days alike. After five lags are used up 6 658 days remain, and because the threshold is the median the two classes are exactly balanced at 0.500, so chance is 0.5 and no majority-class trick is available.

Honest limits: one index along one path through history. We read its numbers as directional, not to the third decimal.

### Activity recognition: UCI HAR

Downloaded once from the UCI archive (dataset 240) and cached to `data/har.npz` (Anguita et al., 2013). Thirty volunteers wore a waist-mounted phone; its accelerometer and gyroscope traces are cut into short overlapping windows, and each window arrives already summarised by 561 precomputed features. The first three rows, with the first six of those features:

| row | subject | activity        | f1     | f2      | f3      | f4      | f5      | f6      |
| --- | ------- | --------------- | ------ | ------- | ------- | ------- | ------- | ------- |
| 0   | **1**   | **4, standing** | 0.2886 | -0.0203 | -0.1329 | -0.9953 | -0.9831 | -0.9135 |
| 1   | **1**   | **4, standing** | 0.2784 | -0.0164 | -0.1235 | -0.9982 | -0.9753 | -0.9603 |
| 2   | **1**   | **4, standing** | 0.2797 | -0.0195 | -0.1135 | -0.9954 | -0.9672 | -0.9789 |

The structure sits in the two bold columns, and in none of the 561 features. Three rows in a row: all subject 1, all standing, and their features agree with each other to two decimals. Rows arrive in runs, one person doing one thing for hundreds of windows at a time.

The rest in words: 10 299 windows, 561 features, six activities (walking, upstairs, downstairs, sitting, standing, laying), thirty subjects with ids 1 to 30, and between 281 and 409 windows each, 343 on average. That last figure is the whole point: a row is not a person, it is a moment of a person, and thirty people is all we have.

The archive ships a train folder and a test folder that are already subject-disjoint. We pool them into one bag of 10 299 windows so we can draw the subject split ourselves instead of inheriting somebody else's, which is what lets us put a record-wise draw and a subject-wise draw against each other on identical data.

Honest limits: thirty people, all healthy adults, one phone in one position.

---

## Appendix D: the code for the real-data split experiments

*Outside the page limit. §2.3 changes one thing at a time on real data; this is the code that did it and the output it printed. Four short files in `code/`, numpy only, all reusing the same MLP from `code/lab_demo.py`: `code/loan_split.py`, `code/finance_split.py`, `code/har_split.py` for the split (A), and `code/scaling_split.py` for the scaling (B). Each prints the table quoted in the body.*

**The shared shape.** Every experiment is the same three thoughts: load a real dataset, build two splits, run the identical MLP under each. Only the split differs. Features are standardised with the mean and standard deviation of the training side only, so the standardising itself never leaks (that is §2.3-B's problem, deliberately kept out of this one).

```python
def standardize(Xtr, Xte):                 # fit on TRAIN only
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s
```

**Piece 1: loan, the control.** One row per client, the last column is the default label. The two splits are a plain random shuffle and a stratified draw that keeps the class ratio on both sides.

```python
def score(X, y, how, seed, width=16, lr=0.3, epochs=300):
    n = len(y); ntest = n // 5; rng = np.random.default_rng(seed)
    if how == "stratified":                # keep the 22% default rate on each side
        te = np.concatenate([rng.permutation(np.where(y == c)[0])[:int(round(ntest * np.mean(y == c)))] for c in (0, 1)])
        tr = np.setdiff1d(np.arange(n), te)
    else:                                  # plain random shuffle
        idx = rng.permutation(n); te, tr = idx[:ntest], idx[ntest:]
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed + 1))
    return accuracy(p, Xte, y[te])
```

```
n 30000 | features 23 | default rate 0.221
10 INDEPENDENT random splits:
  0.814 0.807 0.813 0.814 0.809 0.815 0.809 0.816 0.815 0.815
MEAN 0.813 | sd 0.003 | range 0.009
stratified split: 0.814
```

Ten random splits and a stratified split all land inside 0.009 of each other. The split does not move the number, which is what exchangeable rows look like from the outside.

One thing we tried and dropped: splitting loan by row position gave 0.828, well outside that spread. That is not real structure, it is incidental ordering in the CSV. Loan has no time axis, so a position split measures nothing; the honest control is random versus stratified.

**Piece 2: finance, time order.** Closing prices become returns; the target is a busy day, tomorrow's move larger than the median, read from the last five days' move sizes.

```python
def load_finance(target="volatility", L=5):
    close = np.loadtxt("data/gspc_2026-07-03.csv", skiprows=1)
    ret = np.diff(close) / close[:-1]
    s = np.abs(ret)                                   # volatility proxy
    X = np.column_stack([s[i:len(s)-L+i] for i in range(L)])
    fut = s[L:]; X = X[:len(fut)]
    y = (fut > np.median(s)).astype(int)              # busy day?
    return X, y

def score(X, y, how, seed, width=16, lr=0.5, epochs=300):
    n = len(y); ntest = n // 5
    if how == "chronological":                        # RIGHT: past -> train, future -> test
        tr, te = np.arange(n - ntest), np.arange(n - ntest, n)
    else:                                             # WRONG: shuffle -> the future leaks into train
        idx = np.random.default_rng(seed).permutation(n); te, tr = idx[:ntest], idx[ntest:]
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], width, lr, epochs, np.random.default_rng(seed + 1))
    return accuracy(p, Xte, y[te])
```

The only difference between the two readings is which indices go where. Walk-forward is the same honest idea rolled forward, an expanding past tested on the next block:

```python
def walk_forward(X, y, folds=5, ...):
    n = len(y); bs = n // (folds + 1); out = []
    for k in range(1, folds + 1):
        tr, te = np.arange(0, k*bs), np.arange(k*bs, (k+1)*bs)
        ...
```

```
days 6658 | high-vol rate 0.500
seed | shuffle |  chrono |   diff
   0 |   0.612 |   0.583 | +0.029
 ...
   9 |   0.631 |   0.588 | +0.044
MEAN |   0.615 |   0.585 | +0.030      (shuffle > chrono in 10/10 seeds)
walk-forward (rolling origin, 5 folds): 0.603
CONTRAST, next-day DIRECTION: shuffle 0.529 | chrono 0.533 | diff -0.003   (up-rate 0.537)
```

**Piece 3: HAR, repeated people.** The dataset ships a train folder and a test folder; we pool them so all thirty subjects sit in one bag and we split it ourselves. The subject id attached to every row is what makes the honest split possible at all.

```python
def score(X, y, subj, how, seed, K=6, width=64, lr=0.1, epochs=400):
    n = len(y); rng = np.random.default_rng(seed)
    if how == "subject-wise":              # RIGHT: hold out whole SUBJECTS -> test on new people
        subs = rng.permutation(np.unique(subj)); test_subs = subs[:max(1, len(subs)//5)]
        te = np.isin(subj, test_subs)
    else:                                  # WRONG: shuffle windows -> the same person on both sides
        te = np.zeros(n, bool); te[rng.permutation(n)[:n//5]] = True
    tr = ~te
    Xtr, Xte = standardize(X[tr], X[te])
    p = train(Xtr, y[tr], K, width, lr, epochs, np.random.default_rng(seed + 1))
    return accuracy(p, Xte, y[te])
```

```
windows 10299 | subjects 30 | classes 6
MEAN | record-wise 0.973 | subject-wise 0.946 | diff +0.027   (record > subject in 9/10 seeds)
```

This is the one place the network is not literally the §2.2 model: HAR has six activities, so the output layer has six units instead of two. Everything else, one hidden layer, ReLU, softmax with cross-entropy, full-batch gradient descent, is unchanged.

One warning worth recording, because it nearly fooled us. With 561 features a learning rate of 0.5 (the value the market task uses) makes the training diverge: the accuracies come back as noise between 0.0 and 0.35, and the split comparison becomes meaningless. At 0.1 the model trains cleanly (0.98 on its own training set). We only found this by checking that the model learns at all before trusting any comparison between splits. A split experiment on a model that is not training is measuring nothing.

**Piece 4: scaling, and a world that moves (§2.3-B).** The file is `code/scaling_split.py`. Here the task, the label and the split never move; only the feature's unit and the scaling do. The label is always built from the percent size, so it stays stationary and exactly balanced whichever feature the model is fed, which is what makes the three rows comparable.

```python
def load(kind="percent"):
    close = np.loadtxt("data/gspc_2026-07-03.csv", skiprows=1)
    r = np.diff(close) / close[:-1]
    a = np.abs(r)                                            # percent size: stationary
    src = a if kind == "percent" else np.abs(np.diff(close))  # dollar size: drifts with the level
    X = np.column_stack([src[i:len(src)-L+i] for i in range(L)])
    fut = a[L:]; X = X[:len(fut)]
    y = (fut > np.median(a)).astype(int)                      # label always from the percent size
    return X, y
```

The two scalings. The first is the textbook rule. The second is the drift-aware one, and the thing to check in it is that it is causal: row `t` is built only from rows before `t`, never from `t` itself or later, so it cannot leak the future the way a global scaler would.

```python
def frozen(Xtr, Xte):          # the textbook rule: fit on train, freeze, apply to test
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s

def rolling_z(X, W=250):       # drift-aware: every row is scaled by its OWN trailing window
    for t in range(len(c)):
        lo = max(0, t - W); n = t - lo
        if n < 30: continue                          # not enough past yet: leave at 0
        mu  = (cs[t] - cs[lo]) / n                   # mean of the W days BEFORE t
        var = (cs2[t] - cs2[lo]) / n - mu * mu
        Z[t, j] = (c[t] - mu) / (np.sqrt(max(var, 1e-18)) + 1e-9)
```

```
### under the frozen rule, how far outside the training range does the test land?
  percent (stationary)   test |z|: mean  0.57   max   9.19
  dollars (drifts)       test |z|: mean  1.67   max  27.69

### same MLP, same chronological split, only the feature and the scaling change
feature                | scaling  | accuracy over 10 seeds
percent (stationary)   | frozen   | mean 0.585   sd 0.004
dollars (drifts)       | frozen   | mean 0.512   sd 0.003
dollars (drifts)       | rolling  | mean 0.549   sd 0.007
```

The first block is the mechanism and the second is the damage. Nothing in this experiment leaks: the split is chronological in all three rows, and the rolling scaler is causal. The 0.512 is what the textbook rule does to a working model when the feature underneath it drifts.

**Reproduce.** `python code/loan_split.py`, `python code/finance_split.py`, `python code/har_split.py`, `python code/scaling_split.py`. The loan and market data are frozen in `data/`. HAR is downloaded once from the UCI archive (id 240) and cached to `data/har.npz`.

---

