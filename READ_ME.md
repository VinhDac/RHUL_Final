# Data Snooping in Deep Learning: dissertation (working draft)

*Working draft, written as a journey rather than a proof: we start from the ordinary, by-the-book way of judging a model by its number, and follow honestly where it leads. Reasoning first; the code behind every number is in the appendices. Compass in `NORTH_STAR.md`; the underlying argument in `KEY_CORE.html`.*

*Constraint (handbook): final submission ≤ 50 pages, including bibliography, tables and figures, excluding appendices.*

---

## 0. The number we are about to trust

We have a model. Is it any good? How would we even know?

We do the obvious thing. We hide some data from it, let it guess on that data, and count the hits. That gives us a number. Is it high? Good. Low? Not good. So we read the number and we decide.

But we never build just one. We build many and keep the best number. So what are we really trusting in the end? One number.

And here is the question we almost never stop to ask: can we trust it?

Let us find out honestly. We will not bend anything to force a problem into view. We will do the opposite, and follow every rule the books give us, exactly. Then we stand back and ask, together, what that honest number is really worth.

(One limit, so we are not doing everything at once. We stay with deep learning that forecasts and classifies: is tomorrow a busy day on the market, did this borrower default, which activity is this. Not vision, not language, not the rest. Just putting a label on the next case.)

So, where does everyone start? By the book. Let us start there too.

---

## 1. What deep learning really is, and the plan

So we have a job: take the next case and put a label on it. What do we build?

Almost always, the same thing: a small neural network. It is a stack of simple parts. A few features go in, get mixed and bent through a hidden layer, and come out as a guess between the classes.

![The model: a small network](figures/mlp.svg)

How does it learn? We do not program it. We show it examples, and a procedure called gradient descent nudges its many little knobs, again and again, until its guesses on the training data come out mostly right. Give it more knobs, make it wider or deeper, and it can fit more. That is the whole appeal: with enough knobs, a network can fit almost any pattern we hand it.

And there is the catch, the one that shadows this whole report. If it can fit almost any pattern, then it can also fit patterns that are not really there. Show it pure noise for long enough and it will "learn" the noise, memorising which random point got which random label, and score beautifully on the training data.

![Fitting the training data is not the same as learning](figures/loss_curve.svg)

So here is the question everything turns on: if the model can fit anything, even nonsense, how do we ever know it learned something real and not just the noise? We cannot tell from the training score, because it can ace that either way. We can only tell by trying it on data it has never seen. That single idea is the root of the whole recipe that follows: hold some data back, judge the model on that, and never let it peek while it is learning.

Which brings us to the plan. To turn a network into a number we can show anyone, we follow a recipe, the same one printed in every course:

![The normal way to build a model](figures/pipeline.svg)

Five steps. Frame the data. Split it into training, validation and test. Scale the features. Build and search for a good model. Measure how good it turned out. Every step is standard, careful, exactly what we are told to do, and the number at the end is meant to be honest.

So here is the plan for the whole report. We are not going to attack this recipe. We are going to follow it, faithfully, and carry it to one real problem after another. And at each problem we ask the same plain, stubborn question: is this really as safe as it looks?

Let us meet the first one.

---

## 2. Loan: the well-behaved one that still lies

Our first problem is the friendliest one we have. A bank hands us thirty thousand credit-card clients and one question: which of them will miss their next payment? One row is one person.

| client | age | credit limit | latest bill | latest payment | defaulted? |
| ------ | --- | ------------ | ----------- | -------------- | ---------- |
| 1 | 24 | 20 000 | 3 913 | 0 | yes |
| 2 | 26 | 120 000 | 2 682 | 0 | yes |
| 3 | 34 | 90 000 | 29 239 | 1 518 | no |

Twenty-three numbers about a person, and a yes or no. About one in five defaults. Nothing tricky here: separate people, no time, no repeats. So we do the first steps of the recipe by the book, shuffle and split, and it feels almost too easy.

Still, we are careful, so before trusting anything we poke at the split. Does it matter how we cut the deck? It should not, one client tells us nothing about the next, but we check rather than assume. We cut it eleven different ways, ten random shuffles and one that forces the same one-in-five default rate onto both sides, and hold everything else fixed.

| how we split | test accuracy |
| ------------ | ------------- |
| ten random shuffles | 0.813 ± 0.003 |
| forced-ratio split | 0.814 |

Every cut lands in the same place, within a whisker. The split is a formality here, exactly as the book promises. So we build the model, and it scores 0.813. A fifth of clients default, so stamping "no default" on everyone would already score 0.779; our model beats that. We could write 0.813 down and move on, and almost everyone would.

But a number this smooth deserves one hard question. **The whole point of this model was to find the people who will default. So of the clients who actually did default, how many did our 0.813 model catch?**

We open it up and count.

![Accuracy looks fine while most defaulters slip through](figures/confusion.svg)

Out of 1 353 real defaulters, it caught 438 and missed 915. It waved two in three of them straight through, stamped safe. That shiny 0.813 is almost entirely the big green box, the paying clients it correctly cleared. On the very people we built the model for, it is close to a coin.

And now the number stops looking smooth and starts looking dishonest. A model that catches nobody at all, that stamps "will pay" on all thirty thousand, scores 0.779. Ours catches a third of the defaulters and scores 0.813. Three points apart on paper, and worlds apart in the job that mattered. Accuracy cannot tell a working model from one that does nothing, because it counts the huge easy majority and lets the rare, costly cases dissolve inside it.

The strange part is that nothing here was rigged. The split was honest, we checked it eleven ways. Nothing leaked. The number was measured perfectly. It is exactly right, and it answers the wrong question. The fix is to stop letting the crowd decide and weigh both kinds of client equally, balanced accuracy, which scores the defaulters and the payers on their own and averages the two. Ours is 0.633, not 0.813. Far less flattering, and far more honest. Or skip the single number and just read the boxes; nothing hides in a confusion table.

So the friendliest problem taught us something we did not expect. The trap was not in the data, its rows were as clean as they come. It was in what we chose to measure. One problem in, and the recipe has already slipped a false number past us.

## 3. Market: the edge that was too easy

Our second problem barely looks like data at first: one long column of numbers, the daily closing price of the S&P 500, about six and a half thousand days in a row.

| day | close |
| --- | ----- |
| 1 | 1 455 |
| 2 | 1 399 |
| 3 | 1 402 |
| 4 | 1 403 |
| 5 | 1 441 |

So what can we even predict from a column of prices? The obvious first try is direction: will tomorrow close up or down? We build it, split it by the book, and look. It scores 0.53, no better than betting "up" on every single day. No edge at all. Dead end.

But a market carries more than direction. It has a mood: some stretches are calm, some are stormy, and the storms seem to bunch together, a wild day sitting near other wild days. So we change the question. Never mind which way tomorrow moves; can we say how much? From the sizes of the last five days' moves, we predict whether tomorrow is a busy day, its move bigger than usual.

| row  | last five move sizes              | busy tomorrow? |
| ---- | --------------------------------- | -------------- |
| X[0] | 0.038  0.002  0.001  0.027  0.011 | yes |
| X[1] | 0.002  0.001  0.027  0.011  0.013 | no  |
| X[2] | 0.001  0.027  0.011  0.013  0.004 | yes |

We split it by the book, shuffle and cut, and this time it reads 0.615. That is a real edge, well clear of the coin's 0.5, and on the market an edge like that would be worth a fortune. **Which is exactly the problem. A real edge on the market, won this easily? That is far too good to be true. What is leaking in?**

So instead of celebrating, we ask one plain question: is 0.615 what we would actually get in practice? In real life we would train on the days we have and predict days that have not happened. So we measure it that way too, train on the first 80% in date order, test on the newest 20%, everything else untouched.

| how we split      | test accuracy |
| ----------------- | ------------- |
| shuffle           | 0.615 |
| past, then future | 0.585 |

They do not match. And that stopped us, because it should not happen. Back on the loan clients, eleven different ways of cutting the data agreed to the third decimal. Here, two cuts of the same days, the same model, disagree by three whole points. One of the numbers is lying, and at that moment we did not know which one, or why.

So we finally did the thing we should have done first: we stopped fiddling with the model and looked at the data, the rows themselves, side by side. And there it was, in plain sight the whole time. Each row is the last five days; the very next row is those same days slid along by one. Two neighbouring rows share four of their five numbers. They are near-twins. We had been so busy with the model that we never once looked at what a single row actually was.

That one look explains both numbers. When we shuffle, a day and its near-twin can fall on opposite sides of the cut, so at test time the model is handed a day whose near-double it already studied in training. Its 0.615 was never skill at seeing the future; it was skill at recognising a neighbour. The honest split, past then future, leaves it no twin to lean on, and that is why it settles at 0.585 (roll that cut forward through the years and it hovers near 0.60, drifting a little as the market itself changes, but it never climbs back to 0.615).

One doubt still nagged: how do we know the gap is a real leak, and not just the shuffle getting lucky once? The direction task from the very start gives us the answer. A leak can only inflate an edge that is really there, so on the direction target, which we already know is a coin flip, the gap should vanish. It does: shuffle and honest split both land near 0.53, that same base rate. The gap shows up only where there is a real pattern to steal. That is how we know it is a true leak.

![Cornering the market data, one try at a time](figures/market_tree.svg)

Look back at the trail. We did not reason our way to "respect the order of time." We tried the obvious thing and hit a wall, tried another and got a number too good to trust, checked it and got a contradiction, and only then looked at the data and found the answer sitting there. The lesson is not really about time series. It is that we had spent all our care on the model and none on the data, and the data was exactly where the trap was hiding.

We are not done with the market, though, because there is a step of the recipe we have not questioned yet: scaling. Features arrive on all sorts of sizes, so the book says standardise them, subtract the average, divide by the spread, and fit those numbers on the training data only, never the test. We have done exactly that all along, and it is the careful thing to do.

So let us keep being careful, and change one tiny, innocent thing. Our feature was the size of each day's move, in percent. What if we measure it in points instead, the raw change in the index, which is just what you get by subtracting two prices and forgetting to divide? Same days, same information, a different unit. Everything else stays fixed, the honest split included.

| feature         | test accuracy |
| --------------- | ------------- |
| size in percent | 0.585 |
| size in points  | 0.512 |

Read the second row twice, because it should not be possible. **Nothing leaked, the split is honest, the scaler is frozen exactly as taught, and yet the same model that scored 0.585 now scores 0.512, a coin. How can changing a feature's unit kill a working model?**

By now we know where to look: not at the model, at the data. The two units behave completely differently over the years. A one-percent day is a one-percent day whether the index sits at 1 500 or at 7 000. In points it is not, because the index climbs from about 1 455 to 7 483 across the data, so the same one-percent day is worth about 15 points early on and 75 points near the end. The percent feature holds still; the points feature drifts upward (this is drift, or covariate shift: Shimodaira, 2000; Gama et al., 2014).

![Why a drifting feature breaks the frozen scaler](figures/drift.svg)

Now the frozen scaler is the trap. It learned what "normal" looks like from the early, low years, then applied that to the late, high years. For percent, fine. For points, a disaster: the test days sit far outside anything the model met in training, and it has no idea what to make of them. The number is not lying; it is truthfully reporting a model the recipe broke. The fix is to stop freezing one average for all time and scale each day by its own recent past, a window that slides forward and only ever looks backward, which lifts 0.512 back to 0.549. Better, though the real fix was never to build a drifting feature at all.

So the market cost us twice. The split leaked the future and pushed the number too high; the scaler froze a moving world and pushed it too low. Two opposite failures, one root: we kept trusting the recipe without ever looking hard at the data underneath it.

## 4. Phone: reading the person, not the task

The third problem has nothing to do with time. Thirty people wore a phone while they walked, sat, stood, and so on, and each short moment of motion becomes a row of 561 numbers, with a label for what they were doing. The first three rows:

| row | person | activity | f1   | f2    | f3    |
| --- | ------ | -------- | ---- | ----- | ----- |
| 1   | 1      | standing | 0.29 | -0.02 | -0.13 |
| 2   | 1      | standing | 0.28 | -0.02 | -0.12 |
| 3   | 1      | standing | 0.28 | -0.02 | -0.11 |

Before anything else, we tried to train the network, and it embarrassed us. We reused the settings that had worked on the market, hit go, and the accuracies came back as pure noise, bouncing between 0.0 and 0.35, worse than guessing. For a moment we thought the task was hopeless. It was not: the learning rate was simply too large for 561 features, and the training was blowing up instead of settling down. We turned it down, the model trained cleanly, and it reached 0.98 on its own training data. A small, ordinary mistake, but it left a rule worth keeping: before you trust any number a model prints, check that the model is actually learning at all. A contest between two splits means nothing if neither model has learned a thing.

With that sorted, back to the recipe. Notice, again, before we cut anything: all three rows above are the same person, standing, their numbers barely moving. That is how the data comes, in long runs of one person doing one thing, hundreds of rows at a stretch. Thirty people, but thousands of rows.

By the book: shuffle all the rows, take a random fifth as test. It reads 0.973. Almost perfect, the best number in the whole report.

And that is the moment to be suspicious, not pleased. **0.973 at reading human activity, from a phone in a pocket? Is the model really that good at telling walking from sitting, or is it just recognising these particular thirty people?** Because here is what we should have asked sooner: how would this model ever be used? On someone new, a person it has never met. And shuffle does not test that at all, because once everyone's rows are scattered, almost every one of the thirty people sits on both sides of the cut.

So we ask the honest question instead. We hold out whole people: pick six of the thirty, put every row they ever produced into the test set, and let the model meet them for the very first time at test.

| how we split    | test accuracy |
| --------------- | ------------- |
| shuffle rows    | 0.973 |
| hold out people | 0.946 |

Down it comes again. Part of that shiny 0.973 was never reading activities at all; it was the model recognising these particular thirty people. Shuffle let it study a person in training, then meet the same person again at test, and it happily used the second look. Ask it about a stranger, which is the only thing we actually wanted, and it honestly does worse. A third dataset, a third structure, and the same shuffle leaks all over again, this time not through time but through people.

## 5. The lab: the trap that needs no data

Three datasets, three traps, and every one of them lived in the data: its order in time, its people, its rare cases. So here is a comforting thought: if the traps are in the data, then clean, honest data should be safe. But there is one more trap, and it is the deepest of the lot, because it needs no bad data whatsoever. It is in us.

Look back at what we actually did in every chapter. We never built just one model. We tried a setting, trained, kept the best, tried more. That is the busiest step of the recipe, and the most natural thing in the world: the harder we search, the better the model we walk away with. Or so it feels.

**But here is the question we never stopped to ask: when the best score climbs as we try more, is the search finding a better model, or are we just keeping the luckiest of many noisy numbers?** From the outside, luck looks exactly like getting better. On real data we cannot tell the two apart, because we never know the true skill to hold the number against.

So let us build a case where we do know. We make a dataset out of pure noise: random numbers for features, and a label that is a coin flip, decided without ever looking at the features. There is nothing in it to learn, and because we built it that way, we know the exact truth: no model can beat 0.5. Any score above 0.5 is luck, and nothing else.

Now we search this noise the way we always search. Draw a setting at random, train it, score it on a small validation set, and keep the best. Then do it again with more settings, and more again, and watch what the best score does.

![The winner's curse](figures/winners_curse.svg)

One setting scores near 0.5, but never exactly, it wobbles a little by luck. Try ten and keep the best, and the best of ten sits a bit above 0.5. Try a hundred and keep the best of those, and it sits higher still. Nothing was learned, nothing improved; we just reached further into the lucky tail each time. Put numbers on it, opening a large sealed test on each winner to read its real value:

| settings tried | best score we keep | the truth |
| -------------- | ------------------ | --------- |
| 1   | 0.510 | 0.500 |
| 10  | 0.553 | 0.500 |
| 100 | 0.577 | 0.500 |

The score we keep climbs from 0.51 to 0.58 as we try more. The truth never moves off 0.50, not once. Every point above 0.5 was luck, picked out by keeping the best of many tries. (Exactly how fast that best drifts is plain first-year probability, worked out in Appendix A.)

And this is not a quirk of noise. It is what searching does to any score. On real data the truth is not 0.5, but the same thing happens: the harder we search, the further our reported number floats above the model's real skill. That gap is not bad luck. It is a measure of how hard we looked.

*The machine snoops by itself.*

Searching by hand, one setting at a time, is bad enough. But a deep network does the same thing automatically, and hides most of it from us.

Take early stopping, the normal, sensible habit: we train for many epochs and keep the one where the validation score looked best. That is one peek at the validation set per epoch, and we keep the luckiest. Or take the random seed: the very same setting, started differently, gives a different score, so we run a few seeds and keep the best of those too. None of this feels like searching. All of it is.

So count honestly. We say we tried five settings. But inside each we kept the best of a few seeds, and inside each of those the best of many epochs. Five settings quietly become a hundred and fifty tries, and the number we report is the best of a hundred and fifty, not of five. The tool we reached for at the very start, so flexible, so powerful, turns out to be a machine for driving that one validation score as high as it will go. It is a machine for doing exactly what we just watched inflate a number out of pure noise. There is an old name for it: data snooping. A deep network snoops for a living, and never tells us how many times it looked.

There is no clever fix, only discipline. Search less, and search in the open, not with a hundred hidden seeds and epochs. And keep one sealed test that you open exactly once, on the single model you already chose, never as a dial you turn until you like the number, because a test you select on lies just like anything else you select on.

And that is the last trap, and the deepest, because it was never in the data at all. It was in us, in our own eagerness to keep the best. So we have now watched the number lie in every way it can: through what we chose to measure, through the hidden structure of the data, and through our own searching. It is time to stand back and ask what, if anything, is left.

## 6. So what is left?

Let us count up the damage. Four problems, all handled by the book, and the book betrayed us on every one. On the loan clients the number measured the wrong thing entirely, missing most of the defaulters while it still looked fine. On the market it came out too high, an edge that was really the future leaking backward, and then too low, a working model killed by a drifting feature. On the phone it read the people instead of the task. And on pure noise, with nothing to learn at all, our own searching pushed the number up out of thin air. Every safeguard we trusted failed somewhere, and not one of them said so out loud. We were not careless. We did everything by the book, and the number lied anyway.

**So which number, exactly, are we still allowed to believe?** The score we set out to chase and report and trust cannot carry that trust, because a hidden assumption can quietly rot it and our own searching can quietly inflate it, and from the outside a rotten number looks exactly like a good one. We have run clean out of numbers to trust. If it does not live in the number, then where does trust live?

## 7. What we can trust

Here is the answer, and it was hiding in plain sight the whole time. Look back at the honest fix at each step. Not one of them was a secret technique. The careful practitioner who got fooled and the honest one who did not use the very same five steps: both shuffle, both scale, both search, both measure. The only difference between them, the whole difference, is understanding.

![The same five steps, and the hidden assumption in each](figures/pipeline_traps.svg)

One person reached for shuffle because the book said shuffle. The other looked at the data first, saw that its rows came in time order, and chose a cut that respected time. Same step, opposite result, because one of them understood what the data was before deciding what to do to it. That is the lesson of every problem we walked. The trap was never in the step. It was in doing the step without asking what it quietly assumed, the four tags above, and whether the data in front of us could bear it.

So we do not trust the number, and we do not even trust the recipe on its own, because a recipe followed blindly is exactly what broke on us four times over. What we trust is understanding, made solid and repeatable in the shape of a procedure. State each assumption and check it against the data. Match the method to the structure in front of you. Search less, and look once. Do that, and the number at the end is one you have earned the right to believe, not because it is high, but because you can trace, step by honest step, why the way you made it could not have lied.

That is the whole of it. The recipe is a fine place to start and a dangerous place to stop. We cannot trust the goal. We can trust the understanding that earns it.

---

## References

*Sources for the four working steps we walked, §2 (split) through §5 (metric). The formatting is to be fixed to the handbook style at the end.*

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

**The metric, and imbalanced classes.**

- Provost, F., Fawcett, T. and Kohavi, R. (1998). The case against accuracy estimation for comparing induction algorithms. *ICML*. (Why accuracy is the wrong yardstick when classes or error costs are uneven.)
- He, H. and Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9). (Rare-class classification, and what goes wrong when the positive class is small.)
- Brodersen, K. H., Ong, C. S., Stephan, K. E. and Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. *ICPR*. (Balanced accuracy, the honest metric reported in §5.)

**The data, and the effect the market task leans on.**

- Yeh, I.-C. and Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2). (The UCI loan-default set.)
- Anguita, D. et al. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN*. (The UCI HAR set.)
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4). (Volatility clustering.)

---

## Appendix A: Where the two formulas come from

*Outside the page limit; here so the two formulas the lab leans on can be checked, not taken on trust. Nothing below goes past first-year probability, we derive it in place.*

The lab in §4 used two formulas: the spread of a chance score, σ = 0.5/√n, and the best of N scores, about 0.5 + σ√(2 ln N). Here is where each one comes from.

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
- **It scales with σ = 0.5/√n.** A smaller validation set means a larger σ means a larger gap. The small validation set is the fuel, exactly as §4 said.

Two honest limits, both pushing the real gap a little below the formula:

- **Independence.** We assumed the N draws were independent. Configurations that differ only by a seed, or by one more epoch, are near-copies, not fresh draws, so N of them count as fewer than N independent tries, and the real best sits a touch below σ√(2 ln N). (This is the hidden-search point of §1: seeds and epochs are extra draws, but correlated ones.)
- **The Gaussian approximation.** The Binomial is only approximately a bell curve, and √(2 ln N) is the leading term of a longer expression. Both are close for the n and N we use, not exact.

So we do not lean on the formula as truth. We use it to see the mechanism, and we measure the real gap in the lab (§4). The two agree in shape, and that agreement is the point: the gap is not a quirk of one dataset, it is the biggest of many noisy draws, behaving as the biggest of many noisy draws must.

---

## Appendix B: the code that ran the lab

*Outside the page limit; the measurement half of the grounding. Appendix A shows why the formulas hold; this shows the code that measured the real gap, and that running it reproduces the numbers in §4.*

The whole lab is one short file, `code/lab_demo.py`, that runs on numpy alone, `python code/lab_demo.py`, with `seed = 0`, so every number below reproduces exactly. Here it is in the five pieces of §4. The helper functions (`init`, `forward`, `softmax`, `accuracy`, `sample_config`) are in the file; `forward` is just `h = ReLU(X·W1 + b1)` then `z = h·W2 + b2`, and `sample_config` draws a width from {4, 8, 16, 32, 64} and a learning rate around 0.01–0.3.

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

**Piece 4: the sweep.** Run the gap machine over a grid of N, averaging over fifteen repeats. This is the table in §4.

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

*Outside the page limit. The report leans on these three datasets, so here they are as they actually arrive: where each came from, what the raw rows hold, what makes its structure the structure we claim it is, and how each becomes the task we run. Every number below is read straight off `python code/data_peek.py`.*

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

*Outside the page limit. The working steps change one thing at a time on real data; this is the code that did it and the output it printed. Five short files in `code/`, numpy only, all reusing the same MLP from `code/lab_demo.py`: `code/loan_split.py`, `code/finance_split.py`, `code/har_split.py` for the split (§2), `code/scaling_split.py` for the scaling (§3), and `code/metric_loan.py` for the metric (§5). Each prints the numbers quoted in the body.*

**The shared shape.** Every experiment is the same three thoughts: load a real dataset, build two splits, run the identical MLP under each. Only the split differs. Features are standardised with the mean and standard deviation of the training side only, so the standardising itself never leaks (that is §3's problem, deliberately kept out of this one).

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

This is the one place the network is not literally the §4 model: HAR has six activities, so the output layer has six units instead of two. Everything else, one hidden layer, ReLU, softmax with cross-entropy, full-batch gradient descent, is unchanged.

One warning worth recording, because it nearly fooled us. With 561 features a learning rate of 0.5 (the value the market task uses) makes the training diverge: the accuracies come back as noise between 0.0 and 0.35, and the split comparison becomes meaningless. At 0.1 the model trains cleanly (0.98 on its own training set). We only found this by checking that the model learns at all before trusting any comparison between splits. A split experiment on a model that is not training is measuring nothing.

**Piece 4: scaling, and a world that moves (§3).** The file is `code/scaling_split.py`. Here the task, the label and the split never move; only the feature's unit and the scaling do. The label is always built from the percent size, so it stays stationary and exactly balanced whichever feature the model is fed, which is what makes the three rows comparable.

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

**Piece 5: the metric (§5).** The file is `code/metric_loan.py`. Nothing about the split or the scaling changes here: the loan data is split honestly at random and standardised on the training side, exactly as in Piece 1, so nothing leaks and nothing drifts. The only new thing is that we stop reading the single accuracy number and count what the model actually did with the rare class, the defaulters.

```python
def predict(p, X):
    return forward(X, p)[2].argmax(1)

def run(seed, width=16, lr=0.3, epochs=300):
    ...                                          # honest random split; standardise on TRAIN only
    yhat, ytrue = predict(p, Xte), y[te]
    TP = ((yhat == 1) & (ytrue == 1)).sum()      # default (1) = the positive class
    FP = ((yhat == 1) & (ytrue == 0)).sum()
    TN = ((yhat == 0) & (ytrue == 0)).sum()
    FN = ((yhat == 0) & (ytrue == 1)).sum()
    recall = TP / (TP + FN)                       # of real defaulters, the fraction caught
    spec   = TN / (TN + FP)                       # of non-defaulters, the fraction cleared
    bal    = 0.5 * (recall + spec)                # balanced accuracy: both classes weighed equally
    return acc, recall, spec, prec, bal, (TN, FP, FN, TP)
```

```
n 30000 | features 23 | default rate 0.221
baseline "always predict NO default": accuracy 0.779, defaulters caught 0 of 6636

same MLP, honest random split, mean over 10 seeds (no leakage, no drift)
  accuracy            0.813   (baseline 0.779, so only +0.034)
  recall (defaulters) 0.307   <- of the real defaulters, the fraction actually caught
  precision           0.688
  balanced accuracy   0.633   <- treats both classes equally
one split's confusion (seed 0): TN 4448  FP 199  FN 915  TP 438
  of 1353 real defaulters in this test set, the model caught 438 and missed 915
```

The 0.813 is measured perfectly, and it is still the wrong number. Balanced accuracy, which scores each class on its own and averages the two, drops the model to 0.633, and the confusion row shows the reason in plain counts: 915 of the 1 353 defaulters walked through unflagged. Accuracy, weighing every client the same, simply could not see them.

**Reproduce.** `python code/loan_split.py`, `python code/finance_split.py`, `python code/har_split.py`, `python code/scaling_split.py`, `python code/metric_loan.py`. The loan and market data are frozen in `data/`. HAR is downloaded once from the UCI archive (id 240) and cached to `data/har.npz`.

---

