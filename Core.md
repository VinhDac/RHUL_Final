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

## 1. The tool we reach for

So we have a task: take the next case and put a label on it. What do we build?

These days the answer is almost automatic. We reach for a deep network. Why that one, out of everything? Because it bends to almost anything. Give it enough data and enough room and it will find some shape that fits the problem, whatever the problem happens to be. It is the most flexible tool we have, and flexibility is exactly what we want when we do not yet know what the answer looks like.

And where does all that flexibility come from? From the knobs. A deep network is a wall of things we get to set. How many layers? How wide is each one? Which activation? How fast should it learn, the learning rate? How long do we train it, how many epochs? How hard do we hold it back from overfitting, the regularisation, L1, L2, and more. Turn any one of these and we get a different model.

That is the power, and honestly it feels wonderful. Nothing is fixed. If the model is not good enough yet, there is always another knob to turn, another setting to try. We are never truly stuck.

But stay with that picture for a moment. A wall of knobs, and every combination of them is a different model. How many combinations is that? Far more than we could ever try. So how do we ever land on the right setting?

We do the only thing we can. We try some. We turn a few knobs, train, look at the number, turn a few more, train again. And here is the quiet part, the part we will come back to: just to find a good model, we are already trying many of them.

Hold on to that. It is going to matter more than it looks.

---

## 2. Chasing a better number

### 2.1 The habit

We have our tool and its wall of knobs. Now we want the model to actually be good. So what do we do?

We try things. We pick a setting: this many layers, this width, this learning rate. Call one such choice a configuration. We train it, we score it on data the model never saw, and we write the number down. Not good enough? We change something and try again, another configuration, another number. At the end we keep the configuration with the best number, and we let the rest go.

There is nothing strange about this. It is simply how the work is done. Nobody trains a single model, shrugs, and ships it. We try ten, or a hundred, and we keep the best one. That is not laziness or cheating, it is being thorough. The harder we look, the better the model we walk away with. Who would fault us for looking harder?

So we look harder. We keep going until the best number looks good, and then we write that number down as the result. The model that earned it is our model. The number it scored is the thing we report, and the thing we believe.

And why would we doubt it? We were careful. We held data out, we tried honestly, we kept the winner, and the number came out high. Everything was done exactly the way it is supposed to be done.

So we trust it.

### 2.2 Trying harder, or getting luckier?

But let us pull on one thread before we walk away happy.

We trusted the number because we were thorough: we tried many models and kept the best one. And the harder we looked, the higher the best number climbed. That felt like progress. Each time we tried a little more, the winner scored a little better, so surely trying more was making the model better.

Or was it?

Here is the uncomfortable question. When the best number goes up after we search harder, two very different things could be going on. Maybe the search really is turning up better models. Or maybe we are just keeping the luckiest of a lot of noisy scores, and from the outside luck looks exactly like progress. Either way we see the same thing: a higher number.

On real data we cannot tell the two apart. We do not know the model's true skill, so we have nothing to hold the number up against. The number is all we have, and the number is the very thing we are unsure of.

So let us settle it in the one place where we can. Let us build a world where we already know the truth. If we know for a fact that no model can be any good, and the best number still climbs the harder we search, then we have caught it in the open: that climb was never skill. It was luck, wearing the face of progress.

So how do we build a world with no truth to find? We make the data pure noise. The features are just random numbers. The label is a coin flip, 0 or 1, tossed without ever looking at the features. Nothing in the inputs points to the answer, because we made sure there is no answer to point to. There is, quite simply, nothing to learn.

And that is the whole point. Because we built it, we know the truth exactly: no model, however clever, can do better than a coin. Every model's real accuracy is 0.5, and no amount of trying can move it. So 0.5 is our fixed mark, painted on the wall. Whatever a number claims above 0.5 is, by construction, not skill.

Now we cut the data into three parts. A training part, to fit each model on. A small validation set, 200 points, the set we look at to pick our winner. And a large sealed test, 10 000 points, locked away and opened only right at the end. (The labels are already random, so there is nothing to gain by shuffling; we just cut by position.)

Two of those sizes were chosen on purpose, and they will matter. The validation set is small, so its score is jumpy, and that jumpiness turns out to be the whole engine of what follows. The test set is large, so its score sits almost still, close to the truth, which is exactly what lets it stand in for the 0.5 we already know.

One last thing to name. A configuration here is one setting of the little network: a width and a learning rate, picked at random. To try one more is to draw another such pair, train it, and read its score on those 200 validation points. But on random labels the network learns nothing real, so no configuration is truly better than any other. Its validation score is not skill at all. It is just the fraction of 200 coin-guesses it happened to get right.

That last line is the key to everything, so let us sit with it. Our validation score is nothing but the fraction of n coin-guesses that come up right. So the real question is simple: how far from 0.5 does that fraction tend to land, and what makes it land closer or further?

Think of actual coins. Flip four of them and ask what fraction are heads. It is easy to get three out of four, a wild 0.75, miles from a half. Now flip a thousand. Landing anywhere near 750 heads would be a miracle; you will come out very close to 0.5. So the answer is already in view: with few flips the fraction jumps around, with many it hugs the middle. More points, tighter wobble.

How much tighter, exactly? For a fair coin, the spread of that fraction over n flips works out to

> σ = 0.5 / √n.

(That is a standard fact about coin flips, first-year probability, and we derive it from scratch in Appendix A, so nothing here rests on trust.) The shape is the thing to notice: the wobble shrinks like √n, so to halve it we need four times the data.

Now put in our number. The validation set has n = 200 points, so σ = 0.5 / √200, which is about 0.035. In plain words: a single configuration, with no skill whatsoever, does not sit politely at 0.5. It lands somewhere around 0.5 give or take 0.035, purely by luck, before we have done a single thing wrong.

So that is one try. But we do not keep one try, do we? We keep the best of many. So let us do exactly what we always do, the thing that felt so thorough back in §2.1: draw N configurations, and keep the one with the best validation score.

Watch what that does. Each configuration is a wobbly draw around 0.5, give or take σ. We are no longer asking where a single draw lands. We are asking where the highest of N draws lands, and the highest of many sits above the average, further above the more we draw, because with more tries we reach further into the lucky tail. Worked out (again, first-year probability, Appendix A), the best of N comes to about

> 0.5 + σ · √(2 ln N).

That is the whole engine, in one line: keep the best of more tries, and the best runs further above 0.5. Nothing learned, nothing improved, just the top of a taller pile of luck.

So let us actually run it. We build the no-signal data, we sweep N from 1 upward, and for each N we keep the best model on validation and then, only then, open the sealed test once on that winner (n = 200, averaged over a few repeats so the numbers are steady):

| N   | apparent (best val) | true (sealed test) | gap    |
| --- | ------------------- | ------------------ | ------ |
| 1   | 0.510               | 0.500              | +0.010 |
| 2   | 0.521               | 0.500              | +0.021 |
| 5   | 0.548               | 0.500              | +0.048 |
| 10  | 0.553               | 0.500              | +0.053 |
| 20  | 0.558               | 0.499              | +0.059 |
| 50  | 0.568               | 0.500              | +0.068 |
| 100 | 0.577               | 0.500              | +0.077 |

Read the two middle columns side by side, because this is the thing we came here to see. The number we would report, the best validation score, climbs and climbs, from 0.51 up to 0.58, as we try more. It looks exactly like progress. And the truth, the sealed test, never moves. Every single time, it drops the winner straight back onto 0.50, chance, where we know it belongs.

The gap between them grows from +0.01 to +0.08, widening with N just as √(ln N) said it would. (It runs a touch under the bare formula because our configurations share one training set, so their scores are not quite independent draws; Appendix A is honest about this, and Appendix B has the code.)

So there it is, in the open. Every point above 0.5 was luck, and nothing more. We did not build a better model by trying harder. We just kept reaching deeper into the lucky tail and reporting what we found there as if we had earned it.

Sit with what that gap actually is. It is not noise or sloppiness. The gap measures exactly one thing: how hard we searched. Try once and it is tiny; try a hundred times and it is huge. The number did not fail us quietly, by drifting off. It failed us in the most convincing way possible, telling us we were getting better at the precise moments we were only getting luckier.

And the little formula even tells us where the lie is fed from. The gap rides on σ, and σ = 0.5/√n. Shrink the validation set and σ grows, and the gap grows with it. So that small validation set from earlier was never an innocent detail. It is the fuel. The less data we judge on, the more room luck has to move.

So the validation score is out; we cannot trust the very thing we selected on. But we held back a second set, remember, the large sealed test, the one we swore to open only once. Surely that one is still honest? It is, for a single reason: the model never touched it while we were choosing. That, and nothing else, is the source of its honesty.

So here is the trap that springs next. We open the test, and suppose the score is bad. What do we naturally do? We go back, change the model, and try again until the test looks better. It feels responsible. But look at what we just did: we used the test to steer us. It is not untouched anymore.

We do not even need fresh data to watch this happen, because we still know the truth is 0.5. So this time, instead of looking at the test once, let us reuse it: keep the configuration that scores best on the test, which is exactly what "retrain until the test looks good" comes down to. Put it right next to the validation search:

| N   | select on validation | reuse the test | truth |
| --- | -------------------- | -------------- | ----- |
| 1   | 0.510                | 0.500          | 0.500 |
| 10  | 0.553                | 0.508          | 0.500 |
| 100 | 0.577                | 0.512          | 0.500 |

And there goes the last safe place. Selected on, the test inflates too: 0.500 creeps up to 0.512. It creeps more slowly than the validation only because it is bigger, its wobble σ = 0.5/√10 000 is about seven times smaller, so it holds out longer. But it holds out; it does not hold. The test was never honest because it was "the test." It was honest because we looked at it once, and because it was large enough to sit still.

So the real lesson is quieter, and sharper, than "trust the test instead." It is this: the poison is the selecting, not the set. Any set we choose against will lie to us, and the only guard against it is to look once, on a set big enough to be quiet, and then stop.

So this is the first crack in the number, and it is worth naming plainly. It was not fraud, and it was not incompetence. It was just searching, and then reporting the luckiest thing the search turned up. The cure looks simple too, now that we have seen the disease: stop fishing. Do not chase the number through a hundred tries. Look once, on a set large enough to be quiet, and live with what it tells us.

And here is the comforting thought we reach for next. Everything that just went wrong, we brought on ourselves by being greedy, and we did it on a pile of pure noise, data with nothing in it to find. Real problems are not like that. Real data has real signal. So if we take a real dataset, follow every rule in the book, keep our discipline, and look only once, then surely the number it gives us is honest.

That is a very reasonable thing to believe. So let us go and check it.

### 2.3 By the book, on real data

Good. We have our resolution: no fishing, follow the rules, look once. So let us find out what the rules actually are, and follow them to the letter.

The book is not shy about them. There is a standard recipe, taught in every course and printed in every textbook, and it runs like this. Take the data and shuffle it well. Cut it into three parts: a training set to learn on, a validation set to tune on, and a test set to judge on. Better still, do this many times over and average, k-fold cross-validation, so that no single unlucky cut can fool us. Standardise the features, but fit the scaler on the training data only, never peeking at the test. And at the very end, report accuracy, the fraction we got right. (Stone, 1974; Kohavi, 1995.)

Read that again and notice how careful it is. Every piece of it exists to stop us fooling ourselves. We shuffle so each part looks like the whole. We cross-validate so we never lean on one lucky cut. We fit the scaler on the training set alone so the test stays untouched, which is exactly the discipline we just paid to learn. This is not a lazy recipe. It is decades of hard-won caution, boiled down to a handful of rules, and if we follow it, we are doing everything right.

So let us do exactly that. We take a real dataset, one with genuine signal in it. We build the same little network from §2.2, follow the recipe with full discipline, hold every knob fixed, and open the test exactly once. And to be as careful as we can, we change only one ingredient of the recipe at a time and pin all the rest down, so that if the number ever moves, we know precisely what moved it.

*Three real problems.*

One dataset would not convince anyone. If we tried the method on a single problem and it held, we could always be accused of getting lucky with our example. So let us take three, as different from one another as we can find, and put the exact same by-the-book method through all of them. Here they are.

| problem | what one row is | source and size |
| ------- | --------------- | --------------- |
| **Loan default** | one credit-card client: 23 features, and whether they defaulted | UCI credit-card default (Yeh and Lien, 2009); 30 000 rows, 22% default |
| **The market** | one trading day of the S&P 500 | ^GSPC daily closes, frozen at 2026-07-03; 6 658 days |
| **Activity recognition** | one window of phone motion: 561 features, and which activity it was | UCI HAR (Anguita et al., 2013); 10 299 rows, 6 activities |

Three ordinary, respectable datasets: some bank clients, some years of a stock index, some people moving with a phone in their pocket. Each one has real signal in it, a real pattern worth learning. Appendix C opens each up close, the raw rows exactly as they arrive and how each becomes the task we run; Appendix D holds the code for every experiment below.

So we have our method and our three problems. Let us run it, changing one ingredient at a time, and start with the very first choice the recipe asks of us: how to split the data.

**(A) The split.**

The first thing the recipe asks is that we shuffle the data and cut it into parts. Why shuffle first? So that each part looks like the whole, with no accidental clump of one kind of row landing all in the test. It is the obvious, sensible thing, and the textbook hands it to us as the default: a random split, or k-fold cross-validation (Stone, 1974; Kohavi, 1995). Nobody argues with it. Neither will we.

Still, we are being careful now, so let us not simply assume the split is harmless. Let us check. We take the loan data and, instead of cutting it once, we cut it eleven different ways and hold everything else dead still. Ten of them are ordinary random 80/20 cuts, one per seed. The eleventh is a different kind of cut altogether: a stratified draw that forces the 22% default rate to appear on both sides. Every one of the eleven rebuilds the same network from scratch (one hidden layer of 16 units, learning rate 0.3, 300 full-batch steps), and standardises using the training side of that particular cut only, never the test side.

The logic is simple. If the split really is just a formality, then eleven very different ways of cutting the same data should all land on the same number. If the split secretly mattered, at least one of them would break ranks.

| split              | test accuracy |
| ------------------ | ------------- |
| ten random splits  | 0.813 ± 0.003 |
| stratified split   | 0.814         |

Not one breaks ranks. The ten random cuts run from 0.807 to 0.816, a spread of 0.009, and the stratified cut lands right in the middle at 0.814. That little spread is nothing more than the sampling wobble of a 6 000-row test set; it is not a difference between methods. The split is a formality, exactly as the book promised. On the loan data, shuffle is simply correct.

So there it is. We were disciplined, we followed the recipe, we even stress-tested the one choice we had to make, and the method came through clean. One problem down, and the book has passed. Let us try the next one.

*The market.*

On to the second problem, and we set it up the same careful way. What can the market actually tell us? Not direction: whether tomorrow closes up or down is, on this series, very close to a coin flip (we will lean on that fact shortly). But there is something it does carry. We predict whether tomorrow is a busy day, its move bigger than the median day's, from the sizes of the last five days' moves. Busy days come in clusters, a stormy week tends to stay stormy (volatility clustering, Engle, 1982), so there is a real, repeating pattern here for a model to find. This is not a noise dataset like our first lab. There is genuine signal.

So we follow the recipe, exactly as before. We shuffle the 6 658 days, cut 80/20, standardise on the training side only, build the same little network (16 hidden units, learning rate 0.5, 300 steps), and open the test once. It reads 0.615. Comfortably above the 0.5 of a coin, a real edge. The method has delivered again.

But we are disciplined now, so we do the loan thing: we do not trust a single number, we check it another way. And there is an obvious second way to cut this particular data, because we know how the model would really be used. In the real world we would train it on the days we already have and ask it about days that have not happened yet, the future. So let us measure it exactly like that: train on the first 80% of the days in date order, test on the last 20%, the newest ones. Same network, same starting weights, same everything, only the cut changed. We run the pair ten times, seed against seed, so nothing rests on one lucky draw.

| split                                   | test accuracy |
| --------------------------------------- | ------------- |
| shuffle (random)                        | 0.615         |
| chronological (train past, test future) | 0.585         |

And they do not agree.

Not by a rounding error, either. The shuffle reads higher every single one of the ten times, by +0.030 on average. Stop and feel how strange that is. With the loan data, eleven different cuts of the same rows all landed on the same number, to within a whisper. Here, two cuts of the same 6 658 days, the same network, the same discipline, come out three whole points apart, and in the same direction every time. Nothing changed but where we drew the line. Something about this data is different, and the recipe never said a word about it.

So what is different about this data? Go back to what a row actually is. In the loan file, each row was its own client, a separate person, with no reason for one to resemble the next. Here a row is a day, and days do not arrive in a jumble. They come in order, and, this is the whole thing, today tends to look like yesterday. A calm day sits among calm days, a wild one among wild ones (that clustering again). On top of that, our features are the last five days' moves, so two neighbouring days literally share four of their five numbers. Consecutive days are near-twins.

Now the disagreement explains itself. When we shuffle, we scatter the days at random, so tomorrow's near-twin can easily land in the training set while tomorrow itself sits in the test set. The model is then asked about a day whose almost-double it has already studied. It scores well, but not by seeing the future; it scores well by recognising a neighbour it more or less memorised. That is the extra 0.030. It is not skill, it is the future quietly leaking backwards across the cut (leakage: Kaufman, Rosset and Perlich, 2012).

The chronological cut cannot do this. Training is strictly the past and the test is strictly the newer days, so no future twin is ever in the room while the model learns. It is the honest measurement, and the honest measurement is the lower one. The shuffle was never measuring how well the model reads the future. It was measuring how well the model remembers its neighbours.

There is one more honest question hiding here: which honest number? Testing only on the final stretch gives 0.585. But we can do better than a single window: roll the cut forward through the series, training on an expanding past and testing on the next block, again and again, then average. This is walk-forward, or rolling-origin, the standard honest way to score a time series (Tashman, 2000; Bergmeir and Benitez, 2012), and it reads 0.603. The honest number wobbles a little between the two, because the market itself drifts, a later year is genuinely not the same market as an earlier one. But every honest reading sits below the shuffled 0.615. So we are careful not to blame the whole gap on leakage alone: the shuffle flatters us in two ways at once, it lets near-twin days leak across the cut, and it quietly pretends the future market is the same as the past one. Walk-forward is what proves the shortfall is real and steady, not one unlucky window.

Before we move on, let us make sure we have not fooled ourselves in the other direction, because we are being careful now. Here is a way to double-check the whole story. The leak only worked because there was a real pattern for a near-twin to give away: volatility clusters, so a memorised neighbour genuinely helps. But leakage cannot invent an edge that is not there; it can only inflate one that is. So if we point the very same machinery at a target with no edge at all, the gap should simply vanish.

We have one to hand. Remember that whether the market closes up or down tomorrow is, on this series, a coin flip, with no real pattern to find. So we swap the busy-day target for the up-or-down target and run the identical pair of splits. And the gap disappears: shuffle and chronological now land within a whisker of each other, both hovering near the 0.537 you would get by always guessing up. Nothing to learn, so nothing to leak.

That is the clincher. The busy-day task, which has real signal, shows a gap; the direction task, which has none, shows no gap at all. If our chronological-versus-shuffle setup were simply wired wrong, it would have manufactured a gap on both. It did not. The gap is a real effect, and it turns up exactly where a real pattern exists for the shuffle to leak.

*The activity data.*

Maybe that was just time series being awkward. Time is a special thing, after all; perhaps ordinary data is safe. So let us try the third problem, which has no time in it at all. Thirty people wore a phone and did six everyday things, walking, sitting, standing, and the rest, and the task is to read the activity off a short window of the phone's motion. The activities really are distinguishable, so once again there is genuine signal.

By the book, then. We shuffle all 10 299 windows, take a random fifth as the test, standardise on the training side, build the network (this time with six outputs, one per activity, 64 hidden units, learning rate 0.1, 400 steps), and look once. It reads 0.973. Almost every window classified correctly. This is the best number we have seen yet; the method looks not just sound but excellent.

But we have been burned once, so we do not celebrate. We ask the market's question again: how would this model actually be used? We would put it on some new person and ask what they are doing. And that word, person, is the thread to pull. What is a single row here? It is not an independent example; it is one moment of one particular person, and each of the thirty people supplies hundreds of these moments, all of them near-copies, the same gait, the same way of holding still. Shuffle the windows and almost every person ends up on both sides of the cut. So at test time the model is shown a fresh window of a person it already studied. Is it reading the activity, or is it just recognising the person?

There is a clean way to ask. Instead of shuffling windows, we hold out whole people: draw six of the thirty at random, hand every window they ever produced to the test set, and let the model meet those six for the very first time at test. That is the honest question, can it read a stranger, and it is a known, named method (group the rows by subject and leave whole subjects out, GroupKFold or leave-one-subject-out; Saeb et al., 2017). Everything else stays pinned: same network, same standardisation, ten paired seeds.

| split                          | test accuracy |
| ------------------------------ | ------------- |
| shuffle rows (record-wise)     | 0.973         |
| hold out people (subject-wise) | 0.946         |

And down it comes again, 0.973 to 0.946, lower in nine of the ten runs. Part of that shiny 0.973 was never activity-reading at all; it was the model recognising these particular thirty people. And look at the second thing the honest split reveals. The shuffle readings barely twitch off 0.973, but the subject-wise readings scatter widely, from 0.881 all the way to 0.973, depending on which six strangers we happened to draw. That spread is real information: some people are simply harder to read than others, and the by-the-book split had painted over it completely. The honest question is harder, and its answer is both lower and less sure.

So it was not a quirk of time. Here is data with no time in it whatsoever, and the very same recipe lies again, in the very same direction, for a completely different reason. There the rows were days that resembled their neighbours; here they are moments that belong to a person. Two different structures, one identical failure.

*What the three say together.*

Step back and look at the three at once, because one model and one changed ingredient have told us something general. On the loan data the split did not matter at all. On the market and on the phone data it moved the number by about three points, and always the same way round: the shuffled cut read higher than the honest one.

So the culprit was never shuffling itself. Shuffling was exactly right for the loan data and quietly wrong for the other two, and the recipe has no way to tell those cases apart, because the difference does not live in the recipe. It lives in the data. Loan's rows are separate clients; the market's rows are days in a row; the phone's rows belong to people. Whether rows are interchangeable is a fact about the problem, and the recipe simply assumes it, for free, every time, without ever asking.

And notice that the loan data earns its keep here. It held. If we had only ever shown the two problems that broke, you could fairly say we went looking for trouble and picked our examples. But one of the three stood perfectly still, and it stood still for a reason: its rows really are independent, so shuffling really is correct for it. One case where the recipe is right and two where it is wrong, all three handled identically, is how we know the failure comes from the structure of the problem and not from a rigged demonstration.

Which is why there is no single split that is safe for everything. Each kind of structure has its own matched method, and none of them is exotic; they are all standard, and surveyed in one place if you want them (Roberts et al., 2017). Rows in time order want a chronological cut, or walk-forward. Rows that belong to people, or to any entity, want whole entities held out, GroupKFold or leave-one-subject-out. Rows that truly are independent want exactly the random split the textbook teaches. The skill was never in finding a clever method. It is in recognising which situation we are actually in.

So here is the uncomfortable shape coming into view. Back in §2.2 the number lied because we looked too many times. Here we looked exactly once, in perfect discipline, and it lied anyway, because the procedure that produced it had quietly assumed something false about the data. The same inflated gap, in the same direction, from a completely different cause. There, too much searching. Here, an assumption nobody ever said out loud.

**(B) Standardising.**

The split was only the first ingredient. The recipe has a next step, one we have quietly obeyed in every experiment so far. Features arrive on wildly different scales, so before training we standardise them: subtract the mean, divide by the standard deviation, so each feature sits near zero with a spread of one. And the book is careful about how. Fit the scaler on the training data only, freeze those two numbers, and apply that same frozen transform to the test. Never let the scaler so much as glance at the test set. This is the very discipline we paid for in §2.2, and it is right; we have followed it faithfully all the way through.

So let us keep following it, and let us do something perfectly innocent while we are at it. Back on the market our feature was the size of each day's move, in percent. Suppose we build it a hair differently: the size of the move in points, the raw change in the index, which is just what anyone gets by subtracting two closing prices and not bothering to divide. Same information about the same days, only in a different unit. The task is unchanged, the label is unchanged, the split stays honest and chronological, and we standardise on the training side only, exactly as the book demands. Nothing leaked, nothing fished. We change the unit of one feature, and nothing else.

| feature         | scaling                   | test accuracy |
| --------------- | ------------------------- | ------------- |
| size in percent | frozen, the textbook rule | 0.585         |
| size in points  | frozen, the textbook rule | 0.512         |

Read the second row slowly, because it should not be possible. The exact model that read 0.585 a moment ago now reads 0.512. That is chance, a coin. By changing one feature from percent to points, with an honest split, nothing leaked, the frozen rule followed to the letter, we appear to have destroyed the model outright. And here is the quiet horror of it: a careful practitioner who did every single thing the book asks would read that 0.512, conclude the market has no signal to give, and walk away for good. But we happen to know better, because we watched the percent version work not five minutes ago. The signal is still in there. Something in the recipe just reached in and killed it.

So what did? Look again at the two units, and this time ask not what they mean but how they behave over the years. The percent size holds still: a one-percent day is a one-percent day whether the index sits at 1 500 or at 7 000. But the points size does not, because the index itself grows. Over this stretch it climbs from about 1 455 to 7 483, five times larger. So the very same one-percent day is worth about fifteen points near the start and about seventy-five near the end. The percent feature stands still through the years; the points feature drifts steadily upward with the market.

Now remember what the frozen scaler did. It measured the mean and spread of the feature on the training years, the early ones, then froze those numbers and applied them ever after. For the percent feature that is fine, because the early years describe the later ones. For the points feature it is a disaster, because the training statistics belong to a fifteen-points world and the test belongs to a seventy-five-points world. We can even measure how far off the model is thrown. Under the frozen scaler, how far outside the training range do the test days land?

| feature         | test distance, average | worst  |
| --------------- | ---------------------- | ------ |
| size in percent | 0.57σ                  | 9.19σ  |
| size in points  | 1.67σ                  | 27.69σ |

The drifting feature drops the test days three times further out, the worst of them near twenty-eight standard deviations from where the model was taught that "average" sits. The network is being quizzed on a stretch of the number line it never once saw in training. It learned a world where a busy day was fifteen points, it is asked about a world where a busy day is seventy-five, and the frozen scaler is still standing at the door insisting that fifteen is normal. No wonder it guesses like a coin.

And there is the hidden assumption of this whole step, finally out in the open. When we froze the training statistics and carried them into the future, we assumed, without ever once saying it, that the world would stand still, that whatever we measured in the early years would go on describing the later ones. Sometimes it does. Here it does not.

So can we save it? The disease is obvious now, so the cure suggests itself. The trouble was freezing one mean and one spread for all time, on a feature whose scale keeps moving. So let us stop freezing. Instead, scale every day by its own recent past, a trailing window of the days just before it, and let that window slide forward with the series. Crucially it only ever looks backward, at days that have already happened, so it leaks nothing, exactly the discipline we have kept all along. And it works, mostly: the 0.512 climbs back to 0.549.

Here is the whole picture, now that we can read the labels for what they are:

| feature                      | scaling                    | test accuracy |
| ---------------------------- | -------------------------- | ------------- |
| size in percent (stationary) | frozen, the textbook rule  | 0.585         |
| size in points (drifts)      | frozen, the textbook rule  | 0.512         |
| size in points (drifts)      | rolling, a trailing window | 0.549         |

Mostly, but not all the way. 0.549 is still short of the 0.585 the stable feature reached, and that shortfall is worth saying out loud, because it is the honest lesson of this step. Patching the scaler is only second best. The real cure was never to build a drifting feature in the first place: divide by the price, keep the size in percent, and the drift is gone before any scaler has to meet it. The rolling window rescues a feature we should not have made this way.

This step's hidden assumption has a name: stationarity, the belief that the statistics of the training data still describe the future (Shimodaira, 2000; Quiñonero-Candela et al., 2009; Gama et al., 2014). Where it holds, the frozen rule is exactly right. Where it fails, that same rule quietly kills a working model, and, this is the sting, no honest split will ever warn us, because nothing was leaked. The number is not lying the way the shuffle's number lied. It is perfectly truthful. It truthfully reports a model that the recipe itself broke.

And look at the shape of this next to (A), because together they are the whole trap. In (A), a hidden assumption in the split pushed the number too **high**, handing us an edge that was never there. In (B), a hidden assumption in the scaling pushed the number too **low**, destroying an edge that genuinely was there. The very same disease, with opposite symptoms: one recipe, meant to fit every problem, meeting a problem it does not fit.

**(C) The metric.**

By now we are wary of the market, with its drifting features and its treacherous order in time. So let us go back to the loan data, the one problem that behaved. Its rows are independent, we saw that ourselves; shuffle is correct, there is no time to leak and no scale to drift. If any number in this whole report is safe, it is this one. We split it honestly, we standardise on the training side, we look exactly once, and we report what the book tells us to report: accuracy, the fraction of clients we called correctly. It comes to 0.813.

That looks respectable. But to be sure, let us compare it against doing nothing at all. What would we score by not building a model, by simply stamping "will not default" on every single client? Since about 78% of clients indeed do not default, that lazy stamp scores 0.779. So our model, at 0.813, beats the do-nothing stamp by 0.034. A modest, honest edge. We would write it down and move on.

But wait. What was the whole point of this model? A bank does not pay to be told that most people pay their bills. It builds the model to find the ones who will not, the defaulters. So let us ask the only question that actually matters: of the clients who really did default, how many did our 0.813 model catch?

In one test set there were 1 353 real defaulters. The model caught 438 of them. It missed 915. It waved more than two thirds of the defaulters straight through, stamped safe. Across ten runs it is the same story: of every hundred real defaulters, the model flags about thirty and lets the other seventy pass.

Now set the two facts side by side and feel how strange this is. Our model catches about thirty of every hundred defaulters and scores 0.813. The do-nothing stamp catches none of them and scores 0.779. A model that at least does part of the job and a rule that does not even try sit barely three points apart on the number we chose to report. Accuracy simply cannot tell the difference between a model that works and a model that does nothing at all. And nothing here is leaking, nothing is drifting; the split is the honest one, on the best-behaved dataset we own. The number is exactly right. It is answering the wrong question.

So why does accuracy do this to us? Because of what it quietly treats as equal. It counts every client the same and every mistake the same: one correct call is one point, whoever the client is, and one wrong call is one point off, whichever way it went. That only makes sense under two assumptions nobody says out loud. First, that the classes are roughly balanced, so no single group can drown out the rest. Second, that the two kinds of mistake cost the same, that waving a defaulter through is no worse than pestering a good client. On this problem both are false. Defaulters are the minority, barely a fifth, so getting the easy majority right is enough to carry the score. And missing a defaulter can cost the bank a loan, while a false alarm costs it a phone call. The two errors are nowhere near equal, and accuracy weighs them as if they were.

Once we see that, the repair is not hard. Report a number that puts the two classes on equal footing instead of letting the crowd decide. The simplest is balanced accuracy: score each class on its own, how many defaulters we caught and how many good clients we cleared, and average the two. Our model's balanced accuracy is 0.633, not 0.813. That is the honest figure, and it is far less flattering, because it refuses to let the easy majority hide the hard minority. Or we skip the single number altogether and just look at the count that mattered all along: of 1 353 defaulters, 438 caught, 915 missed. Nothing can hide inside a confusion table.

And notice how this third failure is not like the other two, which is exactly why it earns its own step. In (A) something leaked across the split. In (B) something drifted out from under the scaler. Both were faults in how the number was measured. Here nothing leaks and nothing drifts. This is the loan data, the very control that sailed through (A), split honestly, scaled honestly, looked at once. The measurement is flawless. The 0.813 is true to the last digit. It is simply the answer to a question we did not mean to ask. And it only gets worse the rarer the thing we care about: make defaulters one client in a hundred instead of one in five, as fraud or a rare disease would be, and the do-nothing stamp scores 0.99, while a model that genuinely finds half of them still looks worse on paper. The rarer and more precious the case, the more completely accuracy hides whether we found it at all.

---

## 3. So what is left?

Let us stop and take stock, because we are further from where we started than it looks.

We came into this with a simple, reasonable faith. We would be careful. We would follow the book, every rule of it, and the number the book handed us would be honest. That was the whole plan, and we carried it out to the letter.

And look what happened. First, back on the noise, we learned that just trying hard enough pushes the number up on its own, so a high score can be pure luck wearing the face of skill. We told ourselves that was only greed on a toy, that real data and real discipline would save us. Then we took real data and real discipline to three honest problems, and watched the recipe betray us three separate times. The way we split the data made the number too high, an edge that was never there. The way we scaled a feature made it too low, killing an edge that was. And the way we scored the model handed us a number that was measured perfectly and still could not tell a working model from one that did nothing at all.

Sit in that for a moment, because it is worse than any single failure. It is not that we found one bad rule to cross off a list. Every safeguard we reached for turned in our hand. The held-out validation set, gamed by searching. The sealed test, poisoned the instant we peeked at it twice. The shuffle, the scaler, the metric, each one correct in the place it was born and quietly lying everywhere else. We were never careless. We did everything right, and the number lied anyway.

So where does that leave us? If searching can inflate the number, and a hidden assumption can corrupt it, and a flawless measurement can still answer the wrong question, then what number, exactly, are we allowed to believe? We followed the recipe all the way to the end and found no safe ground at the bottom of it. We are standing here holding the one thing we set out to trust, the score, and we no longer have the faintest idea what to put in its place.

So let us say the hard thing plainly, now that we have earned it. We cannot trust the goal. The number, the score, the thing we set out to chase and report and believe, cannot carry our trust, because searching quietly inflates it and a hidden assumption quietly corrupts it, and from the outside a corrupted number looks exactly like a good one.

And here is the cruelest turn of all, the one that has been waiting since the first page. Go back to the tool we reached for without a second thought: the deep network, so flexible, so powerful, with its wall of a million knobs. How does it actually decide when it is finished? It watches the validation score and keeps the moment that scores best. That is early stopping, and it is normal, careful practice. But look at what it means. Every epoch we let it run is one more peek at the validation set, one more try kept because it happened to look good. Add the seeds we quietly re-rolled and the architectures we quietly swept, and the true count of tries is nothing like the number we admit to. We say we tried five configurations; but inside each we kept the best of a few seeds, and inside each of those the best of many epochs, and five becomes a hundred and fifty without our ever noticing. The tool we trusted most is, by its own design, a machine for driving that one validation number as high as it will go. It is a machine for doing the exact thing we watched inflate a score out of pure noise. That thing has a name, the oldest sin in the book: data snooping. The deep network snoops for a living, and it never once tells us how many times it looked.

So the verdict is not really about carelessness. We were not careless; we were meticulous, and it did not save us. The trouble runs deeper than effort. We never made our assumption plain. We shuffled without asking whether the data was ours to shuffle, we froze a scaler without asking whether the world would hold still, we reported accuracy without asking whether the classes were even. A goal built on an assumption we never stated, and never checked, is not measuring what we believe it measures. And a number that measures the wrong thing is not a target worth hitting. It is not really a goal at all.

Which leaves us just one honest place to go. If we cannot trust the number, and we have now run clean out of numbers to trust, then trust has to live somewhere other than the number. If not in the score, then where? That is the only question left, and it is worth the whole rest of the report.

---

## 4. What we can trust

So we sit in the wreckage and ask the only question left: if not the number, then what?

Start by looking again at how each failure actually happened, because they are more alike than they first seemed. The shuffle assumed the rows were interchangeable, and never checked. The frozen scaler assumed the world would hold still, and never checked. Accuracy assumed the classes were even and the two errors equally costly, and never checked. Even the search betrayed us the same way, by assuming that a higher validation score meant a better model, without ever asking how many times we had looked. Every single collapse was one unstated assumption, carried quietly into a problem it did not fit.

And that, at last, is the way out, because it tells us exactly what we were doing wrong. We were trusting the number, the output, the thing at the very end. But the number was only ever as good as the string of assumptions that produced it, and those we never once examined. So we stop trusting the number, and we start trusting the thing that actually decides whether it is honest: the way it was made. Not the score, but the procedure. The score is only what a procedure produces; the procedure is what determines whether that score means anything at all.

What, then, makes a procedure one we can trust? Everything we just suffered through answers it, because it is those same failures turned right-side up. Two things.

First, its assumptions are said out loud, and checked. We do not shuffle and hope; we ask, in plain words, is this data mine to shuffle, and then we go and look. If the rows come in time, we admit it. If a feature drifts, we measure the drift. If a class is rare, we say so before we ever pick a metric. The assumption stops being a trapdoor hidden under the method and becomes a claim we have written down and tested. That step alone would have caught all three of our disasters, because all three began with an assumption nobody ever stated.

Second, those assumptions have to fit the data actually in front of us. There is no recipe that is honest for every problem, because the honest method is the one shaped to the structure of this problem. A random split for independent clients. A walk-forward split for days in a row: train on the old, test on the newest, never letting the future leak back. Whole people held out when the rows belong to people. The point is not to memorise which method goes where. It is that the method must answer to the data, and never the other way round.

That handles the assumptions that broke us on the real data. But it leaves the first betrayal, the one on the noise, where the number climbed simply because we kept looking. So a trustworthy procedure needs one more habit, and it is plain discipline. Search less, and search in the open. Do not fish with a hundred hidden seeds and epochs and report the luckiest. When we do choose between models, judge them on a less jittery estimate than one small validation set, so the winner is picked by skill and not by wobble. And keep a single sealed test, opened exactly once, on the one model the procedure has already settled on, never as a dial we turn until we like what we see. A number we looked at a single time, made by a method whose assumptions were stated and right, is a number we are allowed to believe. A number we chased around until it looked good is not.

Notice that this quietly rewrites what "the best model" even means. Set two honest workers on the same problem. One searches hard, fishes the seeds and the epochs, and comes back with the prettier number. The other searches modestly, chooses in the open, and comes back with a plainer one. We have spent this whole report learning that the prettier number is the inflated one: its real performance is actually worse, and its report is wrong by roughly the size of the search behind it. The plainer number sits close to the truth. So the disciplined worker wins twice over, with a model that genuinely performs as well or better, and a figure we can actually stand behind. The prize for honesty is not a smaller number. It is a true one, and usually a better model besides.

So we stop accepting a model just because its score is high. We accept it only when that score came out of a procedure that could not have inflated it: an honest method with its assumptions stated and matched to the problem, a sealed test opened one time, and a result that clears the do-nothing baseline by more than the noise of the measurement itself. The number worth believing is the one the procedure earned the right to report.

And here, at the very top, is the thing worth the whole climb. Look back at the villain of this story, the careful practitioner who followed the book to the letter and was betrayed anyway. Now look at the honest procedure we have just built. They use the same steps. Both shuffle, or split, or standardise, or report a score. There is no secret technique in the honest one, no method the careless one had never heard of. The only difference between the two, the whole of it, is understanding. One reached for shuffle because the book said shuffle. The other reached for shuffle only after looking at the data and seeing that its rows really were interchangeable. The steps are identical. What changed is that somebody understood what the data was before choosing what to do to it.

So "trust the procedure" was never quite the whole of it. A procedure followed blindly is exactly the thing that collapsed on us three times over. What we actually trust is understanding, made solid and repeatable in the shape of a procedure. The stated assumption, checked against the data, is understanding written down. The matched method is understanding turned into a choice. The single honest look at the sealed test is understanding of what a measurement can and cannot be. Take the understanding away and the procedure is just the recipe again, waiting to betray the next person who runs it without asking why.

That is where trust actually lives. Not in the number, which we now know can be inflated, corrupted, or simply beside the point. It lives in the understanding of the data in front of us, the structure it has and the assumptions it will and will not bear, and in the logic that carries us from that understanding to a method that fits it. A result becomes meaningful, and worth believing, not when its score is high, but when we can trace, step by honest step, why the way we made it could not have lied to us. We cannot trust the goal. We can trust the understanding that earns it.

---

## References

*These are the sources for §2.3-A, §2.3-B and §2.3-C. The list grows as the other sections are written; the formatting is to be fixed to the handbook style at the end.*

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
- Brodersen, K. H., Ong, C. S., Stephan, K. E. and Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. *ICPR*. (Balanced accuracy, the honest metric reported in §2.3-C.)

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

*Outside the page limit. §2.3 changes one thing at a time on real data; this is the code that did it and the output it printed. Five short files in `code/`, numpy only, all reusing the same MLP from `code/lab_demo.py`: `code/loan_split.py`, `code/finance_split.py`, `code/har_split.py` for the split (A), `code/scaling_split.py` for the scaling (B), and `code/metric_loan.py` for the metric (C). Each prints the numbers quoted in the body.*

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

**Piece 5: the metric (§2.3-C).** The file is `code/metric_loan.py`. Nothing about the split or the scaling changes here: the loan data is split honestly at random and standardised on the training side, exactly as in Piece 1, so nothing leaks and nothing drifts. The only new thing is that we stop reading the single accuracy number and count what the model actually did with the rare class, the defaulters.

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

