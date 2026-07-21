# Data Snooping in Deep Learning: dissertation (working draft)

*Working draft, written as a journey rather than a proof: we start from the ordinary, by-the-book way of judging a model by its number, and follow honestly where it leads. Reasoning first; the code behind every number is in the appendices. Compass in `NORTH_STAR.md`; the underlying argument in `KEY_CORE.html`.*

*Constraint (handbook): final submission ≤ 50 pages, including bibliography, tables and figures, excluding appendices.*

---

## 0. The number we are about to trust

We have a model. Is it any good? How would we even know?

We do the obvious thing. We hide some data from it, let it guess on that data, and count the hits. That gives us a number. Is it high? Good. Low? Not good. So we read the number and we decide.

But we never build just one. We build many and keep the best number. So what are we really trusting in the end? *One number.*

And here is the question we almost never stop to ask.

> **Can we trust it?**

Let us find out honestly. We will not bend anything to force a problem into view. We will do the opposite, and *follow every rule the books give us, exactly.* Then we stand back and ask, together, what that honest number is really worth.

(One limit, so we are not doing everything at once. We stay with deep learning that forecasts and classifies: is tomorrow a busy day on the market, did this borrower default, which activity is this. Not vision, not language, not the rest. Just putting a label on the next case.)

So, where does everyone start? By the book. Let us start there too.

---

## 1. What deep learning really is, and the plan

So we have a job: take the next case and put a label on it. What do we build?

Almost always, the same thing: a small neural network. It is a stack of simple parts. A few features go in, get mixed and bent through a hidden layer, and come out as a guess between the classes.

![The model: a small network](figures/mlp.svg)

How does it learn? Not by us programming it, but by correcting itself, the same four small moves over and over. It takes the numbers in and mixes them into a guess (the forward pass). We score how wrong that guess is, in one number (the loss). We work out which way to nudge each knob to shrink that number (the gradient, the downhill direction). Then we take one small step that way (the update). A few thousand rounds of those four, and the guesses come out mostly right.

![Deep learning is one loop, repeated](figures/training_loop.svg)

One small marvel hides in those four moves: when we score wrongness this way, the correction each step makes is *exactly how far off the guess was*, no more. That is all the training is really doing, over and over. The full working, for anyone who wants it, is in Appendix A.

Give it more knobs, make it wider or deeper, and it can fit more. That is the whole appeal: **with enough knobs, a network can fit almost any pattern we hand it.**

And there is the catch. If it can fit almost any pattern, then **it can also fit patterns that are not really there.** Show it pure noise for long enough and it will "learn" the noise, memorising which random point got which random label, and score beautifully on the training data.

![Fitting the training data is not the same as learning](figures/loss_curve.svg)

So the training score can look flawless and prove nothing, which leaves a nagging question: *if the model can fit anything, even nonsense, how do we know it learned something real, and not just the noise?* We cannot tell from the training score, it does well either way. The only way to know is to try the model on data it has never seen: keep some data back, judge it there, and never let it look while it learns.

Which brings us to the plan. To turn a network into a number we can show anyone, we follow a recipe, the same one printed in every course:

![The normal way to build a model](figures/pipeline.svg)

Five steps. Frame the data. Split it into training, validation and test. Scale the features. Build and search for a good model. Measure how good it turned out. Every step is standard, careful, exactly what we are told to do, and the number at the end is meant to be honest.

So here is the plan for the whole report. We are not out to attack this recipe, we want it to work. So we follow it faithfully, and carry it to one real problem after another. At each one we ask the same plain question.

> **Is this really as safe as it looks?**

By the end we will know the answer, and what it takes to walk away with a number we can actually trust.

---

We have three real problems to carry it to, and we take the friendliest first. We start with a bank's loan clients, the ordinary, clean kind of data the recipe was built for. Then the stock market, one long column of daily prices. Then thirty people carrying a phone through a handful of everyday activities. Each is its own journey, with its own dead ends and its own moment of doubt, and the pipeline stays the map that keeps us oriented across all three. We are not here to collect three tidy results, we are walking three roads to the same place.

Let us meet the first one.

---

## 2. Loan: the well-behaved one that still lies

### a. Reading the problem

A bank hands us thirty thousand credit-card clients and one question: which of them will miss their next payment?

So who are these people? Let us look at one, top to bottom.

![One loan client, row 0](figures/loan_client.svg)

We can almost picture them: twenty-four, a small limit, a modest bill, nothing paid back last month, and a yes at the end. They defaulted. So one row is one person: twenty-three plain numbers about them, and the answer we want to predict. There are thirty thousand of them, and about one in five default.

What exactly is the job? From those twenty-three numbers, for a client we have never seen, say yes or no: will they default? A plain two-way guess, and nothing about it is exotic.

What do we build? The small network from Section 1. Twenty-three numbers go in, pass through one hidden layer of sixteen units with a ReLU, and come out as two scores that softmax turns into a probability for "yes" and "no"; we keep whichever is larger.

And how do we turn that into a number we can put in front of the bank? We run the recipe from Section 1, step for step:

![The plan for the loan clients](figures/loan_pipeline.svg)

1. **Frame** it as the yes-or-no question it already is.
2. **Split** the thirty thousand clients once, into training (60%), validation (20%), and test (20%); the test part stays sealed until the end, and the validation part waits unused until Section 5.
3. **Scale** the twenty-three features, subtracting the mean and dividing by the spread, with those statistics measured on the training part only.
4. **Build and train**: from small random weights, run three hundred passes over the training clients, scoring the guess with cross-entropy, finding the downhill direction by the backprop of Section 1, and stepping every weight a little that way with plain gradient descent (a learning rate of 0.3).
5. **Measure** how good it is by accuracy: the fraction of the sealed test clients it labels correctly.

Two numbers, then, and worth keeping straight. While it trains, the network drives down one loss, cross-entropy, the score of how wrong its probabilities are. When it is done, we judge it by another, accuracy, the plain fraction it gets right. It learns by the first and is graded by the second.

So what do we expect? Honestly, not much drama. The data is clean, the question is plain, the model is standard, and every step is straight from the book. We expect the network to learn the pattern and hand us a high, honest number. So we run it, and see.

### b. By the book

So we run it, exactly as planned. Nothing fights us: epoch after epoch the network settles into the training clients, its accuracy climbing and then levelling off, the clean shape of something that is learning.

![The network learns: training accuracy climbs, then settles](figures/loan_learning.svg)

And the number it hands back is good. On the sealed test clients, the ones it never trained on, it is right **81.9%** of the time. Into the eighties, on our very first honest try.

One careful thought before we celebrate. Is that 82% real, or a fluke of the particular fifth we happened to hold out? Easy enough to check: we cut the deck again and again, a fresh random split each time, then a stratified one that forces the same default rate onto both sides, then even a cut by raw file order. If the score is solid, none of that should move it.

![Every way of splitting lands in the same place](figures/loan_stability.svg)

It does not move. Every cut lands between 0.81 and 0.82, the whole spread no wider than a few thousandths. The score is no accident of one lucky split; it is rock-steady, which is just what a trustworthy result looks like. So there it is: a clean, standard model, trained by the book, scoring 82% on clients it has never seen, and holding that score no matter how we slice the data. Everything points the same way. Trust this number.

**But** wait. That 82%, how much of it is really ours?

Try answering with no model at all. Train nothing, look at nothing; for every client, just answer "no default," the same answer every time. How often is that right? It is right about every client who never defaults, and most of them never do.

![Out of every 100 clients, about 78 never default](figures/loan_baseline.svg)

Of the 30,000, 23,364 never default; only 6,636 do. Answer "no default" for all of them and you are right about the 23,364 and wrong about the 6,636: that is 23,364 out of 30,000, or **0.779**. That score is free. It takes no model, no training, no thought at all; the data gives it away because most people simply pay.

Now stand our trained network next to that free answer.

![Our model against answering "no" to everyone](figures/loan_deflate.svg)

Our network, twenty-three features and three hundred epochs of training: **0.819**. The free answer, which never looks at the data at all: **0.779**. **Everything we built sits four points above what the data was handing out for nothing.**

Four points. Where did even those come from, and what did all that training actually add? And a colder doubt behind it: is our model any good at all, or is it mostly just riding that free 78%?

And then the question that should stop us cold. We have judged this entire thing on one number, its accuracy. But if that number can barely tell our careful, trained model apart from an answer that never even looked at the data, then what is it really measuring? What have we been trusting this whole time?

> **If one number cannot tell a trained model from an answer that never looked at the data, what is it measuring, and can we trust it at all?**

### c. What the number was hiding

So we stop trusting the single number, and go looking for what it hides.

First, rule out the obvious. Is the score high because we split the data badly, or because something leaked from the training set into the test? No. We watched it hold rock-steady across every cut we tried, and there is nothing in this data to leak: no dates, no ids, nothing that puts one client ahead of another or ties a training row to a test row. The data is clean and the split is honest. The fault is not there. It is in the number itself.

So take the number apart. Accuracy asks one plain thing: of all the clients, how many did we label correctly? Notice what it does not ask. It does not care who. **Getting a payer right and getting a defaulter right count for exactly the same.** And there is the crack, because the two are not the same job at all. We did not build this to wave through the people who pay; we built it to catch the people who will not. So we ask the only question that was ever the point.

**Of the 1,353 clients in the test set who really did default, how many did our model catch?**

We stop reading the blended score and count, group by group.

![What the model did with each group: most defaulters slip through](figures/loan_recall.svg)

It caught 472 of them. It missed 881. Two of every three people who default, our carefully trained, 82%-scoring model waved straight through, stamped safe. Then look at the payers, and the picture turns over: it cleared 4,445 of the 4,647. That is where the 82% lives. Almost the whole score is the big, easy group it got right; almost none of it is the small, hard group we actually cared about.

Put it in the plainest terms. Picture an illness that 78% of people do not have. A lazy doctor tells everyone the same thing, "you are healthy," and is right 78% of the time, because most people really are healthy. **And yet that doctor catches zero sick people.** A 78% that looks perfectly respectable, and is useless at the one job that mattered: finding the sick.

Our model is a little better than that lazy doctor: it does catch about one in three of the sick. But its accuracy barely moves, **82% against the doctor's 78%**, because accuracy is mostly measuring how well it labels the healthy majority, which was always the easy part.

That is what makes the four points so slippery. The number was never measuring the thing we cared about. It answers "did we get most clients right?", where the answer is easily yes, and it stays silent on "did we catch the defaulters?", where the answer is mostly no. So the entire gap between our model and doing nothing, those four points, is almost nothing by the number and almost everything in the job, and the number gives us no way to tell which.

> **Four points over doing nothing: nearly nothing, or nearly everything? And if our one number cannot tell the two apart, how are we ever meant to judge this model?**

### d. The number we can trust

So how do we judge this model without being fooled? We stop handing the whole verdict to one blended number, and we measure in a way that gives the rare, costly cases their due.

The fix is small. Instead of asking "how many clients did we get right," a question the majority always wins, we score the two groups apart and then average them. How well did we do on the people who default? How well on the people who pay? Average those two, and each group counts the same, however many are in it. That is balanced accuracy.

Now judge our model and the "always no" answer both ways at once:

![Same model, same data. Change the ruler, and the gap appears.](figures/loan_balanced.svg)

On plain accuracy they are near twins, 0.817 and 0.779, four points apart. On balanced accuracy they split wide open: our model scores **0.646**, and the "always no" answer scores exactly **0.500**, a flat coin, because it never catches a single defaulter. Same model, same data. We changed only the ruler, and a difference that was four grudging points became fifteen. The skill that accuracy was drowning is suddenly there to see.

**So which number do we keep?** Here is the honest answer: **neither**, if we keep it on faith. We are not saying trust 0.646 instead of 0.819. That would be the same mistake in new clothes, swapping one number we do not really understand for another.

What we actually trust is not a number at all. It is the four counts underneath both of them: **472 caught, 881 missed, 4,445 cleared, 202 flagged.** Those are not a summary of anything, they are simply what the model did to 6,000 real people, and nothing can hide inside four counts. Accuracy blended them so the majority buried the failure; balanced accuracy weighs the two groups evenly so the failure shows. We keep the 0.646 only because we can see it is an honest reading of the counts, and we drop the 0.819 only because we can see it is not. The counts are the evidence. A number is just shorthand, trustworthy exactly as far as we have checked what it is made of.

And there, at last, is the assumption we never knew we had made. **Reaching for accuracy without a second thought, we assumed that every client counts the same, and that getting most of them right means the job is done.** That holds only when the two outcomes are roughly balanced and the two mistakes cost the same. Here neither did, and accuracy was blind to both. This is the shape of the whole report in miniature: a plain step of the recipe, followed without question, carrying a hidden assumption that the data quietly breaks.

So notice, in the end, what is worth trusting and what is not. The number itself wobbles: 0.646 today, a hair different tomorrow under another split, and none of it shakes us, because the number was never what we were trusting. We trusted the way we made it. We looked at the real counts, we chose a measure that credits the job we actually cared about, and we can trace, step by step, why it cannot be quietly lying. 

> **The score, the goal we chase, we cannot trust on its own. The process that earns it, we can.**

If a clean, honest process is the only thing worth trusting, then at the next problem, where inside that process is the next **hidden assumption** hiding?

## 3. Market: the edge that was too easy

### a. Reading the problem

We come to the second problem already a little wary. The last one taught us that a number can look clean, steady, and high, and still be lying, so this time we make ourselves a promise: we will not trust a score just because it looks good.

And then we open the problem, and there is almost nothing there to trust. No neat rows of features like the loan clients. Just one long column of numbers: the closing price of the S&P 500 at the end of each day, 6,664 days in a row. A single price that wanders across the years, from a low near 677 to a high near 7,610.

![All we were handed: one drifting column of prices, and what we choose to look at instead](figures/market_data.svg)

So what can we predict from one column of prices? With the loan clients we had twenty-three numbers about each person. Here we have one number a day, and nothing else.

The first idea is the obvious one: predict tomorrow's price. But 677 back then and 7,610 now are not the same kind of thing, and a model fed the raw prices would learn only that the numbers drift upward with the years. Useless. We need to turn the price into something that means the same thing in any year.

So we stop looking at the price, and look at how much it moved. Not "the price is 1,402" but "today it moved 0.2%." A one-percent day is a one-percent day whether the index sits at 1,500 or at 7,000, and measured that way every day becomes comparable. Then one more step: forget which way it went, up or down, and keep only the size of the move. A calm day is a small number, a wild day a large one. That is the lower panel above: the same years, seen as sizes, long quiet stretches broken by a few violent bursts.

From that we build the task. Take the sizes of the last five days, and guess one thing about tomorrow: a busy day, a move bigger than usual, or a calm one? "Usual" we pin at the middle day, so that half of all days count as busy and half as calm. A blind guess is then a coin, 0.5, with no lopsided majority to lean on the way the loan data had.

![The task: from the last five days' sizes, guess whether tomorrow is busy or calm](figures/market_task.svg)

And here is where the last problem has changed us. With the loan clients we built first and asked questions afterwards. This time, before we train anything, we stop and ask the plainest question there is: is there really a pattern here to find, or are we about to spend three hundred epochs chasing noise? A network can always fit something. That is not the same as there being something to fit.

So we look before we build. Does a big day tend to be followed by another big day? We put every day's size against the next day's size, and see whether they lean together.

![Big days tend to be followed by big days: the link is 0.29, well clear of zero](figures/market_signal.svg)

They do. The cloud tilts upward: when today's move is large, tomorrow's leans large too, a link of 0.29 where zero would have meant no pattern at all. Storms cluster, and quiet follows quiet. It is not a strong effect, but it is real, and it is honest, built from nothing but days that have already closed.

So this time we have earned the right to try. There is a genuine signal in the sizes, we are using the same small network as before with nothing tuned or added to help it, and a blind guess scores 0.5. We build it the same careful way, and see what an honest model can do.

### b. The edge that was too easy

The obvious first try is direction: will tomorrow close up or down? We build it, split it by the book, and look.

```
direction:  shuffle 0.533,  chronological 0.540
```

A coin flip, both ways. No edge at all. Dead end.

But a market has more than a direction. It has a mood: some stretches are calm, some stormy, and the storms bunch together, a wild day sitting near other wild days. That is the 0.287 echo we found. So we change the question. Never mind which way tomorrow moves, can we say how much? From the sizes of the last five days, predict whether tomorrow is a busy day.

```
volatility, shuffle (by the book):  0.618
```

A real edge, well clear of the coin's 0.5, and on the market an edge like that would be worth a fortune. **Which is exactly the problem. A real edge on the market, won this easily? That is far too good to be true. What is leaking in?**

So instead of celebrating, we ask one plain question: is 0.618 what we would actually get in practice? In real life we train on the days we have and predict days that have not happened yet. So we measure it that way too, past to future.

```
volatility, shuffle       :  0.618
volatility, chronological :  0.587
```

They do not match. And that stopped us. Back on the loan clients, every way of cutting the data agreed to the third decimal. Here, two cuts of the same days, the same model, disagree by three whole points. **One of the numbers is lying**, and we did not know which one, or why.

So we finally did the thing we should have done first. We stopped fiddling with the model and looked at the rows themselves, side by side:

```
X[0]: [0.0383, 0.0019, 0.0010, 0.0271, 0.0112]
X[1]: [0.0019, 0.0010, 0.0271, 0.0112, 0.0131]
```

There it was, in plain sight the whole time. Each row is the last five days; the very next row is those same days slid along by one. Two neighbouring rows share four of their five numbers. They are near-twins. We had been so busy with the model that we never once looked at what a single row actually was.

That one look explains both numbers. When we shuffle, a day and its near-twin can land on opposite sides of the cut, so at test time the model is handed a day whose near-double it already studied in training. The 0.618 was never skill at seeing the future; it was skill at recognising a neighbour. The honest split, past then future, leaves no twin to lean on, which is why it settles lower, near 0.587, or 0.603 when we roll the cut forward through the years.

One doubt still nagged: how do we know the gap is a real leak, and not the shuffle getting lucky once? The direction task answers it. A leak can only inflate an edge that is really there, so on direction, which we already know is a coin flip, the gap should vanish. It does: shuffle and honest split both land near 0.53. The gap shows up only where there is a real pattern to steal. That is how we know it is a true leak.

### c. A working model, killed by a unit

We are not done with the market. There is a step of the recipe we have not questioned yet: scaling. Features arrive on all sorts of sizes, so the book says standardise them, subtract the average, divide by the spread, and fit those numbers on the training data only, never the test. We have done exactly that all along, and it is the careful thing to do.

So let us keep being careful, and change one tiny, innocent thing. Our feature was the size of each day's move, in percent. What if we measure it in points instead, the raw change in the index, which is just what you get by subtracting two prices and forgetting to divide? Same days, same information, a different unit. Everything else stays fixed, the honest split included.

```
same split, only the feature's unit changes:
  size in percent :  0.586
  size in points  :  0.510
```

Read the second row twice, because it should not be possible. **Nothing leaked, the split is honest, the scaler is frozen exactly as taught, and yet the same model that scored 0.586 now scores 0.510, a coin. How can changing a feature's unit kill a working model?**

By now we know where to look. Not at the model, at the data. The two units behave completely differently over the years. A one-percent day is a one-percent day whether the index sits at 1500 or at 7000. In points it is not, because the index climbs from about 677 to 7610 across the data, so the same one-percent day is worth a handful of points early on and dozens near the end. The percent feature holds still; the points feature drifts upward with the market.

![Why a drifting feature breaks the frozen scaler](figures/drift.svg)

We can even measure how badly. The frozen scaler learned what "normal" looks like from the early, low years. Ask it to score the late, high years, and see how far outside its training range the test days land:

```
frozen scaler, test values as z-scores:
  percent :  mean 0.57,  max  9.19
  points  :  mean 1.67,  max 27.69
```

For percent, the test days sit about where the model expects, under one standard deviation on average. For points, they are miles out, nearly twenty-eight standard deviations at the extreme, values the model never met in training and has no idea what to make of. The number is not lying; it is truthfully reporting a model the recipe broke.

The fix is to stop freezing one average for all time, and scale each day by its own recent past, a window that slides forward and only ever looks backward, so it cannot peek at the future:

```
  points, rolling scaler :  0.555
```

That lifts the coin back to 0.555. Better, though the real fix was never to build a drifting feature in the first place.

### d. What the market cost us

So the market cost us twice. The split leaked the future and pushed the number too high, an edge of 0.618 that was really 0.60. The frozen scaler froze a moving world and pushed the number too low, a working 0.586 collapsed to a coin. Two opposite failures, and the same root under both: we kept trusting the recipe without ever looking hard at the data underneath it.

![The market trail, two traps at once](figures/market_tree.svg)

**So which number do we keep, and what do we give up?** We give up the flattering 0.618, which was the leak, and the broken 0.510; both of those were the recipe reporting on a world it had misread. We keep around 0.60, the honest walk-forward score, once the leak is stripped out and the feature is left in percent where it does not drift. And unlike the loan clients, where the trap was in the yardstick, here the number itself was the honest messenger. It came out too high when we leaked, too low when we drifted, and each time it was telling the truth about a model we had quietly broken.

Two datasets in, and the pattern is starting to show. On the loan clients the number lied through the measure; on the market it never lied at all, it was knocked off course twice by the structure of the data. Each time we did everything by the book, and each time the book was not enough on its own. Next we hand the model to thirty people, and watch the same kind of trap open a third way.

## 4. Phone: reading the person, not the task

### a. Reading the problem

This third problem has nothing to do with time. Thirty people strapped a phone to their waist and did a few ordinary things: walking, sitting, standing, lying down. The phone's motion sensors watched, and every short moment of movement was boiled down to 561 numbers, with a label for what the person was doing. Ten thousand such moments in all.

Before we plan anything, we read one row first, top to bottom, to see what we are holding:

```
one window (row 0):
  person   : 1
  activity : standing
  561 features, first six: 0.2886, -0.0203, -0.1329, -0.9953, -0.9831, -0.9135
```

So one row is a single moment of one person, 561 numbers wide, tagged with one of six activities. Nothing odd yet. Then we glance at the next few rows:

```
row 0: person 1, standing
row 1: person 1, standing
row 2: person 1, standing
...
row 5: person 1, standing
```

All still person 1, all still standing. That is how the data arrives: long runs of one person doing one thing, hundreds of rows at a stretch. Thirty people, but ten thousand rows, about three hundred and forty each. A row is not a person. It is a moment of a person, and we only ever have thirty people.

The job itself is plain: from the 561 numbers, name the activity. There are six, so a guess has something to beat. Do nothing clever, call every moment "laying", the commonest, and we are already right 19% of the time. That is the floor. Anything worth building has to clear it.

### b. A first run, and a scare

Now the by-the-book move. Scale the 561 features so none of them shouts louder than the rest, build the small network from Section 1, and train it. We reuse the learning rate that had worked on the market, and set it going.

It falls apart. We watch the training accuracy as it goes:

```
step too big (learning rate 0.5):
  epoch   0: 0.441
  epoch  30: 0.002
  epoch  90: 0.167
  epoch 149: 0.000
```

Worse than a coin, worse than guessing "laying" every time. For a moment the whole task looks hopeless.

But before giving up, we stop and ask a smaller question: is the model even learning, or is something broken? Think about what training does. It walks the model downhill, one step at a time, toward the bottom of a valley where its guesses are best. The learning rate is the size of that step. Make it too big, and each step overshoots the bottom and lands further up the far side. The model never descends; it bounces across the valley and flies apart.

That is what those numbers are. Not a hard problem, just a step too large for it.

So we change one thing, the step, and try again:

```
step made smaller (learning rate 0.1):
  epoch   0: 0.323
  epoch  30: 0.878
  epoch  90: 0.948
  epoch 149: 0.962
```

This time it walks down cleanly, climbing to 0.962 on its own training data. Same network, same features, same data. The only difference was the size of the step.

A small, ordinary mistake, but it leaves a rule we will need in a minute: before we trust any number a model prints, we check that it is learning at all. A contest between two ways of splitting the data means nothing if one of the models never learned a thing.

### c. The number that was too good

We know the model learns now, so back to the recipe. By the book: shuffle all ten thousand rows, keep a random fifth for the test, and read the score.

```
shuffle rows (record-wise):  0.966
```

Almost perfect. The best number in the whole report. That should thrill us.

It does not, quite. **Reading what a person is doing, from a phone in their pocket, right almost every single time? Is the model that good at telling walking from sitting, or is it just recognising these particular thirty people?**

Here is what we should have asked sooner: how would we ever use this model? On someone new, a person it has never met. And the shuffle never tested that. The rows come in long runs of one person; scatter them at random and almost every one of the thirty lands on both sides of the cut. So at test time the model is handed a moment from someone whose other moments it already studied in training.

So we ask the honest question instead. We hold out whole people: six of the thirty go entirely into the test, met for the first time only there. If part of that 0.966 was really recognising people, holding them out should knock it down.

```
hold out people (subject-wise):  0.948   (shaky: 0.93 to 0.97, depending on which six we hold out)
```

Two things happen at once. The number drops, from 0.966 to about 0.948. And it wobbles: hold out one set of six and you get 0.967, another set and you get 0.930. So something did change when we stopped letting the model meet its test people in advance. But the drop is small, and the honest number jumps around depending on who we pick. Did holding out the people really cost us, or did we just land on a harder handful of strangers? We could not yet say.

### d. Is it the activity, or the person?

We have a doubt, not an answer. The honest number was lower, but only by a little, and it wobbled. Maybe holding out whole people really did cost us, because the model had been leaning on faces it already knew. Or maybe we just drew six harder strangers. From the outside the two look the same. Staring at 0.948 will not tell us which.

So we build a test that changes only the thing in question. We fix the exact same test windows, from the same six people, and train the model two ways: once with those people's other moments allowed into training, once with those people kept out entirely. Same test, same people, everything the same, except whether the model got to know them first.

```
same test people, seen in training :  0.968
same test people, never seen       :  0.947
gap                                :  0.022   (positive in all five tries, 0.007 to 0.047)
```

There it is. Knowing a person in advance is worth about two points, every single time; not once did seeing them fail to help. So the doubt is settled. Part of that shiny 0.966 really was the model recognising these particular thirty people, not reading their activity.

![The phone trail, one try at a time](figures/phone_tree.svg)

**So which number do we keep, and how worried should we be?** We keep the honest one, about 0.947, the score on people the model has never met, because that is the only way it will ever be used. We give back the two points the shuffle handed us for free. But this trap is not the market's. There, the shuffle invented a whole edge out of nothing. Here it only padded a real skill: take the padding away and the model still names a stranger's activity, from a phone in their pocket, about 95 times in 100, far above the 19 of a do-nothing guess. The honest number is lower, and shakier, and still very good. Honesty cost us two points and a little certainty. It was worth paying.

That is the third leak, and the same careless move behind all three: shuffle without looking. Time, rare cases, now people. Next we go somewhere the data is clean and the trap is still waiting, this time in us.

## 5. The search: the trap in us

### a. The question, and a clean test

So far, every trap has lived outside us, in the data or in the measure: the order of time, the people inside it, the rare cases, the yardstick we judged by. Fix all of those, the comforting thought goes, and surely we are safe.

But look at what we kept doing in every chapter. We never built just one model. We tried a setting, trained, and kept the best. We trained for many epochs and kept the one where the validation score looked best. We ran a few seeds and kept the luckiest of them. That searching, keeping the best of many tries, was the busiest thing we did, and the most natural thing in the world. Is it safe?

We cannot answer that on real data, because there the true skill and the luck are tangled together; we never know the number the score should have been. So we build a case where we do know. We take the loan clients, their real twenty-three numbers, and throw the labels away, replacing each with a coin flip that has nothing to do with the person:

```
clients: 30000 | features: 23
label balance: 0.499  (a fair coin)
corr(feature 0, label): -0.0022  (nothing to learn)
```

Now there is nothing in the data to find. Any model, however clever, is guessing, and the true accuracy is exactly 0.5. We split the clients three ways: a training set, a small validation set of two hundred, and a large sealed test of ten thousand that we open only at the very end, as the truth.

```
train 4000 | validation 200 (small: the fuel) | sealed test 10000 (large: the truth)
```

The small validation set is deliberate. It is the fuel: the fewer points we score on, the more a number wobbles by luck, and the more room our searching has to find a lucky one. Now we search the way we always search, and watch what our ordinary habits do to a number that should never move off 0.5.

### b. Early stopping is a search

We train one model, nothing fancy, on data with nothing to learn. We let it run for three hundred epochs, and at each one we do the sensible, standard thing: glance at the validation score and remember the epoch where it looked best. That is all early stopping is. To see what it is really doing, we secretly check the sealed test at every epoch too.

Watch the two numbers over the epochs:

```
  epoch   0: validation 0.530   sealed test 0.494
  epoch  50: validation 0.495   sealed test 0.487
  epoch 100: validation 0.505   sealed test 0.494
  epoch 150: validation 0.525   sealed test 0.498
  epoch 200: validation 0.540   sealed test 0.500
  epoch 250: validation 0.495   sealed test 0.499
```

The validation score never settles. It wanders up and down, 0.49 one epoch, 0.54 another, all of it pure noise, because there is nothing to learn. The sealed test sits quietly near 0.500 the whole time, telling the truth. And here is the move we make without thinking: early stopping keeps the best validation epoch. Out of three hundred wobbles, it picks the highest.

```
early stopping keeps the luckiest epoch (164):
  validation  0.545   <- what we would report
  sealed test 0.500   <- the truth
```

**So when we early-stop and proudly write down 0.545, is that a better model, or just the luckiest of three hundred peeks?** The sealed test answers plainly: 0.500. There was never anything to find. The 0.045 above it is not skill, it is the reward for looking three hundred times and keeping the best look. Early stopping did not tune the model toward the truth. It went shopping through the noise and brought back the prettiest number.

### c. Seeds pile on, and the name for it

One early-stopped run already peeked three hundred times. Now add the other ordinary habit: run the same setup under a few different random seeds, and keep the best. Each seed is a fresh handful of luck. Keep the best of several, and we are searching on top of the search we already did. Watch the kept number as we add seeds:

```
   1 seed : kept validation 0.545   sealed test 0.496   (really  200 hidden tries)
   5 seeds: kept validation 0.581   sealed test 0.496   (really 1000 hidden tries)
  20 seeds: kept validation 0.607   sealed test 0.497   (really 4000 hidden tries)
```

![The winner's curse](figures/winners_curse.svg)

The kept number climbs and climbs, 0.545, 0.581, 0.607, and the sealed test never budges off 0.5. Nothing was learned, nothing improved. We just reached further into the lucky tail each time we added a seed. How fast the best of many climbs above the truth is plain probability, worked out in Appendix A. And look at the count on the right. We think we tried one seed, or five, or twenty. But each one already early-stopped over two hundred epochs, so twenty seeds is not twenty tries, it is four thousand. The number we would report is the best of four thousand, and we would call it "the model".

So count honestly the next time. Five settings, three seeds each, three hundred epochs each: that is four and a half thousand hidden tries, and the number you keep is the best of all of them. The tool we reached for at the very start, so flexible, so powerful, turns out to be a machine for driving one validation score as high as it will go. It is doing, automatically and out of sight, exactly what we just watched invent a number out of nothing. There is an old name for it: **data snooping**. A deep network snoops for a living, and never tells us how many times it looked.

### d. But search is how we improve

So is the answer to stop searching? No. That would throw away the very thing that makes the tool worth using. Searching is not the villain; searching in the dark is. On data with a real signal, the same searching that invented a score out of noise can genuinely find a better model, as long as we do it in the open and keep the test sealed.

So we go back to the real loan labels, where there is something to learn, and we search honestly. Two architectures, a shallow net and a deeper one; two optimisers, plain gradient descent and momentum; four models in all. We pick the best on the validation set, and only then, once, open the sealed test:

```
architecture      optimiser   validation   sealed test
1 hidden layer    plain GD     0.624        0.624
1 hidden layer    momentum     0.660        0.650
2 hidden layers   plain GD     0.618        0.621
2 hidden layers   momentum     0.658        0.653
```

The search paid off. Momentum clearly beats plain gradient descent, lifting the honest score from about 0.62 to 0.65. We keep the best on validation, one hidden layer with momentum, for a sealed-test balanced accuracy of 0.650. That is a real gain over the 0.644 we settled for in Section 2, and because we never touched the test while choosing, it is a gain we can believe.

Notice even here the small tax. The winner's validation, 0.660, still sits a touch above its sealed test, 0.650. A little snooping crept in, as it always does when we keep the best of a few. But it is small, we measured it, and the number we walk away with is the honest one, the 0.650 the test gave up only once.

So there was never a war between searching and honesty. **The careless practitioner and the careful one search exactly the same; the difference is that one reports the number the search inflated, and the other reports the number a sealed test gave back.** Search all you like. Search in the open, count your tries, and keep one test you never select on. Then the number you keep is not the luckiest of four thousand. It is one you earned.

And that is the last trap, and the deepest, because it was never in the data at all. It was in us, in our own eagerness to keep the best. We have now watched the number lie in every way it can: through what we chose to measure, through the hidden structure of the data, and through our own searching. It is time to stand back and ask what, if anything, is left.

## 6. So what is left?

Let us count up the damage. Four problems, all handled by the book, and the book fell short on every one. On the loan clients the number measured the wrong thing: 0.819 accuracy, but only a third of the defaulters caught. On the market it came out too high, an edge of 0.618 that was really the future leaking backward, then too low, a working model dropped to a coin by a drifting feature. On the phone it read the thirty people instead of the task, a near-perfect 0.966 that was really 0.947 on a stranger. And on data with nothing to learn, our own searching still found an edge, 0.607 out of a truth of 0.5.

Every safeguard we trusted failed somewhere, and none of them warned us. We were not careless. We did everything by the book, and the number lied anyway.

**So which number, exactly, are we still allowed to believe?** The score we set out to chase and report and trust cannot carry that trust, because a hidden assumption can quietly distort it, and our own searching can quietly inflate it, and from the outside a bad number looks exactly like a good one. We have run clean out of numbers to trust. If trust does not live in the number, then where does it live?

## 7. What we can trust

Here is the answer, and it was hiding in plain sight the whole time. Look back at the honest fix at each step. Not one of them was a secret technique. The careful practitioner who got fooled and the honest one who did not use the very same five steps: both shuffle, both scale, both search, both measure. The only difference between them, the whole difference, is understanding.

![The same five steps, and the hidden assumption in each](figures/pipeline_traps.svg)

One person reached for shuffle because the book said shuffle. The other looked at the data first, saw that its rows came in time order, and chose a cut that respected time. Same step, opposite result, because one of them understood what the data was before deciding what to do to it. That is the lesson of every problem we walked. The trap was never in the step. It was in doing the step without asking what it quietly assumed, and whether the data in front of us could bear it.

Notice what this is not. It is not a counsel of despair, a warning to stop building or stop searching. We searched, and searching worked: on the loan clients, comparing a handful of architectures and optimisers honestly lifted the score we could actually keep from 0.644 to 0.650, a real gain, and one we could believe precisely because we sealed the test and never chose on it. The tool is powerful, and the goal, a good model on real data, is worth chasing. The discipline is simply how we chase it without fooling ourselves.

So state each assumption and check it against the data. Match the method to the structure in front of you. Search all you like, but search in the open, count your tries, and keep one test you open once and never select on. Do that, and the number at the end is one you have earned the right to believe, not because it is high, but because you can trace, step by honest step, why the way you made it could not have lied.

And we already have four of them. Look at what each honest procedure handed back: a balanced 0.644 on the loan defaulters, around 0.60 on the market with the leak stripped out, about 0.947 on a phone strapped to a stranger, and 0.650 on the loan again after searching a few models in the open. Not one is the flattering number we first saw, and not one is high for its own sake. They are simply what was left standing after every assumption was checked and the test was opened only once. That is why we can believe them.

That is the whole of it. The recipe is a fine place to start and a dangerous place to stop. We cannot trust the goal for its own sake, the number chased and reported and hoped over. We can trust the understanding that earns it, and the number an honest procedure hands back is one worth keeping.

---

## References

*The sources behind Sections 0 to 2, the only sections built so far; this list will grow with the report. Formatting to be brought to the handbook style at the end.*

**Deep learning, the standard method (Section 1).**

- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*. MIT Press.
- Chollet, F. (2018). *Deep Learning with Python*. Manning.

**The loan data, and why accuracy misleads (Section 2).**

- Yeh, I.-C. and Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2). (The UCI credit-card default dataset used here.)
- Provost, F., Fawcett, T. and Kohavi, R. (1998). The case against accuracy estimation for comparing induction algorithms. *ICML*. (Why plain accuracy is the wrong yardstick when the classes are imbalanced.)
- Brodersen, K. H., Ong, C. S., Stephan, K. E. and Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. *ICPR*. (Balanced accuracy, the honest metric used in Section 2.)

---

## Appendix A: The four moves, derived

*Outside the page limit. The four moves of Section 1, written out in full, so the network's own formulae can be checked and not taken on trust. One input `x`, one hidden layer, a guess across the classes.*

*Forward: the guess.* The hidden layer mixes the input with weights `W1` and a bias `b1`, then bends it with ReLU, which just replaces negatives with zero:

> a = W1x + b1,   h = ReLU(a).

A second set of weights turns that into one score per class, and softmax turns the scores into probabilities that add to one:

> z = W2h + b2,   p_k = e^(z_k) / Σⱼ e^(z_j).

That vector `p` is the guess.

*Loss: how wrong.* The true class is `c`. Cross-entropy scores the guess by how little weight it put on the truth:

> L = −log p_c.

Put all the weight on the right class and the loss is zero. Put almost none, and it shoots up.

*Gradient: which way is downhill.* This is the line worth the whole appendix. Write the loss in terms of the scores, using `p_c = e^(z_c) / Σⱼ e^(z_j)`:

> L = −z_c + log Σⱼ e^(z_j).

Now take the slope against one score `z_k`. The first term gives −1 only for the true class; the second gives `e^(z_k) / Σⱼ e^(z_j)`, which is `p_k`. So

> ∂L/∂z_k = p_k − y_k,   that is,   ∂L/∂z = p − y,

where `y` is the truth written as a one-hot vector, a 1 in the true class and 0 everywhere else. Softmax and cross-entropy fall away into something clean: the correction is just the gap between what we guessed and the truth. This is the marvel Section 1 pointed at.

*Backprop: pass the gap back.* That gap sits at the output. The chain rule carries it back to every weight.

![The output error flows back to every weight](figures/backprop.svg)

At the second layer the gradient is the gap times the hidden values; then we push the gap through `W2` to the hidden layer, and through the ReLU, which lets it pass only where `a` was positive:

> ∂L/∂W2 = (p − y) hᵀ,   ∂L/∂b2 = p − y,
> ∂L/∂h = W2ᵀ(p − y),   ∂L/∂a = ∂L/∂h  (only where a > 0),

and the first layer gets its gradient the same way, `∂L/∂W1 = (∂L/∂a) xᵀ` and `∂L/∂b1 = ∂L/∂a`. Over a batch we average these across the examples.

*Update: take the step.* Every weight moves a small step `η` against its gradient:

> W ← W − η ∂L/∂W.

That is one round. Do it a few thousand times and the guesses sharpen. The same forward and backward pass, in code, runs in the notebook of Appendix C.

---

## Appendix B: The loan data, up close

*Outside the page limit, and here so the whole of Section 2 can be understood from this document alone. This is the loan data exactly as it arrives: where it came from, what every column holds, and the few facts about its structure the chapter leans on.*

Fetched once and frozen to `data/loan_uci350.csv` (Yeh and Lien, 2009): thirty thousand clients, twenty-four columns, twenty-three features and the label. Here are the first three clients, one column-group per line so a whole row fits on the page:

| column | what it holds | client 1 | client 2 | client 3 |
| --- | --- | --- | --- | --- |
| 0 | LIMIT_BAL, the credit limit | 20 000 | 120 000 | 90 000 |
| 1 | SEX | 2 | 2 | 2 |
| 2 | EDUCATION | 2 | 2 | 2 |
| 3 | MARRIAGE | 1 | 2 | 2 |
| 4 | AGE | 24 | 26 | 34 |
| 5 to 10 | PAY_0 to PAY_6, repayment status, six months | 2, 2, -1, -1, -2, -2 | -1, 2, 0, 0, 0, 2 | 0, 0, 0, 0, 0, 0 |
| 11 to 16 | BILL_AMT1 to 6, the bill, six months | 3 913, 3 102, 689, 0, 0, 0 | 2 682, 1 725, 2 682, 3 272, 3 455, 3 261 | 29 239, 14 027, 13 559, 14 331, 14 948, 15 549 |
| 17 to 22 | PAY_AMT1 to 6, what they paid, six months | 0, 689, 0, 0, 0, 0 | 0, 1 000, 1 000, 1 000, 0, 2 000 | 1 518, 1 500, 1 000, 1 000, 1 000, 5 000 |
| 23 | **default** | **1** | **1** | **0** |

Client 1 is the twenty-four-year-old from Section 2a: a 20 000 limit, small bills, almost nothing paid back, and a 1 at the end. They defaulted. We use the file as it comes, nothing dropped, nothing engineered.

The rest of the file in numbers: 22.1% of clients default (6 636 of 30 000), which is exactly why answering "no default" for everyone already scores 0.779. Thirty-five rows are exact copies of another row. There is no id column and no date column.

Why Section 2c can rule the data out as the source of the trap: with no identifier and no date there is nothing to order the rows by, and the file is a single cross-section, every client watched over the same six months. The thirty-five duplicates are about one in a thousand, far too few to move any split; with features this coarse, two different people can simply land on the same values. So a random split cannot leak, and every way of cutting the data agrees, exactly as we saw.

Honest limits: one bank, one country, one six-month window (Taiwan, 2005). That is a caution about carrying the model elsewhere, but it cannot manufacture a trap; it is why the loan set serves as the clean control.

---

## Appendix C: The code

*Outside the page limit, and the answer to "how do I run this?" Every number and figure in Section 2 is produced by one self-contained notebook, `notebooks/loan.ipynb`. It loads the raw data, trains the model, prints every count, and plots every chart, and it reads as a standalone walk-through: open it and Run All (it needs only numpy and matplotlib), or read it top to bottom without running a thing. From the command line: `jupyter nbconvert --to notebook --execute notebooks/loan.ipynb`.*

*The market, phone, and search chapters (Sections 3 to 5) each get their own notebook under `notebooks/` as they are built.*
