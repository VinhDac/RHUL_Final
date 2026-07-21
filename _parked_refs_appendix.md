# PARKED: old References + Appendices (superseded)

*Moved out of `READ_ME.md` on 2026-07-21. Written for the earlier draft; being rebuilt per section. Kept for salvage, NOT part of the live document.*

---

## References

*Sources for the four traps we walked, from §2 to §5. The formatting is to be fixed to the handbook style at the end.*

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
- Brodersen, K. H., Ong, C. S., Stephan, K. E. and Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. *ICPR*. (Balanced accuracy, the honest metric reported in §2.)

**The data, and the effect the market task leans on.**

- Yeh, I.-C. and Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2). (The UCI loan-default set.)
- Anguita, D. et al. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN*. (The UCI HAR set.)
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4). (Volatility clustering.)

---

## Appendix A: Where the two formulas come from

*Outside the page limit; here so the two formulas the winner's curse leans on can be checked, not taken on trust. Nothing below goes past first-year probability, we derive it in place.*

The winner's curse in §5 leans on two formulas: the spread of a chance score, σ = 0.5/√n, and the best of N scores, about 0.5 + σ√(2 ln N). Here is where each one comes from.

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
- **It scales with σ = 0.5/√n.** A smaller validation set means a larger σ means a larger gap. The small validation set is the fuel, exactly as §5 said.

Two honest limits, both pushing the real gap a little below the formula:

- **Independence.** We assumed the N draws were independent. Configurations that differ only by a seed, or by one more epoch, are near-copies, not fresh draws, so N of them count as fewer than N independent tries, and the real best sits a touch below σ√(2 ln N). (This is the hidden-search point of §1: seeds and epochs are extra draws, but correlated ones.)
- **The Gaussian approximation.** The Binomial is only approximately a bell curve, and √(2 ln N) is the leading term of a longer expression. Both are close for the n and N we use, not exact.

So we do not lean on the formula as truth. We use it to see the mechanism, and we measure the real gap in §5. The two agree in shape, and that agreement is the point: the gap is not a quirk of one dataset, it is the biggest of many noisy draws, behaving as the biggest of many noisy draws must.

---

## Appendix B: How to run the code

*Outside the page limit. All the code lives in four Jupyter notebooks, one per chapter, under `notebooks/`. Each notebook runs its chapter's whole trail from top to bottom and prints every number the chapter quotes, so nothing in the body has to be taken on trust. Run any one with `jupyter nbconvert --to notebook --execute notebooks/<file>.ipynb`, or open it and run all the cells.*

| Notebook | Chapter | What it produces |
| --- | --- | --- |
| `notebooks/loan.ipynb` | §2 Loan | one client read top to bottom, the eleven split checks, the confusion counts, and balanced accuracy |
| `notebooks/market.ipynb` | §3 Market | the price-to-return-to-size transform traced day by day, the shuffle-versus-honest leak with its near-twins, and the drift under a frozen scaler |
| `notebooks/phone.ipynb` | §4 Phone | the learning-rate divergence and recovery, record-wise versus subject-wise splits, and the paired identity-leak control |
| `notebooks/lab.ipynb` | §5 The search | early stopping and seeds inflating a score out of pure noise, and the honest architecture-and-optimiser comparison |

*The notebooks use only numpy and matplotlib. The three real datasets are described in Appendix C; the network's own formulae are derived in Appendix E.*

---

## Appendix C: the three datasets, up close

*Outside the page limit. The report leans on these three datasets, so here they are as they actually arrive: where each came from, what the raw rows hold, what makes its structure the structure we claim it is, and how each becomes the task we run. Every number below is printed by the notebooks (Appendix B), which read the raw files and show one row moving through each transform.*

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

## Appendix E: the four moves, derived

*Outside the page limit. The four moves of §1, written out in full, so the network's own formulae can be checked and not taken on trust. One input `x`, one hidden layer, a guess across the classes.*

*Forward: the guess.* The hidden layer mixes the input with weights `W₁` and a bias `b₁`, then bends it with ReLU, which just replaces negatives with zero:

> a = W₁x + b₁,   h = ReLU(a).

A second set of weights turns that into one score per class, and softmax turns the scores into probabilities that add to one:

> z = W₂h + b₂,   p_k = e^(z_k) / Σⱼ e^(z_j).

That vector `p` is the guess.

*Loss: how wrong.* The true class is `c`. Cross-entropy scores the guess by how little weight it put on the truth:

> L = −log p_c.

Put all the weight on the right class and the loss is zero. Put almost none, and it shoots up.

*Gradient: which way is downhill.* This is the line worth the whole appendix. Write the loss in terms of the scores, using `p_c = e^(z_c) / Σⱼ e^(z_j)`:

> L = −z_c + log Σⱼ e^(z_j).

Now take the slope against one score `z_k`. The first term gives −1 only for the true class; the second gives `e^(z_k) / Σⱼ e^(z_j)`, which is `p_k`. So

> ∂L/∂z_k = p_k − y_k,   that is,   ∂L/∂z = p − y,

where `y` is the truth written as a one-hot vector, a 1 in the true class and 0 everywhere else. Softmax and cross-entropy fall away into something clean: the correction is just the gap between what we guessed and the truth. This is the marvel §1 pointed at.

*Backprop: pass the gap back.* That gap sits at the output. The chain rule carries it back to every weight.

![The output error flows back to every weight](figures/backprop.svg)

At the second layer the gradient is the gap times the hidden values; then we push the gap through `W₂` to the hidden layer, and through the ReLU, which lets it pass only where `a` was positive:

> ∂L/∂W₂ = (p − y) hᵀ,   ∂L/∂b₂ = p − y,
> ∂L/∂h = W₂ᵀ(p − y),   ∂L/∂a = ∂L/∂h  (only where a > 0),

and the first layer gets its gradient the same way, `∂L/∂W₁ = (∂L/∂a) xᵀ` and `∂L/∂b₁ = ∂L/∂a`. Over a batch we average these across the examples.

*Update: take the step.* Every weight moves a small step `η` against its gradient:

> W ← W − η ∂L/∂W.

That is one round. Do it a few thousand times and the guesses sharpen. The same forward and backward pass, in code, runs in the notebooks listed in Appendix B.

---
