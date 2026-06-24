"""lab.py - the synthetic laboratory (known-truth data).

Builds a Gaussian dataset and the supervisor's three labelling regimes, so the
true performance is known by construction and the gap can be measured exactly.

Design note (locked): EVERYTHING is a variable. Each repeat re-samples fresh
train / validation / sealed-test from the generator. The sealed test stays LARGE
and is opened only once, after selection - re-drawing it is fine, leaking it is not.

------------------------------------------------------------------------------
Functions to implement (contract - bodies are yours):

  make_X(n, d, rng) -> X : (n, d) array of i.i.d. N(0, 1)
      tool : rng.standard_normal((n, d))   (rng passed in, never seeded inside)
      check: X.shape == (n, d); column mean ~ 0; column std ~ 1

  labels_random(n, rng) -> y : random labels, independent of X      [Case 1]
      meaning: no signal -> true generalisation = chance (~50%)
      check  : class balance ~ 50/50

  labels_sign(X) -> y = sign(X[:, 0])                                [Case 2]
      meaning: a clean, axis-aligned linear rule on feature 0
      check  : logistic regression ~ 100% when flip_y = 0 (linearly separable)

  random_isometry(d, rng) -> R : (d, d) orthogonal (rotation +/- reflection)
      tool : QR factorisation of a random Gaussian matrix
      check: R @ R.T ~ I

  rotate(X, R) -> X @ R                                              [Case 3]
      meaning: same rule as Case 2, now hidden along an oblique direction
      check  : kNN accuracy on Case 3 == kNN accuracy on Case 2
               (an isometry preserves distances - this is the supervisor's
                insight, proven by numbers)

  inject_noise(y, flip_y, rng) -> y with a fraction flip_y of labels flipped
      check: fraction actually flipped ~ flip_y

  make_dataset(case, n, d, flip_y, sizes, rng)
        -> (X_train, y_train), (X_val, y_val), (X_test, y_test)
      val is SMALL (snooping engine); test is LARGE (truth ~ no sampling error).
      re-sample fresh on every call.
------------------------------------------------------------------------------
"""
