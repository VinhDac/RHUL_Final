"""test_lab.py - sanity checks for the lab.

These are not just tests: the rotate() check IS the supervisor's isometry
insight, proven by numbers. Keep them as plain asserts - no framework needed.

Run from the repo root:   python -m tests.test_lab

------------------------------------------------------------------------------
Checks to implement (bodies are yours):

  make_X            -> shape == (n, d); column mean ~ 0; column std ~ 1
  labels_random     -> class balance ~ 50/50
  labels_sign       -> logistic regression ~ 100% accuracy when flip_y = 0
  random_isometry   -> R @ R.T ~ I (allclose with a tolerance)
  rotate            -> kNN accuracy on Case 3 == kNN accuracy on Case 2
                       (the distances are preserved by the rotation)
  inject_noise      -> fraction of labels actually flipped ~ flip_y
------------------------------------------------------------------------------
"""
