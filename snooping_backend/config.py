"""config.py - the single source of truth for split sizes and the sealed-test size.

Every experiment (Core.md, notebooks, backend) reads split sizes from HERE, so the
same quantity never appears with two different values across the report and the code.

A split is (n_train, n_val, n_test):
  n_val  = 200 everywhere  -> the validation set is SMALL on purpose (the "snoop"):
                              a small val is noisy, so the best-of-N validation score
                              overshoots and the gap becomes visible.
  n_test = as LARGE as the data allows -> the sealed test is the ruler; the sampling
                              error of an accuracy is at most 0.5/sqrt(n_test), so a big
                              test pins the truth. Synthetic can be made huge; real
                              data cannot, so its ruler is blunter (stated honestly).

Derived (standard error of an accuracy <= 0.5/sqrt(n_test)):
  synthetic n_test=100_000 -> SE <= 0.0016   (truth pinned to ~a thousandth)
  loan      n_test= 10_000 -> SE <= 0.005
  finance   n_test=  1_000 -> SE <= 0.016     (blunt; a walk-forward split can't grow)
All are far below the gaps we measure (~0.01-0.09) EXCEPT finance, where the
comparable SE is exactly why finance is the "the ruler is blunt" warning case.
"""

#                     (n_train, n_val, n_test)
SIZES_SYNTHETIC =     (2000,    200,   100_000)
SIZES_LOAN      =     (2000,    200,    10_000)
SIZES_FINANCE   =     (4000,    200,     1_000)   # walk-forward; test = newest 1000 days
