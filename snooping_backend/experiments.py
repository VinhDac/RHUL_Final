"""experiments.py - the two new experiments that need bespoke search loops.

The rest of the report reuses pipeline.run_once / sweep / gap_kfold unchanged. These
two need model checkpoints during training (E-2) or two whole procedures side by side
(H5), so they live here. Both are deterministic given the rng seed.

  hidden_n_levels(...) -> the E-2 result (Core.md section 6.3): the gap as the hidden
                          search dimensions (seed, epoch) are counted one at a time.
  remedy_AB(...)       -> the H5 result (Core.md section 6.6): an aggressive procedure
                          (A) vs an honest one (B), both revealed on the sealed test.
"""
import copy
import numpy as np
import torch

from snooping_backend.lab import make_dataset
from snooping_backend.mlp import make_mlp, accuracy, sample_config
from snooping_backend.pipeline import gap_kfold


def _train_checkpoints(X, y, width, lr, seed, checkpoints, epochs=300):
    """Train one MLP and return a copy of the model at each epoch in `checkpoints`."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model = make_mlp(X.shape[1], width)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    snaps = []
    for e in range(1, epochs + 1):
        opt.zero_grad(); loss_fn(model(Xt), yt).backward(); opt.step()
        if e in checkpoints:
            snaps.append(copy.deepcopy(model))
    return snaps


def hidden_n_levels(case, d, sizes, rng, n_configs=5, n_seeds=5,
                    checkpoints=(50, 100, 150, 200, 250, 300), R=20):
    """E-2 (section 6.3): keep the search at `n_configs` configurations, then add the
    hidden dimensions one at a time and measure the gap at each level.

    Level 1 counts configs only; level 2 also keeps the best of `n_seeds` seeds; level 3
    also keeps the best of the epoch checkpoints. Returns a dict with the mean gap and the
    effective N at each level -> {'config': (N_eff, gap), 'seed': ..., 'epoch': ...}.
    """
    checkpoints = tuple(checkpoints)
    last = len(checkpoints) - 1
    g1, g2, g3 = [], [], []
    for _ in range(R):
        (Xtr, ytr), (Xv, yv), (Xte, yte) = make_dataset(case, d, 0.0, sizes, rng)
        cands = []                                   # (val_acc, model, seed_index, ckpt_index)
        configs = [sample_config(rng) for _ in range(n_configs)]
        for cfg in configs:
            w, lr = cfg["width"], cfg["lr"]
            for si in range(n_seeds):
                seed = int(rng.integers(0, 2**31 - 1))
                for ci, m in enumerate(_train_checkpoints(Xtr, ytr, w, lr, seed, checkpoints)):
                    cands.append((accuracy(m, Xv, yv), m, si, ci))

        def gap_of(subset):
            val, m = max(((c[0], c[1]) for c in subset), key=lambda t: t[0])
            return val - accuracy(m, Xte, yte)

        g1.append(gap_of([c for c in cands if c[2] == 0 and c[3] == last]))   # configs only
        g2.append(gap_of([c for c in cands if c[3] == last]))                 # + seeds
        g3.append(gap_of(cands))                                              # + epochs
    return {"config": (n_configs, float(np.mean(g1))),
            "seed":   (n_configs * n_seeds, float(np.mean(g2))),
            "epoch":  (n_configs * n_seeds * len(checkpoints), float(np.mean(g3)))}


def remedy_AB(case, d, flip_y, sizes, rng, n_a=20, seeds_a=3,
              checkpoints=(50, 100, 150, 200, 250, 300), n_b=10, R=10):
    """H5 (section 6.6): compare an AGGRESSIVE procedure A (search n_a configs, keeping the
    best of seeds and epoch checkpoints on a small validation set) against an HONEST one B
    (search n_b configs by 5-fold cross-validation, no seed/epoch fishing), on the SAME
    split, revealing the sealed test on each. Returns means: {'A': {...}, 'B': {...}}.
    """
    checkpoints = tuple(checkpoints)
    A_app, A_true, B_app, B_true = [], [], [], []
    for _ in range(R):
        (Xtr, ytr), (Xv, yv), (Xte, yte) = make_dataset(case, d, flip_y, sizes, rng)
        best_val, best = -1.0, None                                   # A: aggressive
        for _ in range(n_a):
            c = sample_config(rng)
            for _s in range(seeds_a):
                seed = int(rng.integers(0, 2**31 - 1))
                for m in _train_checkpoints(Xtr, ytr, c["width"], c["lr"], seed, checkpoints):
                    v = accuracy(m, Xv, yv)
                    if v > best_val:
                        best_val, best = v, m
        A_app.append(best_val); A_true.append(accuracy(best, Xte, yte))
        rb = gap_kfold(lambda r: ((Xtr, ytr), (Xv, yv), (Xte, yte)), n_b, rng)   # B: honest
        B_app.append(rb["apparent"]); B_true.append(rb["true"])
    mean = lambda a: float(np.mean(a))
    return {"A": {"apparent": mean(A_app), "true": mean(A_true), "gap": mean(A_app) - mean(A_true)},
            "B": {"apparent": mean(B_app), "true": mean(B_true), "gap": mean(B_app) - mean(B_true)}}
