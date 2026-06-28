"""pipeline.py - the one machine (the snooping experiment).

    split
      -> search N configs (train on train, score on the SMALL validation set)
      -> keep the best-on-validation config            (= the snoop)
      -> reveal it once on the LARGE sealed test set    (= true performance)
      -> gap = best_validation_score - sealed_test_score

The same machine serves every experiment; an experiment just varies one knob.

DISCIPLINE: the sealed test is used ONLY to reveal, never to select / tune /
early-stop. Any leak invalidates the measurement.

------------------------------------------------------------------------------
  run_once(case, N, sizes, d, flip_y, rng, epochs)   -> implemented below
        -> dict(apparent, true, gap, config)
      one full split -> search -> select -> reveal.

  repeat / sweep / log_run  -> Phan 5 (the headline sweep + logging).
------------------------------------------------------------------------------
"""
import numpy as np

from snooping_backend.lab import make_dataset
from snooping_backend.mlp import train, accuracy, sample_config


def run_once(make_splits, N, rng, epochs=300, sample=sample_config):
    """One full gap measurement (the machine of Core.md section 3), for ANY data source.

    make_splits(rng) -> (X_train, y_train), (X_val, y_val), (X_test, y_test).
    Search N MLP configs (score on the SMALL val) -> keep best-on-val -> reveal the
    LARGE sealed test once, on the winner only -> gap. Returns dict(apparent, true, gap, config).
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(rng)

    best_val, best_model, best_config = -1.0, None, None
    for _ in range(N):
        config = sample(rng)
        seed = int(rng.integers(0, 2**31 - 1))         # each config gets its own init
        model = train(X_train, y_train, config["width"], config["lr"], epochs=epochs, seed=seed)
        val = accuracy(model, X_val, y_val)             # SELECTION uses validation only
        if val > best_val:
            best_val, best_model, best_config = val, model, config

    # sealed test touched EXACTLY ONCE, on the winner only (never used to select):
    true = accuracy(best_model, X_test, y_test)
    return {"apparent": best_val, "true": true, "gap": best_val - true, "config": best_config}


def sweep(make_splits, N_values, rng, R=30, epochs=300, sample=sample_config):
    """The headline sweep: for each N, the mean gap over R repeats (Core.md section 5).

    make_splits(rng) -> (train, val, test); called fresh each repeat (a new synthetic
    sample, or a re-shuffled split of fixed real data). Cumulative: each repeat trains
    ONE pool of max(N_values) configs and records each config's validation score; then
    for each N it keeps the best-of-the-first-N by VALIDATION and reveals the sealed
    test ONLY on that winner. Returns N -> {apparent, true, gap, gap_std} (means over R).
    """
    N_max = max(N_values)
    apparent = {N: [] for N in N_values}
    true = {N: [] for N in N_values}
    gap = {N: [] for N in N_values}

    for _ in range(R):
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = make_splits(rng)

        models, vals = [], []
        for _ in range(N_max):
            cfg = sample(rng)
            seed = int(rng.integers(0, 2**31 - 1))
            m = train(X_tr, y_tr, cfg["width"], cfg["lr"], epochs=epochs, seed=seed)
            models.append(m)
            vals.append(accuracy(m, X_val, y_val))            # validation only
        vals = np.array(vals)

        test_of = {}                                          # winner index -> sealed-test score
        for N in N_values:
            i = int(np.argmax(vals[:N]))                      # best-of-first-N, by validation
            if i not in test_of:
                test_of[i] = accuracy(models[i], X_te, y_te)  # test revealed ONLY on a winner
            a, t = float(vals[i]), test_of[i]
            apparent[N].append(a); true[N].append(t); gap[N].append(a - t)

    return {N: {"apparent": float(np.mean(apparent[N])),
                "true": float(np.mean(true[N])),
                "gap": float(np.mean(gap[N])),
                "gap_std": float(np.std(gap[N]))}
            for N in N_values}


def synthetic_splits(case, d, flip_y, sizes):
    """A make_splits provider for the synthetic lab (fresh sample on every call).

    Real data (loan, finance) will supply its own make_splits provider; the machine
    (run_once / sweep) is unchanged - it just receives a different one.
    """
    return lambda rng: make_dataset(case, d, flip_y, sizes, rng)
