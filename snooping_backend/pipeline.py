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
from snooping_backend.lab import make_dataset
from snooping_backend.mlp import train, accuracy, sample_config


def run_once(case, N, sizes, d, flip_y, rng, epochs=300):
    """One full gap measurement (the machine of Core.md section 3).

    split -> search N MLP configs (score on the SMALL val) -> keep best-on-val
    -> reveal the LARGE sealed test once, on the winner only -> gap.

    sizes = (n_train, n_val, n_test). Returns dict(apparent, true, gap, config).
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_dataset(case, d, flip_y, sizes, rng)

    best_val, best_model, best_config = -1.0, None, None
    for _ in range(N):
        config = sample_config(rng)
        seed = int(rng.integers(0, 2**31 - 1))         # each config gets its own init
        model = train(X_train, y_train, config["width"], config["lr"], epochs=epochs, seed=seed)
        val = accuracy(model, X_val, y_val)             # SELECTION uses validation only
        if val > best_val:
            best_val, best_model, best_config = val, model, config

    # sealed test touched EXACTLY ONCE, on the winner only (never used to select):
    true = accuracy(best_model, X_test, y_test)
    return {"apparent": best_val, "true": true, "gap": best_val - true, "config": best_config}
