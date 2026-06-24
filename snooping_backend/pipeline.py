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
Functions to implement (contract - bodies are yours):

  run_once(case, N, flip_y, sizes, rng)
        -> dict(best_val, true, gap, winner_config)
      one full split -> search -> select -> reveal.

  repeat(setting, R, rng) -> list of run_once results
      everything re-drawn each repeat (train, val, test, configs); then we
      average. gap(setting) = mean over R; also record spread (std or min-max).

  sweep(knob, values, R, rng) -> mean + spread of gap per value
      E1 = sweep N over e.g. [1, 2, 5, 10, 20, 50, 100, 200].
      later: sweep flip_y (noise), capacity, selection protocol.

  log_run(record, path) -> append seed + config + both scores + gap to a CSV
      reproducible and auditable.
------------------------------------------------------------------------------
"""
