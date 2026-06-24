"""Data-snooping research harness.

Measure how much of a model's apparent improvement from searching configurations
is real, and how much is luck. validation_score = true_performance + luck_noise;
keeping the best of N is keeping the luckiest, so the best validation score is
inflated. We MEASURE the inflation with a sealed test set:

    gap = best_validation_score - sealed_test_score
"""
