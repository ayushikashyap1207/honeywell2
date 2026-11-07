import numpy as np

from monitoring.drift import population_stability_index


def test_psi_is_near_zero_for_identical_distributions():
    values = np.linspace(0, 1, 500)
    assert population_stability_index(values, values) < 0.01


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 2000)
    shifted = rng.normal(2, 1, 2000)

    assert population_stability_index(reference, shifted) > 0.25
