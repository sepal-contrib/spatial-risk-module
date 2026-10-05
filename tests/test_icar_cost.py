"""Rough iCAR training-time estimate behind the New model dialog's warning.

forestatrisk's sampler recomputes the whole linear predictor for every
coefficient it proposes, so one MCMC iteration costs about
n_obs * (1.7 ns * n_coef**2 + 36 ns * n_coef) (benchmarked 2026-10-01: 18.3 ms
per iteration at 20k samples x 15 coefficients, 41.3 ms at 10k x 40). Every
category of a C() term adds a coefficient, so a many-level categorical such as
a sub-jurisdiction map dominates. The expected values below are worked by hand
from that formula.
"""

import pytest

from spatialrisk.mlmodels.icar_cost import (
    count_coefficients,
    estimate_icar_training,
    estimate_mcmc_seconds,
)

FORMULA = "I(defor) + trial ~ scale(altitude) + scale(roads) + C(subj) + C(pa)"
LEVELS = {"subj": 106, "pa": 2}


def test_mcmc_seconds_match_the_benchmark():
    """One fitted point reproduces its own measurement."""
    # 8000 * 20000 * (1.7e-9 * 225 + 36e-9 * 15) = 147.6 s (measured: 146 s)
    assert estimate_mcmc_seconds(20_000, 15, 8000) == pytest.approx(147.6)


def test_each_category_beyond_the_reference_is_one_coefficient():
    """Treatment coding: levels - 1 columns per C() term."""
    # Intercept + 2 continuous + (106 - 1) + (2 - 1)
    assert count_coefficients(FORMULA, LEVELS) == 109


def test_without_an_intercept_the_first_categorical_keeps_every_level():
    """Without an intercept patsy gives the first factor full rank."""
    assert count_coefficients("y ~ 0 + C(subj) + x", {"subj": 5}) == 6


def test_bare_numeric_use_of_a_categorical_layer_is_one_column():
    """Only C() expands a layer into levels; a bare name is one column."""
    assert count_coefficients("y ~ subj", {"subj": 106}) == 2


def test_the_cell_term_forestatrisk_consumes_is_not_a_coefficient():
    """The trailing cell column is stripped by forestatrisk before sampling."""
    assert count_coefficients("y + trial ~ x + cell", {}) == 2


def test_many_level_categorical_on_a_large_sample_is_slow_and_named():
    """The Peruvian Amazon case: about 69 min, subj named with its speedup."""
    # 113 coefficients -> 8000 * 20000 * (1.7e-9 * 12769 + 36e-9 * 113) = 4124 s
    formula = "y ~ " + " + ".join(f"scale(x{i})" for i in range(6)) + " + C(subj)"
    est = estimate_icar_training(formula, {"subj": 107}, n_obs=20_000, iterations=8000)

    assert est.seconds == pytest.approx(4124, rel=1e-3)
    assert est.is_slow
    top = est.culprits[0]
    assert (top.name, top.levels) == ("subj", 107)
    # Without subj: 7 coefficients -> 1.7e-9 * 49 + 36e-9 * 7 = 3.353e-7 s per
    # sample-iteration, against 2.5775e-5 with it.
    assert top.speedup == pytest.approx(76.87, rel=1e-3)


def test_same_formula_on_a_small_sample_trains_fast():
    """Sample size matters as much as levels: 400 points train in seconds."""
    # MTQ: 34 subj levels but 400 samples -> about 14 s, no warning.
    formula = "y ~ " + " + ".join(f"scale(x{i})" for i in range(7)) + " + C(subj)"
    est = estimate_icar_training(formula, {"subj": 34}, n_obs=400, iterations=8000)

    assert est.seconds < 60
    assert not est.is_slow


def test_binary_categoricals_are_not_culprits():
    """A two-level layer barely moves the estimate, so it is not named."""
    est = estimate_icar_training(FORMULA, LEVELS, n_obs=20_000, iterations=8000)

    assert [c.name for c in est.culprits] == ["subj"]


def test_unparsable_formula_gives_no_estimate():
    """A half-typed formula yields no estimate rather than an exception."""
    assert estimate_icar_training("y ~ C(", LEVELS, n_obs=1000, iterations=100) is None
