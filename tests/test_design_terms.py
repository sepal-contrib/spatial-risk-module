# tests/test_design_terms.py
"""plain_categorical: which patsy terms can skip patsy's per-value level loop."""
import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices

from spatialrisk.mlmodels.design_terms import (
    is_numeric_domain,
    level_positions,
    plain_categorical,
)


def _terms(formula, df):
    """The fitted design info and its terms keyed by name, with their subterms."""
    _, x = dmatrices(formula, df, NA_action="drop")
    info = x.design_info
    return info, {t.name(): (t, s) for t, s in info.term_codings.items()}


def test_a_bare_numeric_categorical_is_plain_and_keeps_patsys_contrast():
    """Levels stay in patsy's order, and the contrast rows follow them."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"y": rng.integers(0, 2, 60), "k": rng.choice([3, 1, 2], 60).astype(float)}
    )
    info, terms = _terms("y ~ C(k, levels=[3, 1, 2])", df)
    plain = plain_categorical(info, *terms["C(k, levels=[3, 1, 2])"])
    assert plain.column == "k"
    assert plain.code == "C(k, levels=[3, 1, 2])"
    np.testing.assert_array_equal(plain.levels, [3.0, 1.0, 2.0])
    # Treatment coding against the first level: level 3 is all zeros.
    np.testing.assert_array_equal(plain.contrast, [[0, 0], [1, 0], [0, 1]])


def test_expressions_interactions_numerics_and_string_domains_are_not_plain():
    """Only a single bare C(<column>) on a numeric domain qualifies."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, 90),
            "k": rng.choice([1, 2, 3], 90).astype(float),
            "a": rng.normal(size=90),
            "s": rng.choice(["1", "2"], 90).astype(object),
        }
    )
    info, terms = _terms(
        "y ~ C(k + 0, levels=[1, 2, 3]) + C(k, levels=[1, 2, 3]):a"
        " + C(s, levels=['1', '2']) + a",
        df,
    )
    names = set(terms) - {"Intercept"}
    assert len(names) == 4
    for name in names:
        assert plain_categorical(info, *terms[name]) is None, name


def test_is_numeric_domain():
    """Finite, duplicate-free numbers only; digit strings are not numbers."""
    assert is_numeric_domain((1, 2, 3))
    assert is_numeric_domain((0.5, 2.0))
    assert not is_numeric_domain(("1", "2"))
    assert not is_numeric_domain((1, 1.0))
    assert not is_numeric_domain((1, float("nan")))
    assert is_numeric_domain((1, 2**53))
    assert not is_numeric_domain((1, 2**53 + 1))
    assert not is_numeric_domain((1, np.int64(2**53 + 1)))


def test_level_positions_match_exactly_and_report_misses_by_factor():
    """Unsorted values find their level; a miss, even past the top, is reported."""
    levels = np.array([0.5, 1.0, 2.0, 7.0])
    column = np.array([7.0, 0.5, 2.0, 1.0, 7.0, 1.0])
    np.testing.assert_array_equal(
        level_positions(levels, column, "C(k)"), [3, 0, 2, 1, 3, 1]
    )
    # 9.0 sorts past the last level: clamped and reported, not an IndexError.
    exact = r"^C\(k\): values not among the training levels: \[1\.5, 9\.0\]$"
    with pytest.raises(ValueError, match=exact):
        level_positions(levels, np.array([1.0, 9.0, 1.5, 9.0]), "C(k)")
