# tests/test_linear_predictor.py
"""compile_linear_predictor: lookups for categoricals, a subset for the rest."""
import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices

from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor

LEVELS = list(range(1, 61))


def _frame(n, rng, levels=LEVELS):
    return pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "pa": rng.integers(0, 2, n).astype(float),
            "k": rng.choice(levels, n).astype(float),
        }
    )


def _design(formula, df):
    _, x = dmatrices(formula, df, NA_action="drop")
    return x


def test_eta_matches_the_full_design_product():
    """Constant + lookups + subset equals X @ coef to float64 precision."""
    rng = np.random.default_rng(1)
    train = _frame(500, rng)
    x = _design(
        f"y ~ scale(a) + C(k, levels={LEVELS}) + C(pa, levels=[0, 1]) + b:scale(a)",
        train,
    )
    coef = rng.normal(size=x.shape[1])
    pred = compile_linear_predictor(x.design_info, coef)
    new = _frame(300, rng)
    from patsy.highlevel import build_design_matrices

    (x_new,) = build_design_matrices([x.design_info], new, NA_action="raise")
    np.testing.assert_allclose(
        pred.eta(new), np.asarray(x_new) @ coef, rtol=1e-12, atol=0
    )
    assert pred.materialised_columns == 2  # scale(a), b:scale(a)
    assert pred.working_set_columns == 4


def test_levels_absent_from_the_training_frame_still_look_up_their_coefficient():
    """The table is built from levels=..., not from the levels the frame contained."""
    rng = np.random.default_rng(2)
    train = _frame(200, rng, levels=[1, 2, 3])  # level 4 never seen
    x = _design("y ~ C(k, levels=[1, 2, 3, 4])", train)
    assert "C(k, levels=[1, 2, 3, 4])[T.4]" in x.design_info.column_names
    coef = np.array([0.5, 1.0, 2.0, 3.0])
    pred = compile_linear_predictor(x.design_info, coef)
    new = pd.DataFrame({"k": [4.0, 1.0]})
    np.testing.assert_allclose(pred.eta(new), [0.5 + 3.0, 0.5])
    assert pred.materialised_columns == 0


def test_sum_coding_is_still_a_lookup():
    """Any contrast works because the table is contrast_matrix @ coef."""
    rng = np.random.default_rng(3)
    train = _frame(300, rng, levels=[1, 2, 3, 4])
    x = _design("y ~ C(k, Sum, levels=[1, 2, 3, 4])", train)
    coef = rng.normal(size=x.shape[1])
    pred = compile_linear_predictor(x.design_info, coef)
    new = _frame(50, rng, levels=[1, 2, 3, 4])
    from patsy.highlevel import build_design_matrices

    (x_new,) = build_design_matrices([x.design_info], new)
    np.testing.assert_allclose(pred.eta(new), np.asarray(x_new) @ coef, rtol=1e-12)
    assert pred.materialised_columns == 0


def test_a_categorical_that_is_not_a_plain_column_is_materialised():
    """C(k + 0) reads an expression, so it goes through patsy, and still matches."""
    rng = np.random.default_rng(4)
    train = _frame(300, rng, levels=[1, 2, 3])
    x = _design("y ~ C(k + 0, levels=[1, 2, 3])", train)
    coef = rng.normal(size=x.shape[1])
    pred = compile_linear_predictor(x.design_info, coef)
    new = _frame(40, rng, levels=[1, 2, 3])
    from patsy.highlevel import build_design_matrices

    (x_new,) = build_design_matrices([x.design_info], new)
    np.testing.assert_allclose(pred.eta(new), np.asarray(x_new) @ coef, rtol=1e-12)
    assert pred.materialised_columns == 2


def test_string_categories_are_materialised_even_when_they_parse_as_numbers():
    """Digit-string levels are not a numeric domain, so patsy keeps building them."""
    rng = np.random.default_rng(8)
    levels = ["1", "2", "3"]
    train = pd.DataFrame(
        {"y": rng.integers(0, 2, 120), "s": rng.choice(levels, 120).astype(object)}
    )
    x = _design("y ~ C(s, levels=['1', '2', '3'])", train)
    coef = rng.normal(size=x.shape[1])
    pred = compile_linear_predictor(x.design_info, coef)
    new = pd.DataFrame({"s": rng.choice(levels, 30).astype(object)})
    from patsy.highlevel import build_design_matrices

    (x_new,) = build_design_matrices([x.design_info], new)
    np.testing.assert_allclose(pred.eta(new), np.asarray(x_new) @ coef, rtol=1e-12)
    assert pred.materialised_columns == 2


def test_an_unseen_level_raises_and_names_the_factor():
    """Same failure class as patsy today: a value outside levels= is an error."""
    rng = np.random.default_rng(5)
    x = _design("y ~ C(k, levels=[1, 2, 3])", _frame(100, rng, levels=[1, 2, 3]))
    pred = compile_linear_predictor(x.design_info, np.ones(x.shape[1]))
    with pytest.raises(ValueError, match=r"C\(k, levels=\[1, 2, 3\]\).*7"):
        pred.eta(pd.DataFrame({"k": [1.0, 7.0]}))


def test_trailing_coefficients_are_ignored():
    """The iCAR betas carry extra entries after the design columns."""
    rng = np.random.default_rng(6)
    x = _design("y ~ scale(a) + C(pa, levels=[0, 1])", _frame(100, rng))
    coef = np.r_[rng.normal(size=x.shape[1]), 99.0, 98.0]
    pred = compile_linear_predictor(x.design_info, coef)
    new = _frame(20, rng)
    from patsy.highlevel import build_design_matrices

    (x_new,) = build_design_matrices([x.design_info], new)
    np.testing.assert_allclose(
        pred.eta(new), np.asarray(x_new) @ coef[: x.shape[1]], rtol=1e-12
    )


def test_describe_counts_buckets():
    """describe() says how many lookups and materialised columns there are."""
    rng = np.random.default_rng(7)
    x = _design(
        "y ~ scale(a) + C(k, levels=[1, 2, 3]) + C(pa, levels=[0, 1])",
        _frame(100, rng, levels=[1, 2, 3]),
    )
    pred = compile_linear_predictor(x.design_info, np.ones(x.shape[1]))
    assert pred.describe() == "2 lookup term(s), 1 materialised col(s)"
