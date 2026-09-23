# tests/test_linear_predictor.py
"""compile_linear_predictor: lookups for categoricals, a subset for the rest."""
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices

from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor

LEVELS = list(range(1, 61))
L106 = list(range(1, 107))
PERU_LIKE = (
    " + ".join(f"scale(x{i})" for i in range(1, 8))
    + f" + C(k, levels={L106}) + C(pa, levels=[0, 1])"
)


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
    # eta + max(lookup 4, 2 matrix cols + 3 x 2 factors (scale(a), b) + 2 build)
    assert pred.working_set_columns == 11


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


def _float_frame(n, rng, n_numeric=7):
    """All-float64 columns, built the way the inference engine builds block_df."""
    cols = {f"x{i}": rng.normal(size=n) for i in range(1, n_numeric + 1)}
    cols["pa"] = rng.integers(0, 2, n).astype(float)
    cols["k"] = rng.choice(L106, n).astype(float)
    cols["y"] = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame(cols)


def _eta_peak_bytes(pred, df):
    """Tracemalloc peak of one eta() call, above what was live before it."""
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        pred.eta(df)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        if started:
            tracemalloc.stop()


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param(f"y ~ scale(x1) + C(k, levels={L106})", id="golden"),
        pytest.param(f"y ~ {PERU_LIKE}", id="peru-like"),
        pytest.param(
            f"y ~ C(k, levels={L106}) + C(pa, levels=[0, 1])", id="lookup-only"
        ),
    ],
)
def test_working_set_columns_covers_the_measured_eta_peak(formula):
    """The planner's charge covers eta()'s real peak without grossly overshooting."""
    rng = np.random.default_rng(9)
    x = _design(formula, _float_frame(2000, rng))
    pred = compile_linear_predictor(x.design_info, rng.normal(size=x.shape[1]))
    n_rows = 300_000
    new = _float_frame(n_rows, rng)
    pred.eta(new.iloc[:100])  # warm patsy's lazy state outside the measurement
    measured = _eta_peak_bytes(pred, new)
    charged = pred.working_set_columns * 8 * n_rows
    assert measured <= charged, (measured / n_rows, pred.working_set_columns)
    assert charged <= 2.5 * measured, (measured / n_rows, pred.working_set_columns)


def test_working_set_columns_does_not_grow_with_the_level_count():
    """A 106-level lookup is charged exactly what a 2-level one is."""
    rng = np.random.default_rng(10)
    train = _float_frame(2000, rng)
    wide = _design(f"y ~ scale(x1) + C(k, levels={L106})", train)
    narrow = _design("y ~ scale(x1) + C(pa, levels=[0, 1])", train)
    pred_wide = compile_linear_predictor(wide.design_info, np.ones(wide.shape[1]))
    pred_narrow = compile_linear_predictor(narrow.design_info, np.ones(narrow.shape[1]))
    assert pred_wide.working_set_columns == pred_narrow.working_set_columns
