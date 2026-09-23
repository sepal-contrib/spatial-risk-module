# tests/test_linear_predictor.py
"""compile_linear_predictor: lookups for categoricals, a subset for the rest."""
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices
from threadpoolctl import threadpool_limits

from spatialrisk.mlmodels import linear_predictor
from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor

LEVELS = list(range(1, 61))
L106 = list(range(1, 107))
PERU_LIKE = (
    " + ".join(f"scale(x{i})" for i in range(1, 8))
    + f" + C(k, levels={L106}) + C(pa, levels=[0, 1])"
)
TEN_NUMERICS = " + ".join(f"scale(x{i})" for i in range(1, 11))


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
    # Per pixel only the eta vector; per chunk row max(lookup 4, 2 matrix cols
    # + 3 x 2 factors (scale(a), b) + 2 build).
    assert pred.working_set_columns == 1
    assert pred.chunk_scratch_bytes == linear_predictor._ETA_CHUNK_ROWS * 10 * 8


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


def test_an_unseen_level_past_the_first_chunk_still_raises(monkeypatch):
    """Every chunk is checked; the error lists the offending chunk's values."""
    rng = np.random.default_rng(13)
    x = _design("y ~ C(k, levels=[1, 2, 3])", _frame(100, rng, levels=[1, 2, 3]))
    pred = compile_linear_predictor(x.design_info, np.ones(x.shape[1]))
    monkeypatch.setattr(linear_predictor, "_ETA_CHUNK_ROWS", 4)
    k = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 7.0, 9.0]  # chunk 3 is bad
    names_the_chunk = r"C\(k, levels=\[1, 2, 3\]\).*\[7\.0, 9\.0\]"
    with pytest.raises(ValueError, match=names_the_chunk):
        pred.eta(pd.DataFrame({"k": k}))


def test_an_empty_frame_gives_an_empty_float64_eta():
    """Zero rows run no chunk and still return a float64 vector."""
    rng = np.random.default_rng(12)
    x = _design(f"y ~ {PERU_LIKE}", _float_frame(500, rng))
    pred = compile_linear_predictor(x.design_info, rng.normal(size=x.shape[1]))
    eta = pred.eta(_float_frame(10, rng).iloc[:0])
    assert eta.shape == (0,)
    assert eta.dtype == np.float64


def test_an_integer_lookup_column_matches_its_float_twin(monkeypatch):
    """An int column is cast chunk by chunk and gives the float column's eta."""
    rng = np.random.default_rng(14)
    formula = f"y ~ scale(x1) + C(k, levels={L106}) + C(pa, levels=[0, 1])"
    x = _design(formula, _float_frame(500, rng))
    pred = compile_linear_predictor(x.design_info, rng.normal(size=x.shape[1]))
    monkeypatch.setattr(linear_predictor, "_ETA_CHUNK_ROWS", 64)
    as_float = _float_frame(1000, rng)
    as_int = as_float.astype({"k": np.int64, "pa": np.int64})
    assert np.array_equal(pred.eta(as_int), pred.eta(as_float))


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
        pytest.param(f"y ~ {PERU_LIKE}", id="7-numeric-2-lookup"),
        pytest.param(
            f"y ~ scale(x1) + x2:scale(x1) + C(k + 0, levels={L106})"
            " + C(pa, levels=[0, 1])",
            id="interaction-and-materialised-categorical",
        ),
        pytest.param(f"y ~ {TEN_NUMERICS}", id="10-numeric"),
    ],
)
def test_row_chunks_do_not_change_a_single_bit_of_eta(formula, monkeypatch):
    """eta() in row chunks equals one whole-frame pass, bit for bit.

    BLAS is pinned to one thread, as the engine pins it: a threaded gemv splits
    the rows at thread boundaries and is not reproducible across thread counts
    even without chunking (seen on the 10-numeric design).
    """
    rng = np.random.default_rng(11)
    x = _design(formula, _float_frame(2000, rng, n_numeric=10))
    pred = compile_linear_predictor(x.design_info, rng.normal(size=x.shape[1]))
    with threadpool_limits(limits=1, user_api="blas"):
        for chunk in (linear_predictor._ETA_CHUNK_ROWS, 4096, 1024):
            # 3.5 chunks and 3 rows: the last chunk is short and not a
            # multiple of 4.
            n_rows = 3 * chunk + chunk // 2 + 3
            new = _float_frame(n_rows, rng, n_numeric=10)
            monkeypatch.setattr(linear_predictor, "_ETA_CHUNK_ROWS", n_rows)
            reference = pred.eta(new)
            monkeypatch.setattr(linear_predictor, "_ETA_CHUNK_ROWS", chunk)
            assert np.array_equal(pred.eta(new), reference), chunk


def test_chunks_reach_patsy_as_pandas_rows(monkeypatch):
    """Chunked eta() equals patsy's whole-frame build, bit for bit.

    scale() keeps float128 state: pandas rounds (x - mean) / sd to float64 once,
    numpy's in-place operators round twice, so numpy rows would move the last
    ulp of about a quarter of the rows.
    """
    rng = np.random.default_rng(15)
    x = _design(f"y ~ {TEN_NUMERICS} - 1", _float_frame(2000, rng, n_numeric=10))
    coef = rng.normal(size=x.shape[1])
    pred = compile_linear_predictor(x.design_info, coef)
    monkeypatch.setattr(linear_predictor, "_ETA_CHUNK_ROWS", 1024)
    new = _float_frame(3 * 1024 + 5, rng, n_numeric=10)
    from patsy.highlevel import build_design_matrices

    with threadpool_limits(limits=1, user_api="blas"):
        (x_new,) = build_design_matrices([x.design_info], new, NA_action="raise")
        assert np.array_equal(pred.eta(new), np.asarray(x_new) @ coef)


# tracemalloc also counts the Python objects patsy and pandas churn per chunk
# (reference cycles in pandas' arithmetic, patsy's recompiled factor code)
# until the cyclic GC takes them back: a few KiB per chunk, measured as up to
# +0.08 B/row on the slope at the 32768-row chunk used below. Half a byte per
# row is still less than the smallest per-pixel array eta() could leave behind
# (a 1-byte bool mask).
_PY_CHURN_BYTES_PER_ROW = 0.5


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
def test_working_set_columns_covers_the_measured_eta_peak(formula, monkeypatch):
    """Per-pixel charge plus the bounded chunk scratch cover eta()'s real peak.

    Measured at 4 and 16 chunks: (a) each peak fits working_set_columns per
    pixel plus chunk_scratch_bytes; (b) the peak grows by no more than
    working_set_columns per added row, so the scratch no longer scales with the
    stripe; and the charge does not grossly overshoot the peak.
    """
    # A chunk smaller than production keeps the frames small;
    # chunk_scratch_bytes scales with the same constant.
    monkeypatch.setattr(linear_predictor, "_ETA_CHUNK_ROWS", 32768)
    rng = np.random.default_rng(9)
    x = _design(formula, _float_frame(2000, rng))
    pred = compile_linear_predictor(x.design_info, rng.normal(size=x.shape[1]))
    chunk = linear_predictor._ETA_CHUNK_ROWS
    n1, n2 = 4 * chunk, 16 * chunk
    frames = {n1: _float_frame(n1, rng), n2: _float_frame(n2, rng)}
    for df in frames.values():
        pred.eta(df)  # warm patsy's lazy state, pandas' caches and free lists
    peaks = {n: _eta_peak_bytes(pred, df) for n, df in frames.items()}
    per_px = pred.working_set_columns * 8
    for n, peak in peaks.items():
        assert peak <= per_px * n + pred.chunk_scratch_bytes, (n, peak / n)
    slope = (peaks[n2] - peaks[n1]) / (n2 - n1)
    assert slope <= per_px + _PY_CHURN_BYTES_PER_ROW, slope
    charged = per_px * n2 + pred.chunk_scratch_bytes
    assert charged <= 2.5 * peaks[n2], (charged, peaks[n2])


def test_working_set_columns_does_not_grow_with_the_level_count():
    """A 106-level lookup is charged exactly what a 2-level one is."""
    rng = np.random.default_rng(10)
    train = _float_frame(2000, rng)
    wide = _design(f"y ~ scale(x1) + C(k, levels={L106})", train)
    narrow = _design("y ~ scale(x1) + C(pa, levels=[0, 1])", train)
    pred_wide = compile_linear_predictor(wide.design_info, np.ones(wide.shape[1]))
    pred_narrow = compile_linear_predictor(narrow.design_info, np.ones(narrow.shape[1]))
    assert pred_wide.working_set_columns == pred_narrow.working_set_columns
    assert pred_wide.chunk_scratch_bytes == pred_narrow.chunk_scratch_bytes
