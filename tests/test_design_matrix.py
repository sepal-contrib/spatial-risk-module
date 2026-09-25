# tests/test_design_matrix.py
"""compile_design_builder: patsy's design matrix as float32 row chunks."""
import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices
from patsy.highlevel import build_design_matrices

from spatialrisk.mlmodels import design_matrix
from spatialrisk.mlmodels.design_matrix import compile_design_builder

L106 = list(range(1, 107))
L117 = list(range(1, 118))
FIVE_NUMERICS = " + ".join(f"scale(x{i})" for i in range(1, 6))

FORMULAS = {
    "bol-like": f"y ~ {FIVE_NUMERICS} + C(pa, levels=[0, 1]) + C(k, levels={L117})",
    "no-intercept": f"y ~ C(k, levels={L106}) + scale(x1) - 1",
    "sum-coding": f"y ~ C(k, Sum, levels={L106}) + scale(x1)",
    "bare-C": "y ~ C(k) + scale(x1)",
    "interaction-and-materialised-categorical": (
        "y ~ scale(x1) + x2:scale(x1) + C(pa, levels=[0, 1]):scale(x3)"
        f" + C(k + 0, levels={L106})"
    ),
    "plain-numeric": "y ~ x1 + x2 + np.log(np.abs(x3) + 1)",
    "only-categorical": f"y ~ C(k, levels={L106})",
    "intercept-only": "y ~ 1",
    "same-column-in-two-C-terms": (
        f"y ~ C(k, levels={L106}) + C(k, Sum, levels={L106}) + scale(x1)"
    ),
    "categorical-and-its-interaction": (
        "y ~ C(pa, levels=[0, 1]) + C(pa, levels=[0, 1]):scale(x1)"
    ),
    "float-levels": "y ~ C(f, levels=[0.5, 1.5, 2.5]) + scale(x1) - 1",
    "reversed-levels": f"y ~ C(k, levels={L106[::-1]}) + scale(x1)",
}


def _frame(n, rng, levels=L106):
    """All-float64 columns, built the way the inference engine builds block_df."""
    cols = {f"x{i}": rng.normal(size=n) for i in range(1, 8)}
    cols["pa"] = rng.integers(0, 2, n).astype(float)
    cols["k"] = rng.choice(levels, n).astype(float)
    cols["y"] = rng.integers(0, 2, n).astype(float)
    cols["f"] = rng.choice([0.5, 1.5, 2.5], n)
    return pd.DataFrame(cols)


def _builder(formula, rng, levels=L106):
    """The design info patsy fits on a training frame, and its compiled builder."""
    _, x = dmatrices(formula, _frame(3000, rng, levels), NA_action="drop")
    return x.design_info, compile_design_builder(x.design_info)


def _stack(builder, df):
    """Every chunk of ``df``'s design, stacked back into one matrix."""
    parts = [x.copy() for _, _, x in builder.chunks(df)]
    if not parts:
        return np.empty((0, builder.n_columns), dtype=np.float32)
    return np.concatenate(parts)


@pytest.mark.parametrize("name", sorted(FORMULAS))
def test_chunks_are_patsys_matrix_cast_to_float32_bit_for_bit(name):
    """Any chunk size gives exactly what sklearn made of patsy's whole matrix."""
    rng = np.random.default_rng(3)
    levels = L117 if name == "bol-like" else L106
    design_info, builder = _builder(FORMULAS[name], rng, levels)
    new = _frame(5003, rng)
    (whole,) = build_design_matrices([design_info], new, NA_action="raise")
    expected = np.asarray(whole, dtype=np.float32)
    for rows in (builder.chunk_rows, 1000, 7):
        builder.chunk_rows = rows
        got = _stack(builder, new)
        assert got.dtype == np.float32
        assert got.flags.c_contiguous
        assert np.array_equal(got, expected), rows


def test_an_unseen_level_raises_and_names_the_factor():
    """Same failure class as patsy: a value outside levels= is an error."""
    rng = np.random.default_rng(5)
    _, builder = _builder("y ~ C(k, levels=[1, 2, 3])", rng, [1, 2, 3])
    with pytest.raises(ValueError, match=r"C\(k, levels=\[1, 2, 3\]\).*7\.0"):
        _stack(builder, pd.DataFrame({"k": [1.0, 7.0]}))


def test_an_unseen_level_past_the_first_chunk_still_raises():
    """Every chunk is checked; the error lists the offending chunk's values."""
    rng = np.random.default_rng(13)
    _, builder = _builder("y ~ C(k, levels=[1, 2, 3])", rng, [1, 2, 3])
    builder.chunk_rows = 4
    k = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 7.0, 9.0]  # chunk 3 is bad
    names_the_chunk = r"C\(k, levels=\[1, 2, 3\]\).*\[7\.0, 9\.0\]"
    with pytest.raises(ValueError, match=names_the_chunk):
        _stack(builder, pd.DataFrame({"k": k}))


def test_a_nan_categorical_value_is_an_unseen_level():
    """NaN never matches a level (patsy would have dropped the row instead)."""
    rng = np.random.default_rng(16)
    _, builder = _builder("y ~ C(k, levels=[1, 2, 3])", rng, [1, 2, 3])
    with pytest.raises(ValueError, match=r"not among the training levels: \[nan\]"):
        _stack(builder, pd.DataFrame({"k": [1.0, np.nan]}))


def test_a_nan_in_a_patsy_built_term_raises_instead_of_dropping_the_row():
    """The old closure's NA_action="drop" would have shortened the output."""
    from patsy import PatsyError

    rng = np.random.default_rng(17)
    _, builder = _builder("y ~ scale(x1) + C(pa, levels=[0, 1])", rng)
    df = _frame(10, rng)
    df.loc[3, "x1"] = np.nan
    with pytest.raises(PatsyError, match="missing values"):
        _stack(builder, df)


def test_an_integer_categorical_column_matches_its_float_twin():
    """An int column is cast chunk by chunk and gives the float column's design."""
    rng = np.random.default_rng(14)
    formula = f"y ~ scale(x1) + C(k, levels={L106}) + C(pa, levels=[0, 1])"
    _, builder = _builder(formula, rng)
    builder.chunk_rows = 64
    as_float = _frame(1000, rng)
    as_int = as_float.astype({"k": np.int64, "pa": np.int64})
    assert np.array_equal(_stack(builder, as_int), _stack(builder, as_float))


def test_an_empty_frame_yields_no_chunk():
    """Zero rows run no chunk."""
    rng = np.random.default_rng(12)
    _, builder = _builder(FORMULAS["bol-like"], rng, L117)
    assert list(builder.chunks(_frame(10, rng).iloc[:0])) == []


def test_describe_counts_the_buckets():
    """The log line counts one-hot terms, materialised and total columns."""
    rng = np.random.default_rng(7)
    _, builder = _builder(
        "y ~ scale(x1) + C(k, levels=[1, 2, 3]) + C(pa, levels=[0, 1])",
        rng,
        [1, 2, 3],
    )
    assert builder.describe() == (
        "2 one-hot term(s) by lookup, 1 materialised col(s), "
        "5 col(s) in 262144-row float32 chunks"
    )


@pytest.mark.parametrize(
    "n_columns, rows",
    [
        (1, 1 << 18),
        (124, 1 << 18),
        (310, 1 << 16),
        (5000, 1 << 12),
        (10**6, 32),
        (10**9, 1),
    ],
)
def test_chunk_rows_keep_one_float32_chunk_near_128_mib(n_columns, rows):
    """The power of two below 128 MiB / (4 B x columns), within [1, 2^18]."""
    assert design_matrix._chunk_rows(n_columns) == rows
