# tests/test_design_matrix.py
"""compile_design_builder: patsy's design matrix as float32 row chunks."""
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrices
from patsy.highlevel import build_design_matrices
from sklearn.ensemble import RandomForestClassifier

from spatialrisk.mlmodels import design_matrix
from spatialrisk.mlmodels.design_matrix import (
    _check_full_coverage,
    compile_design_builder,
)

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
    "one-scale": "y ~ scale(x1)",
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
    parts = []
    for _, _, x in builder.chunks(df):
        assert x.dtype == np.float32 and x.flags.c_contiguous
        parts.append(x.copy())
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


def test_a_bare_c_design_rebuilt_from_a_samples_csv_matches_patsy(tmp_path):
    """RFModel._ensure_design_info's path: pd.read_csv types codes as int64.

    A model reloaded from its pickle rebuilds design info with
    ``pd.read_csv(samples)``, which types integer categorical codes as
    numpy/Python ``int64`` rather than the float64 the live training frame
    used, so patsy's categories come back as Python ``int``. A bare ``C(k)``
    (no ``levels=``) must still route through the one-hot lookup bucket, not
    the patsy fallback, and give a bit-exact match.
    """
    rng = np.random.default_rng(23)
    n = 3000
    train = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "k": rng.integers(1, 6, n),
            "pa": rng.integers(0, 2, n),
            "x1": rng.normal(size=n),
        }
    )
    csv_path = tmp_path / "samples.csv"
    train.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path)

    _, x_ref = dmatrices("y ~ C(k) + C(pa) + scale(x1)", reloaded, NA_action="drop")
    design_info = x_ref.design_info
    builder = compile_design_builder(design_info)
    builder.chunk_rows = 7

    new = _frame(2001, rng, levels=[1, 2, 3, 4, 5])
    got = _stack(builder, new)
    (expected,) = build_design_matrices([design_info], new, NA_action="raise")
    assert np.array_equal(got, np.asarray(expected, dtype=np.float32))
    assert len(builder.one_hots) == 2


def test_check_full_coverage_catches_a_gap():
    """A column no bucket claims is a missing column, not silent garbage."""
    with pytest.raises(RuntimeError, match=r"missing columns \[2\]"):
        _check_full_coverage(4, [slice(0, 2)], [], np.array([3], dtype=np.intp))


def test_check_full_coverage_catches_an_overlap():
    """A column two buckets claim is a duplicate, not a silent overwrite."""
    with pytest.raises(RuntimeError, match=r"duplicated columns \[1\]"):
        _check_full_coverage(
            3, [slice(0, 2)], [slice(1, 2)], np.array([2], dtype=np.intp)
        )


def test_check_full_coverage_accepts_a_valid_partition():
    """Every column claimed exactly once raises nothing."""
    _check_full_coverage(
        5, [slice(0, 1)], [slice(1, 3)], np.array([3, 4], dtype=np.intp)
    )


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


# --------------------------------------------------------------------------- #
# evaluate() and its memory
# --------------------------------------------------------------------------- #
def test_evaluate_runs_fn_per_chunk_into_one_float64_vector():
    """Each chunk reaches fn once, in order; an empty frame never calls it."""
    rng = np.random.default_rng(21)
    design_info, builder = _builder(FORMULAS["bol-like"], rng, L117)
    builder.chunk_rows = 1000
    new = _frame(4321, rng)
    calls = []

    def fn(x):
        """Record each chunk's row count and return a scaled column."""
        calls.append(x.shape[0])
        return x[:, 1] * 2

    got = builder.evaluate(new, fn)
    (whole,) = build_design_matrices([design_info], new, NA_action="raise")
    assert got.dtype == np.float64
    assert np.array_equal(got, np.asarray(whole, dtype=np.float32)[:, 1] * 2)
    assert calls == [1000, 1000, 1000, 1000, 321]
    empty = builder.evaluate(new.iloc[:0], fn)
    assert empty.shape == (0,) and empty.dtype == np.float64
    assert calls == [1000, 1000, 1000, 1000, 321]
    assert builder.working_set_columns == 1


def _peak_bytes(run, df):
    """Tracemalloc peak of ``run(df)``, above what was live before it."""
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        run(df)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        if started:
            tracemalloc.stop()


# Python objects patsy and pandas churn per chunk (see the same allowance in
# tests/test_linear_predictor.py): well under one byte per row.
_PY_CHURN_BYTES_PER_ROW = 0.5
_CHUNK = 32768  # smaller than production, so the frames stay small


def _two_peaks(builder, run, rng, levels):
    """Peaks at 4 and 16 chunks, after one warm-up run of each."""
    n1, n2 = 4 * _CHUNK, 16 * _CHUNK
    frames = {n1: _frame(n1, rng, levels), n2: _frame(n2, rng, levels)}
    for df in frames.values():
        run(df)  # warm patsy's lazy state, pandas' caches and free lists
    return {n: _peak_bytes(run, df) for n, df in frames.items()}


@pytest.mark.parametrize(
    "name", ["bol-like", "plain-numeric", "one-scale", "only-categorical"]
)
def test_chunk_scratch_bytes_covers_the_measured_build_peak(name):
    """One output column per pixel plus the bounded chunk scratch cover the peak.

    (a) each peak fits working_set_columns per pixel plus chunk_scratch_bytes;
    (b) the peak grows by no more than one float64 per added row, so nothing
    but the output scales with the stripe; (c) the charge does not grossly
    overshoot the peak.
    """
    rng = np.random.default_rng(9)
    levels = L117 if name == "bol-like" else L106
    _, builder = _builder(FORMULAS[name], rng, levels)
    builder.chunk_rows = _CHUNK
    peaks = _two_peaks(
        builder, lambda df: builder.evaluate(df, lambda x: x[:, 0]), rng, levels
    )
    n1, n2 = sorted(peaks)
    per_px = builder.working_set_columns * 8
    for n, peak in peaks.items():
        assert peak <= per_px * n + builder.chunk_scratch_bytes, (n, peak / n)
    slope = (peaks[n2] - peaks[n1]) / (n2 - n1)
    assert slope <= per_px + _PY_CHURN_BYTES_PER_ROW, slope
    charged = per_px * n2 + builder.chunk_scratch_bytes
    assert charged <= 2.5 * peaks[n2], (charged, peaks[n2])


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_a_forest_over_the_chunks_holds_one_float64_per_pixel(n_jobs):
    """RF's closure shape: nothing but the output vector scales with the stripe.

    sklearn's predict phase keeps the chunk plus all_proba (16 B/row) and, per
    running joblib worker, one tree's proba and leaf ids (24 B/row), whatever
    the stripe size; RFModel.apply's serial default keeps the pickled n_jobs=-1.
    """
    rng = np.random.default_rng(10)
    design_info, builder = _builder(FORMULAS["bol-like"], rng, L117)
    builder.chunk_rows = _CHUNK
    train = _frame(3000, rng, L117)
    (xt,) = build_design_matrices([design_info], train, NA_action="raise")
    forest = RandomForestClassifier(
        n_estimators=20,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=0,
    ).fit(np.asarray(xt, dtype=np.float32), train["y"].to_numpy())

    def proba(x):
        """Return the forest's positive-class probability for a chunk."""
        return forest.predict_proba(x)[:, 1]

    peaks = _two_peaks(builder, lambda df: builder.evaluate(df, proba), rng, L117)
    n1, n2 = sorted(peaks)
    slope = (peaks[n2] - peaks[n1]) / (n2 - n1)
    assert slope <= 8 + _PY_CHURN_BYTES_PER_ROW, slope
    predict_phase = builder.chunk_rows * (4 * builder.n_columns + 16 + 24 * n_jobs)
    bound = 8 * n2 + max(builder.chunk_scratch_bytes, predict_phase)
    assert peaks[n2] <= bound, (peaks[n2], bound)
