# spatialrisk/mlmodels/design_matrix.py
"""Build a patsy design matrix in float32 row chunks, one-hot columns by lookup.

A random forest reads the whole design matrix, one row per pixel. Built by
patsy over a whole stripe, a ``C(subj, levels=[1..117])`` term made that
matrix 124 float64 columns wide. patsy's per-value level loop (see
:mod:`spatialrisk.mlmodels.design_terms`) took 30 s of a 39 s stripe, on one
core, holding the GIL. The float64 matrix plus sklearn's own float32 copy
peaked 8.9 GB above baseline. Measured 2026-09-25 on one real BOL stripe:
6.39 Mpx valid, 100 trees, 8 joblib threads.

:func:`compile_design_builder` walks a :class:`patsy.DesignInfo` once and
sorts its terms like
:func:`~spatialrisk.mlmodels.linear_predictor.compile_linear_predictor` does:

* **constant** -- the intercept, a column of 1.0;
* **one-hot** -- a plain categorical
  (:func:`~spatialrisk.mlmodels.design_terms.plain_categorical`). Its columns
  are the contrast-matrix row of each value's level, found by exact matching
  against the formula's level tuple;
* **materialised** -- everything else (numeric terms, interactions, categorical
  expressions), built per chunk by patsy on ``design_info.subset(...)``.

:meth:`DesignBuilder.chunks` yields the matrix ``chunk_rows`` rows at a time.
Each chunk is exactly ``np.asarray(<patsy's matrix>, dtype=np.float32)`` for
those rows, which is the conversion sklearn's forests apply to their input. So
a forest predicting chunk by chunk gives the same probabilities, bit for bit,
as one call on patsy's whole matrix. Every step is row-wise, so any chunk size
gives the same result. The materialised part still reaches patsy as pandas
Series over row views of the columns, for the rounding reason
:mod:`spatialrisk.mlmodels.linear_predictor`'s docstring gives.

Speed on the stripe above: patsy's whole matrix took 30.2 s, the builder
3.8 s. The whole RF closure went from 39.2 s to 12.0 s.

Memory: :meth:`DesignBuilder.evaluate` keeps one float64 output per row
(:attr:`DesignBuilder.working_set_columns` is 1). Everything else lives for
one chunk and is bounded by :attr:`DesignBuilder.chunk_scratch_bytes`. The
chunk height (:func:`_chunk_rows`) keeps one float32 chunk near 128 MiB. On
the stripe above the 124-column design got 2^18 rows. The closure took
22.6 / 14.5 / 12.9 / 12.0 / 11.7 s at 2^15 .. 2^19 rows, so 2^18 is the
smallest power of two within 5% of the fastest, and it is also the cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from patsy.highlevel import build_design_matrices

from spatialrisk.mlmodels.design_terms import level_positions, plain_categorical

#: One float32 chunk of the design is sized to about this many bytes.
_CHUNK_TARGET_BYTES = 128 * 2**20
#: Chunk height bounds, in rows (see the module docstring for the cap). No
#: floor above one row: a design too wide for the byte target still gets a
#: chunk near the target, however slow.
_MIN_CHUNK_ROWS = 1
_MAX_CHUNK_ROWS = 1 << 18

# chunk_scratch_bytes' per-row model, on top of the float32 chunk itself.
# One-hot phase: the value column cast to float64 (a copy only for int
# columns), the int64 searchsorted index, the bool miss mask, plus the gathered
# float32 contrast rows (4 B per term column), which numpy materialises before
# copying them into the chunk. Measured 17 B/row on float columns and 25 B/row
# on int columns before the gather, rounded up to 4 float64 columns as in
# linear_predictor.
_LOOKUP_BYTES_PER_ROW = 32
# Materialised phase: the same patsy build as linear_predictor's materialised
# phase (measured there): the float64 subset matrix, 3 float64 columns per
# factor patsy evaluates, 4 float64 columns of build transients (pandas 3.0
# needs all 4 for a single scale() factor).
_COLS_PER_FACTOR = 3
_BUILD_COLS = 4


def _chunk_rows(n_columns: int) -> int:
    """Rows per chunk: the power of two nearest below the byte target, clamped."""
    rows = _CHUNK_TARGET_BYTES // (4 * max(1, int(n_columns)))
    rows = 1 << (int(rows).bit_length() - 1) if rows > 0 else _MIN_CHUNK_ROWS
    return int(min(_MAX_CHUNK_ROWS, max(_MIN_CHUNK_ROWS, rows)))


@dataclass
class _OneHot:
    code: str  # factor code, for error messages
    column: str  # block_df column the factor reads
    dest: slice  # the term's columns in the full design
    levels_sorted: np.ndarray  # float64, ascending
    rows_sorted: np.ndarray  # float32 contrast rows, aligned with levels_sorted


@dataclass
class DesignBuilder:
    """Compiled float32 chunk builder for one design; see the module docstring."""

    n_columns: int
    constant_slices: List[slice]
    one_hots: List[_OneHot]
    subset_design_info: Optional[object]  # patsy DesignInfo or None
    subset_positions: np.ndarray  # intp: full-design column of each subset column
    chunk_rows: int

    @property
    def working_set_columns(self) -> int:
        """Float64 columns per pixel the planner charges for one evaluate() call.

        Only the returned vector scales with the stripe; the chunk and its
        transients are :attr:`chunk_scratch_bytes`, which the planner leaves to
        the memory budget's headroom, as it does for the linear predictor.
        """
        return 1

    @property
    def chunk_scratch_bytes(self) -> int:
        """Upper bound on one chunk's build transients, in bytes.

        ``chunk_rows`` times the float32 chunk (4 B per column) plus the larger
        of the one-hot phase (``_LOOKUP_BYTES_PER_ROW`` + 4 B per column of the
        widest one-hot term) and the materialised phase (float64 subset matrix,
        3 columns per patsy factor, 4 build columns). The caller's ``fn`` runs
        after the build and keeps only the chunk alive, so its own scratch is
        the caller's to account for (sklearn's forest: at most about
        16 + 24 x n_jobs B/row, measured 40 B/row at ``n_jobs=1`` and 302 B/row
        at 16).
        """
        lookup = 0
        if self.one_hots:
            widest = max(oh.rows_sorted.shape[1] for oh in self.one_hots)
            lookup = _LOOKUP_BYTES_PER_ROW + 4 * widest
        materialised = 0
        if self.subset_design_info is not None:
            n_factors = len(self.subset_design_info.factor_infos)
            materialised = 8 * (
                len(self.subset_positions) + _COLS_PER_FACTOR * n_factors + _BUILD_COLS
            )
        return self.chunk_rows * (4 * self.n_columns + max(lookup, materialised))

    def describe(self) -> str:
        """One line for the prediction log."""
        return (
            f"{len(self.one_hots)} one-hot term(s) by lookup, "
            f"{len(self.subset_positions)} materialised col(s), "
            f"{self.n_columns} col(s) in {self.chunk_rows}-row float32 chunks"
        )

    def chunks(self, block_df) -> Iterator[Tuple[int, int, np.ndarray]]:
        """``(start, stop, x)`` for every chunk of ``block_df``'s rows.

        ``x`` is a C-contiguous float32 ``(stop - start, n_columns)`` array,
        valid until the next chunk is requested. A value outside a one-hot
        term's levels raises ``ValueError`` naming the factor and the offending
        values of that chunk. An empty frame yields nothing.
        """
        n = len(block_df)
        # Views of the frame's columns, fetched once: chunks slice them, so no
        # column is cast or copied whole.
        columns = {name: np.asarray(block_df[name]) for name in block_df}
        for start in range(0, n, self.chunk_rows):
            stop = min(start + self.chunk_rows, n)
            x = np.empty((stop - start, self.n_columns), dtype=np.float32)
            for sl in self.constant_slices:
                x[:, sl] = 1.0
            for oh in self.one_hots:
                pos = level_positions(
                    oh.levels_sorted, columns[oh.column][start:stop], oh.code
                )
                x[:, oh.dest] = oh.rows_sorted[pos]
                del pos
            if self.subset_design_info is not None:
                # pandas Series over row views, not numpy rows (scale() would
                # round differently) and not block_df.iloc (each view registers
                # a weakref on block_df); see linear_predictor's docstring.
                rows = {
                    name: pd.Series(column[start:stop], name=name, copy=False)
                    for name, column in columns.items()
                }
                (xs,) = build_design_matrices(
                    [self.subset_design_info], rows, NA_action="raise"
                )
                x[:, self.subset_positions] = np.asarray(xs)
                del rows, xs
            yield start, stop, x
            del x

    def evaluate(self, block_df, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """``fn`` over every chunk's design, as one float64 vector of ``len(block_df)``.

        ``fn(x)`` receives each float32 chunk from :meth:`chunks` and returns
        one value per row. It is not called for an empty frame.
        """
        out = np.empty(len(block_df), dtype=np.float64)
        for start, stop, x in self.chunks(block_df):
            out[start:stop] = fn(x)
            del x
        return out


def _check_full_coverage(
    n_columns: int,
    constant_slices: List[slice],
    one_hot_slices: List[slice],
    subset_positions: np.ndarray,
) -> None:
    """Raise unless the buckets write every column of ``range(n_columns)`` once.

    ``chunks()`` fills its output array bucket by bucket without ever zeroing
    it first, so a bucketing bug that dropped or double-claimed a column would
    otherwise leave uninitialised memory in the array sklearn reads, or
    silently overwrite one bucket's column with another's. A plain ``assert``
    would be stripped by ``python -O``, so this raises ``RuntimeError``
    instead. Cost is O(``n_columns``), paid once per :func:`compile_design_builder`
    call, not per chunk.
    """
    counts = np.zeros(n_columns, dtype=np.int64)
    for sl in constant_slices:
        counts[sl] += 1
    for sl in one_hot_slices:
        counts[sl] += 1
    counts[subset_positions] += 1
    missing = np.flatnonzero(counts == 0)
    duplicated = np.flatnonzero(counts > 1)
    if missing.size or duplicated.size:
        raise RuntimeError(
            "design builder column coverage is broken: "
            f"missing columns {missing.tolist()}, "
            f"duplicated columns {duplicated.tolist()}"
        )


def compile_design_builder(design_info) -> DesignBuilder:
    """Sort ``design_info``'s terms into constant, one-hot and materialised buckets."""
    n_columns = len(design_info.column_names)
    constant_slices: List[slice] = []
    one_hots: List[_OneHot] = []
    materialised_terms = []
    for term, subterms in design_info.term_codings.items():
        sl = design_info.term_slices[term]
        if len(term.factors) == 0:
            constant_slices.append(sl)
            continue
        plain = plain_categorical(design_info, term, subterms)
        if plain is None:
            materialised_terms.append(term)
            continue
        order = np.argsort(plain.levels)
        one_hots.append(
            _OneHot(
                code=plain.code,
                column=plain.column,
                dest=sl,
                levels_sorted=plain.levels[order],
                rows_sorted=np.ascontiguousarray(
                    plain.contrast[order], dtype=np.float32
                ),
            )
        )
    if materialised_terms:
        subset = design_info.subset(materialised_terms)
        positions = np.array(
            [design_info.column_name_indexes[c] for c in subset.column_names],
            dtype=np.intp,
        )
    else:
        subset, positions = None, np.zeros(0, dtype=np.intp)
    _check_full_coverage(
        n_columns, constant_slices, [oh.dest for oh in one_hots], positions
    )
    return DesignBuilder(
        n_columns=n_columns,
        constant_slices=constant_slices,
        one_hots=one_hots,
        subset_design_info=subset,
        subset_positions=positions,
        chunk_rows=_chunk_rows(n_columns),
    )
