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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

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
    return DesignBuilder(
        n_columns=n_columns,
        constant_slices=constant_slices,
        one_hots=one_hots,
        subset_design_info=subset,
        subset_positions=positions,
        chunk_rows=_chunk_rows(n_columns),
    )
