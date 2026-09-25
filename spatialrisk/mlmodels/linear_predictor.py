# spatialrisk/mlmodels/linear_predictor.py
"""Evaluate a patsy design's linear predictor without materialising categoricals.

A treatment-coded categorical with ``L`` levels adds ``L - 1`` float64 columns
per row to a design matrix, yet its contribution to ``X @ coef`` is one number
per level. :func:`compile_linear_predictor` walks a :class:`patsy.DesignInfo`
once and sorts every term into one of three buckets:

* **constant** -- the intercept term, added once;
* **lookup** -- a single-factor categorical ``C(<column>...)`` on a numeric
  level domain: ``table = contrast_matrix @ coef[slice]`` (one entry per
  level), evaluated per row by exact value matching against the level tuple
  patsy recorded from ``levels=...``;
* **materialised** -- everything else (numeric terms, interactions, categorical
  expressions), built per call through ``build_design_matrices`` on
  ``design_info.subset(...)``.

The lookup table is built from ``factor_infos[f].categories``, i.e. from the
formula's ``levels=`` domain, never from the values any frame happened to
contain: a level with no sample support still has a coefficient and still
contributes exactly what its one-hot column would have.

Numerics: term-wise accumulation and BLAS ``X @ coef`` may differ in the last
ulp of eta. Callers who rescale to uint16 gate that with exact golden tests.

Memory: :meth:`LinearPredictor.eta` walks the frame in row chunks of
``_ETA_CHUNK_ROWS``, so its only per-pixel allocation is the returned vector
(:attr:`LinearPredictor.working_set_columns` is 1). Every other transient --
lookup gathers, patsy's per-factor copies, the subset matrix, the product --
lives for one chunk at a time. That scratch is bounded per call by
:attr:`LinearPredictor.chunk_scratch_bytes` (about 4 MiB per ``scale()``
numeric, 42 MiB for ten, ~120 MiB for a 106-level categorical patsy has to
materialise), and the planner does not charge it: one scratch per worker is
left to the headroom the memory budget keeps
(:data:`spatialrisk.gdal_env.INFERENCE_MEMORY_FRACTION` budgets only half the
free memory).

patsy still gets pandas Series, as it did from the whole frame: ``scale()``
keeps its mean and variance as float128, and pandas arithmetic rounds
``(x - mean) / sd`` to float64 once where numpy's in-place operators round
twice, so plain numpy rows would move the last ulp of ~27% of the rows. The
Series are built over row views of the columns rather than taken from
``block_df.iloc[...]``, because every pandas view of ``block_df`` registers a
weak reference on it and those pile up with the chunk count.

Chunking is bit-identical to one whole-frame pass: every step is row-wise, and
a power-of-two chunk keeps OpenBLAS's 4-row gemv blocks where a whole-frame
``X @ coef`` puts them (chunks that are not a multiple of 4 moved the last ulp
of tens to hundreds of rows in 1M on a 10-numeric design). That holds with one
BLAS thread, which the engine pins; a threaded gemv splits the rows at thread
boundaries and is not reproducible across thread counts even unchunked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from patsy.highlevel import build_design_matrices

from spatialrisk.mlmodels.design_terms import level_positions, plain_categorical

# eta() walks block_df in chunks of this many rows: the smallest power of two
# within 5% of the fastest (2^18) on 7 scale() numerics + two lookups over 4M
# rows, and the largest whose scratch stays under 64 MiB for ten numerics.
_ETA_CHUNK_ROWS = 1 << 17

# eta()'s memory in float64 columns (8 B each). The figures come from
# tracemalloc measurements (patsy 1.0.2, pandas 2.3, numpy 2.2), rounded up,
# and tests/test_linear_predictor.py pins them.
#
# Per pixel: the returned eta vector is the only allocation that scales with
# the stripe.
_ETA_COLS = 1
# Per chunk row, two phases that never overlap (chunk_scratch_bytes).
# Lookup phase: the chunk cast to float64 (a copy only when the frame holds
# ints), the int64 searchsorted index, one gathered float64 vector and the bool
# miss mask. Measured: 17 B/row on float columns, 25 B/row on int columns.
_LOOKUP_COLS = 4
# Materialised phase: until the matrix is built, patsy keeps each factor's
# evaluated copy and its NA mask alive, and scale() copies once more. Measured:
# 25 B/row per scale() factor with its matrix column included.
_COLS_PER_FACTOR = 3
# The in-flight factor's evaluation temporaries and the ``X @ coef`` product.
# A single scale() factor peaks at 40 B/row inside patsy.
_BUILD_COLS = 2


@dataclass
class _Lookup:
    code: str  # factor code, for error messages
    column: str  # block_df column the factor reads
    levels_sorted: np.ndarray  # float64, ascending
    table_sorted: np.ndarray  # float64, aligned with levels_sorted


@dataclass
class LinearPredictor:
    """Compiled form of ``X @ coef`` for one design; see the module docstring."""

    constant: float
    lookups: List[_Lookup]
    subset_design_info: Optional[object]  # patsy DesignInfo or None
    subset_coef: np.ndarray
    materialised_columns: int = field(init=False)

    def __post_init__(self):
        """Derive the column counts from the subset."""
        self.materialised_columns = int(self.subset_coef.shape[0])

    @property
    def working_set_columns(self) -> int:
        """Float64 columns per pixel the planner should charge for one eta() call.

        eta() evaluates the frame in chunks of ``_ETA_CHUNK_ROWS`` rows, so the
        only allocation that scales with the stripe is the returned eta vector:
        one column, whatever the design and whatever any categorical's level
        count. The per-chunk transients are bounded separately by
        :attr:`chunk_scratch_bytes`, which the planner does not charge: that
        scratch (one per worker; about 4 MiB per ``scale()`` numeric of the
        formulas ``generate_patsy_formula`` emits, ~120 MiB for a materialised
        106-level categorical) is left to the memory budget's headroom.
        ``test_working_set_columns_covers_the_measured_eta_peak`` pins both
        against tracemalloc.
        """
        return _ETA_COLS

    @property
    def chunk_scratch_bytes(self) -> int:
        """Upper bound on eta()'s per-chunk transients, in bytes.

        ``_ETA_CHUNK_ROWS`` rows times the larger of two phases, in float64
        columns:

        * the lookup phase, a fixed 4 columns whatever the level count;
        * the materialised phase: the subset matrix, 3 columns per factor
          patsy evaluates, and 2 columns of build transients.

        The per-factor figure assumes single-column factors. Those are the
        ``scale(x)``, plain numeric and ``C(...)`` factors that
        ``generate_patsy_formula`` emits. A multi-column basis such as a spline
        would need re-measuring.
        """
        lookup = _LOOKUP_COLS if self.lookups else 0
        materialised = 0
        if self.subset_design_info is not None:
            n_factors = len(self.subset_design_info.factor_infos)
            materialised = (
                self.materialised_columns + _COLS_PER_FACTOR * n_factors + _BUILD_COLS
            )
        return _ETA_CHUNK_ROWS * max(lookup, materialised) * 8

    def describe(self) -> str:
        """One line for the plan log."""
        return (
            f"{len(self.lookups)} lookup term(s), "
            f"{self.materialised_columns} materialised col(s)"
        )

    def eta(self, block_df) -> np.ndarray:
        """The linear predictor for every row of ``block_df`` (float64).

        Rows are evaluated in chunks of ``_ETA_CHUNK_ROWS`` (see the module
        docstring). Each row still accumulates the constant, then the lookups
        in ``term_codings`` order, then the materialised product. A value
        outside a lookup's levels raises ``ValueError``; the message lists the
        offending values of the first chunk that has any, not of the whole
        frame.
        """
        n = len(block_df)
        out = np.full(n, self.constant, dtype=np.float64)
        # Views of the frame's columns, fetched once: chunks slice them, so no
        # column is cast or copied whole.
        columns = {name: np.asarray(block_df[name]) for name in block_df}
        lookup_columns = [columns[lk.column] for lk in self.lookups]
        step = _ETA_CHUNK_ROWS
        for start in range(0, n, step):
            stop = min(start + step, n)
            for lk, column in zip(self.lookups, lookup_columns):
                pos = level_positions(lk.levels_sorted, column[start:stop], lk.code)
                out[start:stop] += lk.table_sorted[pos]
                # Free the chunk's transients before the next lookup or the
                # patsy build below.
                del pos
            if self.subset_design_info is not None:
                # This chunk's rows of block_df, as pandas Series over the column
                # views: not numpy rows (scale() would round differently) and
                # not block_df.iloc (each view registers a weakref on
                # block_df). See the module docstring.
                rows = {
                    name: pd.Series(column[start:stop], name=name, copy=False)
                    for name, column in columns.items()
                }
                (x,) = build_design_matrices(
                    [self.subset_design_info], rows, NA_action="raise"
                )
                out[start:stop] += np.asarray(x, dtype=np.float64) @ self.subset_coef
                # Do not carry the matrix into the next chunk's lookups.
                del rows, x
        return out


def compile_linear_predictor(design_info, coef) -> LinearPredictor:
    """Sort ``design_info``'s terms into constant, lookup and materialised buckets.

    ``coef`` has one entry per ``design_info.column_names`` entry. Trailing
    extra entries are tolerated and ignored (the spec's §4.1 contract, as
    iCAR's old ``betas[:n_cols]`` slice did), so a caller that must catch a
    model fitted on another design checks the width itself, as
    ``GLMModel.apply`` does; iCAR's fitted betas match the design width in
    practice.
    """
    coef = np.asarray(coef, dtype=np.float64)
    n_cols = len(design_info.column_names)
    if coef.shape[0] < n_cols:
        raise ValueError(
            f"coef has {coef.shape[0]} entries but the design has {n_cols} columns"
        )
    coef = coef[:n_cols]

    constant = 0.0
    lookups: List[_Lookup] = []
    materialised_terms = []

    for term, subterms in design_info.term_codings.items():
        sl = design_info.term_slices[term]
        if len(term.factors) == 0:
            constant += float(coef[sl].sum())
            continue
        plain = plain_categorical(design_info, term, subterms)
        if plain is None:
            materialised_terms.append(term)
            continue
        table = plain.contrast @ coef[sl]
        order = np.argsort(plain.levels)
        lookups.append(
            _Lookup(
                code=plain.code,
                column=plain.column,
                levels_sorted=plain.levels[order],
                table_sorted=table[order],
            )
        )

    if materialised_terms:
        subset = design_info.subset(materialised_terms)
        idx = [design_info.column_name_indexes[c] for c in subset.column_names]
        subset_coef = coef[idx]
    else:
        subset, subset_coef = None, np.zeros(0, dtype=np.float64)

    return LinearPredictor(
        constant=constant,
        lookups=lookups,
        subset_design_info=subset,
        subset_coef=subset_coef,
    )
