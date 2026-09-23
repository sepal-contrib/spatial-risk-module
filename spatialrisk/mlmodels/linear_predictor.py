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
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from patsy.highlevel import build_design_matrices

_PLAIN_C = re.compile(r"^C\(\s*([A-Za-z_]\w*)\s*(?:,.*)?\)$", re.S)

# eta()'s peak working set, in float64 columns (8 B/px each). The figures come
# from tracemalloc measurements (patsy 1.0.2, pandas 2.3), rounded up, and
# tests/test_linear_predictor.py pins them. eta() runs two phases that never
# overlap. Both hold the returned eta vector.
_ETA_COLS = 1
# Lookup phase: the column cast to float64 (a copy only when the frame holds
# ints), the int64 searchsorted index, one gathered float64 vector and the bool
# miss mask. Measured: 25 B/px on float columns, 33 B/px on int columns.
_LOOKUP_COLS = 4
# Materialised phase: until the matrix is built, patsy keeps each factor's
# evaluated copy and its NA mask alive, and scale() copies once more. Measured:
# 25 B/px per scale() factor with its matrix column included.
_COLS_PER_FACTOR = 3
# The in-flight factor's evaluation temporaries and the ``X @ coef`` product.
# A single scale() factor peaks at 40 B/px inside patsy.
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
        """Design columns the planner should charge per pixel for one eta() call.

        This is a conservative model of eta()'s measured peak in float64 columns,
        not a count of design columns. It is the eta vector plus the larger of
        two phases:

        * the lookup phase, a fixed 4 columns whatever the level count;
        * the materialised phase: the subset matrix, 3 columns per factor
          patsy evaluates, and 2 columns of build transients.

        The per-factor figure assumes single-column factors. Those are the
        ``scale(x)``, plain numeric and ``C(...)`` factors that
        ``generate_patsy_formula`` emits. A multi-column basis such as a spline
        would need re-measuring.
        ``test_working_set_columns_covers_the_measured_eta_peak`` pins the model
        against tracemalloc.
        """
        lookup = _LOOKUP_COLS if self.lookups else 0
        materialised = 0
        if self.subset_design_info is not None:
            n_factors = len(self.subset_design_info.factor_infos)
            materialised = (
                self.materialised_columns + _COLS_PER_FACTOR * n_factors + _BUILD_COLS
            )
        return _ETA_COLS + max(lookup, materialised)

    def describe(self) -> str:
        """One line for the plan log."""
        return (
            f"{len(self.lookups)} lookup term(s), "
            f"{self.materialised_columns} materialised col(s)"
        )

    def eta(self, block_df) -> np.ndarray:
        """The linear predictor for every row of ``block_df`` (float64)."""
        n = len(block_df)
        out = np.full(n, self.constant, dtype=np.float64)
        for lk in self.lookups:
            values = np.asarray(block_df[lk.column], dtype=np.float64)
            pos = np.searchsorted(lk.levels_sorted, values)
            np.minimum(pos, lk.levels_sorted.shape[0] - 1, out=pos)
            miss = lk.levels_sorted[pos] != values
            if miss.any():
                bad = np.unique(values[miss])[:10].tolist()
                raise ValueError(
                    f"{lk.code}: values not among the training levels: {bad}"
                )
            out += lk.table_sorted[pos]
            # Free the per-pixel transients now, not when the next lookup or the
            # patsy build below rebinds or outlives them.
            del values, pos, miss
        if self.subset_design_info is not None:
            (x,) = build_design_matrices(
                [self.subset_design_info], block_df, NA_action="raise"
            )
            out += np.asarray(x, dtype=np.float64) @ self.subset_coef
        return out


def _is_numeric_domain(categories) -> bool:
    # float() parses digit strings, but patsy matches "1" and 1.0 as different
    # levels; string domains stay with patsy so its errors stay intact.
    if any(isinstance(c, (str, bytes)) for c in categories):
        return False
    try:
        arr = np.asarray(categories, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 1 and np.isfinite(arr).all() and np.unique(arr).size == arr.size


def compile_linear_predictor(design_info, coef) -> LinearPredictor:
    """Sort ``design_info``'s terms into constant, lookup and materialised buckets.

    ``coef`` has one entry per ``design_info.column_names`` entry; trailing extra
    entries are ignored (iCAR's ``betas`` carry more).
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
        lookup = None
        if len(subterms) == 1 and len(term.factors) == 1:
            (factor,) = term.factors
            (subterm,) = subterms
            info = design_info.factor_infos[factor]
            m = _PLAIN_C.match(factor.code.strip())
            if (
                info.type == "categorical"
                and m is not None
                and factor in subterm.contrast_matrices
                and _is_numeric_domain(info.categories)
            ):
                contrast = subterm.contrast_matrices[factor].matrix
                table = np.asarray(contrast, dtype=np.float64) @ coef[sl]
                levels = np.asarray(info.categories, dtype=np.float64)
                order = np.argsort(levels)
                lookup = _Lookup(
                    code=factor.code,
                    column=m.group(1),
                    levels_sorted=levels[order],
                    table_sorted=table[order],
                )
        if lookup is not None:
            lookups.append(lookup)
        else:
            materialised_terms.append(term)

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
