# spatialrisk/mlmodels/design_terms.py
"""Which patsy terms are plain categoricals on a numeric level domain.

patsy turns a categorical factor into level indexes with a Python loop over
every value (``patsy.categorical.categorical_to_int``): about 1.5 s per million
values, holding the GIL (profiled 2026-09-25 on a real BOL stripe). Two
consumers skip that loop for the single-factor ``C(<column>, levels=[...])``
terms ``spatialrisk.far_helpers.generate_patsy_formula`` emits. Both find each
value's level with :func:`level_positions`, an exact vectorised match against
the level tuple patsy recorded:

* :mod:`spatialrisk.mlmodels.linear_predictor` (GLM, iCAR) reduces the term to
  one number per level, ``contrast @ coef``;
* :mod:`spatialrisk.mlmodels.design_matrix` (RF) copies the term's one-hot
  columns out of the contrast rows.

The levels come from ``factor_infos[f].categories``, i.e. the formula's
``levels=`` domain (or, for a bare ``C(x)``, the levels patsy saw in the
training frame), never from the values any prediction frame contains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

PLAIN_C = re.compile(r"^C\(\s*([A-Za-z_]\w*)\s*(?:,.*)?\)$", re.S)


@dataclass(frozen=True)
class PlainCategorical:
    """A single-factor ``C(<column>...)`` term on a numeric level domain."""

    code: str  # factor code, for error messages
    column: str  # block_df column the factor reads
    levels: np.ndarray  # float64, in patsy's level order
    contrast: np.ndarray  # float64 (n_levels, term columns), rows follow levels


def is_numeric_domain(categories) -> bool:
    """True when ``categories`` is a finite, duplicate-free numeric level tuple.

    float() parses digit strings, but patsy matches "1" and 1.0 as different
    levels; string domains stay with patsy so its errors stay intact.
    """
    if any(isinstance(c, (str, bytes)) for c in categories):
        return False
    # An integer past 2**53 has no exact float64 twin, while patsy matches it
    # exactly: leave such a domain to patsy.
    if any(
        isinstance(c, (int, np.integer)) and abs(int(c)) > 2**53 for c in categories
    ):
        return False
    try:
        arr = np.asarray(categories, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 1 and np.isfinite(arr).all() and np.unique(arr).size == arr.size


def level_positions(levels_sorted: np.ndarray, column, code: str) -> np.ndarray:
    """Each value's index in ``levels_sorted``, or ValueError naming ``code``.

    ``levels_sorted`` is a :class:`PlainCategorical`'s levels in ascending
    order. ``column`` is cast to float64 (a copy only when it is not float64
    already) and matched exactly, so a value patsy would reject as an unseen
    level is rejected here too; NaN matches no level. A value above the last
    level is clamped before the comparison, so it is reported, never indexed
    past the end. The message lists at most 10 of the offending values.
    """
    values = np.asarray(column, dtype=np.float64)
    pos = np.searchsorted(levels_sorted, values)
    np.minimum(pos, levels_sorted.shape[0] - 1, out=pos)
    miss = levels_sorted[pos] != values
    if miss.any():
        bad = np.unique(values[miss])[:10].tolist()
        raise ValueError(f"{code}: values not among the training levels: {bad}")
    return pos


def plain_categorical(design_info, term, subterms) -> Optional[PlainCategorical]:
    """``term`` as a :class:`PlainCategorical`, or None when patsy must build it.

    ``subterms`` is ``design_info.term_codings[term]``. A term qualifies when it
    has one factor and one subterm, the factor is categorical, its code is a
    bare ``C(<column>...)`` call on a column name (``C(k + 0)`` reads an
    expression and does not qualify), patsy recorded a contrast matrix for it,
    and its levels are a numeric domain (:func:`is_numeric_domain`).
    """
    if len(subterms) != 1 or len(term.factors) != 1:
        return None
    (factor,) = term.factors
    (subterm,) = subterms
    info = design_info.factor_infos[factor]
    m = PLAIN_C.match(factor.code.strip())
    if (
        info.type != "categorical"
        or m is None
        or factor not in subterm.contrast_matrices
        or not is_numeric_domain(info.categories)
    ):
        return None
    return PlainCategorical(
        code=factor.code,
        column=m.group(1),
        levels=np.asarray(info.categories, dtype=np.float64),
        contrast=np.asarray(subterm.contrast_matrices[factor].matrix, dtype=np.float64),
    )
