"""Stratified sampling — per-class draws via an allocation method."""
from typing import Tuple

import numpy as np

from spatialrisk.sampling import allocation as alloc
from spatialrisk.sampling import blocked
from spatialrisk.sampling.base import SamplingStrategyBase

_ALLOCATORS = {
    "equal": lambda counts, n, adapt, pa: alloc.allocate_equal(counts, n),
    "proportional": lambda counts, n, adapt, pa: alloc.allocate_proportional(counts, n),
    "deforisk": lambda counts, n, adapt, pa: alloc.allocate_deforisk(
        counts, n, adapt=adapt, pixel_area_ha=pa
    ),
}


class StratifiedSampling(SamplingStrategyBase):
    """Stratified sampling: per-class draws sized by an allocation method."""

    def select(
        self,
        valid_indices,
        *,
        n_samples,
        seed=None,
        strata_values=None,
        allocation="equal",
        adapt=False,
        pixel_area_ha=None,
        **_,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (row_indices, col_indices) drawn per class via ``allocation``.

        The in-memory path, kept for callers that already hold index arrays.
        ``generate_points`` uses :meth:`select_blocked` instead.
        """
        if strata_values is None:
            raise ValueError("Stratified sampling requires strata_values.")
        rows, cols = valid_indices
        classes = np.unique(strata_values)
        class_counts = {int(c): int((strata_values == c).sum()) for c in classes}

        if n_samples is None:
            # Uniform with random/systematic: None => draw all available per class.
            per_class = dict(class_counts)
        else:
            allocator = _ALLOCATORS.get(allocation or "equal")
            if allocator is None:
                raise ValueError(f"Unknown allocation method: {allocation}")
            per_class = allocator(class_counts, n_samples, adapt, pixel_area_ha)

        rng = np.random.default_rng(seed)
        # Event-first ordering: descending class value puts 1 (event) before 0.
        out_rows, out_cols = [], []
        for c in sorted(class_counts, reverse=True):
            n_c = per_class.get(c, 0)
            if n_c <= 0:
                continue
            members = np.where(strata_values == c)[0]
            pick = rng.choice(members, size=min(n_c, len(members)), replace=False)
            out_rows.append(rows[pick])
            out_cols.append(cols[pick])
        if not out_rows:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.concatenate(out_rows), np.concatenate(out_cols)

    def select_blocked(
        self,
        scan,
        *,
        n_samples,
        seed=None,
        allocation="equal",
        adapt=False,
        pixel_area_ha=None,
        **_,
    ):
        """Blocked twin of :meth:`select` — same draws, no per-class ``np.where``.

        ``select`` drew ``rng.choice(members, n_c)`` where ``members`` are the
        positions of class ``c`` inside ``np.where(valid)``. Those positions are
        ranks in the global row-major enumeration of that class's valid pixels,
        so drawing ``rng.choice(n_available, n_c)`` from the pass-1 count gives
        the *same* selection on the same stream, and a second stripe pass turns
        the ranks back into coordinates.

        Three behaviours are preserved deliberately rather than tidied up:

        * classes are visited in descending value order (event class first) and
          concatenated in that order, because the output row order is part of
          the stored sample;
        * a saturated class still draws a full permutation instead of shortcutting
          to an ordered copy — the shortcut would consume a different amount of
          the RNG stream and change every later class;
        * allocation caps per class with no redistribution (see
          ``allocation._cap``), so the total may fall short of ``n_samples``.

        Returns ``(rows, cols, values)``.
        """
        raw_counts, _n_valid = blocked.count_values(scan)
        # class_counts carries the historical int()-key collision; match_counts
        # is the population `np.where(sv == c)` actually enumerated. See
        # blocked.class_tables — they differ only for colliding float strata.
        class_counts, match_counts = blocked.class_tables(raw_counts)

        if n_samples is None:
            per_class = dict(class_counts)
        else:
            allocator = _ALLOCATORS.get(allocation or "equal")
            if allocator is None:
                raise ValueError(f"Unknown allocation method: {allocation}")
            per_class = allocator(class_counts, n_samples, adapt, pixel_area_ha)

        rng = np.random.default_rng(seed)
        wanted, orders = {}, {}
        for c in sorted(class_counts, reverse=True):
            n_c = per_class.get(c, 0)
            if n_c <= 0:
                continue
            available = match_counts[c]
            picks = rng.choice(available, size=min(n_c, available), replace=False)
            order = np.argsort(picks)
            wanted[c] = picks[order]
            orders[c] = order

        if not wanted:
            return blocked.empty_result(scan)

        collected = blocked.collect_ranks(scan, wanted)
        parts = [blocked.restore_order(orders[c], collected[c]) for c in wanted]
        return tuple(np.concatenate(arrays) for arrays in zip(*parts))
