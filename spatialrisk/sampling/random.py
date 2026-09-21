"""Simple random sampling over valid pixels."""
from typing import Tuple

import numpy as np

from spatialrisk.sampling import blocked
from spatialrisk.sampling.base import SamplingStrategyBase


class RandomSampling(SamplingStrategyBase):
    """Simple random sampling: every valid pixel is equally likely."""

    def select(
        self, valid_indices, *, n_samples, seed=None, **_
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (row_indices, col_indices) drawn uniformly from the valid pixels.

        The in-memory path, kept for callers that already hold index arrays.
        ``generate_points`` uses :meth:`select_blocked` instead.
        """
        rows, cols = valid_indices
        n_valid = len(rows)
        if n_samples is None or n_samples >= n_valid:
            return valid_indices
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_valid, size=n_samples, replace=False)
        return rows[idx], cols[idx]

    def select_blocked(self, scan, *, n_samples, seed=None, **_):
        """Blocked twin of :meth:`select` — same draw, no whole-raster arrays.

        ``select`` picks positions into ``np.where(valid)``; those positions are
        ranks in the global row-major enumeration of the valid pixels, so the
        identical draw can be made from the *count* alone and converted back to
        coordinates by a second stripe pass. ``rng.choice(n, k, replace=False)``
        is deliberately the same call ``select`` makes, because
        ``rng.choice(arr, k, replace=False)`` is
        ``arr[rng.choice(len(arr), k, replace=False)]`` on the same stream —
        that equivalence is what makes the two paths bit-identical.

        Returns ``(rows, cols, values)``; the band values come free from the
        stripe that was decoded to find the coordinates, which saves a third
        pass just to fill the ``strata`` column.
        """
        n_valid = blocked.count_valid(scan)
        if n_samples is None or n_samples >= n_valid:
            # Historical "take all": every valid pixel in row-major order, and
            # notably WITHOUT drawing, so the RNG stream is never touched.
            return blocked.collect_all(scan)
        rng = np.random.default_rng(seed)
        picks = rng.choice(n_valid, size=n_samples, replace=False)
        order = np.argsort(picks)
        collected = blocked.collect_ranks(scan, {None: picks[order]})[None]
        return blocked.restore_order(order, collected)
