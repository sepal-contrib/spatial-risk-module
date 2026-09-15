"""Systematic sampling — a regular grid intersected with the valid mask."""
from typing import Tuple

import numpy as np

from spatialrisk.sampling import blocked
from spatialrisk.sampling.base import SamplingStrategyBase

#: How many index entries the grid-membership test in :meth:`select` handles at
#: a time, so its modulo temporaries stay a few tens of MB on any raster size.
_MEMBERSHIP_CHUNK = 1 << 22


class SystematicSampling(SamplingStrategyBase):
    """Systematic sampling: a regular grid intersected with the valid mask."""

    def select(
        self,
        valid_indices,
        *,
        n_samples=None,
        shape=None,
        spacing_m=None,
        res_m=None,
        **_
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (row_indices, col_indices) on a regular grid inside the mask.

        The in-memory path, kept for callers that already hold index arrays.
        ``generate_points`` uses :meth:`select_blocked` instead. ``valid_indices``
        must be row-major (as ``np.where`` returns) -- the grid-membership test
        relies on it, which is what lets the full-shape scatter mask go.
        """
        rows, cols = valid_indices
        n_valid = len(rows)

        if spacing_m is not None:
            # Distance mode.
            if spacing_m <= 0:
                raise ValueError("spacing_m must be > 0.")
            if res_m is None:
                raise ValueError(
                    "Distance-based systematic sampling requires res_m "
                    "(pixel size in metres)."
                )
            if shape is None:
                raise ValueError("Systematic sampling requires the raster shape.")
            res_y, res_x = res_m
            step_row = max(1, int(round(spacing_m / res_y)))
            step_col = max(1, int(round(spacing_m / res_x)))
        else:
            # Count mode — keep the original behaviour (no shape needed here).
            if n_samples is None or n_samples >= n_valid:
                return valid_indices
            if shape is None:
                raise ValueError("Systematic sampling requires the raster shape.")
            step = max(1, int(round(np.sqrt(n_valid / n_samples))))
            step_row = step_col = step

        if step_row == 1 and step_col == 1:
            return valid_indices

        # Which valid pixels sit on the grid, tested directly on the index
        # arrays. The previous version scattered a full-shape bool and then
        # built two int64 meshgrids over the whole raster: 2.1 GiB and 33.1 GiB
        # respectively on a 2.22 Gpx raster, the largest allocation in the
        # pipeline. Nothing here is proportional to `shape`, and the modulo
        # temporaries are chunked so they stay small regardless of `n_valid`.
        #
        # This keeps the caller's pixel order, which is the row-major order the
        # old grid enumeration produced as long as `valid_indices` comes from
        # np.where (it always does — see service.generate_points).
        keep = np.empty(n_valid, dtype=bool)
        for start in range(0, n_valid, _MEMBERSHIP_CHUNK):
            stop = start + _MEMBERSHIP_CHUNK
            np.equal(rows[start:stop] % step_row, 0, out=keep[start:stop])
            keep[start:stop] &= cols[start:stop] % step_col == 0
        return rows[keep], cols[keep]

    def select_blocked(self, scan, *, n_samples=None, spacing_m=None, res_m=None, **_):
        """Blocked twin of :meth:`select` — grid nodes per stripe, no full grid.

        Distance mode needs no pass 1 at all (the step comes from the pixel
        size), so it is a single read. Count mode needs the valid-pixel total to
        reproduce ``step = round(sqrt(n_valid / n_samples))`` exactly, including
        its clamp to 1, and that total now comes from the stripe count instead
        of ``len(np.where(valid)[0])``.

        Returns ``(rows, cols, values)``.
        """
        if spacing_m is not None:
            if spacing_m <= 0:
                raise ValueError("spacing_m must be > 0.")
            if res_m is None:
                raise ValueError(
                    "Distance-based systematic sampling requires res_m "
                    "(pixel size in metres)."
                )
            res_y, res_x = res_m
            step_row = max(1, int(round(spacing_m / res_y)))
            step_col = max(1, int(round(spacing_m / res_x)))
        else:
            n_valid = blocked.count_valid(scan)
            if n_samples is None or n_samples >= n_valid:
                return blocked.collect_all(scan)
            step = max(1, int(round(np.sqrt(n_valid / n_samples))))
            step_row = step_col = step
        return blocked.collect_grid(scan, step_row, step_col)
