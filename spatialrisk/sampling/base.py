"""Abstract base class for sampling strategies (pure: array in, indices out)."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class SamplingStrategyBase(ABC):
    """A sampling strategy selects pixel indices from a set of valid pixels.

    Implementations are pure — they take index arrays and return index arrays,
    with no raster or file I/O (that lives in service.py). This keeps them
    unit-testable without fixtures.
    """

    @abstractmethod
    def select(
        self,
        valid_indices: Tuple[np.ndarray, np.ndarray],
        *,
        n_samples: Optional[int],
        seed: Optional[int] = None,
        strata_values: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        allocation: Optional[str] = None,
        adapt: bool = False,
        pixel_area_ha: Optional[float] = None,
        spacing_m: Optional[float] = None,
        res_m: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (row_indices, col_indices) of the selected pixels."""
        raise NotImplementedError

    def select_blocked(self, scan, **kwargs):
        """Select pixels by streaming ``scan``, never materialising all of them.

        This is the memory-bounded counterpart to :meth:`select`, and the path
        ``sampling.service.generate_points`` actually takes. Where ``select``
        receives index arrays already built over every valid pixel (2 x int64 x
        n_valid -- the allocation that made country-scale rasters OOM), this
        walks a ``blocked.RasterScan`` in full-width row stripes.

        It is declared here, deliberately non-abstract, so the ABC describes the
        real contract while a strategy that only implements ``select`` still
        constructs. Concrete strategies must override it.

        Returns ``(row_indices, col_indices, values)`` -- a 3-tuple, unlike
        ``select``'s 2-tuple: the pixel values come free from the second stripe
        pass, so the caller needs no extra read to build ``strata``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement select_blocked(); "
            "generate_points requires it for memory-bounded sampling."
        )
