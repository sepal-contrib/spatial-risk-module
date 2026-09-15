"""Which raw variables still need harmonizing onto the current base grid?

Step 3 re-derived *every* raw variable on every press: with N layers already
aligned, adding one new variable cost N+1 reprojections instead of one.
Re-deriving is a no-op in output terms — ``reproject_and_match`` and
``rasterize`` write a deterministic file from (source, base geobox) — so
skipping an output that is already on the current grid costs nothing but saves
the wall clock.

"Already harmonized" is deliberately a statement about the *files*, not about
the registry. A ``processed_variables`` entry survives a base-raster change and
the old output is then on the wrong grid, so the grid is checked directly. The
mtime check on top of it catches a raw variable edited in place to point at a
different file: ``variables_tile.on_save`` re-registers it under the same
``{name}_{year}`` key, so neither the registry nor the grid would notice.

Solara-free and Project-free on purpose (duck-typed on ``name``/``year``/
``path``/``data_type``), so the whole predicate is unit-testable without a
render harness or a real project on disk.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from spatialrisk.variables.models import DataType, RasterizationMethod, RasterType

logger = logging.getLogger("spatial_risk")


@dataclass(frozen=True)
class HarmonizationStatus:
    """Raw-variable keys split by whether Step 3 still has work to do on them.

    Keys are the *raw* collection's keys (what ``reproject_and_match_all`` and
    ``rasterize_all`` take as ``keys=``), not the processed output keys — the
    two coincide today but the caller filters the raw collection.
    """

    pending: List[str]
    current: List[str]

    @property
    def total(self) -> int:
        """Harmonizable layers in the project (inactive vectors excluded)."""
        return len(self.pending) + len(self.current)


def output_key(var: Any) -> Optional[str]:
    """The ``processed_variables`` key ``add_as_processed`` gives ``var``'s output.

    Mirrors ``LocalRasterVar.add_as_processed`` exactly. Both harmonization
    paths preserve ``name`` and ``year`` (``reproject_and_match`` passes
    ``name=self.name``; ``rasterize`` likewise), so this one mapping covers
    rasters and vectors alike.
    """
    name = getattr(var, "name", None)
    year = getattr(var, "year", None)
    return f"{name}_{year}" if year else name


def expected_raster_type(var: Any):
    """The ``raster_type`` ``var``'s harmonized output will carry.

    Rasters keep their own — ``reproject_and_match`` passes
    ``raster_type=self.raster_type`` — and it decides the resampling method
    (nearest for categorical, bilinear for continuous). Vectors get it from the
    rasterization mode: ``rasterize`` sets categorical for ``unique`` and
    continuous otherwise.

    Comparing this against the registered output is what catches an edit that
    changed *how* a layer is harmonized without changing the file it comes
    from. ``variables_tile._build_variable`` rebuilds the variable from the
    modal's ``raster_type`` / ``rasterization_method`` and re-registers it under
    the same ``{name}_{year}`` key, so neither the key, the output grid, nor
    either mtime would move — the old output would otherwise be kept forever.
    """
    if getattr(var, "data_type", None) == DataType.vector:
        method = getattr(var, "rasterization_method", None)
        return (
            RasterType.categorical
            if method == RasterizationMethod.unique
            else RasterType.continuous
        )
    return getattr(var, "raster_type", None)


def _matches_geobox(path: Path, geobox) -> bool:
    """True when the raster at ``path`` sits exactly on ``geobox``.

    Header read only — no pixels are touched. rasterio's ``CRS.__eq__`` and
    ``Affine.__eq__`` compare correctly against odc-geo's types, and
    ``geobox.shape`` is a ``Shape2d(x=, y=)`` whose ``.yx`` matches rasterio's
    ``(rows, cols)``. An unreadable or corrupt file answers False so the layer
    is re-derived rather than trusted.
    """
    import rasterio

    try:
        with rasterio.open(path) as src:
            return (
                src.crs == geobox.crs
                and src.transform == geobox.transform
                and src.shape == geobox.shape.yx
            )
    except Exception:
        logger.debug("Could not read %s for a grid check", path, exc_info=True)
        return False


def is_current(var: Any, output: Any, geobox) -> bool:
    """True when ``output`` is a harmonized product of ``var`` sitting on ``geobox``.

    Ordered cheapest-first: attribute lookups and two ``stat`` calls gate the
    single file open, so a project whose layers are mostly new pays almost
    nothing.
    """
    if output is None:
        return False
    if getattr(output, "raster_type", None) != expected_raster_type(var):
        return False  # the layer is harmonized differently now
    src_path = getattr(var, "path", None)
    out_path = getattr(output, "path", None)
    if src_path is None or out_path is None:
        return False  # a GEEVar has no local file yet — it must be downloaded
    src_path, out_path = Path(src_path), Path(out_path)
    if not out_path.exists():
        return False
    if not src_path.exists():
        # Left pending on purpose: the run then fails the same way it does
        # today rather than silently trusting an output we cannot verify.
        return False
    if out_path.stat().st_mtime < src_path.stat().st_mtime:
        return False
    return _matches_geobox(out_path, geobox)


def harmonization_status(project: Any) -> HarmonizationStatus:
    """Split the project's harmonizable raw variables into pending / current.

    Mirrors exactly what Step 3 processes: every raw *raster*
    (``reproject_and_match_all`` does not filter on ``active``) plus every
    *active* raw vector (``rasterize_all`` does). Inactive vectors appear in
    neither list, because Step 3 never touches them and counting them would
    make the GUI's "N of M" hint lie.

    With no base raster there is no grid to compare against, so everything
    harmonizable is reported pending; ``run_processing`` raises its own error
    long before it gets that far.
    """
    raw = getattr(project, "raw_variables", None) or {}
    processed = getattr(project, "processed_variables", None) or {}
    base = getattr(project, "base_raster", None)

    candidates = [
        (key, var)
        for key, var in raw.items()
        if getattr(var, "data_type", None) == DataType.raster
        or (
            getattr(var, "data_type", None) == DataType.vector
            and getattr(var, "active", True)
        )
    ]
    if base is None:
        return HarmonizationStatus(pending=[k for k, _ in candidates], current=[])

    geobox = base.get_base_geobox()

    # ``LocalVectorVar.rasterize`` writes ``{name}.tif`` with no year suffix, so
    # two temporal vectors sharing a name also share one output file while
    # holding two registry entries. Task F5 fixes that path, but a project saved
    # before F5 still has such entries, and one of them describes bytes that
    # belong to its sibling year — never claim either current.
    seen = Counter(
        str(path)
        for path in (
            getattr(processed.get(output_key(var)), "path", None)
            for _, var in candidates
        )
        if path is not None
    )
    shared_paths = {path for path, n in seen.items() if n > 1}

    pending: List[str] = []
    current: List[str] = []
    for key, var in candidates:
        output = processed.get(output_key(var))
        shared = str(getattr(output, "path", None)) in shared_paths
        fresh = not shared and is_current(var, output, geobox)
        (current if fresh else pending).append(key)
    return HarmonizationStatus(pending=pending, current=current)
