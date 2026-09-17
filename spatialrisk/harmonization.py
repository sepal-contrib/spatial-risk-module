"""Which raw variables still need harmonizing onto the current base grid?

Step 3 re-derived *every* raw variable on every press: with N layers already
aligned, adding one new variable cost N+1 reprojections instead of one.
Re-deriving is a no-op in output terms — ``reproject_and_match`` and
``rasterize`` write a deterministic file from (source, base geobox) — so
skipping an output that is already on the current grid costs nothing but saves
the wall clock.

"Already harmonized" is deliberately a statement about the *files*, not about
the registry. A ``processed_variables`` entry survives a base-raster change and
the old output is then on the wrong grid, so the grid is checked directly.

The mtime check on top of it catches exactly one thing: a source *newer* than
its output — a re-download, or a file rewritten in place. It cannot detect the
converse, a raw variable re-pointed at an OLDER file, because the output then
stays the newer of the two. That converse is therefore handled upstream, not
here: ``variables_tile.on_save`` (every edit) and ``variables_tile._do_add``
(every add — a re-add after removal, or a confirmed duplicate-key replace)
both unregister the processed entry under the key they just wrote, which trips
condition one, so the layer is always pending regardless of which way the
mtimes fall. What remains is a source replaced in place, at its own
path, with its mtime preserved (``cp -p``, ``rsync --times``) and no edit made
in the GUI — an accepted residual risk. Removing the layer from the harmonized
list (``remove_processed_variable``) forces it through.

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


def geobox_signature(geobox) -> str:
    """A compact, comparable string for the grid ``geobox`` describes.

    Formatted with ``repr``, which is round-trip exact for a Python float and
    survives JSON unchanged. ``%.10g`` was tried and is wrong in the one
    direction that matters: it collapses ``-29.999999999`` to ``-30``, so two
    genuinely different grids would share a signature and a layer would display
    as current while sitting on the wrong grid. Signatures are only ever
    compared for equality, never parsed, so exactness costs nothing.

    The CRS half is canonicalized for the same reason. An odc-geo ``CRS``
    built from ``"EPSG:<code>"`` and one rebuilt from that code's WKT compare
    equal and resolve the same ``to_epsg()``, but ``str()`` them and you get
    two different strings — one short, one the full WKT blob. That gap is not
    hypothetical here: the base geobox is built in code from the user's EPSG
    string, while a harmonized output's geobox is re-read from the file odc-geo
    just wrote, which comes back in WKT form. Interpolating the CRS object
    directly would stamp the two sides of the "is this current" comparison
    with different signatures for the identical grid, and every layer would
    read pending forever — quietly, since the failure never calls a stale file
    current. Preferring the EPSG code when one exists collapses both paths
    onto the same token; only a CRS with no EPSG code (custom or unregistered
    projections) falls through to WKT.

    Duck-typed on ``crs`` / ``transform`` / ``shape.yx`` like the rest of this
    module, so it takes an odc-geo GeoBox without importing one. ``crs``
    itself need not be an odc-geo ``CRS`` — a plain string or ``None`` both
    work — so its ``to_epsg``/``to_wkt`` accessors are checked with
    ``hasattr`` rather than assumed.
    """
    crs = geobox.crs
    epsg = crs.to_epsg() if crs is not None and hasattr(crs, "to_epsg") else None
    if epsg:
        crs_token = f"EPSG:{epsg}"
    elif crs is not None and hasattr(crs, "to_wkt"):
        crs_token = crs.to_wkt()
    else:
        crs_token = str(crs) if crs is not None else ""

    coeffs_tuple = tuple(geobox.transform)[:6]
    if len(coeffs_tuple) < 6:
        raise ValueError(
            "geobox_signature: transform has only "
            f"{len(coeffs_tuple)} coefficients, need 6"
        )
    coeffs = "|".join(repr(float(c)) for c in coeffs_tuple)
    rows, cols = geobox.shape.yx
    return f"{crs_token}|{coeffs}|{rows}x{cols}"


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


def is_harmonizable(var: Any) -> bool:
    """True when Step 3 would process ``var``: any raster, or an active vector.

    ``reproject_and_match_all`` does not filter on ``active`` but
    ``rasterize_all`` does, so an inactive vector is never touched — and must
    not be counted or listed as pending.
    """
    data_type = getattr(var, "data_type", None)
    if data_type == DataType.raster:
        return True
    return data_type == DataType.vector and bool(getattr(var, "active", True))


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

    Comparing this against the registered output catches a layer that is
    harmonized differently now than the output on disk was: neither the
    registry key, the output grid, nor either mtime moves when only the
    ``raster_type`` / ``rasterization_method`` changes. A GUI edit no longer
    needs this — ``variables_tile.on_save`` unregisters the processed entry
    outright — so it stands as the second line of defence, for an entry that
    survived one (a project written by an older version, or a variable changed
    outside the tile).
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
    # Every enum in ``spatialrisk/variables/models.py`` is a ``(str, Enum)``,
    # which is the only reason this comparison (and the ``data_type`` filter in
    # ``harmonization_status``) works across both construction paths: a
    # validated variable stores the plain string (``use_enum_values=True`` ->
    # ``"continuous"``) while a ``model_construct``ed one keeps the member
    # (``RasterType.continuous``). Drop that ``str`` mixin and no layer is ever
    # current here, and — far worse — the ``data_type`` filter matches nothing,
    # so Run silently becomes a no-op.
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
    # Catches a *newer* source only (a re-download, a file rewritten in place):
    # an output older than its source cannot have been derived from the bytes
    # that are there now. The converse is invisible from here — re-pointing the
    # variable at an older file leaves the output the newer of the two — which
    # is why ``variables_tile.on_save`` drops the processed entry on every edit.
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

    candidates = [(key, var) for key, var in raw.items() if is_harmonizable(var)]
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
        # Guarded rather than ``str(getattr(...))``: a missing output would
        # otherwise be compared as the literal string "None".
        out_path = getattr(output, "path", None)
        shared = out_path is not None and str(out_path) in shared_paths
        fresh = not shared and is_current(var, output, geobox)
        (current if fresh else pending).append(key)
    return HarmonizationStatus(pending=pending, current=current)
