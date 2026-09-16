"""Validate and adapt a user-supplied prediction raster to the app's contract.

The app's own predictions are single-band UInt16 rasters, nodata 0, values
1..65535 on a linear relative-probability scale, on the project's base-raster
grid. An import produced elsewhere (QGIS, another pipeline) rarely matches, so
this module (1) inspects the file, (2) checks its value range against the
scale the user declared and (3) warps + rescales it onto that contract.

Solara-free and gui-free by contract. rasterio is imported at module level
(it is a hard dependency of the package); GDAL, odc-geo and forestatrisk are
imported lazily inside the functions that need them.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio

logger = logging.getLogger("spatial_risk")

PathLike = Union[str, "Path"]

#: ``"probability"`` = floats in [0, 1]; ``"risk"`` = whole numbers in [1, 65535].
VALUE_SCALES = ("probability", "risk")

#: Upper bound of the app's risk scale (UInt16 max).
RISK_MAX = 65535


class ImportRasterError(ValueError):
    """The raster cannot be imported as a prediction; message says why."""


@dataclass(frozen=True)
class RasterInfo:
    """Structural metadata of a raster, read without touching pixel data."""

    path: Path
    band_count: int
    dtype: str
    crs: Optional[str]
    resolution: Tuple[float, float]
    nodata: Optional[float]
    width: int
    height: int


def inspect_raster(path: PathLike) -> RasterInfo:
    """Metadata-only check of an import candidate. Safe to call from a UI handler.

    Raises ``ImportRasterError`` for an unreadable file, a band count other
    than one, a missing CRS or a non-numeric dtype.
    """
    path = Path(path)
    try:
        ds = rasterio.open(path)
    except Exception as exc:  # rasterio raises RasterioIOError subclasses
        raise ImportRasterError(f"Cannot open raster: {path} ({exc})") from exc
    with ds:
        if ds.count != 1:
            raise ImportRasterError(
                f"The raster has {ds.count} bands; a prediction must have exactly one."
            )
        if ds.crs is None:
            raise ImportRasterError(
                "The raster has no coordinate reference system, so it cannot be "
                "placed on the project grid. Assign a CRS to the file first."
            )
        dtype = ds.dtypes[0]
        if not np.issubdtype(np.dtype(dtype), np.number):
            raise ImportRasterError(
                f"Unsupported data type {dtype!r}: expected numbers."
            )
        crs = ds.crs.to_string()
        epsg = ds.crs.to_epsg()
        return RasterInfo(
            path=path,
            band_count=ds.count,
            dtype=dtype,
            crs=f"EPSG:{epsg}" if epsg else crs,
            resolution=(abs(ds.transform.a), abs(ds.transform.e)),
            nodata=ds.nodata,
            width=ds.width,
            height=ds.height,
        )


def raster_range(path: PathLike) -> Tuple[float, float]:
    """(min, max) of band 1 with nodata and NaN excluded. Reads the whole file.

    Never call from a Solara handler: a full statistics pass on a country-scale
    raster blocks the websocket loop for the whole session.
    """
    from osgeo import gdal

    # GDAL's default (non-exception) mode is process-global: whether a failed
    # gdal.Open/ComputeRasterMinMax raises or just returns None/logs depends on
    # whether unrelated code already flipped gdal.UseExceptions() elsewhere in
    # the process. Scope exceptions on for this call only, so the failure mode
    # here is deterministic regardless of load order.
    with gdal.ExceptionMgr(useExceptions=True):
        try:
            ds = gdal.Open(str(path))
        except RuntimeError as exc:
            raise ImportRasterError(f"Cannot open raster: {path} ({exc})") from exc
        if ds is None:
            raise ImportRasterError(f"Cannot open raster: {path}")
        try:
            band = ds.GetRasterBand(1)
            vmin, vmax = band.ComputeRasterMinMax(False)
        except RuntimeError as exc:
            raise ImportRasterError(
                f"Could not compute the value range of {path}: {exc}"
            ) from exc
        finally:
            band = None
            ds = None
    return float(vmin), float(vmax)


def check_scale(vmin: float, vmax: float, scale: str) -> None:
    """Refuse a value range that contradicts the declared *scale*.

    The risk scale is bounded on both sides: a file whose maximum sits at or
    below 1 is a probability raster in disguise, and rounding it onto 1..65535
    would flatten every pixel to nodata or to the lowest risk category.
    """
    if scale not in VALUE_SCALES:
        raise ImportRasterError(
            f"Unknown value scale {scale!r}; expected one of {VALUE_SCALES}."
        )
    if vmin < 0:
        raise ImportRasterError(
            f"The raster holds negative values (range {vmin:g} to {vmax:g}); a "
            "risk map cannot. Check that fill values are declared as the "
            "raster's nodata."
        )
    if vmax == vmin:
        raise ImportRasterError(
            f"The raster is constant (range {vmin:g} to {vmax:g}, every pixel "
            "is the same value); it carries no risk information."
        )
    if scale == "probability" and vmax > 1:
        raise ImportRasterError(
            f"Values range from {vmin} to {vmax}, but the raster was declared as "
            "probabilities 0..1. Either pick the 1..65535 risk scale or rescale "
            "the file."
        )
    if scale == "risk" and vmax <= 1:
        raise ImportRasterError(
            f"Values range from {vmin} to {vmax}, at or below 1, but the raster "
            f"was declared on the 1..{RISK_MAX} risk scale — that is a "
            "probability raster. Pick the probability 0..1 scale instead."
        )
    if scale == "risk" and vmax > RISK_MAX:
        raise ImportRasterError(
            f"Values range from {vmin} to {vmax}, above the 1..{RISK_MAX} risk "
            "scale. Rescale the file first."
        )


def adapt_raster(
    src: PathLike, dst: PathLike, geobox, scale: str, *, blk_rows: int = 256
) -> Path:
    """Warp *src* onto *geobox* and write it as a 1..65535 UInt16 raster at *dst*.

    Two passes:

    1. ``xr_reproject`` with nearest-neighbour resampling into a hidden sibling
       temp file (``.<stem>.warp.tif``). Nearest is mandatory for both scales:
       no new values are invented and category boundaries stay sharp.
    2. A row-band pass that maps the warped values onto the contract and writes
       the canonical tiled layout with nodata 0:

       * ``"probability"``: nodata/NaN -> 0, else ``far.misc.rescale``;
       * ``"risk"``: nodata/NaN -> 0, else rounded and cast to uint16 (a value
         that rounds to 0 is nodata, exactly as the contract states).

    On the probability scale 0 is a real value -- the lowest probability, which
    rescales to 1 -- so it must survive the warp. That only needs saying for an
    integer source that declares no nodata: odc fills outside the footprint with
    0 for an integer warp, and nothing on disk would then tell the fill from a
    genuine 0 (see ``_write_uint16``). Warping such a source as float32 makes
    the fill NaN instead, which keeps the two apart. The risk scale needs no
    such care: there 0 *is* nodata by contract, so fill and 0 mean the same.

    ``blk_rows`` defaults to the canonical destination block height
    (``raster_profile.BLOCK_SIZE``), so one band covers whole tile rows instead
    of read-modify-writing every one of them twice.

    The temp file is removed whatever happens. A *dst* this call created is
    removed on failure; one that was already on disk is left alone, since pass 1
    writes only to the temp file and a warp error must not destroy an unrelated
    artifact. Cleanup never masks the error it is cleaning up after.

    The adapted raster is a display artifact like any other prediction, so the
    viewer's external overview pyramid is built once it is written -- best
    effort, exactly as ``mlmodels.base.BaseModel._ensure_display_overviews``
    does it: a raster that cannot be optimised still imports, it just draws
    slower.
    """
    if scale not in VALUE_SCALES:
        raise ImportRasterError(
            f"Unknown value scale {scale!r}; expected one of {VALUE_SCALES}."
        )
    from spatialrisk.geo_utils import xr_reproject

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Whether *we* created dst decides whether we may delete it on failure.
    dst_existed = dst.exists()
    tmp = dst.with_name(f".{dst.stem}.warp.tif")
    try:
        xr_reproject(
            raster_path=str(src),
            geobox=geobox,
            resampling_method="nearest",
            output_path=str(tmp),
            cast_dtype=_warp_cast_dtype(src, scale),
        )
        _write_uint16(tmp, dst, scale, blk_rows)
    except Exception:
        # suppress(): a read-only folder must surface as the warp/write error
        # the user can act on, not as a PermissionError from the cleanup.
        if not dst_existed:
            with suppress(OSError):
                dst.unlink(missing_ok=True)
        raise
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
    _build_display_overviews(dst)
    return dst


def _warp_cast_dtype(src: Path, scale: str) -> Optional[str]:
    """Return the dtype to warp *src* in, or ``None`` to keep its own.

    ``"float32"`` only for the one combination whose genuine zeros the warp
    would otherwise eat: a probability source in an integer dtype that declares
    no nodata. Deliberately narrow -- casting a source that *does* declare
    nodata could round the declared value away from the one on disk and turn
    every nodata pixel back into data, which is worse than the bug being fixed.
    float32 holds a probability far more precisely than the 16-bit scale it is
    headed for, so nothing is lost on the way.
    """
    if scale != "probability":
        return None
    with rasterio.open(src) as ds:
        if ds.nodata is None and np.issubdtype(np.dtype(ds.dtypes[0]), np.integer):
            return "float32"
    return None


def _build_display_overviews(path: Path) -> None:
    """Build the viewer's overview pyramid for the adapted raster. Best effort.

    Without it a zoomed-out tile of a 256 px tiled raster decodes the whole
    file (see :mod:`spatialrisk.overviews`). A failure to optimise must not
    fail the import, so it is logged and swallowed.
    """
    import spatialrisk.overviews as overviews

    if not path.exists():
        return
    try:
        overviews.ensure_overviews(path, min_pixels=overviews.OVERVIEW_MIN_PIXELS)
    except Exception:
        logger.exception("Could not build overviews for %s", path)


def _write_uint16(warped: Path, dst: Path, scale: str, blk_rows: int) -> None:
    """Second pass of :func:`adapt_raster`: contract mapping + canonical write."""
    from rasterio.windows import Window

    from spatialrisk.raster_profile import rasterio_profile

    with rasterio.open(warped) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint16", count=1, nodata=0)
        profile.update(rasterio_profile("uint16"))
        src_nodata = src.nodata
        # odc fills the pixels outside the source footprint with its
        # `resolve_fill_value`: NaN for a float warp (caught by ~isfinite
        # below), but 0 for an integer one — and when the source declares no
        # nodata, that 0 is carried into the warped file as an ordinary value.
        # Nothing on disk tells it apart from real data, so an integer warp
        # with no declared nodata treats 0 as nodata. Keyed on the warped
        # DTYPE, not on the scale, because it is the warp that decides the
        # fill. A probability source never reaches here as an integer —
        # `_warp_cast_dtype` sends it through the warp as float32 precisely so
        # that its genuine zeros stay data and rescale to 1 — so in practice
        # this drops the fill of a risk source, where 0 is nodata regardless.
        zero_is_fill = src_nodata is None and np.issubdtype(
            np.dtype(src.dtypes[0]), np.integer
        )
        # One normalisation for both the step and the window height: a 0 or a
        # float would otherwise yield empty or fractional windows silently.
        step = max(1, int(blk_rows))
        with rasterio.open(dst, "w", **profile) as out:
            for row0 in range(0, src.height, step):
                rows = min(step, src.height - row0)
                win = Window(0, row0, src.width, rows)
                arr = src.read(1, window=win).astype(np.float64)
                invalid = ~np.isfinite(arr)
                if src_nodata is not None and np.isfinite(src_nodata):
                    invalid |= arr == src_nodata
                if zero_is_fill:
                    invalid |= arr == 0
                out.write(_to_contract(arr, invalid, scale), 1, window=win)


def _to_contract(arr: np.ndarray, invalid: np.ndarray, scale: str) -> np.ndarray:
    """Map a float64 block to uint16 on the 1..65535 scale; ``invalid`` -> 0."""
    result = np.zeros(arr.shape, dtype=np.uint16)
    valid = ~invalid
    if not valid.any():
        return result
    if scale == "probability":
        import forestatrisk as far

        # rescale() mutates its input and clamps p < 1e-6 to 1e-6, so nodata
        # never lands on 0 by accident; clip guards floating noise above 1.
        values = np.clip(arr[valid], 0.0, 1.0)
        result[valid] = far.misc.rescale(values).astype(np.uint16)
    else:
        # 0..RISK_MAX, not 1..: on the risk scale 0 IS nodata by contract, and
        # the dialog says so. A pixel that rounds to 0 must therefore stay a
        # visible hole rather than be floored into valid lowest-risk data —
        # that floor is what turns a warp fill into silently wrong numbers
        # downstream. check_scale already refuses a file whose whole range
        # sits at or below 1, so this only ever bites stray sub-half pixels.
        values = np.clip(np.rint(arr[valid]), 0, RISK_MAX)
        result[valid] = values.astype(np.uint16)
    return result
