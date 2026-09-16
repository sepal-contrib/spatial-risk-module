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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio

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
    """Refuse a value range that contradicts the declared *scale*."""
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
    if scale == "risk" and vmax > RISK_MAX:
        raise ImportRasterError(
            f"Values range from {vmin} to {vmax}, above the 1..{RISK_MAX} risk "
            "scale. Rescale the file first."
        )


def adapt_raster(
    src: PathLike, dst: PathLike, geobox, scale: str, *, blk_rows: int = 128
) -> Path:
    """Warp *src* onto *geobox* and write it as a 1..65535 UInt16 raster at *dst*.

    Two passes:

    1. ``xr_reproject`` with nearest-neighbour resampling into a hidden sibling
       temp file (``.<stem>.warp.tif``). Nearest is mandatory for both scales:
       no new values are invented and category boundaries stay sharp.
    2. A row-band pass that maps the warped values onto the contract and writes
       the canonical tiled layout with nodata 0:

       * ``"probability"``: nodata/NaN -> 0, else ``far.misc.rescale``;
       * ``"risk"``: nodata/NaN -> 0, else rounded and cast to uint16.

    The temp file is removed whatever happens. A partially written *dst* is
    removed on failure.
    """
    if scale not in VALUE_SCALES:
        raise ImportRasterError(
            f"Unknown value scale {scale!r}; expected one of {VALUE_SCALES}."
        )
    from spatialrisk.geo_utils import xr_reproject

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.warp.tif")
    try:
        xr_reproject(
            raster_path=str(src),
            geobox=geobox,
            resampling_method="nearest",
            output_path=str(tmp),
        )
        _write_uint16(tmp, dst, scale, blk_rows)
    except Exception:
        if dst.exists():
            dst.unlink()
        raise
    finally:
        if tmp.exists():
            tmp.unlink()
    return dst


def _write_uint16(warped: Path, dst: Path, scale: str, blk_rows: int) -> None:
    """Second pass of :func:`adapt_raster`: contract mapping + canonical write."""
    from rasterio.windows import Window

    from spatialrisk.raster_profile import rasterio_profile

    with rasterio.open(warped) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint16", count=1, nodata=0)
        profile.update(rasterio_profile("uint16"))
        src_nodata = src.nodata
        with rasterio.open(dst, "w", **profile) as out:
            for row0 in range(0, src.height, max(1, int(blk_rows))):
                rows = min(blk_rows, src.height - row0)
                win = Window(0, row0, src.width, rows)
                arr = src.read(1, window=win).astype(np.float64)
                invalid = ~np.isfinite(arr)
                if src_nodata is not None and np.isfinite(src_nodata):
                    invalid |= arr == src_nodata
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
        values = np.clip(np.rint(arr[valid]), 0, RISK_MAX)
        result[valid] = values.astype(np.uint16)
    return result
