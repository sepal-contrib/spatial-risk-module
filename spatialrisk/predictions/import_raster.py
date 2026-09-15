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
