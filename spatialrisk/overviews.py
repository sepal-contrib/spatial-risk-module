"""Build external raster overviews (pyramids) for fast low-zoom tiling.

Predictions are written as 256 px tiles (see ``spatialrisk.raster_profile``),
which is the right layout for the reads the app does at native zoom — but the
worst one for a *zoomed-out* tile. A decimated read of a row-stripe file only
touches the rows it samples; a tiled file has to decode every tile those
samples cross, i.e. the whole raster, for every tile the viewer asks for.

Measured on a 30000x30000 uint16 map, one tile covering the whole raster
through localtileserver's tiler: 0.19 s from LZW row strips, 2.6 s from 256 px
tiles, 0.09 s from either one once overviews exist. That cost scales with the
pixel count, so on a country-scale raster a zoomed-out map simply never draws.

``ensure_overviews`` builds them once, as **external** ``.ovr`` sidecars:
opening the dataset read-only makes GDAL write ``<path>.ovr`` and leaves the
source GeoTIFF byte-identical (predictions are registered project artifacts and
must not be mutated). rio-tiler / localtileserver read ``.ovr`` sidecars
automatically.
"""

import logging
from pathlib import Path

from osgeo import gdal

logger = logging.getLogger("spatial_risk")

# Below roughly 5000x5000 a full-resolution decimated read is already fast
# (~0.07 s), so the pyramid costs more than it saves.
OVERVIEW_MIN_PIXELS = 25_000_000


def _overview_options(ds) -> list:
    """Compress the pyramid the way the rasters themselves are compressed.

    A deflate sidecar of a ZSTD raster is no small extra: the overview levels
    hold a third of the base pixel count, so at the wrong codec the ``.ovr``
    can outweigh the raster it belongs to.
    """
    from spatialrisk.raster_profile import compression

    codec = compression("gdal")
    # Predictor from the band type rather than numpy: gdal_array is the one
    # GDAL binding that breaks when a stale pip wheel shadows the conda build.
    type_name = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    predictor = 3 if "Float" in type_name else 2
    return [
        f"COMPRESS_OVERVIEW={codec.upper()}",
        f"PREDICTOR_OVERVIEW={predictor}",
    ]


def overview_path(path) -> Path:
    """Path of the external overview sidecar GDAL writes for ``path``."""
    return Path(str(path) + ".ovr")


def ensure_overviews(
    path,
    resampling="average",
    levels=(2, 4, 8, 16, 32),
    *,
    min_pixels: int = 0,
) -> bool:
    """Build external overviews for ``path`` if it has none. Idempotent.

    A sidecar older than its raster is *stale*: GDAL serves whatever ``.ovr``
    sits next to the file, with no freshness check of its own, so a raster
    rewritten in place (``gdal.Translate`` + ``os.replace`` leaves the sidecar
    behind) would be drawn from the previous run's pixels. Such a sidecar is
    deleted and rebuilt — or just deleted, when the raster is below
    ``min_pixels``.

    Parameters
    ----------
    path : str | Path
        GeoTIFF to add overviews to.
    resampling : str
        GDAL resampling algorithm. ``average`` is nodata-aware because prediction
        rasters carry ``nodata=0``.
    levels : tuple[int, ...]
        Decimation factors for the overview pyramid.
    min_pixels : int
        Skip rasters smaller than this many pixels — they are already fast to
        decimate. ``0`` (the default) builds for any size.

    Returns ``True`` when a pyramid was built, ``False`` when one was already
    there or the raster is below ``min_pixels``.
    """
    path = str(path)
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster for overviews: {path}")
    try:
        pixels = ds.RasterXSize * ds.RasterYSize
        has_overviews = ds.GetRasterBand(1).GetOverviewCount() > 0
    finally:
        ds = None

    sidecar = overview_path(path)
    stale = sidecar.exists() and sidecar.stat().st_mtime < Path(path).stat().st_mtime
    if stale:
        logger.info("Overviews for %s are older than the raster; rebuilding", path)
        sidecar.unlink(missing_ok=True)
        has_overviews = False

    if min_pixels and pixels < min_pixels:
        return False
    if has_overviews:
        return False

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    try:
        # Per-call options (not the process-global config) keep the .ovr small
        # without leaking COMPRESS_OVERVIEW to later BuildOverviews calls.
        ds.BuildOverviews(resampling, list(levels), options=_overview_options(ds))
        return True
    finally:
        if ds is not None:
            ds.FlushCache()
            ds = None  # explicit flush then close
