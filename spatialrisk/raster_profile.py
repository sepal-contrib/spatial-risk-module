"""One GeoTIFF layout for every raster this package writes for the app.

Predictions, density maps and the MW rate rasters used to be written the way
each writer happened to: inherited from the target's profile (ML models),
LZW row strips (MW), deflate strips (allocation). Row strips are the worst
layout for the two ways the app reads them back: the map viewer pulls random
256 px tiles, and a 256 px tile on a strip file decodes 256 full-width strips.

Measured 2026-09-15 on a 20000x20000 uint16 risk map with realistic spatial
autocorrelation (30% nodata), 16-core box::

    layout                          write   size   200 random tiles
    strips, LZW + predictor 2       13.7 s  570 MB  6.2 s
    strips, deflate                  8.1 s  526 MB  4.6 s
    tiles 256, deflate + pred 2     10.2 s  439 MB  0.24 s
    tiles 256, ZSTD + pred 2         2.3 s  433 MB  0.17 s

So: 256 px tiles with a predictor and BIGTIFF. Nearly all of that win is the
tiling; the codec is the small remainder. Head to head at the same tiling
(2026-09-16, same raster, encode time excluding data generation)::

    codec     encode   size   200 tiles   whole-raster decode   band scan
    deflate     5.4 s  493 MB     0.20 s                1.6 s      1.8 s
    ZSTD        1.4 s  493 MB     0.14 s                1.4 s      1.5 s

Deflate is the default anyway: predictions leave this app (QGIS, ArcGIS,
collaborators on older GDAL builds) and deflate is readable everywhere, while
ZSTD needs a libgdal built with libzstd. It buys a ~4x faster encode and a
15-40% faster decode at the same size, so ``SPATIAL_RISK_RASTER_COMPRESS=zstd``
turns it on where the outputs stay inside a SEPAL sandbox; a forced codec the
writing backend cannot produce falls back to deflate rather than failing.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger("spatial_risk")

COMPRESS_ENV = "SPATIAL_RISK_RASTER_COMPRESS"
"""Environment override for the codec (``zstd``, ``deflate``, ``lzw``)."""

DEFAULT_COMPRESS = "deflate"
"""Readable by any GDAL build — the outputs are shared outside this app."""

BLOCK_SIZE = 256
_CODECS = ("zstd", "deflate", "lzw")


@lru_cache(maxsize=None)
def rasterio_supports(codec: str) -> bool:
    """Whether rasterio's own libgdal can *create* a GeoTIFF with ``codec``.

    A real in-memory write rather than a version check: rasterio may ship its
    own libgdal (the pip wheel does), so ``osgeo.gdal`` is not evidence.
    """
    from rasterio.io import MemoryFile

    try:
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                compress=codec,
            ) as dst:
                dst.write(np.zeros((1, 2, 2), dtype="uint8"))
            with mem.open() as src:
                return (src.profile.get("compress") or "").lower() == codec
    except Exception:  # any failure means "not supported"
        return False


@lru_cache(maxsize=None)
def gdal_supports(codec: str) -> bool:
    """Whether ``osgeo.gdal``'s libgdal offers ``codec`` for GeoTIFF creation."""
    try:
        from osgeo import gdal

        opts = gdal.GetDriverByName("GTiff").GetMetadataItem("DMD_CREATIONOPTIONLIST")
        return codec.upper() in (opts or "")
    except Exception:
        return False


def compression(backend: str = "rasterio") -> str:
    """The codec to write with: deflate, or the env override where it works.

    ``backend`` names the library that will do the writing (``"rasterio"`` or
    ``"gdal"``), because in some environments they are two different libgdal
    builds with different codec sets — an overridden codec has to be probed
    against the one that will actually write the file.
    """
    forced = os.environ.get(COMPRESS_ENV, "").strip().lower()
    if not forced:
        return DEFAULT_COMPRESS
    if forced not in _CODECS:
        raise ValueError(f"{COMPRESS_ENV}={forced!r}: expected one of {_CODECS}")
    supports = rasterio_supports if backend == "rasterio" else gdal_supports
    if supports(forced):
        return forced
    logger.info(
        "GDAL (%s) has no %s support; writing rasters with %s instead.",
        backend,
        forced.upper(),
        DEFAULT_COMPRESS.upper(),
    )
    return DEFAULT_COMPRESS


def predictor_for(dtype) -> int:
    """TIFF predictor: 3 (floating point) for floats, 2 (horizontal) otherwise."""
    return 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2


def rasterio_profile(dtype) -> dict:
    """Creation keywords to ``update()`` a rasterio profile with."""
    return dict(
        driver="GTiff",
        tiled=True,
        blockxsize=BLOCK_SIZE,
        blockysize=BLOCK_SIZE,
        compress=compression("rasterio"),
        predictor=predictor_for(dtype),
        BIGTIFF="YES",
    )


def gdal_creation_options(dtype) -> list:
    """The same layout as ``KEY=VALUE`` strings for ``gdal.Driver.Create``."""
    return [
        "TILED=YES",
        f"BLOCKXSIZE={BLOCK_SIZE}",
        f"BLOCKYSIZE={BLOCK_SIZE}",
        f"COMPRESS={compression('gdal').upper()}",
        f"PREDICTOR={predictor_for(dtype)}",
        "BIGTIFF=YES",
    ]


def is_canonical(path) -> bool:
    """True when ``path`` already has the tiling and codec this module writes."""
    with rasterio.open(path) as src:
        if not src.is_tiled or src.block_shapes[0] != (BLOCK_SIZE, BLOCK_SIZE):
            return False
        return (src.profile.get("compress") or "").lower() == compression("gdal")


def reencode_in_place(path) -> bool:
    """Rewrite ``path`` in the canonical layout; True if it was rewritten.

    For rasters produced by code we do not own (``riskmapjnr`` writes the MW
    and JNR maps with its own creation options). ``gdal.Translate`` preserves
    georeferencing, nodata and band values; the rewrite lands in a sibling
    temp file and replaces the original atomically, so a crash mid-way leaves
    the upstream file intact.
    """
    from osgeo import gdal

    path = Path(path)
    if is_canonical(path):
        return False
    with rasterio.open(path) as src:
        dtype = src.dtypes[0]
    tmp = path.with_name(f".{path.stem}.reencode{path.suffix}")
    try:
        ds = gdal.Translate(
            str(tmp), str(path), creationOptions=gdal_creation_options(dtype)
        )
        if ds is None:
            raise RuntimeError(f"gdal.Translate failed for {path}")
        ds = None  # flush + close before the rename
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    logger.info("Re-encoded %s as tiled %s.", path.name, compression("gdal").upper())
    return True
