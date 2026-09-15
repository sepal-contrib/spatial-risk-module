"""External (.ovr) overview generation for prediction rasters.

Idempotent, size-gated, and non-destructive to the source GeoTIFF.
"""

import hashlib

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spatialrisk.overviews import ensure_overviews


def _write_raster(path):
    """A small UInt16 raster with nodata=0 (mirrors prediction outputs)."""
    data = (np.arange(256 * 256, dtype=np.uint16).reshape(256, 256) % 65535) + 1
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "height": 256,
        "width": 256,
        "nodata": 0,
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1 / 256, 1 / 256),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_creates_external_overviews(tmp_path):
    """The pyramid lands in a sidecar; the raster keeps its bytes."""
    tif = tmp_path / "pred.tif"
    _write_raster(tif)
    before = _digest(tif)

    built = ensure_overviews(str(tif))

    assert built is True
    # external sidecar written, source bytes untouched
    assert (tmp_path / "pred.tif.ovr").exists()
    assert _digest(tif) == before

    with rasterio.open(tif) as src:
        assert len(src.overviews(1)) > 0


def test_second_call_is_noop(tmp_path):
    """A raster that already has overviews is left alone."""
    tif = tmp_path / "pred.tif"
    _write_raster(tif)

    assert ensure_overviews(str(tif)) is True
    assert ensure_overviews(str(tif)) is False  # idempotent
