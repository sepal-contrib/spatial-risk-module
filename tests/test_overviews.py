"""External (.ovr) overview generation for prediction rasters.

Idempotent, size-gated, and non-destructive to the source GeoTIFF.
"""

import hashlib
import os

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


def test_a_stale_sidecar_is_rebuilt(tmp_path):
    """A prediction rewritten under the same name must not keep old overviews.

    GDAL trusts any ``.ovr`` next to the raster, so a sidecar left over from a
    previous run would be served (wrong pixels) forever.
    """
    tif = tmp_path / "pred.tif"
    _write_raster(tif)
    assert ensure_overviews(str(tif)) is True
    sidecar = tmp_path / "pred.tif.ovr"
    built_at = sidecar.stat().st_mtime

    # The raster was rewritten in place (gdal.Translate + os.replace keeps the
    # sidecar), so it is now newer than the overviews built from it.
    os.utime(tif, (built_at + 10,) * 2)

    assert ensure_overviews(str(tif)) is True  # rebuilt, not skipped
    assert sidecar.exists()
    assert sidecar.stat().st_mtime != built_at  # a fresh pyramid, not the old one


def test_small_rasters_are_skipped_under_a_threshold(tmp_path):
    """Overviews only pay off on rasters big enough to be slow to decimate."""
    tif = tmp_path / "pred.tif"
    _write_raster(tif)  # 256x256

    assert ensure_overviews(str(tif), min_pixels=1_000_000) is False
    assert not (tmp_path / "pred.tif.ovr").exists()


def test_a_stale_sidecar_is_dropped_even_when_skipped(tmp_path):
    """A stale sidecar is wrong pixels, so it goes even when nothing is built."""
    tif = tmp_path / "pred.tif"
    _write_raster(tif)
    ensure_overviews(str(tif))
    sidecar = tmp_path / "pred.tif.ovr"
    os.utime(sidecar, (1, 1))

    assert ensure_overviews(str(tif), min_pixels=1_000_000) is False
    assert not sidecar.exists()


def test_min_pixels_counts_the_raster(tmp_path):
    """A raster at or above the threshold is built."""
    tif = tmp_path / "pred.tif"
    _write_raster(tif)  # 256*256 = 65536 px

    assert ensure_overviews(str(tif), min_pixels=65_536) is True


def test_sidecar_uses_the_canonical_codec(tmp_path):
    """The pyramid is compressed like the raster it belongs to.

    A deflate sidecar of a ZSTD raster can outweigh the raster itself.
    """
    from spatialrisk.raster_profile import compression

    tif = tmp_path / "pred.tif"
    _write_raster(tif)
    ensure_overviews(str(tif))

    with rasterio.open(tmp_path / "pred.tif.ovr") as ovr:
        assert ovr.profile["compress"].lower() == compression("gdal")
        assert ovr.profile.get("predictor", 2) == 2  # uint16 -> horizontal
