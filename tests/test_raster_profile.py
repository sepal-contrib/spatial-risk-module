"""The shared GeoTIFF layout: ZSTD tiles by default, deflate when unavailable."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from spatialrisk import raster_profile as rp  # noqa: E402


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(rp.COMPRESS_ENV, raising=False)


def test_prefers_zstd_when_both_backends_support_it(monkeypatch):
    """Default codec is ZSTD for rasterio and GDAL writers alike."""
    monkeypatch.setattr(rp, "rasterio_supports", lambda codec: True)
    monkeypatch.setattr(rp, "gdal_supports", lambda codec: True)
    assert rp.compression("rasterio") == "zstd"
    assert rp.compression("gdal") == "zstd"
    assert rp.rasterio_profile("uint16")["compress"] == "zstd"
    assert "COMPRESS=ZSTD" in rp.gdal_creation_options("uint16")


def test_falls_back_to_deflate_per_backend(monkeypatch):
    """A backend without ZSTD gets deflate; the other keeps ZSTD."""
    monkeypatch.setattr(rp, "rasterio_supports", lambda codec: codec != "zstd")
    monkeypatch.setattr(rp, "gdal_supports", lambda codec: True)
    assert rp.compression("rasterio") == "deflate"
    assert rp.compression("gdal") == "zstd"


def test_env_override_wins_and_is_validated(monkeypatch):
    """An explicit codec beats the probes; unknown names are rejected."""
    monkeypatch.setattr(rp, "rasterio_supports", lambda codec: True)
    monkeypatch.setenv(rp.COMPRESS_ENV, "lzw")
    assert rp.compression() == "lzw"
    monkeypatch.setenv(rp.COMPRESS_ENV, "bogus")
    with pytest.raises(ValueError):
        rp.compression()


def test_layout_is_256_tiles_with_a_dtype_aware_predictor():
    """Both profile flavours share the tiling and pick the predictor by dtype."""
    p = rp.rasterio_profile("float64")
    assert (p["tiled"], p["blockxsize"], p["blockysize"]) == (True, 256, 256)
    assert p["predictor"] == 3 and p["BIGTIFF"] == "YES"
    assert rp.rasterio_profile("uint16")["predictor"] == 2
    opts = rp.gdal_creation_options("uint32")
    assert {"TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "PREDICTOR=2"} <= set(opts)


def test_probes_agree_with_a_real_write(tmp_path):
    """The live probe matches what a real write with that codec produces."""
    codec = rp.compression("rasterio")
    assert rp.rasterio_supports(codec)
    path = tmp_path / "probe.tif"
    with rasterio.open(
        path,
        "w",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 4, 1, 1),
        **rp.rasterio_profile("uint8"),
    ) as dst:
        dst.write(np.arange(16, dtype="uint8").reshape(1, 4, 4))
    with rasterio.open(path) as src:
        assert src.profile["compress"] == codec
        assert src.is_tiled and src.block_shapes[0] == (256, 256)


def _strip_lzw(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype.name,
        crs="EPSG:32631",
        nodata=0,
        transform=from_origin(500000.0, 5000000.0, 30.0, 30.0),
        tiled=False,
        compress="lzw",
        predictor=2,
    ) as dst:
        dst.write(data, 1)


def test_reencode_in_place_converts_strips_and_keeps_everything_else(tmp_path):
    """An upstream LZW strip file becomes canonical with identical content."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 65535, size=(300, 290)).astype("uint16")
    path = tmp_path / "prob_mw.tif"
    _strip_lzw(path, data)
    assert not rp.is_canonical(path)

    assert rp.reencode_in_place(path) is True
    assert rp.is_canonical(path)
    with rasterio.open(path) as src:
        assert src.nodata == 0 and src.crs.to_epsg() == 32631
        assert src.transform == from_origin(500000.0, 5000000.0, 30.0, 30.0)
        np.testing.assert_array_equal(src.read(1), data)
    assert not list(tmp_path.glob(".*reencode*"))

    # Already canonical: a no-op that leaves the file untouched.
    before = path.stat().st_mtime_ns
    assert rp.reencode_in_place(path) is False
    assert path.stat().st_mtime_ns == before
