"""Inspection and scale checks for user-imported prediction rasters."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from spatialrisk.predictions.import_raster import (
    ImportRasterError,
    RasterInfo,
    check_scale,
    inspect_raster,
    raster_range,
)


def _write(path, data, *, count=1, crs="EPSG:4326", nodata=None, res=0.01):
    """Tiny GeoTIFF; ``data`` is 2-D and repeated for every band."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(-55.0, -24.0, res, res),
        nodata=nodata,
    ) as dst:
        for b in range(1, count + 1):
            dst.write(data, b)
    return path


# --- inspect_raster -----------------------------------------------------------


def test_inspect_reports_structure(tmp_path):
    """inspect_raster reports band count, dtype, CRS, nodata and geometry."""
    src = _write(tmp_path / "a.tif", np.ones((4, 6), dtype=np.uint16), nodata=0)
    info = inspect_raster(src)
    assert isinstance(info, RasterInfo)
    assert info.band_count == 1
    assert info.dtype == "uint16"
    assert info.crs == "EPSG:4326"
    assert info.nodata == 0
    assert info.width == 6 and info.height == 4
    assert info.resolution == pytest.approx((0.01, 0.01))


def test_inspect_rejects_two_bands(tmp_path):
    """A multi-band raster is not a valid single-band prediction."""
    src = _write(tmp_path / "two.tif", np.ones((4, 4), dtype=np.uint8), count=2)
    with pytest.raises(ImportRasterError, match="2 band"):
        inspect_raster(src)


def test_inspect_rejects_missing_crs(tmp_path):
    """A raster with no CRS cannot be placed on the project grid."""
    src = _write(tmp_path / "nocrs.tif", np.ones((4, 4), dtype=np.uint8), crs=None)
    with pytest.raises(ImportRasterError, match="coordinate reference system"):
        inspect_raster(src)


def test_inspect_rejects_missing_file(tmp_path):
    """An unreadable path raises ImportRasterError instead of propagating."""
    with pytest.raises(ImportRasterError, match="Cannot open"):
        inspect_raster(tmp_path / "nope.tif")


def test_inspect_reads_no_pixel_data(tmp_path, monkeypatch):
    """Safe for a Solara handler: metadata only."""
    src = _write(tmp_path / "a.tif", np.ones((4, 4), dtype=np.uint16))
    real_open = rasterio.open

    class _Guard:
        def __init__(self, ds):
            self._ds = ds

        def read(self, *a, **k):
            raise AssertionError("inspect_raster must not read pixels")

        def __getattr__(self, name):
            return getattr(self._ds, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._ds.__exit__(*exc)

    monkeypatch.setattr(
        "spatialrisk.predictions.import_raster.rasterio.open",
        lambda *a, **k: _Guard(real_open(*a, **k)),
    )
    inspect_raster(src)


# --- raster_range -------------------------------------------------------------


def test_range_excludes_nodata_and_nan(tmp_path):
    """raster_range ignores nodata and NaN pixels when computing min/max."""
    data = np.array([[0.2, 0.8], [np.nan, -9999.0]], dtype=np.float32)
    src = _write(tmp_path / "f.tif", data, nodata=-9999.0)
    vmin, vmax = raster_range(src)
    assert vmin == pytest.approx(0.2, abs=1e-6)
    assert vmax == pytest.approx(0.8, abs=1e-6)


def test_range_rejects_unreadable_file(tmp_path):
    """A corrupt file raises ImportRasterError instead of a silent bogus range.

    Without scoping gdal.ExceptionMgr, gdal.Open on a corrupt file just
    returns None (or logs and returns None) depending on process-wide GDAL
    exception state, rather than reliably raising; this pins the
    deterministic, always-raises behaviour.
    """
    src = tmp_path / "garbage.tif"
    src.write_bytes(b"not a real tiff file" * 20)
    with pytest.raises(ImportRasterError, match="Cannot open"):
        raster_range(src)


# --- check_scale --------------------------------------------------------------


def test_probability_scale_accepts_unit_interval():
    """0..1 values are valid for the probability scale."""
    check_scale(0.0, 1.0, "probability")


def test_probability_scale_rejects_values_above_one():
    """A range above 1 contradicts a declared probability scale."""
    with pytest.raises(ImportRasterError, match=r"0\.0 to 100\.0"):
        check_scale(0.0, 100.0, "probability")


def test_risk_scale_rejects_values_above_65535():
    """A range above the UInt16 max contradicts a declared risk scale."""
    with pytest.raises(ImportRasterError, match="65535"):
        check_scale(1.0, 70000.0, "risk")


def test_any_scale_rejects_negative_values():
    """Negative values are invalid under either declared scale."""
    with pytest.raises(ImportRasterError, match="negative"):
        check_scale(-1.0, 0.5, "probability")
    with pytest.raises(ImportRasterError, match="negative"):
        check_scale(-1.0, 10.0, "risk")


def test_negative_value_message_states_observed_range():
    """The negative-values error names both the min and the max, not just min."""
    with pytest.raises(ImportRasterError, match=r"-1(\.0)? to 0\.5"):
        check_scale(-1.0, 0.5, "probability")


def test_constant_raster_is_rejected():
    """A raster with no value spread carries no risk information."""
    with pytest.raises(ImportRasterError, match="constant"):
        check_scale(0.4, 0.4, "probability")


def test_unknown_scale_is_rejected():
    """An unrecognized scale token is rejected outright."""
    with pytest.raises(ImportRasterError, match="value scale"):
        check_scale(0.0, 1.0, "percent")
