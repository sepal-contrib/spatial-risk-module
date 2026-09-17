"""``reproject_and_match`` must fail loudly when it writes an all-nodata raster.

2026-08-03: an app run silently wrote ``temperature_2m_max_reprojected_matched``
as 27.5M pixels of constant nodata while the input was valid. Nothing raised;
the corruption surfaced later as an empty map layer. The guard turns that into
an immediate error — but must stay quiet when the output has any valid pixel,
and when the *input* was already all-nodata (then the output is faithful).
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import spatialrisk.variables.local_raster_var as lrv_module
from spatialrisk import Project
from spatialrisk.geo_utils import raster_is_all_nodata
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import RasterType

NODATA = -32768.0


def _write_raster(path: Path, data: np.ndarray, nodata=NODATA):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(-55.0, -24.0, 0.01, 0.01),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_raster_is_all_nodata_detects_constant_fill(tmp_path):
    """A constant-nodata raster is detected."""
    path = tmp_path / "bad.tif"
    _write_raster(path, np.full((64, 64), NODATA, dtype=np.float64))
    assert raster_is_all_nodata(path) is True


def test_raster_is_all_nodata_passes_partial_data(tmp_path):
    """A single valid pixel is enough to pass the check."""
    data = np.full((64, 64), NODATA, dtype=np.float64)
    data[10, 10] = 26.0  # a single valid pixel must be enough
    path = tmp_path / "sparse.tif"
    _write_raster(path, data)
    assert raster_is_all_nodata(path) is False


def test_raster_is_all_nodata_handles_nan_nodata(tmp_path):
    """NaN nodata needs isnan, not equality."""
    path = tmp_path / "nan.tif"
    _write_raster(path, np.full((16, 16), np.nan, dtype=np.float64), nodata=np.nan)
    assert raster_is_all_nodata(path) is True


@pytest.fixture()
def local_var(tmp_path):
    """LocalRasterVar over a small valid raster with stub project folders."""
    Project._ensure_model_schemas()
    raw = tmp_path / "raw.tif"
    data = np.full((32, 32), 26.0, dtype=np.float64)
    _write_raster(raw, data)

    var = LocalRasterVar.model_construct(
        name="temperature",
        path=raw,
        raster_type=RasterType.continuous,
        project=None,
    )
    processed = tmp_path / "data"
    processed.mkdir()

    class _Folders:
        processed_data_folder = processed

    class _Project:
        folders = _Folders()

    var.project = _Project()
    return var


class _FakeGeobox:
    """Just enough geobox for the metadata extraction and the grid stamp."""

    class crs:
        @staticmethod
        def to_epsg():
            return 32721

    class resolution:
        x = 30.0

    # A real GeoBox always carries these; reproject_and_match stamps the
    # output's grid_signature from them (see harmonization.geobox_signature).
    transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)

    class shape:
        yx = (32, 32)


def _fake_xr_reproject_writing(value):
    def fake(raster_path, geobox, resampling_method, output_path):
        _write_raster(Path(output_path), np.full((32, 32), value, dtype=np.float64))

    return fake


def test_reproject_and_match_raises_on_all_nodata_output(local_var, monkeypatch):
    """All-nodata output from a valid input must raise."""
    monkeypatch.setattr(lrv_module, "xr_reproject", _fake_xr_reproject_writing(NODATA))
    with pytest.raises(ValueError, match="nodata"):
        local_var.reproject_and_match(geobox=_FakeGeobox())


def test_reproject_and_match_accepts_valid_output(local_var, monkeypatch):
    """A valid output passes the guard untouched."""
    monkeypatch.setattr(lrv_module, "xr_reproject", _fake_xr_reproject_writing(26.0))
    result = local_var.reproject_and_match(geobox=_FakeGeobox())
    assert result.path.exists()


def test_reproject_and_match_allows_all_nodata_when_input_was(local_var, monkeypatch):
    """A faithful all-nodata copy of an all-nodata input is not corruption."""
    _write_raster(local_var.path, np.full((32, 32), NODATA, dtype=np.float64))
    monkeypatch.setattr(lrv_module, "xr_reproject", _fake_xr_reproject_writing(NODATA))
    result = local_var.reproject_and_match(geobox=_FakeGeobox())
    assert result.path.exists()
