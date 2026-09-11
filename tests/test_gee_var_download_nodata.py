"""``GEEVar._download`` must pick a nodata/unmask fill the layer cannot contain.

The old behaviour hardcoded ``unmask_value=255, nodata_value=255`` for every
raster download — a byte/categorical convention. For continuous layers 255 is
a plausible real value (altitude in metres, precipitation in mm/year), so
genuine pixels became indistinguishable from fill: ``altitude.tif`` for an AOI
spanning 83-461 m lost every true 255 m pixel.

The rule under test:

- explicit ``export_nodata`` on the GEEVar always wins;
- continuous raster_type -> -32768 (int16-safe, impossible for altitude m,
  precipitation mm, slope degrees, temperature degC);
- categorical or unset raster_type -> 255 (existing byte convention).

``to_local_raster(raster_type=...)`` must resolve the override *before*
downloading, so a GEEVar without ``raster_type`` set still downloads with the
continuous sentinel when converted as continuous.
"""

from types import SimpleNamespace

import pytest

import spatialrisk.variables.gee_var as gee_var_module
from spatialrisk import Project
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.models import DataType, RasterType


class _FakeImage:
    """Stand-in for an ee.Image."""


class _FakeAoi:
    def geometry(self):
        return "geom"


@pytest.fixture()
def captured_download(tmp_path, monkeypatch):
    """Patch download_ee_image where gee_var imports it; capture kwargs."""
    calls = []

    def fake_download(image, path, **kwargs):
        calls.append(kwargs)
        path.touch()  # _download verifies the file exists afterwards

    monkeypatch.setattr(gee_var_module, "download_ee_image", fake_download)
    return calls


def _gee_var(tmp_path, **overrides) -> GEEVar:
    Project._ensure_model_schemas()
    kwargs = dict(
        name="altitude",
        data_type=DataType.raster,
        gee_images=[_FakeImage()],
    )
    kwargs.update(overrides)
    var = GEEVar(**kwargs)
    var.aoi = _FakeAoi()
    var.project = SimpleNamespace(
        folders=SimpleNamespace(data_raw_folder=tmp_path / "data_raw")
    )
    return var


def test_continuous_downloads_with_int16_sentinel(tmp_path, captured_download):
    """Continuous layers export with -32768, not the byte fill."""
    var = _gee_var(tmp_path, raster_type=RasterType.continuous)
    var._download()

    assert captured_download[0]["unmask_value"] == -32768
    assert captured_download[0]["nodata_value"] == -32768


def test_categorical_keeps_byte_convention(tmp_path, captured_download):
    """Categorical layers keep the established 255 byte convention."""
    var = _gee_var(tmp_path, name="subj", raster_type=RasterType.categorical)
    var._download()

    assert captured_download[0]["unmask_value"] == 255
    assert captured_download[0]["nodata_value"] == 255


def test_unset_raster_type_keeps_byte_convention(tmp_path, captured_download):
    """No raster_type means no safe assumption: keep the old behaviour."""
    var = _gee_var(tmp_path, raster_type=None)
    var._download()

    assert captured_download[0]["unmask_value"] == 255
    assert captured_download[0]["nodata_value"] == 255


def test_export_nodata_override_wins(tmp_path, captured_download):
    """An explicit export_nodata beats the raster_type-derived default."""
    var = _gee_var(tmp_path, raster_type=RasterType.continuous, export_nodata=-9999)
    var._download()

    assert captured_download[0]["unmask_value"] == -9999
    assert captured_download[0]["nodata_value"] == -9999


def test_to_local_raster_raster_type_override_reaches_download(
    tmp_path, captured_download
):
    """The conversion-time raster_type must be resolved BEFORE downloading."""
    var = _gee_var(tmp_path, raster_type=None)
    local = var.to_local_raster(raster_type=RasterType.continuous)

    assert captured_download[0]["unmask_value"] == -32768
    assert captured_download[0]["nodata_value"] == -32768
    assert local.raster_type == RasterType.continuous


def test_asset_id_path_resolves_to_image_for_download(
    tmp_path, captured_download, monkeypatch
):
    """A GEEVar built from an asset id (``path``, no ``gee_images``) must download.

    The validator accepts ``path`` set to an asset id, so ``_download`` has to
    resolve it to an ee.Image instead of raising
    "gee_images must be provided for download".
    """
    loaded = []

    def fake_image(asset_id):
        loaded.append(asset_id)
        return _FakeImage()

    monkeypatch.setattr(gee_var_module.ee, "Image", fake_image)
    var = _gee_var(tmp_path, name="custom", gee_images=None, path="projects/p/assets/x")

    paths = var._download()

    assert loaded == ["projects/p/assets/x"]
    assert len(captured_download) == 1
    assert paths[0].name == "custom.tif"
    # The resolved image is kept so the layer is mappable afterwards.
    assert var.gee_images and isinstance(var.gee_images[0], _FakeImage)
