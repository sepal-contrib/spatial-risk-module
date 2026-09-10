"""A variable can carry its own map visualization (``vis_params``).

Custom layers have no catalogue entry to take a palette from, so the user
picks one in the Variables modal. It must survive the project manifest, follow
the layer through the GEE download into the local raster, and a GEE asset id
must resolve to an image without going through ``_download`` first (the map
toggle needs it before the layer is downloaded).
"""

from types import SimpleNamespace

import pytest

import spatialrisk.variables.gee_var as gee_var_module
from spatialrisk import Project
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

MASK_VIS = {"palette": ["ffffff", "7b1fa2"], "min": 0, "max": 1}


class _FakeImage:
    """Stand-in for an ee.Image."""


class _FakeAoi:
    def geometry(self):
        return "geom"


@pytest.fixture()
def captured_download(monkeypatch):
    """Patch download_ee_image where gee_var imports it; capture (image, kwargs)."""
    calls = []

    def fake_download(image, path, **kwargs):
        calls.append((image, kwargs))
        path.touch()

    monkeypatch.setattr(gee_var_module, "download_ee_image", fake_download)
    return calls


def _gee_var(tmp_path, **overrides) -> GEEVar:
    """A GEEVar with a fake AOI and project folders."""
    Project._ensure_model_schemas()
    kwargs = dict(
        name="mangrove_mask",
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        gee_images=[_FakeImage()],
    )
    kwargs.update(overrides)
    var = GEEVar(**kwargs)
    var.aoi = _FakeAoi()
    var.project = SimpleNamespace(
        folders=SimpleNamespace(data_raw_folder=tmp_path / "data_raw")
    )
    return var


def test_vis_params_survives_the_manifest_round_trip():
    """What the user picked is saved with the project and restored on load."""
    Project._ensure_model_schemas()
    var = LocalRasterVar(
        name="x",
        path="/tmp/x.tif",
        raster_type=RasterType.categorical,
        vis_params=MASK_VIS,
    )
    data = var.model_dump(mode="json")
    assert data["vis_params"] == MASK_VIS
    assert LocalRasterVar.model_validate(data).vis_params == MASK_VIS


def test_to_local_raster_carries_vis_params(tmp_path, captured_download):
    """The downloaded raster renders with the palette chosen for the GEE layer."""
    var = _gee_var(tmp_path, vis_params=MASK_VIS)
    local = var.to_local_raster()
    assert local.vis_params == MASK_VIS


def test_resolve_images_builds_an_image_from_an_asset_id(monkeypatch):
    """An asset-id GEEVar resolves to an ee.Image on demand and keeps it."""
    built = []

    def fake_image(asset_id):
        built.append(asset_id)
        return _FakeImage()

    monkeypatch.setattr(gee_var_module.ee, "Image", fake_image)
    var = GEEVar(
        name="custom_layer",
        data_type=DataType.raster,
        path="projects/p/assets/x",
    )

    images = var.resolve_images()

    assert built == ["projects/p/assets/x"]
    assert len(images) == 1 and isinstance(images[0], _FakeImage)
    assert var.gee_images is images  # kept, so the next call is free
    var.resolve_images()
    assert built == ["projects/p/assets/x"]  # not rebuilt

    # A catalogue GEEVar already holds its image; nothing is rebuilt either.
    image = _FakeImage()
    ready = GEEVar(name="altitude", data_type=DataType.raster, gee_images=[image])
    assert ready.resolve_images()[0] is image
    assert built == ["projects/p/assets/x"]


def test_download_goes_through_resolve_images(tmp_path, captured_download, monkeypatch):
    """The lazy asset-id resolution is shared, not duplicated in ``_download``."""
    image = _FakeImage()
    monkeypatch.setattr(gee_var_module.ee, "Image", lambda _id: image)
    var = _gee_var(tmp_path, gee_images=None, path="projects/p/assets/x")
    var._download()
    assert captured_download[0][0] is image
