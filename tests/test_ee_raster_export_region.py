"""``download_ee_image`` must never put the AOI's *computed* geometry in the request.

Earth Engine rejects a download request whose region (or clip area) is the
``.geometry()`` of a large table asset with "Description length exceeds
maximum" — the server inlines the materialised geometry into the request
description. Reproduced 2026-09-11 with a single-feature asset of ~1.1M
coordinates: every variant using ``fc.geometry()`` failed, while clipping with
the collection itself and exporting over ``geometry().bounds()`` succeeded and
produced the very same export grid.

The rule under test:

- the clip uses the region object as given (Geometry, Feature or
  FeatureCollection are all valid ``clip`` areas);
- the export region handed to geedim is the *bounds* of the region — a
  four-corner rectangle regardless of how heavy the AOI is.
"""

import pytest

import spatialrisk.gee.ee_raster_export as export_module
from spatialrisk.gee.ee_raster_export import download_ee_image


class _Bounds:
    """Sentinel for ``<geometry>.bounds()``."""


class _FakeGeometry:
    """Stand-in for an ee.Geometry: has ``bounds`` but no ``geometry``."""

    def __init__(self):
        self.bounds_result = _Bounds()

    def bounds(self):
        return self.bounds_result


class _FakeCollection:
    """Stand-in for an ee.FeatureCollection / ee.Feature."""

    def __init__(self):
        self.geom = _FakeGeometry()

    def geometry(self):
        return self.geom


class _FakeAccessor:
    """Stand-in for the ``image.gd`` geedim accessor."""

    def __init__(self, image):
        self._image = image

    def prepareForExport(self, **kwargs):
        self._image.calls.append(("prepareForExport", kwargs))
        return self._image  # geedim returns an ee.Image carrying ``.gd`` again

    def toGeoTIFF(self, **kwargs):
        self._image.calls.append(("toGeoTIFF", kwargs))


class _FakeImage:
    """Stand-in for an ee.Image recording clip/unmask and the geedim calls."""

    def __init__(self):
        self.calls = []
        self.gd = _FakeAccessor(self)

    def clip(self, area):
        self.calls.append(("clip", area))
        return self

    def unmask(self, value, sameFootprint):
        self.calls.append(("unmask", value))
        return self


@pytest.fixture()
def fake_ee(monkeypatch):
    """Make the helper's ``isinstance(image, ee.Image)`` guard accept the fake."""
    monkeypatch.setattr(export_module.ee, "Image", _FakeImage)


def _run(region):
    image = _FakeImage()
    download_ee_image(
        image,
        "out.tif",
        region=region,
        scale=30,
        crs="EPSG:4326",
        unmask_value=255,
        nodata_value=255,
    )
    return image.calls


def test_collection_region_clips_with_collection_and_exports_its_bounds(fake_ee):
    """Collection/Feature AOI: clip(fc), export region = fc.geometry().bounds()."""
    fc = _FakeCollection()
    calls = _run(fc)

    clip_areas = [area for name, area in calls if name == "clip"]
    assert clip_areas == [fc], "clip must use the collection, not its geometry"

    ((_, prepare_kwargs),) = [c for c in calls if c[0] == "prepareForExport"]
    assert prepare_kwargs["region"] is fc.geom.bounds_result


def test_geometry_region_exports_its_bounds(fake_ee):
    """A bare ee.Geometry AOI: clip(geom), export region = geom.bounds()."""
    geom = _FakeGeometry()
    calls = _run(geom)

    clip_areas = [area for name, area in calls if name == "clip"]
    assert clip_areas == [geom]

    ((_, prepare_kwargs),) = [c for c in calls if c[0] == "prepareForExport"]
    assert prepare_kwargs["region"] is geom.bounds_result


def test_no_region_leaves_export_unbounded(fake_ee):
    """Without a region nothing is clipped and geedim gets region=None."""
    calls = _run(None)
    assert not [c for c in calls if c[0] == "clip"]
    ((_, prepare_kwargs),) = [c for c in calls if c[0] == "prepareForExport"]
    assert prepare_kwargs["region"] is None
