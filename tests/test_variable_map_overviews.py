"""Variable and derived layers get an overview pyramid before they reach the map.

Without one, a zoomed-out tile of a large raster decodes the whole file: the
2.2 Gpx BOL slope variable took ~50 s a tile. Unlike predictions, variables are
often categorical (forest masks, class maps), so the pyramid must pick real
pixel values (``nearest``), never average classes into ones that don't exist.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import gui.scripts.variable_map as vmap
import spatialrisk.overviews as overviews

CLASSES = {3, 5}


class FakeMap:
    """Accepts the helper's layer calls; the tests assert on files, not the map."""

    def remove_layer(self, key, none_ok=False):
        """Accept the replace-by-key removal."""

    def add_layer(self, layer, key=""):
        """Accept the added layer."""


@pytest.fixture
def fake_tiles(monkeypatch):
    """Keep the real overview build but skip the tile server itself."""
    import localtileserver

    monkeypatch.setattr(
        localtileserver, "TileClient", lambda path: object(), raising=False
    )
    monkeypatch.setattr(
        localtileserver,
        "get_leaflet_tile_layer",
        lambda client, **kwargs: "FAKE_LAYER",
        raising=False,
    )


def _class_checkerboard_tif(path, size=64):
    """A categorical raster whose neighbouring pixels alternate class 3 and 5."""
    rows, cols = np.indices((size, size))
    data = np.where((rows + cols) % 2 == 0, 3, 5).astype("uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 1 / size, 1 / size),
        nodata=255,
    ) as dst:
        dst.write(data, 1)


def _categorical_var():
    return type(
        "LocalRasterVar",
        (),
        {"name": "landcover", "raster_type": type("RT", (), {"value": "categorical"})},
    )()


def _add(path):
    return vmap.add_raster_var_on_map(
        FakeMap(), str(path), var=_categorical_var(), layer_name="lc", key="var_lc"
    )


def test_large_variable_gets_a_pyramid(fake_tiles, monkeypatch, tmp_path):
    """A raster over the size threshold is drawn with an overview sidecar."""
    monkeypatch.setattr(overviews, "OVERVIEW_MIN_PIXELS", 0)
    tif = tmp_path / "landcover.tif"
    _class_checkerboard_tif(tif)

    _add(tif)

    assert overviews.overview_path(tif).exists()


def test_pyramid_keeps_only_real_classes(fake_tiles, monkeypatch, tmp_path):
    """Zoomed-out levels show classes 3 and 5, never an averaged 4."""
    monkeypatch.setattr(overviews, "OVERVIEW_MIN_PIXELS", 0)
    tif = tmp_path / "landcover.tif"
    _class_checkerboard_tif(tif)

    _add(tif)

    with rasterio.open(tif) as src:
        factors = src.overviews(1)
        assert factors
        for f in factors:
            level = src.read(1, out_shape=(src.height // f, src.width // f))
            assert set(np.unique(level)) <= CLASSES, f"level {f} invented values"


def test_small_variable_is_left_alone(fake_tiles, tmp_path):
    """Below the threshold a full-resolution read is already fast: no sidecar."""
    tif = tmp_path / "landcover.tif"
    _class_checkerboard_tif(tif)

    _add(tif)

    assert not overviews.overview_path(tif).exists()


def test_a_failed_pyramid_still_draws_the_layer(fake_tiles, monkeypatch, tmp_path):
    """The pyramid is an optimisation: a failure to build it must not block the map."""

    def boom(*args, **kwargs):
        raise RuntimeError("read-only folder")

    monkeypatch.setattr(overviews, "ensure_overviews", boom)
    tif = tmp_path / "landcover.tif"
    _class_checkerboard_tif(tif)

    assert _add(tif) == "FAKE_LAYER"
