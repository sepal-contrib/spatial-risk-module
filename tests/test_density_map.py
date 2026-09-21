"""Continuous styling for the deforestation-density raster."""

import numpy as np

from gui.scripts.density_map import (
    density_colormap,
    density_layer_key,
    density_value_range,
)


def test_layer_keys_are_namespaced():
    """Density layers never collide with prediction layers on the map."""
    assert density_layer_key("reserve_abc123") == "density_reserve_abc123"


def test_colormap_is_a_matplotlib_colormap_object():
    """get_leaflet_tile_layer rejects dict colormaps — it needs the object."""
    from matplotlib.colors import Colormap

    assert isinstance(density_colormap(), Colormap)


def test_colormap_runs_green_to_yellow_to_red():
    """Low density is green, mid is yellow, high is red (not a yellow→red ramp).

    Checks hue via channel dominance rather than exact hex values, so a swap
    to another green-yellow-red ramp keeps passing while a yellow-only low
    end (the previous YlOrRd) fails.
    """
    cmap = density_colormap()
    low, mid, high = (cmap(x)[:3] for x in (0.0, 0.5, 1.0))
    assert low[1] > low[0] and low[1] > low[2]  # green dominates at the bottom
    assert mid[0] > 0.8 and mid[1] > 0.8 and mid[2] < 0.8  # yellow in the middle
    assert high[0] > high[1] and high[0] > high[2]  # red dominates at the top


def test_value_range_ignores_nodata(tmp_path):
    """vmin/vmax come from the valid pixels, not the -9999 fill."""
    from osgeo import gdal

    from spatialrisk.allocation import DENSITY_NODATA

    path = tmp_path / "d.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, 1, gdal.GDT_Float64)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(DENSITY_NODATA)
    arr = np.full((4, 4), DENSITY_NODATA)
    arr[0, 0] = 0.25
    arr[1, 1] = 0.75
    band.WriteArray(arr)
    ds = None

    vmin, vmax = density_value_range(path)

    assert vmin == 0.25
    assert vmax == 0.75


def test_value_range_falls_back_when_every_pixel_is_nodata(tmp_path):
    """An all-nodata raster still yields a usable range instead of raising."""
    from osgeo import gdal

    from spatialrisk.allocation import DENSITY_NODATA

    path = tmp_path / "empty.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, 1, gdal.GDT_Float64)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(DENSITY_NODATA)
    band.WriteArray(np.full((4, 4), DENSITY_NODATA))
    ds = None

    assert density_value_range(path) == (0.0, 1.0)


def test_add_density_on_map_returns_layer_and_value_range(monkeypatch, tmp_path):
    """The range it already computed is handed back so the legend can reuse it."""
    from gui.scripts import density_map

    monkeypatch.setattr(density_map, "density_value_range", lambda path: (0.5, 7.25))

    class FakeClient:
        def center(self):
            """Fixed map center used only when fit_bounds is requested."""
            return (0, 0)

        default_zoom = 5

    class FakeMap:
        def __init__(self):
            """Track the keys of layers added to this fake map."""
            self.added = []

        def remove_layer(self, key, none_ok=False):
            """No-op: nothing has been added yet in this test."""

        def add_layer(self, layer, key=None):
            """Record the layer key that was added."""
            self.added.append(key)

    fake_layer = object()
    monkeypatch.setitem(
        __import__("sys").modules,
        "localtileserver",
        type(
            "M",
            (),
            {
                "TileClient": lambda path: FakeClient(),
                "get_leaflet_tile_layer": lambda *a, **k: fake_layer,
            },
        ),
    )

    layer, value_range = density_map.add_density_on_map(
        FakeMap(), tmp_path / "d.tif", key="density_x", layer_name="run"
    )
    assert layer is fake_layer
    assert value_range == (0.5, 7.25)


def _fake_legend_port():
    """A minimal LegendPort double.

    Records calls instead of touching the app_state singleton — tiles get an
    explicit handle, not a global.
    """
    from gui.scripts.legend_registry import LegendPort

    registered = []
    unregistered = []
    return (
        LegendPort(
            register=lambda *legends: registered.extend(legends),
            unregister=lambda *ids: unregistered.extend(ids),
            generation=lambda: 0,
        ),
        registered,
        unregistered,
    )


def test_drop_density_layer_removes_layer_and_legend():
    """toggle-off and delete both go through _drop_density_layer."""
    from gui.tile import toolbox_tile

    removed = []
    port, _registered, unregistered = _fake_legend_port()

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """Record the key removed from the fake map."""
            removed.append(key)

    toolbox_tile._drop_density_layer(FakeMap(), "density_run1", port)

    assert removed == ["density_run1"]
    assert unregistered == ["density_run1"]


def test_drop_density_layer_tolerates_a_missing_port():
    """A None port is a no-op, not a crash — tiles can render without one."""
    from gui.tile import toolbox_tile

    removed = []

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """Record the key removed from the fake map."""
            removed.append(key)

    toolbox_tile._drop_density_layer(FakeMap(), "density_run1", None)

    assert removed == ["density_run1"]


def test_toggle_density_registers_through_the_port(monkeypatch, tmp_path):
    """The on-branch publishes a legend via legend_port, not app_state."""
    from gui.scripts import density_map
    from gui.tile import toolbox_tile

    monkeypatch.setattr(density_map, "density_value_range", lambda path: (0.0, 3.0))
    monkeypatch.setitem(
        __import__("sys").modules,
        "localtileserver",
        type(
            "M",
            (),
            {
                "TileClient": lambda path: type(
                    "C", (), {"center": lambda self: (0, 0), "default_zoom": 5}
                )(),
                "get_leaflet_tile_layer": lambda *a, **k: object(),
            },
        ),
    )

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """No-op: nothing is on the map yet in this test."""

        def add_layer(self, layer, key=None):
            """No-op: the layer object is not inspected in this test."""

    port, registered, _unregistered = _fake_legend_port()

    _layer, (vmin, vmax) = density_map.add_density_on_map(
        FakeMap(), tmp_path / "d.tif", key="density_run1", layer_name="Run 1"
    )
    port.register(toolbox_tile._density_legend("run1", "Run 1", vmin, vmax))

    assert len(registered) == 1
    legend = registered[0]
    assert legend.layer_id == "density_run1"
    assert [label.literal for label in legend.spec.labels] == ["0", "3"]


def test_density_legend_is_keyed_by_the_map_layer_key():
    """The legend's layer_id matches the key the raster was added under."""
    from gui.scripts.density_map import density_layer_key
    from gui.tile import toolbox_tile

    legend = toolbox_tile._density_legend("run1", "Run 1", 0.0, 3.0)
    assert legend.layer_id == density_layer_key("run1")
    assert legend.label.literal == "Run 1"
    assert [label.literal for label in legend.spec.labels] == ["0", "3"]
