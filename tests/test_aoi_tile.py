"""AoiTile restores through ``AoiView(spec=)`` and resets the picker on a switch.

pysepal 4 restores a picker by writing its ``spec`` channel, not by remounting
it, and treats ``spec=None`` from outside as a no-op. So the tile resets the
picker itself (``clear_ref``) on a project switch that ``AoiView`` would not
act on: a project with no spec (New, Close, no AOI, a legacy SHAPE/POINTS
manifest) and the same non-DRAW spec again.

``AoiView`` is replaced by a stand-in that records the props it gets and
answers ``clear_ref`` the way v4's clear does (value and spec to None), so
these tests pin the tile's own decision, not pysepal's picker.
"""

import json

import geopandas as gpd
import reacton
import solara
from pysepal.solara.components.aoi import AoiResult, AoiSpec
from shapely.geometry import box

import gui.tile.aoi_tile as aoi_tile

_DRAWING = [{"type": "Feature", "properties": {}, "geometry": None}]


class _FakeDrawControl:
    def __init__(self):
        self.data = list(_DRAWING)


class _FakeMap:
    """Just the map surface the tile touches: the draw control and keyed layers."""

    def __init__(self):
        self.dc = _FakeDrawControl()
        self.layers = {}

    def remove_layer(self, key, none_ok=False):
        self.layers.pop(key, None)

    def add_layer(self, layer, key=None):
        self.layers[key] = layer


def _draw_spec(bounds=(12.40, 43.89, 12.52, 43.99)):
    gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")
    return AoiSpec(method="DRAW", name="x", geo_json=json.loads(gdf.to_json()))


def _admin_spec(code="197"):
    return AoiSpec(method="ADMIN0", admin_codes=(code,))


def _result(spec):
    return AoiResult(method=spec.method, name="x", spec=spec)


class _Harness:
    """Render AoiTile under a host that feeds it the switch signal."""

    def __init__(self, monkeypatch, spec=None, result=None):
        self.map = _FakeMap()
        self.aoi_result = solara.reactive(result)
        self.aoi_spec = solara.reactive(spec)
        self.signal = solara.reactive(0)
        self.loading = solara.reactive(False)
        self.clears = []
        self.props = {}

        harness = self

        @solara.component
        def FakeAoiView(value, spec, clear_ref=None, map_=None, **props):
            harness.props.update(value=value, spec=spec, clear_ref=clear_ref, **props)

            def register():
                def clear():
                    harness.clears.append(True)
                    value.set(None)
                    spec.set(None)
                    map_.layers.pop("aoi", None)  # v4 drops every "aoi" layer

                clear_ref.current = clear

            solara.use_effect(register, [])

        monkeypatch.setattr(aoi_tile, "AoiView", FakeAoiView)

        @solara.component
        def Host():
            aoi_tile.AoiTile(
                map_=self.map,
                gee_interface=None,
                aoi_result=self.aoi_result,
                aoi_spec=self.aoi_spec,
                restore_signal=self.signal.value,
                loading=self.loading,
            )

        _, self.rc = reacton.render(Host(), handle_error=False)

    def switch(self, result):
        """What a project switch does: install the result, then bump."""
        self.aoi_result.set(result)
        self.signal.set(self.signal.value + 1)

    def close(self):
        self.rc.close()


def test_the_picker_gets_the_spec_channel_and_a_clear_handle(monkeypatch):
    """AoiView is driven through spec= (restore) and clear_ref= (reset)."""
    h = _Harness(monkeypatch)
    try:
        assert h.props["spec"] is h.aoi_spec
        assert h.props["value"] is h.aoi_result
        assert h.props["autoselect"] is True
        assert callable(h.props["clear_ref"].current)
    finally:
        h.close()


def test_mounting_never_resets_the_picker(monkeypatch):
    """A mounting picker restores from aoi_spec itself; a reset would wipe it."""
    h = _Harness(monkeypatch, spec=_admin_spec(), result=_result(_admin_spec()))
    try:
        assert h.clears == []
        assert h.aoi_spec.value == _admin_spec()
    finally:
        h.close()


def test_new_project_resets_the_picker_and_its_drawing(monkeypatch):
    """The DRAW -> New case: no stale selection, no stale editable shape."""
    h = _Harness(monkeypatch, spec=_draw_spec(), result=_result(_draw_spec()))
    try:
        h.switch(None)

        assert h.clears == [True]
        assert h.aoi_spec.value is None
        assert h.map.dc.data == []
    finally:
        h.close()


def test_a_project_without_a_spec_resets_the_picker_but_keeps_its_aoi(monkeypatch):
    """A legacy SHAPE project: blank picker, but the loaded AOI stays in use.

    The shell's switch effect draws the loaded AOI under the "aoi" key, and in
    the real tree it runs BEFORE this tile's (verified against MapApp): the
    reset's clear() drops that layer, so the tile has to draw it again.
    """
    h = _Harness(monkeypatch, spec=_draw_spec(), result=_result(_draw_spec()))
    try:
        gdf = gpd.GeoDataFrame(
            {"name": ["shp"]}, geometry=[box(1, 1, 2, 2)], crs="EPSG:4326"
        )
        legacy_shape = AoiResult(method="SHAPE", name="shp", gdf=gdf, spec=None)
        h.map.layers["aoi"] = "drawn by the shell's switch effect"
        h.switch(legacy_shape)

        assert h.clears == [True]
        assert h.aoi_result.value is legacy_shape
        assert h.map.dc.data == []
        assert h.map.layers["aoi"].data["features"][0]["geometry"]["type"] == (
            "Polygon"
        )
    finally:
        h.close()


def test_a_different_spec_is_left_to_the_picker(monkeypatch):
    """AoiView restores a changed spec itself: no reset, drawing untouched."""
    h = _Harness(monkeypatch, spec=_admin_spec("197"), result=_result(_admin_spec()))
    try:
        h.switch(_result(_admin_spec("178")))

        assert h.clears == []
        assert h.aoi_spec.value == _admin_spec("197")  # do_load writes it next
        assert h.map.dc.data == _DRAWING
    finally:
        h.close()


def test_the_same_admin_spec_again_is_reset_so_it_restores(monkeypatch):
    """Equal specs are not re-applied; the reset makes the next write count."""
    h = _Harness(monkeypatch, spec=_admin_spec(), result=_result(_admin_spec()))
    try:
        # Another project with the same AOI (a distinct name, so the reactive
        # does not swallow the write as an equal value).
        again = AoiResult(method="ADMIN0", name="other_project", spec=_admin_spec())
        h.switch(again)

        assert h.clears == [True]
        assert h.aoi_spec.value is None
        assert h.aoi_result.value is again
    finally:
        h.close()


def test_a_drawing_is_never_reset_before_its_restore(monkeypatch):
    """A reset's dc.clear() lands after the re-seed in a browser and wipes it."""
    h = _Harness(monkeypatch, spec=_draw_spec(), result=_result(_draw_spec()))
    try:
        h.switch(_result(_draw_spec()))  # the same drawing again
        h.switch(_result(_draw_spec(bounds=(20.0, 10.0, 20.2, 10.2))))

        assert h.clears == []
        assert h.map.dc.data == _DRAWING
    finally:
        h.close()
