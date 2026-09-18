"""Tests for displaying variable layers on the map.

Covers predefined-catalogue visualization (palettes / randomVisualizer), the
black/white fallback, and the threaded GEE-layer add that avoids the
cross-event-loop crash.
"""

import inspect


def _named(name, **attrs):
    """Build a throwaway object whose ``type(obj).__name__`` equals ``name``."""
    return type(name, (), attrs)()


def test_is_mappable_predicate():
    """A variable is mappable if it carries a GEE image or is a local raster/vector."""
    from gui.scripts.map_helpers import is_mappable

    assert is_mappable(_named("GEEVar", gee_images=["img"])) is True
    assert is_mappable(_named("LocalRasterVar", gee_images=None)) is True
    assert is_mappable(_named("LocalVectorVar", gee_images=None)) is True
    # A GEE variable with no fetched image and an unknown type are not mappable.
    assert is_mappable(_named("GEEVar", gee_images=None)) is False
    assert is_mappable(_named("SomethingElse", gee_images=None)) is False


def test_styled_layer_categorical_is_black_white_not_random():
    """Categorical rasters must render as a 0=black, 1=white palette.

    Never the old ``ee.Image.randomVisualizer()`` (random RGB).
    """
    from gui.tile.variables_tile import _styled_layer

    class FakeImage:
        def randomVisualizer(self):  # pragma: no cover - must never be reached
            raise AssertionError("randomVisualizer must not be used")

    var = _named("LocalRasterVar", raster_type=_named("RT", value="categorical"))
    image = FakeImage()

    out_image, vis, render_kind = _styled_layer(image, var, None)

    assert out_image is image
    assert vis.get("palette") == ["000000", "ffffff"]
    assert vis.get("min") == 0
    assert vis.get("max") == 1
    assert render_kind == "categorical_fallback"


def test_styled_layer_predefined_binary_uses_catalogue_palette():
    """A predefined binary mask (e.g. rivers) renders with its catalogue palette.

    White background, feature colour — not the generic black/white default.
    """
    from gui.scripts.predefined_variables import PREDEFINED_CATALOGUE
    from gui.tile.variables_tile import _styled_layer

    var = _named("GEEVar", raster_type=_named("RT", value="categorical"))
    var.name = "rivers"
    image = object()

    out_image, vis, render_kind = _styled_layer(image, var, None)

    assert out_image is image
    assert vis["palette"] == PREDEFINED_CATALOGUE["rivers"]["vis_params"]["palette"]
    assert vis["palette"][0] == "ffffff"
    assert vis["min"] == 0 and vis["max"] == 1
    assert render_kind == "catalogue_palette"


def test_styled_layer_predefined_continuous_stretches_palette():
    """A predefined continuous var (altitude) keeps its terrain palette.

    Omits min/max when the AOI stretch can't be computed (no gee_interface).
    """
    from gui.tile.variables_tile import _styled_layer

    var = _named("GEEVar", raster_type=_named("RT", value="continuous"))
    var.name = "altitude"

    out_image, vis, render_kind = _styled_layer(object(), var, None)

    assert vis["palette"][0] == "006633"  # terrain ramp, not grayscale
    assert "min" not in vis and "max" not in vis
    assert render_kind == "catalogue_palette"


def test_styled_layer_subj_uses_random_visualizer():
    """The multi-class subjurisdiction layer routes through randomVisualizer().

    Random RGB per class, and is added with empty vis params.
    """
    from gui.tile.variables_tile import _styled_layer

    class FakeImage:
        def randomVisualizer(self):
            return "RGB_IMAGE"

    var = _named("GEEVar", raster_type=_named("RT", value="categorical"))
    var.name = "subj"

    out_image, vis, render_kind = _styled_layer(FakeImage(), var, None)

    assert out_image == "RGB_IMAGE"
    assert vis == {}
    assert render_kind == "random_visualizer"


def test_gee_layer_add_uses_sync_api_off_the_solara_loop():
    """GEE layers must be added via the blocking interface offloaded to a thread.

    Never the async map API. ``add_ee_layer_async`` awaited on Solara's loop
    touches eeclient session locks bound to the GEE interface's private loop and
    crashes with "bound to a different event loop".
    """
    from gui.tile.variables_tile import _add_gee_layer, _toggle_var_on_map

    # The toggle now runs entirely on a worker thread (one per click, via
    # spawn_in_context) rather than via asyncio.to_thread from an awaited
    # component-scoped task, so look for the call in the worker itself.
    toggle_src = inspect.getsource(_toggle_var_on_map)
    assert "add_ee_layer_async(" not in toggle_src  # not called on Solara's loop
    assert "_add_gee_layer" in toggle_src

    add_src = inspect.getsource(_add_gee_layer)
    assert "add_ee_layer_async(" not in add_src  # uses the blocking API, not async
    assert "add_ee_layer(" in add_src


def test_add_vector_on_map_registers_layer_under_key(tmp_path):
    """A local vector file is drawn as a GeoJSON layer registered under its key.

    Replacing any prior layer with the same key.
    """
    import geopandas as gpd
    from shapely.geometry import box

    from gui.scripts.map_helpers import add_vector_on_map

    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    path = tmp_path / "v.geojson"
    gdf.to_file(path, driver="GeoJSON")

    calls = {}

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            calls["removed"] = key

        def add_layer(self, layer, key=""):
            calls["added"] = key

    add_vector_on_map(FakeMap(), str(path), "myvar", "var_myvar")

    assert calls["removed"] == "var_myvar"
    assert calls["added"] == "var_myvar"


def test_variables_tile_does_not_render_derived_list():
    """The Variables tile must not show derived (processed) variables.

    That list now lives only in the Process tile.
    """
    from gui.tile import variables_tile

    src = inspect.getsource(variables_tile.VariablesTile)
    assert "DerivedVariableList" not in src


def test_source_toggle_labels_layers_as_raw():
    """The source-variable toggle must label its layers with an origin marker.

    It must hand the renderers a "[R] "-prefixed name, not the bare registry
    key — otherwise a raw variable and its harmonized counterpart render
    under identical names.
    """
    from gui.tile.variables_tile import _toggle_var_on_map

    # The labeling logic now lives in the module-level worker (see the
    # to_thread-removal note above).
    src = inspect.getsource(_toggle_var_on_map)
    assert "raw_layer_label(key)" in src
    assert "layer_name=key," not in src  # bare key must be gone
    assert "add_vector_on_map(map_, str(var.path), key," not in src
    assert "_add_gee_layer(map_, images[0], var, key, layer_key" not in src
