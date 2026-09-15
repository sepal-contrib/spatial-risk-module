"""Prediction map layers.

Predictions are drawn with the QGIS-faithful palette (pinned vmin/vmax), and
overviews are built only when the opt-in flag is set.
"""

import gui.scripts.prediction_map as pm


class FakeClient:
    """Stand-in for localtileserver's TileClient."""

    def __init__(self, path):
        """Record the raster path a real TileClient would open."""
        self.path = path

    def center(self):
        """Report a fixed center, like a real TileClient would."""
        return (0.0, 0.0)

    default_zoom = 5


class FakeMap:
    """Stand-in for the ipyleaflet map passed to add_prediction_on_map."""

    def __init__(self):
        """Track layers added/removed instead of touching a real map."""
        self.removed = []
        self.added = []
        self.center = None
        self.zoom = None

    def remove_layer(self, key, none_ok=False):
        """Record the removed layer key."""
        self.removed.append(key)

    def add_layer(self, layer, key=""):
        """Record the added (layer, key) pair."""
        self.added.append((layer, key))


def _patch_localtileserver(monkeypatch):
    """Stub TileClient + get_leaflet_tile_layer; capture the tile-layer kwargs."""
    captured = {}

    def fake_get_layer(client, **kwargs):
        captured.update(kwargs)
        return "FAKE_LAYER"

    import localtileserver

    monkeypatch.setattr(localtileserver, "TileClient", FakeClient, raising=False)
    monkeypatch.setattr(
        localtileserver, "get_leaflet_tile_layer", fake_get_layer, raising=False
    )
    return captured


def test_prediction_added_with_pinned_far_palette(monkeypatch, tmp_path):
    """A FAR prediction gets the QGIS-faithful ramp pinned to 1..65535."""
    captured = _patch_localtileserver(monkeypatch)
    tif = tmp_path / "p.tif"
    tif.write_bytes(b"")
    fake_map = FakeMap()

    layer = pm.add_prediction_on_map(
        fake_map,
        str(tif),
        model_key="glm_m1",
        layer_name="glm_m1__d",
        key="pred_glm_m1__d",
    )

    assert layer == "FAKE_LAYER"
    assert captured["vmin"] == 1 and captured["vmax"] == 65535
    assert captured["nodata"] == 0
    from matplotlib.colors import Colormap

    assert isinstance(captured["colormap"], Colormap)
    assert tuple(round(x * 255) for x in captured["colormap"](0.0)[:3]) == (
        34,
        139,
        34,
    )  # FAR green
    assert fake_map.removed == ["pred_glm_m1__d"]  # replaced existing
    assert fake_map.added[0][1] == "pred_glm_m1__d"


def test_stretch_palette_omits_vmin_vmax_for_autostretch(monkeypatch, tmp_path):
    """display_palette='stretch' auto-stretches the ramp to the file's range.

    No pinned vmin/vmax are passed to the tile layer.
    """
    captured = _patch_localtileserver(monkeypatch)
    tif = tmp_path / "p.tif"
    tif.write_bytes(b"")

    pm.add_prediction_on_map(
        FakeMap(),
        str(tif),
        model_key="imported-map",
        layer_name="n",
        key="k",
        display_palette="stretch",
    )
    from matplotlib.colors import Colormap

    assert isinstance(captured["colormap"], Colormap)  # still a ramp, just unpinned
    assert captured["vmin"] is None and captured["vmax"] is None


def test_far_palette_pins_range_regardless_of_model_key(monkeypatch, tmp_path):
    """display_palette='far' pins the 1..65535 FAR ramp regardless of model_key.

    Even an imported name that wouldn't resolve to the far family on its own
    still gets the pinned range.
    """
    captured = _patch_localtileserver(monkeypatch)
    tif = tmp_path / "p.tif"
    tif.write_bytes(b"")

    pm.add_prediction_on_map(
        FakeMap(),
        str(tif),
        model_key="imported-map",
        layer_name="n",
        key="k",
        display_palette="far",
    )
    assert captured["vmin"] == 1 and captured["vmax"] == 65535


def test_overviews_are_always_built(monkeypatch, tmp_path):
    """The pyramid is not optional.

    A tiled prediction without one makes every zoomed-out tile decode the whole
    raster, so there is no useful way to display one un-optimised.
    """
    _patch_localtileserver(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "spatialrisk.overviews.ensure_overviews",
        lambda p, **k: calls.append((p, k)) or True,
    )
    tif = tmp_path / "p.tif"
    tif.write_bytes(b"")

    pm.add_prediction_on_map(
        FakeMap(), str(tif), model_key="mw_5", layer_name="n", key="k"
    )

    assert [c[0] for c in calls] == [str(tif)]
    # ...but small rasters stay untouched: they decimate fast enough already.
    import spatialrisk.overviews as ov

    assert calls[0][1]["min_pixels"] == ov.OVERVIEW_MIN_PIXELS


def test_inference_tile_uses_palette_helper_and_overview_option():
    """Predictions route through the QGIS-faithful helper, not bare add_raster.

    The helper always builds the overview pyramid, and the add runs off the
    Solara loop.
    """
    import inspect

    from gui.tile import inference_tile

    src = inspect.getsource(inference_tile.InferenceTile)
    assert "add_prediction_on_map" in src  # value-pinned palette path
    assert "map_.add_raster(" not in src  # no more bare grayscale add
    # No overview opt-out: a tiled prediction with no pyramid makes every
    # zoomed-out tile decode the whole raster, so a country-scale map never draws.
    assert "gen_overviews" not in src
    assert "build_overviews" not in src
    assert "to_thread" in src  # add offloaded to a thread
    assert "use_task" in src  # threaded via solara.lab.use_task
    assert "pending_toggle" in src  # toggle routed through the reactive
    # Adding a prediction must NOT recenter/rezoom the map — keep the user's view.
    assert "fit_bounds=False" in src
    assert "fit_bounds=True" not in src
    # Each prediction's stored palette drives its map display (imports vs computed).
    assert "display_palette" in src


def test_inference_tile_supports_local_prediction_import():
    """Step 7 lets the user import a local raster as a prediction.

    The New prediction dialog has an import mode (file picker + palette
    choice), and the import script is wired to the registry + reactive. The
    picker/palette form lives in PredictionFormDialog (unified creation
    dialog).
    """
    import inspect

    from gui.tile import inference_tile
    from gui.widget import prediction_form_dialog

    src = inspect.getsource(inference_tile)
    assert "import_prediction" in src  # routes through the import adapter
    assert "sepal_client" in src  # picker needs the SEPAL client
    # New prediction is published so the outputs list + Evaluation maps update.
    assert "project.set(" in src or "project_reactive.set(" in src

    dialog_src = inspect.getsource(prediction_form_dialog)
    assert "FileInputComponent" in dialog_src  # local raster file picker
    assert "_import_palette_items" in dialog_src  # palette choice


def _fake_legend_port():
    """A minimal LegendPort double with a bumpable generation.

    Records calls instead of touching the app_state singleton — tiles get an
    explicit handle, not a global (see tests/test_density_map.py's
    ``_fake_legend_port`` for the sibling shape used by the density tile;
    this one also exposes a ``bump`` to drive the staleness guard, which
    density's synchronous add does not need).
    """
    from gui.scripts.legend_registry import LegendPort

    registered = []
    unregistered = []
    state = {"generation": 0}

    def bump():
        state["generation"] += 1

    port = LegendPort(
        register=lambda *legends: registered.extend(legends),
        unregister=lambda *ids: unregistered.extend(ids),
        generation=lambda: state["generation"],
    )
    return port, registered, unregistered, bump


def test_drop_pred_layers_removes_layers_and_legends():
    """_drop_pred_layers removes every map layer a row added and its legends."""
    from gui.tile import inference_tile

    removed = []

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """Record the removed layer key."""
            removed.append(key)

    port, registered, unregistered, _bump = _fake_legend_port()
    row = {"key": "row1", "storage_keys": ["pred_a", "pred_b"]}
    for storage_key in row["storage_keys"]:
        port.register(
            inference_tile._pred_legend(
                storage_key, "rf_2020", None, "my prediction", True
            )
        )
    assert len(registered) == 2

    inference_tile._drop_pred_layers(row, FakeMap(), port)

    assert removed == [
        inference_tile._pred_layer_key("pred_a"),
        inference_tile._pred_layer_key("pred_b"),
    ]
    assert unregistered == [
        inference_tile._pred_layer_key("pred_a"),
        inference_tile._pred_layer_key("pred_b"),
    ]


def test_drop_pred_layers_tolerates_a_missing_port():
    """A None legend_port is a no-op, not a crash — tiles render without one."""
    from gui.tile import inference_tile

    removed = []

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """Record the removed layer key."""
            removed.append(key)

    row = {"key": "row1", "storage_keys": ["pred_a"]}
    inference_tile._drop_pred_layers(row, FakeMap(), None)

    assert removed == [inference_tile._pred_layer_key("pred_a")]


def test_pred_legend_is_keyed_by_the_map_layer_key():
    """_pred_legend keys its LayerLegend by the map layer key, not raw name."""
    from gui.tile import inference_tile

    legend = inference_tile._pred_legend("pred_a", "rf_2020", None, "My run", False)
    assert legend.layer_id == inference_tile._pred_layer_key("pred_a")
    assert legend.label.literal == "My run"
    assert legend.spec.kind == "gradient"
    assert legend.spec.title.key == "legend.prediction.title"


def test_pred_legend_uses_the_display_name_not_the_storage_key():
    """A single-raster row shows the prediction's display name alone."""
    from gui.tile import inference_tile

    legend = inference_tile._pred_legend(
        "glm__ds1_2020", "glm", None, "My glm run", False
    )
    assert legend.label.literal == "My glm run"


def test_pred_legend_disambiguates_multi_raster_rows_with_the_storage_key():
    """A multi-raster row appends the storage key so entries stay distinct."""
    from gui.tile import inference_tile

    legend = inference_tile._pred_legend(
        "glm__ds1_2020", "glm", None, "My glm run", True
    )
    assert legend.label.literal == "My glm run — glm__ds1_2020"


def test_pred_legend_honours_an_imported_display_palette():
    """_pred_legend forwards an imported prediction's display_palette."""
    from gui.scripts.legend_data import prediction_spec
    from gui.tile import inference_tile

    legend = inference_tile._pred_legend(
        "imported", "imported", "jnr", "My import", False
    )
    assert legend.spec.colors == prediction_spec("jnr_x").colors


# --- add-branch staleness guard (fix round 1) ---------------------------------
#
# These render the real InferenceTile and drive its use_task through a worker
# thread, mirroring tests/test_postprocess_tile_threading.py's approach to
# pinning behaviour that only shows up once real background execution is
# involved. `InferenceOutputList` is stubbed out (as in
# test_postprocess_tile_threading's `_StubDialog`) purely to capture
# `on_toggle_map` without needing the full products table to render.


def _stale_guard_project(storage_key, path="/tmp/pred_a.tif"):
    """A minimal fake Project with one prediction.

    Shaped like PredictionFormDialog and InferenceTile expect (see
    tests/test_inference_edit_failed.py's `_project` helper for the same
    shape).
    """
    import types

    pred = types.SimpleNamespace(path=path, display_palette=None)
    return types.SimpleNamespace(
        models={},
        datasets={},
        processed_variables={},
        predictions={storage_key: pred},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder="/tmp"),
    )


def _render_capturing_on_toggle_map(monkeypatch, project, map_, legend_port=None):
    """Render InferenceTile with InferenceOutputList stubbed out.

    Hands back on_toggle_map, so a test can drive the toggle without a real
    table.
    """
    import reacton
    import solara

    from gui.tile import inference_tile

    captured = {}

    @solara.component
    def _StubList(
        project,
        inference_jobs,
        preds_on_map=None,
        on_toggle_map=None,
        on_dismiss=None,
        on_delete=None,
        on_edit=None,
        on_open=None,
    ):
        captured["on_toggle_map"] = on_toggle_map
        solara.Text("")

    monkeypatch.setattr(inference_tile, "InferenceOutputList", _StubList)

    box, rc = reacton.render(
        inference_tile.InferenceTile(
            project=solara.reactive(project), map_=map_, legend_port=legend_port
        ),
        handle_error=False,
    )
    return captured["on_toggle_map"], rc


def test_stale_project_generation_rolls_back_without_registering_a_legend(
    monkeypatch,
):
    """A project switch mid-add is rolled back: layer removed, no legend.

    Regression for the fix-round-1 defect: the staleness check used to live
    inside the add branch's `finally:` block and `return` from there, which
    (on the non-exception path this test exercises) rolled back correctly —
    this pins that the rollback itself still works after moving the check
    out of `finally`. Uses a fake LegendPort (round-2: tiles get an explicit
    handle, not the app_state singleton) whose generation the test bumps
    directly, standing in for a project switch during the await.
    """
    import threading
    import time

    from gui.tile import inference_tile

    class FakeMap:
        def __init__(self):
            self.removed = []

        def remove_layer(self, key, none_ok=False):
            """Record the removed layer key."""
            self.removed.append(key)

    fake_map = FakeMap()
    project = _stale_guard_project("pred_a")
    port, registered, _unregistered, bump = _fake_legend_port()

    started = threading.Event()
    release = threading.Event()

    def fake_add(map_, path, **kwargs):
        # Stands in for the blocking add; holds until the test has bumped
        # the port's generation, simulating a project switch mid-await.
        started.set()
        release.wait(5.0)
        return "FAKE_LAYER"

    monkeypatch.setattr("gui.scripts.prediction_map.add_prediction_on_map", fake_add)

    on_toggle_map, rc = _render_capturing_on_toggle_map(
        monkeypatch, project, fake_map, legend_port=port
    )
    try:
        row = {"key": "row1", "storage_keys": ["pred_a"], "model_key": "rf_2020"}
        on_toggle_map(row)

        assert started.wait(5.0), "add_prediction_on_map never ran"
        bump()
        release.set()

        deadline = time.time() + 5.0
        while time.time() < deadline and not fake_map.removed:
            time.sleep(0.01)

        assert fake_map.removed == [inference_tile._pred_layer_key("pred_a")]
        assert registered == []
        assert "row1" not in inference_tile.preds_on_map.value
    finally:
        rc.close()
        inference_tile.preds_on_map.set(set())


def test_add_exception_still_reaches_the_error_path(monkeypatch, caplog):
    """A raised add error propagates to the outer handler, not swallowed.

    Regression for the fix-round-1 defect: `return` inside the `finally:`
    block discards any exception in flight from the `try` body. The bug only
    shows up on the *combination* the review flagged — an add that raises
    AND a project switch in the same await window — so this bumps the
    port's generation before raising, exactly like
    `test_stale_project_generation_rolls_back_without_registering_a_legend`
    does, but with a failing add instead of a successful one. Before the
    fix, the generation mismatch made the old `finally:` block `return`
    before the exception could propagate, so neither `logger.exception` nor
    `set_form_error` ever ran.
    """
    import logging

    from gui.tile import inference_tile

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """No-op remove; this scenario never lands a layer."""

    project = _stale_guard_project("pred_a")
    port, registered, _unregistered, bump = _fake_legend_port()

    def fake_add(map_, path, **kwargs):
        # Simulate a project switch landing in the same await window as a
        # failing add — the exact combination that used to be swallowed.
        bump()
        raise RuntimeError("boom")

    monkeypatch.setattr("gui.scripts.prediction_map.add_prediction_on_map", fake_add)

    on_toggle_map, rc = _render_capturing_on_toggle_map(
        monkeypatch, project, FakeMap(), legend_port=port
    )
    try:
        with caplog.at_level(logging.ERROR, logger="spatial_risk"):
            row = {"key": "row1", "storage_keys": ["pred_a"], "model_key": "rf_2020"}
            on_toggle_map(row)

            import time

            deadline = time.time() + 5.0
            while time.time() < deadline and not caplog.records:
                time.sleep(0.01)

        assert any(
            "prediction map toggle failed" in r.message for r in caplog.records
        ), "the exception was swallowed instead of reaching the error path"
        # No legend for a layer that never successfully landed.
        assert registered == []
    finally:
        rc.close()
        inference_tile.preds_on_map.set(set())
