"""Every "add layer to map" button shows an hourglass while its layer is added (#38).

Adding a layer can take seconds (overview build, tile-server start, a GEE
min/max stretch) and the map-plus row action gave no sign it had been clicked.
Each list now takes the keys whose map toggle is in flight and renders that
row's button as an hourglass that ignores clicks (no Vuetify spinner); the
tiles feed the lists from the same ``InflightKeys`` their handlers claim, so
the hourglass appears on the click and goes when the worker releases the key,
on success and on failure alike.

Rendered, not grepped: a prop that never reaches the widget is invisible to a
source check (see reacton-ipyvuetify-import-required).
"""

import threading
import time
import types

import ipyvuetify as vw
import pytest
import reacton
import solara
from _notification_host import render_under_notifications

from gui.i18n import t

# Warm the translator before the first render (see test_model_form_dialog_render).
t("common.cancel")

from gui.scripts.density_map import density_layer_key  # noqa: E402
from gui.scripts.product_rows import inference_rows  # noqa: E402
from gui.tile import (  # noqa: E402
    derived_map,
    inference_tile,
    postprocess_tile,
    process_tile,
    sampling_tile,
    toolbox_tile,
    variables_tile,
)
from gui.widget.allocation_list import AllocationList  # noqa: E402
from gui.widget.inference_output_list import InferenceOutputList  # noqa: E402
from gui.widget.product_table import BUSY_ICON  # noqa: E402
from gui.widget.sample_set_list import SampleSetList  # noqa: E402
from gui.widget.variable_list import (  # noqa: E402
    DerivedVariableList,
    HarmonizationVariableList,
    SourceVariableList,
)
from spatialrisk.harmonization import HarmonizationStatus  # noqa: E402
from spatialrisk.predictions.prediction import Prediction  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.sample import Sample  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402

Project._ensure_model_schemas()

TIMEOUT = 10.0
MAP_ICONS = ("mdi-map-plus", "mdi-map-minus", BUSY_ICON)
IDLE = (False, False)
BUSY = (True, True)


@pytest.fixture(autouse=True)
def _drain_module_state():
    """Leave no in-flight claim or on-map key behind in the tiles' modules."""
    yield
    for keys in (
        variables_tile.vars_inflight,
        derived_map.derived_toggle_inflight,
        sampling_tile.samples_pending,
        inference_tile.preds_inflight,
        toolbox_tile.density_inflight,
    ):
        keys.release(*keys.value)
    variables_tile.vars_on_map.set(set())
    toolbox_tile.density_on_map.set(set())


# --- helpers -----------------------------------------------------------------


def _find(widget, cls, out=None):
    """Every ``cls`` widget under ``widget``, depth first, in render order."""
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    children = list(getattr(widget, "children", []) or [])
    for slot in getattr(widget, "v_slots", None) or []:
        children.extend(slot.get("children", []) or [])
    for child in children:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _map_buttons(box):
    """Every map-toggle button in the rendered tree, in row order."""
    out = []
    for btn in _find(box, vw.Btn):
        icons = [str(i.children[0]) for i in _find(btn, vw.Icon) if i.children]
        if any(name in MAP_ICONS for name in icons):
            out.append(btn)
    return out


def _states(box):
    """``(hourglass, ignores clicks)`` of each map-toggle button, in row order.

    A busy button must show the hourglass *instead of* Vuetify's spinner, and
    ignore clicks through ``sr-busy`` rather than the faded ``disabled`` look,
    so a button that is ``loading`` or ``disabled`` fails here outright.
    """
    out = []
    for b in _map_buttons(box):
        assert not b.loading, "map toggle shows the spinner, not the hourglass"
        assert not b.disabled, "busy map toggle fades to disabled grey"
        icons = [str(i.children[0]) for i in _find(b, vw.Icon) if i.children]
        out.append((BUSY_ICON in icons, "sr-busy" in (b.class_ or "")))
    return out


def _render(element):
    """Render a list component and return its widget tree."""
    box, rc = reacton.render(element, handle_error=False)
    return box, rc


def _wait(pred):
    """Poll ``pred`` until it is truthy or TIMEOUT elapses."""
    tick = threading.Event()
    for _ in range(int(TIMEOUT * 100)):
        if pred():
            return True
        tick.wait(0.01)
    return pred()


def _raster(p, name, path="/tmp/x.tif", year=None):
    """A mappable local raster variable bound to project ``p``."""
    return LocalRasterVar.model_construct(
        name=name,
        year=year,
        path=path,
        project=p,
        data_type="raster",
        raster_type="categorical",
        active=True,
    )


class _Notifier:
    """Records error toasts; everything else is a no-op."""

    def __init__(self):
        """Start with no recorded errors."""
        self.errors = []

    def error(self, message, *, timeout=None):
        """Record the error message."""
        self.errors.append(message)

    def success(self, message, *, timeout=None):
        """Ignore success toasts."""

    def warning(self, message, *, timeout=None):
        """Ignore warning toasts."""


class _FakeMap:
    """Map stand-in that records added and removed layer keys."""

    def __init__(self):
        """Start with an empty map."""
        self.added = []
        self.removed = []

    def add_layer(self, layer, key=None):
        """Record the added layer key."""
        self.added.append(key)

    def remove_layer(self, key, none_ok=False):
        """Record the removed layer key."""
        self.removed.append(key)


def _legend_port():
    """A LegendPort stand-in that records registrations."""
    from gui.scripts.legend_registry import LegendPort

    registered = []
    port = LegendPort(
        register=lambda *legends: registered.extend(legends),
        unregister=lambda *ids: None,
        generation=lambda: 0,
    )
    return port, registered


# --- lists: only the row being added is busy ---------------------------------


def test_source_list_hourglass_only_on_the_row_being_added():
    """Step 2: the clicked row's map button is an hourglass ignoring clicks."""
    p = Project(project_name="p")
    p.raw_variables["roads"] = _raster(p, "roads")
    p.raw_variables["rivers"] = _raster(p, "rivers")
    box, _rc = _render(
        SourceVariableList(
            project=solara.reactive(p),
            on_remove=lambda k: None,
            on_toggle_map=lambda k: None,
            vars_on_map=solara.reactive(set()),
            toggling_keys=frozenset({"roads"}),
        )
    )
    assert _states(box) == [BUSY, IDLE]


def test_source_list_is_idle_by_default():
    """No in-flight keys given: no row is busy (the prop is optional)."""
    p = Project(project_name="p")
    p.raw_variables["roads"] = _raster(p, "roads")
    box, _rc = _render(
        SourceVariableList(
            project=solara.reactive(p),
            on_remove=lambda k: None,
            on_toggle_map=lambda k: None,
        )
    )
    assert _states(box) == [IDLE]


def test_derived_list_hourglass_only_on_the_row_being_added():
    """Step 4: derived-layer rows follow their own in-flight keys."""
    p = Project(project_name="p")
    p.processed_variables["roads_dist"] = _raster(p, "roads_dist")
    p.processed_variables["rivers_dist"] = _raster(p, "rivers_dist")
    box, _rc = _render(
        DerivedVariableList(
            project=solara.reactive(p),
            on_toggle_map=lambda k: None,
            derived_on_map=solara.reactive(set()),
            toggling_keys=frozenset({"rivers_dist"}),
        )
    )
    assert _states(box) == [IDLE, BUSY]


def test_harmonization_list_hourglass_on_the_output_key():
    """Step 3: the toggle acts on the harmonized output, so it keys on that."""
    p = Project(project_name="p")
    p.raw_variables["fc_2020"] = _raster(p, "fc", year=2020)
    p.raw_variables["alt"] = _raster(p, "alt")
    p.processed_variables["fc_2020"] = _raster(p, "fc", year=2020)
    p.processed_variables["alt"] = _raster(p, "alt")
    box, _rc = _render(
        HarmonizationVariableList(
            project=solara.reactive(p),
            status=HarmonizationStatus(pending=[], current=["fc_2020", "alt"]),
            on_harmonize=lambda k: None,
            on_toggle_map=lambda k: None,
            derived_on_map=solara.reactive(set()),
            toggling_keys=frozenset({"fc_2020"}),
        )
    )
    assert _states(box) == [BUSY, IDLE]


def test_sample_list_hourglass_on_the_pending_row():
    """Step 5: the pending set it already had now shows the hourglass too."""
    p = Project(project_name="p")
    for key in ("rand_1", "rand_2"):
        p.samples[key] = Sample(
            name=key, raster_var_name="fcc", strategy="random", n_samples=10
        )
    box, _rc = _render(
        SampleSetList(
            project=solara.reactive(p),
            sampling_jobs=solara.reactive([]),
            on_toggle_map=lambda k: None,
            pending=frozenset({"rand_2"}),
        )
    )
    assert _states(box) == [IDLE, BUSY]


def test_inference_list_hourglass_only_on_the_row_being_added():
    """Step 7: a prediction row is busy while its rasters are being added."""
    p = Project(project_name="p")
    for name in ("run_a", "run_b"):
        p.predictions[name] = Prediction(
            name=name,
            path=f"/tmp/{name}.tif",
            model_key="glm_glm_v1",
            dataset_name="calibration",
            model_snapshot={"model_type": "glm", "name": "glm_v1"},
        )
    keys = [r["key"] for r in inference_rows(p, [])]
    assert len(keys) == 2
    box, _rc = _render(
        InferenceOutputList(
            project=solara.reactive(p),
            inference_jobs=solara.reactive([]),
            on_toggle_map=lambda row: None,
            toggling_keys=frozenset({keys[0]}),
        )
    )
    assert _states(box) == [BUSY, IDLE]


def _allocation_row(key):
    """A saved allocation run that wrote a density raster."""
    return {
        "kind": "record",
        "key": key,
        "name": key,
        "created_at": None,
        "annual_ha": 1.0,
        "total_ha": 4.0,
        "years_forecast": 4,
        "density_map_path": f"/tmp/{key}.tif",
        "warnings": [],
        "provenance": "user",
    }


def test_allocation_list_hourglass_only_on_the_density_being_added():
    """Toolbox: the density toggle keys on the density layer key."""
    box, _rc = _render(
        AllocationList(
            rows=[_allocation_row("r1"), _allocation_row("r2")],
            on_delete=lambda key: None,
            on_toggle_density=lambda row: None,
            toggling_keys=frozenset({density_layer_key("r2")}),
        )
    )
    assert _states(box) == [IDLE, BUSY]


# --- tiles: the hourglass follows the real claim/release--------------------


def _mount_capturing(monkeypatch, module, list_name, element, **patches):
    """Render ``element`` with ``module.list_name`` replaced by a recorder.

    Returns ``(captured, rc)``; ``captured`` always holds the latest props the
    tile handed the list, so a re-render shows up there.
    """
    captured = {}

    @solara.component
    def _StubList(**kwargs):
        """Record the props and render nothing."""
        captured.clear()
        captured.update(kwargs)
        solara.Text("")

    monkeypatch.setattr(module, list_name, _StubList)
    for name, value in patches.items():
        monkeypatch.setattr(module, name, value)
    _box, rc = render_under_notifications(
        lambda: solara.Column(children=[element]), handle_error=False
    )
    return captured, rc


def _mount_variables_tile(monkeypatch, map_, notifier):
    """VariablesTile over two local rasters, with its list and modal stubbed."""

    @solara.component
    def _StubModal(*args, **kwargs):
        """Render nothing in place of the real modal."""
        solara.Text("")

    p = Project(project_name="demo")
    p.raw_variables["roads"] = _raster(p, "roads")
    p.raw_variables["rivers"] = _raster(p, "rivers")
    project = solara.reactive(p, equals=lambda a, b: a is b)
    monkeypatch.setattr(variables_tile, "_var_legend", lambda *a, **k: None)
    return _mount_capturing(
        monkeypatch,
        variables_tile,
        "SourceVariableList",
        variables_tile.VariablesTile(project=project, map_=map_),
        VariableModal=_StubModal,
        use_notifications=lambda: notifier,
    )


def test_variables_tile_hourglass_from_click_until_the_layer_lands(monkeypatch):
    """The row is busy while the add runs and clears once the layer is on."""
    gate = threading.Event()
    started = threading.Event()

    def slow_add(map_, path, *, var=None, layer_name=None, key=None, fit_bounds=False):
        """Hold the add until the test opens the gate."""
        started.set()
        gate.wait(TIMEOUT)

    monkeypatch.setattr(variables_tile, "add_raster_var_on_map", slow_add)
    captured, rc = _mount_variables_tile(monkeypatch, _FakeMap(), _Notifier())
    try:
        assert captured["toggling_keys"] == frozenset()
        captured["on_toggle_map"]("roads")
        assert started.wait(TIMEOUT)
        assert _wait(lambda: captured["toggling_keys"] == frozenset({"roads"}))
        gate.set()
        assert _wait(lambda: captured["toggling_keys"] == frozenset())
        assert "roads" in variables_tile.vars_on_map.value
    finally:
        gate.set()
        rc.close()


def test_variables_tile_clears_the_hourglass_when_the_add_fails(monkeypatch):
    """A failed add clears the hourglass and reports through a toast."""

    def broken_add(*args, **kwargs):
        """Fail the way an unreadable raster does."""
        raise RuntimeError("no tiles")

    monkeypatch.setattr(variables_tile, "add_raster_var_on_map", broken_add)
    notifier = _Notifier()
    captured, rc = _mount_variables_tile(monkeypatch, _FakeMap(), notifier)
    try:
        captured["on_toggle_map"]("roads")
        assert _wait(lambda: bool(notifier.errors))
        assert _wait(lambda: captured["toggling_keys"] == frozenset())
        assert "no tiles" in notifier.errors[0]
        assert "roads" not in variables_tile.vars_on_map.value
    finally:
        rc.close()


def _derived_project():
    """A project with one source raster and its harmonized output."""
    p = Project(project_name="proj")
    p.raw_variables["forest_2020"] = _raster(p, "forest_2020")
    p.processed_variables["forest_2020"] = _raster(p, "forest_2020")
    return solara.reactive(p, equals=lambda a, b: a is b)


@pytest.mark.parametrize(
    ("module", "list_name", "make_tile"),
    [
        (
            process_tile,
            "HarmonizationVariableList",
            lambda project: process_tile.ProcessTile(
                project=project, processing=solara.reactive(False)
            ),
        ),
        (
            postprocess_tile,
            "DerivedVariableList",
            lambda project: postprocess_tile.PostProcessTile(project=project),
        ),
    ],
)
def test_derived_tiles_feed_their_in_flight_toggles_to_the_list(
    monkeypatch, module, list_name, make_tile
):
    """Steps 3 and 4 hand the shared derived in-flight set to their lists."""
    from gui.scripts import process_actions

    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=[], current=list(p.raw_variables)),
    )
    monkeypatch.setattr(process_actions, "auto_utm_epsg", lambda path: "EPSG:5490")
    monkeypatch.setattr(process_actions, "base_raster_resolution", lambda var: 30.0)
    captured, rc = _mount_capturing(
        monkeypatch, module, list_name, make_tile(_derived_project())
    )
    inflight = derived_map.derived_toggle_inflight
    try:
        assert captured["toggling_keys"] == frozenset()
        assert inflight.claim("forest_2020")
        assert _wait(lambda: captured["toggling_keys"] == frozenset({"forest_2020"}))
        inflight.release("forest_2020")
        assert _wait(lambda: captured["toggling_keys"] == frozenset())
    finally:
        rc.close()


def test_inference_tile_feeds_its_in_flight_toggles_to_the_list(monkeypatch):
    """Step 7 hands preds_inflight to the list; the row hourglass is the only cue.

    The tile used to draw an indeterminate bar above the list while any
    prediction was being added; with the row's hourglass it was redundant.
    """
    captured, rc = _mount_capturing(
        monkeypatch,
        inference_tile,
        "InferenceOutputList",
        inference_tile.InferenceTile(
            project=solara.reactive(Project(project_name="p")), map_=_FakeMap()
        ),
    )
    inflight = inference_tile.preds_inflight
    try:
        assert captured["toggling_keys"] == frozenset()
        assert inflight.claim("rowA")
        assert _wait(lambda: captured["toggling_keys"] == frozenset({"rowA"}))
        assert not rc.find(vw.ProgressLinear).widgets
        inflight.release("rowA")
        assert _wait(lambda: captured["toggling_keys"] == frozenset())
    finally:
        rc.close()


# --- sampling: a failed add now tells the user --------------------------------


def test_sample_toggle_failure_is_toasted_and_released(monkeypatch):
    """The hourglass clears on failure, and the error reaches a toast."""

    def boom(*args, **kwargs):
        """Fail both rendering paths."""
        raise RuntimeError("no tiles")

    monkeypatch.setattr(
        "gui.scripts.pmtiles_map.add_sample_pmtiles_on_map", boom, raising=False
    )
    monkeypatch.setattr(
        "gui.scripts.map_helpers.add_sample_points_on_map", boom, raising=False
    )
    ss = types.SimpleNamespace(
        points_path="/tmp/s.gpkg", pmtiles_path="/tmp/s.pmtiles", ensure_pmtiles=None
    )
    project = solara.reactive(types.SimpleNamespace(samples={"s": ss}))
    notifier = _Notifier()
    assert sampling_tile.samples_pending.claim("s")
    sampling_tile._toggle_sample_on_map("s", project, _FakeMap(), True, notifier)
    assert "s" not in sampling_tile.samples_pending
    assert len(notifier.errors) == 1
    assert "no tiles" in notifier.errors[0]


# --- toolbox: the density toggle leaves the websocket loop --------------------


def _allocation_project():
    """A project whose only allocation run wrote a density raster."""
    from spatialrisk.allocations import AllocationRun

    p = Project(project_name="p")
    p.allocations["reserve_bbb22222"] = AllocationRun(
        name="reserve",
        run_id="bbb22222",
        created_at="2026-07-29T10:00:00",
        borders_file="/b.gpkg",
        defor_juris_ha=20000.0,
        years_forecast=4,
        annual_ha=312.4,
        total_ha=1249.6,
        out_dir="/out",
        csv_path="/out/defor_project.csv",
        density_map_path="/out/density.tif",
    )
    return solara.reactive(p)


def _mount_toolbox(monkeypatch, notifier, legend_port=None):
    """ToolboxTile with a map, its AllocationList replaced by a recorder."""
    return _mount_capturing(
        monkeypatch,
        toolbox_tile,
        "AllocationList",
        toolbox_tile.ToolboxTile(
            project=_allocation_project(), map_=_FakeMap(), legend_port=legend_port
        ),
        use_notifications=lambda: notifier,
    )


def _density_row(captured):
    """The one density-bearing record row the tile hands its list."""
    rows = [r for r in captured["rows"] if r.get("density_map_path")]
    assert len(rows) == 1, captured["rows"]
    return rows[0]


def test_density_toggle_returns_at_once_and_hourglass_until_added(monkeypatch):
    """The click returns immediately; the row is busy until the layer lands."""
    gate = threading.Event()
    calls = []

    def slow_add(map_, path, *, key, layer_name, fit_bounds=False, opacity=1.0):
        """Hold the add until the test opens the gate."""
        calls.append(key)
        gate.wait(TIMEOUT)
        return object(), (0.0, 2.5)

    monkeypatch.setattr(toolbox_tile, "add_density_on_map", slow_add)
    port, registered = _legend_port()
    captured, rc = _mount_toolbox(monkeypatch, _Notifier(), legend_port=port)
    try:
        row = _density_row(captured)
        key = density_layer_key(row["key"])
        assert captured["toggling_keys"] == frozenset()

        started = time.monotonic()
        captured["on_toggle_density"](row)
        # The old handler ran the add inline: it would sit on the gate here.
        assert time.monotonic() - started < TIMEOUT / 2
        assert _wait(lambda: captured["toggling_keys"] == frozenset({key}))
        captured["on_toggle_density"](row)  # a re-click while loading: ignored
        assert key not in toolbox_tile.density_on_map.value

        gate.set()
        assert _wait(lambda: captured["toggling_keys"] == frozenset())
        assert key in toolbox_tile.density_on_map.value
        assert [legend.layer_id for legend in registered] == [key]
        assert calls == [key]
    finally:
        gate.set()
        rc.close()


def test_density_toggle_failure_clears_the_hourglass_and_toasts(monkeypatch):
    """A failed add clears the hourglass, stays off the map and says why."""

    def broken_add(*args, **kwargs):
        """Fail the way an unreadable raster does."""
        raise RuntimeError("density.tif: no such file")

    monkeypatch.setattr(toolbox_tile, "add_density_on_map", broken_add)
    notifier = _Notifier()
    captured, rc = _mount_toolbox(monkeypatch, notifier)
    try:
        row = _density_row(captured)
        captured["on_toggle_density"](row)
        assert _wait(lambda: bool(notifier.errors))
        assert _wait(lambda: captured["toggling_keys"] == frozenset())
        assert "no such file" in notifier.errors[0]
        assert not toolbox_tile.density_on_map.value
    finally:
        rc.close()


def test_density_toggle_off_removes_the_layer(monkeypatch):
    """The second click takes the layer and its on-map mark back off."""
    monkeypatch.setattr(
        toolbox_tile,
        "add_density_on_map",
        lambda map_, path, **kw: (object(), (0.0, 1.0)),
    )
    captured, rc = _mount_toolbox(monkeypatch, _Notifier())
    try:
        row = _density_row(captured)
        key = density_layer_key(row["key"])
        captured["on_toggle_density"](row)
        assert _wait(lambda: key in toolbox_tile.density_on_map.value)
        assert _wait(lambda: not toolbox_tile.density_inflight.value)
        captured["on_toggle_density"](row)
        assert _wait(lambda: key not in toolbox_tile.density_on_map.value)
        assert _wait(lambda: not toolbox_tile.density_inflight.value)
    finally:
        rc.close()
