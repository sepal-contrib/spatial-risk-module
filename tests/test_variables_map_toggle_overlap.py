"""Toggling layer B while layer A is still loading must not lose A's bookkeeping.

With one shared ``use_task`` the second toggle cancelled the first coroutine
after its blocking add had been handed to a thread: the layer landed on the
map, but ``vars_on_map`` and the legend registration after the ``await``
were skipped — layer visible, toggle off, no legend, and the next click
tried to add it again.
"""

import threading

import reacton
import solara

TIMEOUT = 10.0


class FakeMap:
    """Minimal map stand-in that records removed layer keys."""

    def __init__(self):
        """Start with no removed layers recorded."""
        self.removed = []

    def remove_layer(self, key, none_ok=False):
        """Record the removed layer key."""
        self.removed.append(key)


def _project():
    """Build a Project with two mappable local raster variables."""
    from spatialrisk.project import Project
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    p = Project(project_name="demo")
    for key in ("roads", "rivers"):
        p.raw_variables[key] = LocalRasterVar.model_construct(
            name=key,
            data_type="raster",
            raster_type="categorical",
            path=f"/tmp/{key}.tif",
            project=p,
        )
    return p


def _render(monkeypatch, project, map_):
    """Render VariablesTile with stubbed children and return on_toggle_map + rc."""
    from gui.tile import variables_tile

    captured = {}

    @solara.component
    def _StubList(
        project,
        on_remove,
        on_edit=None,
        on_toggle_map=None,
        vars_on_map=None,
        on_download=None,
        downloading_keys=frozenset(),
    ):
        """Capture on_toggle_map and render nothing."""
        captured["on_toggle_map"] = on_toggle_map
        solara.Text("")

    @solara.component
    def _StubModal(*args, **kwargs):
        """Render nothing in place of the real modal."""
        solara.Text("")

    class _Notifier:
        """No-op notifier stand-in."""

        def error(self, message, *, timeout=None):
            """Swallow error notifications."""
            pass

        def success(self, message, *, timeout=None):
            """Swallow success notifications."""
            pass

    monkeypatch.setattr(variables_tile, "SourceVariableList", _StubList)
    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    monkeypatch.setattr(variables_tile, "use_notifications", lambda: _Notifier())
    box, rc = reacton.render(
        variables_tile.VariablesTile(project=project, map_=map_), handle_error=False
    )
    return captured["on_toggle_map"], rc


def _wait(pred):
    """Poll ``pred`` until it is truthy or TIMEOUT elapses."""
    tick = threading.Event()
    for _ in range(int(TIMEOUT * 100)):
        if pred():
            return True
        tick.wait(0.01)
    return pred()


def test_second_toggle_does_not_drop_the_first_toggles_bookkeeping(monkeypatch):
    """A second toggle must not skip the first toggle's on-map bookkeeping."""
    from gui.tile import variables_tile

    gate_roads = threading.Event()
    started_roads = threading.Event()
    added = []

    def fake_add_raster(
        map_, path, *, var=None, layer_name=None, key=None, fit_bounds=False
    ):
        """Block on the roads key until the test releases the gate."""
        if key == variables_tile._map_layer_key("roads"):
            started_roads.set()
            gate_roads.wait(TIMEOUT)
        added.append(key)

    monkeypatch.setattr(variables_tile, "add_raster_var_on_map", fake_add_raster)
    monkeypatch.setattr(variables_tile, "is_mappable", lambda var: True)
    monkeypatch.setattr(variables_tile, "_var_legend", lambda *a, **k: None)

    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    on_toggle_map, rc = _render(monkeypatch, project, FakeMap())
    try:
        on_toggle_map("roads")
        assert started_roads.wait(TIMEOUT)
        on_toggle_map("rivers")
        assert _wait(lambda: "rivers" in variables_tile.vars_on_map.value)
        gate_roads.set()
        assert _wait(
            lambda: "roads" in variables_tile.vars_on_map.value
        ), "roads landed on the map but its on-map state was never recorded"
        assert _wait(lambda: not variables_tile.vars_inflight.value)
    finally:
        gate_roads.set()
        rc.close()
        variables_tile.vars_on_map.set(set())
        variables_tile.vars_inflight.release("roads", "rivers")


def test_reclick_on_a_loading_layer_is_ignored(monkeypatch):
    """A re-click on a layer still loading must not re-add or remove it."""
    from gui.tile import variables_tile

    gate = threading.Event()
    started = threading.Event()
    added = []

    def fake_add_raster(map_, path, **kwargs):
        """Record the add and block until the test releases the gate."""
        added.append(kwargs.get("key"))
        started.set()
        gate.wait(TIMEOUT)

    monkeypatch.setattr(variables_tile, "add_raster_var_on_map", fake_add_raster)
    monkeypatch.setattr(variables_tile, "is_mappable", lambda var: True)
    monkeypatch.setattr(variables_tile, "_var_legend", lambda *a, **k: None)

    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    fake_map = FakeMap()
    on_toggle_map, rc = _render(monkeypatch, project, fake_map)
    try:
        on_toggle_map("roads")
        assert started.wait(TIMEOUT)
        on_toggle_map("roads")  # still loading: neither a second add nor a remove
        gate.set()
        assert _wait(lambda: "roads" in variables_tile.vars_on_map.value)
        assert added == [variables_tile._map_layer_key("roads")]
        assert fake_map.removed == []
    finally:
        gate.set()
        rc.close()
        variables_tile.vars_on_map.set(set())
        variables_tile.vars_inflight.release("roads")
