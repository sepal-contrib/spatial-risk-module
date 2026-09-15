"""Toggling processed layer B while A loads must not lose A's on-map state."""

import threading

import reacton
import solara

TIMEOUT = 10.0


class FakeMap:
    """Minimal map stand-in that records ``remove_layer`` calls."""

    def __init__(self):
        """Start with no removed layers."""
        self.removed = []

    def remove_layer(self, key, none_ok=False):
        """Record the removed layer key."""
        self.removed.append(key)


class _Notifier:
    """Minimal notifier stand-in that records ``error`` calls."""

    def __init__(self):
        """Start with no errors."""
        self.errors = []

    def error(self, message, *, timeout=None):
        """Record the error message."""
        self.errors.append(message)


def _project():
    """Build a demo Project with two mappable processed raster variables."""
    from spatialrisk.project import Project
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    p = Project(project_name="demo")
    for key in ("roads_dist", "rivers_dist"):
        p.processed_variables[key] = LocalRasterVar.model_construct(
            name=key,
            data_type="raster",
            raster_type="continuous",
            path=f"/tmp/{key}.tif",
            project=p,
            processing_history=["dist"],
        )
    return p


def _render(project, map_, notifier):
    """Render a Host component that captures ``use_derived_map_toggle``'s callback."""
    from gui.tile.derived_map import use_derived_map_toggle

    captured = {}

    @solara.component
    def Host():
        """Call the hook once and stash its returned callback."""
        captured["on_toggle_map"] = use_derived_map_toggle(project, map_, notifier)
        solara.Text("")

    box, rc = reacton.render(Host(), handle_error=False)
    return captured["on_toggle_map"], rc


def _wait(pred):
    """Poll ``pred`` until it is True or TIMEOUT elapses."""
    tick = threading.Event()
    for _ in range(int(TIMEOUT * 100)):
        if pred():
            return True
        tick.wait(0.01)
    return pred()


def test_second_toggle_keeps_the_first_toggles_bookkeeping(monkeypatch):
    """A second toggle started while the first is mid-add must not cancel it."""
    from gui.tile import derived_map

    gate = threading.Event()
    started = threading.Event()

    def fake_add_raster(
        map_, path, *, var=None, layer_name=None, key=None, fit_bounds=False
    ):
        """Block the roads_dist add until the test releases ``gate``."""
        if key == derived_map.derived_layer_key("roads_dist"):
            started.set()
            gate.wait(TIMEOUT)

    monkeypatch.setattr(derived_map, "add_raster_var_on_map", fake_add_raster)
    monkeypatch.setattr(derived_map, "is_mappable", lambda var: True)
    monkeypatch.setattr(derived_map, "_derived_legend", lambda *a, **k: None)

    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    on_toggle_map, rc = _render(project, FakeMap(), _Notifier())
    try:
        on_toggle_map("roads_dist")
        assert started.wait(TIMEOUT)
        on_toggle_map("rivers_dist")
        assert _wait(lambda: "rivers_dist" in derived_map.derived_on_map.value)
        gate.set()
        assert _wait(lambda: "roads_dist" in derived_map.derived_on_map.value)
        assert _wait(lambda: not derived_map.derived_toggle_inflight.value)
    finally:
        gate.set()
        rc.close()
        derived_map.derived_on_map.set(set())
        derived_map.derived_toggle_inflight.release("roads_dist", "rivers_dist")


def test_add_failure_toasts_and_releases(monkeypatch):
    """A failing add toasts the error and still releases the in-flight claim."""
    from gui.tile import derived_map

    def boom(*a, **k):
        """Simulate a broken tile server."""
        raise RuntimeError("tile server down")

    monkeypatch.setattr(derived_map, "add_raster_var_on_map", boom)
    monkeypatch.setattr(derived_map, "is_mappable", lambda var: True)
    notifier = _Notifier()
    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    on_toggle_map, rc = _render(project, FakeMap(), notifier)
    try:
        on_toggle_map("roads_dist")
        assert _wait(lambda: notifier.errors)
        assert "tile server down" in notifier.errors[0]
        assert _wait(lambda: not derived_map.derived_toggle_inflight.value)
        assert "roads_dist" not in derived_map.derived_on_map.value
    finally:
        rc.close()
        derived_map.derived_on_map.set(set())
