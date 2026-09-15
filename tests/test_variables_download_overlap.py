"""Two per-row downloads started back to back must both republish the project.

Same defect as the derived layers: one shared ``use_task`` per tile, so the
second row's click cancelled the first row's continuation and its
``publish_if_current`` never ran — the layer was on disk but the source list
still showed it as cloud-backed.
"""

import threading

import pytest
import reacton
import solara

TIMEOUT = 10.0


class GEEVar:
    """Stands in for GEEVar: materialize_raw_layers is monkeypatched anyway.

    Named ``GEEVar`` on purpose — the tile picks the pending (cloud-backed)
    variables with ``type(v).__name__ == "GEEVar"``.
    """

    def __init__(self, name):
        """Record the variable's name; the tile only reads ``name``."""
        self.name = name
        self.data_type = "raster"


_GEEVar = GEEVar


@pytest.fixture(autouse=True)
def _drain_download_inflight():
    """Leave no claim behind in the tile's module-level in-flight set."""
    yield
    from gui.tile import variables_tile

    variables_tile.download_inflight.release(*variables_tile.download_inflight.value)


def _project():
    from spatialrisk.project import Project

    p = Project(project_name="demo")
    p.raw_variables["roads"] = _GEEVar("roads")
    p.raw_variables["rivers"] = _GEEVar("rivers")
    return p


class _Notifier:
    """Stands in for pysepal's Notifier; records the toasts the tile raises."""

    def __init__(self):
        """Start with no recorded error toasts."""
        self.errors = []

    def error(self, message, *, timeout=None):
        self.errors.append(message)

    def success(self, message, *, timeout=None):
        pass

    def track(self, title, total_steps=None):
        from pysepal.solara.notifications.notifier import _NoopTaskTrackerContextManager

        return _NoopTaskTrackerContextManager()


def _render(monkeypatch, project, notifier):
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
        captured["on_download"] = on_download
        captured["downloading_keys"] = downloading_keys
        solara.Text("")

    @solara.component
    def _StubModal(*args, **kwargs):
        solara.Text("")

    monkeypatch.setattr(variables_tile, "SourceVariableList", _StubList)
    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    monkeypatch.setattr(variables_tile, "use_notifications", lambda: notifier)
    box, rc = reacton.render(
        variables_tile.VariablesTile(project=project, map_=None), handle_error=False
    )
    return captured, rc


def _wait(pred):
    tick = threading.Event()
    for _ in range(int(TIMEOUT * 100)):
        if pred():
            return True
        tick.wait(0.01)
    return pred()


def test_two_row_downloads_both_republish(monkeypatch):
    """A second row's download must not drop the first row's republish."""
    from gui.scripts import process_actions
    from gui.tile import variables_tile

    gate_roads = threading.Event()
    started_roads = threading.Event()
    done = []

    def _fake_materialize(project, keys=None, on_progress=None):
        for k in keys:
            if k == "roads":
                started_roads.set()
                gate_roads.wait(TIMEOUT)
            done.append(k)
        return list(keys)

    monkeypatch.setattr(process_actions, "materialize_raw_layers", _fake_materialize)
    monkeypatch.setattr("spatialrisk.project.Project.save", lambda self, *a, **k: None)

    proj = _project()
    project = solara.reactive(proj, equals=lambda a, b: a is b)
    publishes = []
    project.subscribe(lambda v: publishes.append(len(publishes)))

    captured, rc = _render(monkeypatch, project, _Notifier())
    try:
        captured["on_download"]("roads")
        assert started_roads.wait(TIMEOUT)
        captured["on_download"]("rivers")
        assert _wait(lambda: "rivers" in done)
        gate_roads.set()
        assert _wait(lambda: "roads" in done)
        assert _wait(lambda: len(publishes) >= 2), f"republishes: {len(publishes)}"
        assert _wait(lambda: not variables_tile.download_inflight.value)
    finally:
        gate_roads.set()
        rc.close()
        variables_tile.download_inflight.release("roads", "rivers")


def test_a_row_already_downloading_is_not_started_twice(monkeypatch):
    """A re-click on a running row starts nothing, and bulk skips that row."""
    from gui.scripts import process_actions
    from gui.tile import variables_tile

    gate = threading.Event()
    started = threading.Event()
    calls = []

    def _blocking(project, keys=None, on_progress=None):
        calls.append(tuple(keys))
        started.set()
        gate.wait(TIMEOUT)
        return list(keys)

    monkeypatch.setattr(process_actions, "materialize_raw_layers", _blocking)
    monkeypatch.setattr("spatialrisk.project.Project.save", lambda self, *a, **k: None)
    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    captured, rc = _render(monkeypatch, project, _Notifier())
    try:
        captured["on_download"]("roads")
        assert started.wait(TIMEOUT)
        captured["on_download"]("roads")
        captured["on_download"](None)  # bulk must skip the in-flight key
        assert _wait(lambda: len(calls) == 2)
        assert calls[0] == ("roads",)
        assert calls[1] == ("rivers",)
        gate.set()
        assert _wait(lambda: not variables_tile.download_inflight.value)
    finally:
        gate.set()
        rc.close()
        variables_tile.download_inflight.release("roads", "rivers")
