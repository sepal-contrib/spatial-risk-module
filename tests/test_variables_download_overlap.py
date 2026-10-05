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
        toggling_keys=frozenset(),
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

    def _fake_materialize(
        project, keys=None, on_progress=None, overwrite=False, on_wait=None
    ):
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

    def _blocking(project, keys=None, on_progress=None, overwrite=False, on_wait=None):
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


def test_download_all_is_the_hourglass_while_a_download_runs(monkeypatch):
    """Download-all shows the hourglass, not a spinner, while any row downloads.

    With a layer still idle it stays clickable; with every layer downloading it
    ignores clicks (``sr-busy``) rather than fading to the disabled grey (#38).
    """
    import ipyvuetify as vw

    from gui.i18n import t
    from gui.tile import variables_tile
    from gui.widget.product_table import BUSY_ICON

    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    _captured, rc = _render(monkeypatch, project, _Notifier())

    def button():
        def leaves(w):
            for c in getattr(w, "children", None) or []:
                if isinstance(c, str):
                    yield c
                else:
                    yield from leaves(c)

        hits = [
            b
            for b in rc.find(vw.Btn).widgets
            if any(
                s.startswith(t("tiles.variables.download_button", count=n))
                for s in leaves(b)
                for n in (0, 1, 2)
            )
        ]
        assert len(hits) == 1
        return hits[0]

    def icon(btn):
        return [str(i.children[0]) for i in btn.children if isinstance(i, vw.Icon)]

    inflight = variables_tile.download_inflight
    try:
        assert icon(button()) == ["mdi-cloud-download-outline"]

        inflight.claim("roads")  # one running, "rivers" still idle
        assert _wait(lambda: icon(button()) == [BUSY_ICON])
        assert "sr-busy" not in (button().class_ or "")
        assert button().disabled is False and not button().loading
        assert button().outlined is True and not button().color  # grey, not blue
        assert not rc.find(vw.ProgressLinear).widgets  # the hourglass says it

        inflight.claim("rivers")  # nothing left to start
        assert _wait(lambda: "sr-busy" in (button().class_ or ""))
        assert button().disabled is False and not button().loading

        inflight.release("roads", "rivers")
        assert _wait(lambda: icon(button()) == ["mdi-cloud-download-outline"])
        assert button().color == "primary" and not button().outlined
    finally:
        rc.close()
