"""Two derived layers submitted back to back must BOTH reach the list.

Regression for the 2026-09-14 report: two derived layers finished (raster
written, registered, project saved) and never appeared. The tile drove every
submission through one ``solara.lab.use_task``; re-invoking it cancelled the
first coroutine at its ``await`` — ``CancelledError`` is a BaseException, so
``except Exception`` missed it and ``publish_if_current`` never ran — while
the worker thread finished anyway. Now each submission is its own
``spawn_in_context`` worker and the republish happens on that thread.
"""

import threading

import reacton
import solara

TIMEOUT = 10.0


def _project():
    from spatialrisk.project import Project
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    p = Project(project_name="demo")
    for key in ("forest_2010", "rivers_2010"):
        p.processed_variables[key] = LocalRasterVar.model_construct(
            name=key.split("_")[0],
            year=2010,
            data_type="raster",
            raster_type="categorical",
            path=None,
            project=p,
        )
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
    from gui.tile import postprocess_tile

    captured = {}

    @solara.component
    def _StubDialog(project, open_, on_submit):
        captured["on_submit"] = on_submit
        solara.Text("")

    monkeypatch.setattr(postprocess_tile, "DerivedLayerDialog", _StubDialog)
    monkeypatch.setattr(postprocess_tile, "use_notifications", lambda: notifier)
    box, rc = reacton.render(
        postprocess_tile.PostProcessTile(project=project, map_=None),
        handle_error=False,
    )
    return captured["on_submit"], rc


def _wait(pred):
    tick = threading.Event()
    for _ in range(int(TIMEOUT * 100)):
        if pred():
            return True
        tick.wait(0.01)
    return pred()


def _entry(pp_key):
    return {"op": "dist", "start_key": "", "end_key": "", "pp_key": pp_key}


def test_first_job_still_publishes_when_a_second_is_submitted(monkeypatch):
    """A second submission must not drop the first job's republish."""
    from gui.scripts import process_actions
    from gui.tile import postprocess_tile
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    gate_a = threading.Event()
    started_a = threading.Event()
    finished = []

    def _fake_apply(project, key, step):
        if key == "forest_2010":
            started_a.set()
            gate_a.wait(TIMEOUT)
        name = f"{project.processed_variables[key].name}_{step}"
        project.processed_variables[name] = LocalRasterVar.model_construct(
            name=name,
            data_type="raster",
            raster_type="continuous",
            path=None,
            project=project,
            processing_history=[step],
        )
        finished.append(name)

    monkeypatch.setattr(process_actions, "apply_post_processing", _fake_apply)

    proj = _project()
    project = solara.reactive(proj, equals=lambda a, b: a is b)
    publishes = []
    project.subscribe(lambda v: publishes.append(sorted(v.processed_variables)))

    on_submit, rc = _render(monkeypatch, project, _Notifier())
    try:
        on_submit(_entry("forest_2010"))
        assert started_a.wait(TIMEOUT), "job A never started"
        on_submit(_entry("rivers_2010"))  # B: a second submission while A runs
        assert _wait(lambda: "rivers_dist" in finished)
        gate_a.set()  # A finishes last
        assert _wait(lambda: "forest_dist" in finished)
        assert _wait(
            lambda: len(publishes) >= 2
        ), f"only {len(publishes)} republish(es) reached the UI: {publishes}"
        assert "forest_dist" in publishes[-1]
        assert "rivers_dist" in publishes[-1]
        assert _wait(lambda: not postprocess_tile.derived_inflight.value)
    finally:
        gate_a.set()
        rc.close()
        postprocess_tile.derived_inflight.release("forest_dist", "rivers_dist")


def test_resubmitting_an_in_flight_layer_is_refused_with_a_toast(monkeypatch):
    """Re-submitting a layer that is still running starts nothing and toasts."""
    from gui.scripts import process_actions
    from gui.tile import postprocess_tile

    gate = threading.Event()
    started = threading.Event()
    calls = []

    def _blocking_apply(project, key, step):
        calls.append(key)
        started.set()
        gate.wait(TIMEOUT)

    monkeypatch.setattr(process_actions, "apply_post_processing", _blocking_apply)
    notifier = _Notifier()
    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, notifier)
    try:
        on_submit(_entry("forest_2010"))
        assert started.wait(TIMEOUT)
        on_submit(_entry("forest_2010"))  # same output name, still running
        assert _wait(lambda: len(notifier.errors) == 1)
        assert "forest_dist" in notifier.errors[0]
        assert calls == ["forest_2010"]  # the second click started nothing
    finally:
        gate.set()
        rc.close()
        assert _wait(lambda: not postprocess_tile.derived_inflight.value)


def test_a_failing_job_releases_its_key(monkeypatch):
    """A failed job must give its key back so the layer can be retried."""
    from gui.scripts import process_actions
    from gui.tile import postprocess_tile

    def _boom(project, key, step):
        raise RuntimeError("gdal exploded")

    monkeypatch.setattr(process_actions, "apply_post_processing", _boom)
    notifier = _Notifier()
    project = solara.reactive(_project(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, notifier)
    try:
        on_submit(_entry("forest_2010"))
        assert _wait(lambda: notifier.errors)
        assert _wait(lambda: not postprocess_tile.derived_inflight.value)
        # a retry is possible right away
        on_submit(_entry("forest_2010"))
        assert _wait(lambda: len(notifier.errors) == 2)
    finally:
        rc.close()
        _wait(lambda: not postprocess_tile.derived_inflight.value)
        postprocess_tile.derived_inflight.release("forest_dist")
