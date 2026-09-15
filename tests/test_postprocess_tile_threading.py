"""Regression: edge/dist must not run on the Solara event-handler thread.

``PostProcessTile.on_submit`` sent change detection to a background task but ran
edge/dist inline, straight in the widget callback. Solara executes widget
callbacks inside ``process_kernel_messages`` — called synchronously, while
holding ``context.lock``, from the per-session websocket thread's
``while True: message = await ws.receive()`` loop (solara/server/server.py).
Blocking there means the session reads no further websocket messages for the
whole run: every button goes dead and no re-render or notification milestone
reaches the browser. Over a large AOI ``gdal.ComputeProximity`` takes minutes,
so the app simply appeared to hang; over a small test AOI it returned in under a
second and nobody noticed.

A worker thread is genuinely enough here — GDAL releases the GIL for
``ComputeProximity`` (measured: a ticker thread kept its 5 ms cadence for the
whole call), unlike the forestatrisk MCMC case that needed a subprocess.
"""

import threading

import pytest
import reacton
import solara

# Long enough that a synchronous handler visibly blocks, short enough that the
# regression case fails fast instead of hanging the suite.
BLOCK_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def _drain_derived_inflight():
    """Leave no claim behind in the tile's module-level in-flight set."""
    yield
    from gui.tile import postprocess_tile

    postprocess_tile.derived_inflight.release(*postprocess_tile.derived_inflight.value)


def _project_with_processed_var():
    from spatialrisk.project import Project
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    p = Project(project_name="demo")
    p.processed_variables["forest_2010"] = LocalRasterVar.model_construct(
        name="forest",
        year=2010,
        data_type="raster",
        raster_type="categorical",
        path=None,
        project=p,
    )
    return p


class _RecordingNotifier:
    """Stands in for pysepal's Notifier; records what the tile publishes."""

    def __init__(self):
        self.errors = []

    def error(self, message, *, timeout=None):
        self.errors.append((message, timeout))

    def success(self, message, *, timeout=None):
        pass

    def track(self, title, total_steps=None):
        from pysepal.solara.notifications.notifier import _NoopTaskTrackerContextManager

        return _NoopTaskTrackerContextManager()


def _render_capturing_on_submit(monkeypatch, project, notifier=None):
    """Render PostProcessTile with the dialog stubbed out to hand back on_submit."""
    from gui.tile import postprocess_tile

    captured = {}

    @solara.component
    def _StubDialog(project, open_, on_submit):
        captured["on_submit"] = on_submit
        solara.Text("")

    monkeypatch.setattr(postprocess_tile, "DerivedLayerDialog", _StubDialog)
    if notifier is not None:
        monkeypatch.setattr(postprocess_tile, "use_notifications", lambda: notifier)

    box, rc = reacton.render(
        postprocess_tile.PostProcessTile(project=project, map_=None),
        handle_error=False,
    )
    return captured["on_submit"], rc


def test_edge_dist_does_not_block_the_event_handler(monkeypatch):
    """on_submit must hand edge/dist to a worker and return immediately."""
    from gui.scripts import process_actions

    started = threading.Event()
    release = threading.Event()
    ran_on = {}

    def _blocking_apply(project, key, step):
        ran_on["ident"] = threading.get_ident()
        started.set()
        release.wait(BLOCK_TIMEOUT)

    monkeypatch.setattr(process_actions, "apply_post_processing", _blocking_apply)

    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render_capturing_on_submit(monkeypatch, project)

    handler_ident = threading.get_ident()
    on_submit({"op": "dist", "start_key": "", "end_key": "", "pp_key": "forest_2010"})
    handler_returned_early = not release.is_set() and started.wait(BLOCK_TIMEOUT)

    try:
        assert started.is_set(), "edge/dist work never ran"
        assert ran_on["ident"] != handler_ident, (
            "edge/dist ran on the event-handler thread — that blocks solara's "
            "websocket message loop and freezes the whole session"
        )
        assert handler_returned_early, (
            "on_submit only returned after the work finished — the handler is "
            "still synchronous"
        )
    finally:
        release.set()
        rc.close()


def test_edge_dist_reports_failures_as_an_error_toast(monkeypatch):
    """Moving to a thread must not lose the error path the sync version had."""
    from gui.scripts import process_actions
    from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT

    def _boom(project, key, step):
        raise RuntimeError("gdal exploded")

    monkeypatch.setattr(process_actions, "apply_post_processing", _boom)

    notifier = _RecordingNotifier()
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render_capturing_on_submit(monkeypatch, project, notifier)

    on_submit({"op": "dist", "start_key": "", "end_key": "", "pp_key": "forest_2010"})

    deadline = threading.Event()
    for _ in range(int(BLOCK_TIMEOUT * 100)):
        if notifier.errors:
            break
        deadline.wait(0.01)

    assert notifier.errors, "the failure never reached the UI"
    message, timeout = notifier.errors[0]
    assert "gdal exploded" in message
    assert timeout == ERROR_TOAST_TIMEOUT
    rc.close()
