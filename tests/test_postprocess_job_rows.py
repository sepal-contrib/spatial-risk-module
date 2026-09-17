"""A submitted derived layer is a job row for as long as it is processing.

Before this, ``processed_variables`` only gained the entry once the raster had
been written, so the Derived layers list stayed empty for the whole run and the
user had no idea whether the form submission had taken. The tile now publishes
a session job the moment the dialog validates, exactly as the Train, Sampling
and Inference tabs do, and the row carries the run's status through to the end.
"""

import threading

import pytest
import reacton
import solara

# Long enough that a slow machine still observes the running row, short enough
# that a regression fails fast instead of hanging the suite.
TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def _clean_session_state():
    """Reset the module-level job list and claims between tests."""
    from gui.tile import postprocess_tile as mod

    mod.derived_jobs.set([])
    mod.derived_inflight.release(*mod.derived_inflight.value)
    yield
    mod.derived_jobs.set([])
    mod.derived_inflight.release(*mod.derived_inflight.value)


class _RecordingNotifier:
    """Stands in for pysepal's Notifier; records what the tile publishes."""

    def __init__(self):
        self.errors = []

    def error(self, message, *, timeout=None):
        self.errors.append((message, timeout))

    def success(self, message, *, timeout=None):
        pass

    def track(self, title, total_steps=None):
        from pysepal.solara.notifications.notifier import (
            _NoopTaskTrackerContextManager,
        )

        return _NoopTaskTrackerContextManager()


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


def _render(monkeypatch, project, notifier=None):
    """Render PostProcessTile with the dialog stubbed to hand back on_submit."""
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


def _dist_entry(pp_key="forest_2010"):
    return {"op": "dist", "start_key": "", "end_key": "", "pp_key": pp_key}


def _jobs():
    from gui.tile.postprocess_tile import derived_jobs

    return derived_jobs.value


def _wait_for(predicate, timeout=TIMEOUT):
    """Poll until ``predicate()`` is truthy (background workers are threads)."""
    clock = threading.Event()
    for _ in range(int(timeout * 100)):
        if predicate():
            return True
        clock.wait(0.01)
    return predicate()


def test_submitting_lists_the_layer_before_the_work_finishes(monkeypatch):
    """The whole point: the row exists while the raster is still being made."""
    from gui.scripts import process_actions

    started, release = threading.Event(), threading.Event()

    def _blocking_apply(project, key, step):
        started.set()
        release.wait(TIMEOUT)

    monkeypatch.setattr(process_actions, "apply_post_processing", _blocking_apply)
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, _RecordingNotifier())

    on_submit(_dist_entry())
    try:
        assert started.wait(TIMEOUT), "the worker never ran"
        assert len(_jobs()) == 1
        job = _jobs()[0]
        assert job["status"] == "running"
        assert job["name"] == "forest_dist"
        assert job["output_key"] == "forest_dist_2010"
        assert "forest_dist_2010" not in project.value.processed_variables
    finally:
        release.set()
        rc.close()


def test_a_finished_job_row_is_marked_completed(monkeypatch):
    """The row reaches a terminal state instead of spinning forever."""
    from gui.scripts import process_actions

    monkeypatch.setattr(
        process_actions, "apply_post_processing", lambda project, key, step: None
    )
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, _RecordingNotifier())

    on_submit(_dist_entry())
    assert _wait_for(lambda: _jobs() and _jobs()[0]["status"] == "completed")
    rc.close()


def test_a_failed_job_row_keeps_the_error_message(monkeypatch):
    """The reason survives the toast, in the row itself."""
    from gui.scripts import process_actions

    def _boom(project, key, step):
        raise RuntimeError("gdal exploded")

    monkeypatch.setattr(process_actions, "apply_post_processing", _boom)
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, _RecordingNotifier())

    on_submit(_dist_entry())
    assert _wait_for(lambda: _jobs() and _jobs()[0]["status"] == "failed")
    assert "gdal exploded" in _jobs()[0]["error"]
    rc.close()


def test_the_same_output_cannot_be_submitted_twice_at_once(monkeypatch):
    """Two GDAL runs writing one .tif — the second submission is refused."""
    from gui.scripts import process_actions

    started, release = threading.Event(), threading.Event()
    calls = []

    def _blocking_apply(project, key, step):
        calls.append(key)
        started.set()
        release.wait(TIMEOUT)

    monkeypatch.setattr(process_actions, "apply_post_processing", _blocking_apply)
    notifier = _RecordingNotifier()
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, notifier)

    on_submit(_dist_entry())
    assert started.wait(TIMEOUT)
    on_submit(_dist_entry())
    try:
        assert len(_jobs()) == 1, "the refused submission must not add a second row"
        assert notifier.errors, "the user was not told why nothing happened"
        assert len(calls) == 1
    finally:
        release.set()
        rc.close()


def test_two_different_layers_run_at_the_same_time(monkeypatch):
    """Different outputs are independent — only the same key is exclusive."""
    from gui.scripts import process_actions

    both_started = threading.Barrier(3, timeout=TIMEOUT)

    def _blocking_apply(project, key, step):
        both_started.wait()

    monkeypatch.setattr(process_actions, "apply_post_processing", _blocking_apply)
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, _RecordingNotifier())

    on_submit(_dist_entry())
    on_submit({"op": "edge", "start_key": "", "end_key": "", "pp_key": "forest_2010"})
    try:
        both_started.wait()  # raises BrokenBarrierError if they serialized
        assert {j["name"] for j in _jobs()} == {"forest_dist", "forest_edge"}
    finally:
        rc.close()


def test_a_dismissed_job_row_goes_away(monkeypatch):
    """A failed row is the user's to clear."""
    from gui.scripts import process_actions
    from gui.tile import postprocess_tile

    def _boom(project, key, step):
        raise RuntimeError("nope")

    monkeypatch.setattr(process_actions, "apply_post_processing", _boom)
    project = solara.reactive(_project_with_processed_var(), equals=lambda a, b: a is b)
    on_submit, rc = _render(monkeypatch, project, _RecordingNotifier())

    on_submit(_dist_entry())
    assert _wait_for(lambda: _jobs() and _jobs()[0]["status"] == "failed")
    postprocess_tile.dismiss_derived_job(_jobs()[0]["id"])
    assert _jobs() == []
    rc.close()


def test_removing_a_derived_layer_purges_its_job_row(monkeypatch):
    """A stale completed row must not resurface once its entry is deleted."""
    from gui.tile import postprocess_tile

    postprocess_tile.derived_jobs.set(
        [
            {
                "id": "j1",
                "name": "forest_dist",
                "output_key": "forest_dist_2010",
                "status": "completed",
                "error": None,
            }
        ]
    )
    postprocess_tile.forget_derived_jobs_for("forest_dist_2010")
    assert _jobs() == []
