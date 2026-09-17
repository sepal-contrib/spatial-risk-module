"""Reference & projection live behind a strip, not permanently in the tile body.

Step 3 is a list you work in; the reference grid is a setting chosen once. The
Select + EPSG + resolution form cost 11 of the tile's ~20 rows for a control
that is read once and then never touched, so it moved into a dialog. The strip
states the current choice and opens the form on demand — the shape of Step 2,
whose variable form is likewise a dialog.
"""

import threading
import time
from pathlib import Path

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t
from gui.scripts import process_actions
from gui.tile import process_tile
from spatialrisk.harmonization import HarmonizationStatus
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar

Project._ensure_model_schemas()

STRIP_CLASS = "sr-reference-strip"


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """Every path the tile takes on mount is disk I/O — stub the lot.

    The strip renders from the project model alone; the status check, the
    UTM/resolution autofill and the reprojection all read real rasters.
    """
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=[], current=list(p.raw_variables)),
    )
    monkeypatch.setattr(process_actions, "auto_utm_epsg", lambda path: "EPSG:5490")
    monkeypatch.setattr(process_actions, "base_raster_resolution", lambda var: 30.0)


def _raster(p, name="fc", **extra):
    return LocalRasterVar.model_construct(
        name=name,
        data_type="raster",
        raster_type="continuous",
        path=Path(f"/x/{name}.tif"),
        year=2020,
        project=p,
        **extra,
    )


def _project(with_base=True):
    p = Project(project_name="ref-strip")
    p.raw_variables["fc_2020"] = _raster(p)
    if with_base:
        p.base_raster = _raster(p, default_crs=5490, default_resolution=30.0)
    return solara.reactive(p, equals=lambda a, b: a is b)


def _render(project):
    t("common.close")  # prime the catalog (first t() inside a first render)
    box, rc = reacton.render(
        process_tile.ProcessTile(project=project, processing=solara.reactive(False)),
        handle_error=False,
    )
    _settle(rc)
    return rc


def _settle(rc, timeout: float = 5.0):
    """Wait for the tile's threaded tasks to land before interacting with it.

    ProcessTile starts ``autofill_base`` and ``harmonization_hint`` on worker
    threads, and each re-renders when it resolves. The real app serialises
    those re-renders against widget callbacks under the session's kernel
    context lock; a bare ``reacton.render()`` has no such lock, so a synthetic
    ``.click()`` can interleave with one landing and be dropped. That showed up
    as these tests failing only under a loaded machine.

    Polling until the tree stops moving removes the interleaving without
    weakening anything the tests assert.
    """
    deadline = time.time() + timeout
    stable, previous = 0, None
    while time.time() < deadline and stable < 3:
        snapshot = (tuple(_texts(rc)), len(rc.find(vw.Btn).widgets))
        stable = stable + 1 if snapshot == previous else 0
        previous = snapshot
        time.sleep(0.02)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until it holds — worker threads land asynchronously."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _texts(rc):
    """Every string leaf in render order (``rc.find`` does not descend cells)."""
    out = []

    def walk(w):
        for c in getattr(w, "children", None) or []:
            if isinstance(c, str):
                out.append(c)
            else:
                walk(c)

    for root in rc.find(vw.Html).widgets:
        walk(root)
    return out


def _leaves(w):
    """Every string leaf under a widget, in render order."""
    for c in getattr(w, "children", None) or []:
        if isinstance(c, str):
            yield c
        else:
            yield from _leaves(c)


def _strip(rc):
    """The reference strip button, found by the class only it carries."""
    hits = [
        b
        for b in rc.find(vw.Btn).widgets
        if STRIP_CLASS in str(getattr(b, "class_", "") or "").split()
    ]
    assert len(hits) == 1, f"expected exactly one reference strip, got {len(hits)}"
    return hits[0]


def _btn_by_label(rc, label):
    """A button whose rendered label is exactly ``label``."""
    hits = [b for b in rc.find(vw.Btn).widgets if label in list(_leaves(b))]
    assert hits, f"no button labelled {label!r}"
    return hits[0]


def _reference_dialog(rc):
    """The dialog whose card carries the reference form's title."""
    title = t("tiles.process.reference_dialog_title")
    hits = [d for d in rc.find(vw.Dialog).widgets if title in list(_leaves(d))]
    assert len(hits) == 1, f"expected one reference dialog, got {len(hits)}"
    return hits[0]


def test_strip_states_the_current_reference():
    """The trigger itself carries the choice: name, CRS and pixel size.

    Asserted on the strip's own subtree, not the tile's: a summary rendered
    somewhere else on the page would leave the button blank, which is the one
    thing the strip exists to avoid.
    """
    rc = _render(_project())
    try:
        stated = [s for s in _leaves(_strip(rc)) if "fc" in s and "5490" in s]
        assert stated, f"strip does not state the reference; it reads {_texts(rc)}"
        assert "30" in stated[0], f"strip omits the pixel size: {stated[0]!r}"
    finally:
        rc.close()


def test_strip_prompts_when_nothing_is_set_yet():
    """With no reference the strip asks for one instead of stating one."""
    rc = _render(_project(with_base=False))
    try:
        assert t("tiles.process.reference_unset") in list(_leaves(_strip(rc)))
        assert not [s for s in _texts(rc) if "5490" in s], _texts(rc)
    finally:
        rc.close()


def test_the_form_is_behind_the_strip_not_in_the_tile_body():
    """Closed on mount, open after a click — that is the whole simplification."""
    rc = _render(_project())
    try:
        assert _reference_dialog(rc).v_model is not True
        _strip(rc).click()
        assert _reference_dialog(rc).v_model is True
    finally:
        rc.close()


def test_submitting_the_dialog_warps_off_the_render_thread(monkeypatch):
    """The Set button hands the form values to a worker and returns at once.

    ``set_base_raster`` runs a full GDAL warp. Solara widget callbacks run
    inside ``process_kernel_messages`` under the session's context lock, so
    doing it inline froze the whole UI for its duration. The stub blocks until
    this test lets it go: the click returning while it is still inside the warp
    is the property, and the strip must say so meanwhile.
    """
    calls = []
    started, release = threading.Event(), threading.Event()

    def _slow_set_base(p, key, epsg, res, auto_save=True):
        calls.append((key, epsg, res, auto_save))
        started.set()
        release.wait(5.0)

    monkeypatch.setattr(process_actions, "set_base_raster", _slow_set_base)
    # The project is still the open one, so the worker goes on to save it —
    # Project.save() would create a real project folder under the data root.
    monkeypatch.setattr(Project, "save", lambda self, *a, **kw: None)

    rc = _render(_project(with_base=False))
    try:
        _strip(rc).click()
        rc.find(vw.Select).widget.v_model = "fc_2020"
        # Choosing the raster kicks off autofill_base on a worker thread; let
        # it land before grabbing widget refs it would re-render out from under.
        _settle(rc)
        epsg, resolution = rc.find(vw.TextField).widgets[:2]
        epsg.v_model = "EPSG:5490"
        resolution.v_model = "30"
        _settle(rc)

        _btn_by_label(rc, t("tiles.process.set_base_button")).click()

        # The click has already returned; the warp is still running.
        assert started.wait(5.0), "the warp never reached a worker thread"
        # auto_save=False: the worker checks the project is still the open one
        # before anything is written to disk.
        assert calls == [("fc_2020", "EPSG:5490", 30.0, False)]
        _settle(rc)
        assert t("tiles.process.reference_pending") in list(_leaves(_strip(rc)))
        assert rc.find(vw.ProgressLinear).widgets, "no progress bar while warping"
    finally:
        release.set()
        # Let the worker release its in-flight key: it is module-level, so a
        # thread still holding it would refuse the next test's submit.
        assert _wait_until(lambda: "reference" not in process_tile.reference_inflight)
        rc.close()


def test_a_second_submit_is_dropped_while_the_first_warp_runs(monkeypatch):
    """One reference warp at a time — two would fight over the same output.

    The dialog's ``disabled=`` is a render-time prop and reaches the browser a
    round-trip after the click, so the in-flight key is the real guard.
    """
    calls = []
    release = threading.Event()

    def _slow_set_base(p, key, epsg, res, auto_save=True):
        calls.append(key)
        release.wait(5.0)

    monkeypatch.setattr(process_actions, "set_base_raster", _slow_set_base)
    monkeypatch.setattr(Project, "save", lambda self, *a, **kw: None)

    rc = _render(_project(with_base=False))
    try:
        _strip(rc).click()
        rc.find(vw.Select).widget.v_model = "fc_2020"
        _settle(rc)
        epsg, resolution = rc.find(vw.TextField).widgets[:2]
        epsg.v_model = "EPSG:5490"
        resolution.v_model = "30"
        _settle(rc)

        _btn_by_label(rc, t("tiles.process.set_base_button")).click()
        assert _wait_until(lambda: calls == ["fc_2020"]), "the first warp never ran"
        _btn_by_label(rc, t("tiles.process.set_base_button")).click()

        time.sleep(0.2)  # a second worker would have appended by now
        assert calls == ["fc_2020"], f"the second submit started a warp too: {calls}"
    finally:
        release.set()
        assert _wait_until(lambda: "reference" not in process_tile.reference_inflight)
        rc.close()


def test_the_dialog_refuses_an_empty_form_instead_of_silently_disabling():
    """A disabled button never says why; the dialog names the missing field."""
    rc = _render(_project(with_base=False))
    try:
        _strip(rc).click()
        assert not rc.find(vw.Alert).widgets, "the form complains before submit"

        _btn_by_label(rc, t("tiles.process.set_base_button")).click()

        shown = [s for a in rc.find(vw.Alert).widgets for s in _leaves(a)]
        assert t("tiles.process.error_pick_reference") in shown, shown
    finally:
        rc.close()
