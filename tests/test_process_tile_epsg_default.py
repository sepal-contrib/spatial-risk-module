"""The reference form's EPSG starts empty; the ⌖ button suggests the UTM zone.

Choosing a reference raster used to pre-fill the EPSG field with that
raster's UTM zone, so the analysis CRS was decided for the user the moment
they picked a layer (issue #37). The field now stays empty until the user
types a code or asks for the suggestion with the ⌖ button, which carries a
tooltip saying what it does. A project that already has a reference keeps
showing the EPSG it was set with.
"""

import threading
import time
from pathlib import Path

import ipyvuetify as vw
import pytest
import reacton
import solara
from pysepal.solara.notifications.bus import NotificationBus
from pysepal.solara.notifications.notifier import Notifier
from pysepal.solara.notifications.state import ToastType

from gui.i18n import t
from gui.scripts import process_actions
from gui.tile import process_tile
from spatialrisk.harmonization import HarmonizationStatus
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar

Project._ensure_model_schemas()

STRIP_CLASS = "sr-reference-strip"
UTM_ICON = "mdi-crosshairs-gps"
# Every reference the form submitted (set_base_raster is stubbed to record).
SUBMITTED = []


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """Stub every raster read and write the tile can reach from the form.

    ``set_base_raster`` and ``Project.save`` are stubbed too: a submit that
    slipped through would otherwise warp and create a real project folder.
    """
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=[], current=list(p.raw_variables)),
    )
    monkeypatch.setattr(process_actions, "auto_utm_epsg", lambda path: "EPSG:5490")
    # 10, not the form's default 30: the pre-fill must visibly land.
    monkeypatch.setattr(process_actions, "base_raster_resolution", lambda var: 10.0)
    monkeypatch.setattr(
        process_actions,
        "set_base_raster",
        lambda *a, **kw: SUBMITTED.append(a),
    )
    monkeypatch.setattr(Project, "save", lambda self, *a, **kw: None)
    SUBMITTED.clear()
    yield
    # Module-level state: a leaked claim would refuse other tests' submits.
    process_tile.reference_inflight.release("reference")


@pytest.fixture
def bus(monkeypatch):
    """Route the tile's toasts onto a bus the test can read."""
    bus = NotificationBus()
    monkeypatch.setattr(process_tile, "use_notifications", lambda: Notifier(bus))
    return bus


def _raster(p, name, **extra):
    """A raw raster variable that claims to be on disk (nothing reads it)."""
    return LocalRasterVar.model_construct(
        name=name,
        data_type="raster",
        raster_type="continuous",
        path=Path(f"/x/{name}.tif"),
        year=2020,
        project=p,
        **extra,
    )


def _make(name="epsg-default", crs=None, res=30.0):
    """A project with two raw rasters and, given ``crs``, a reference on fc."""
    p = Project(project_name=name)
    p.raw_variables["fc_2020"] = _raster(p, "fc")
    p.raw_variables["dem_2020"] = _raster(p, "dem")
    if crs is not None:
        p.base_raster = _raster(p, "fc", default_crs=crs, default_resolution=res)
    return p


def _reactive(p):
    """The open-project reactive, as the app holds it (identity equality)."""
    return solara.reactive(p, equals=lambda a, b: a is b)


def _project(with_base=False):
    """A project with two raw rasters and, optionally, a saved reference."""
    return _reactive(_make(crs=5490 if with_base else None))


def _leaves(w):
    """Every string leaf under a widget, in render order."""
    for c in getattr(w, "children", None) or []:
        if isinstance(c, str):
            yield c
        else:
            yield from _leaves(c)


def _state(rc):
    """What the form shows: every text leaf, plus the field values."""
    fields = tuple(f.v_model for f in rc.find(vw.TextField).widgets)
    texts = tuple(s for root in rc.find(vw.Html).widgets for s in _leaves(root))
    return fields, texts, len(rc.find(vw.Btn).widgets)


def _settle(rc, timeout: float = 5.0):
    """Wait until the threaded tasks stop re-rendering the tree."""
    deadline = time.time() + timeout
    stable, previous = 0, None
    while time.time() < deadline and stable < 3:
        snapshot = _state(rc)
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


def _open(rc):
    """Open the reference dialog from the tile's reference strip."""
    strips = [
        b
        for b in rc.find(vw.Btn).widgets
        if STRIP_CLASS in str(getattr(b, "class_", "") or "").split()
    ]
    assert len(strips) == 1, f"expected one reference strip, got {len(strips)}"
    strips[0].click()
    _settle(rc)


def _render(project):
    """Render the tile and open the reference dialog."""
    t("common.close")  # prime the catalog (first t() inside a first render)
    _box, rc = reacton.render(
        process_tile.ProcessTile(project=project, processing=solara.reactive(False)),
        handle_error=False,
    )
    _settle(rc)
    _open(rc)
    return rc


def _switch(rc, project, new):
    """Install another project in the open-project reactive, then reopen."""
    project.set(new)
    _settle(rc)
    if new is not None:
        _open(rc)


def _field(rc, label_key):
    """The form's text field labelled by ``label_key``."""
    label = t(label_key)
    hits = [f for f in rc.find(vw.TextField).widgets if f.label == label]
    assert len(hits) == 1, f"expected one {label!r} field, got {len(hits)}"
    return hits[0]


def _epsg(rc):
    """The EPSG field's current value."""
    return _field(rc, "tiles.process.epsg_label").v_model


def _choose(rc, key):
    """Pick a reference raster in the form's Select."""
    rc.find(vw.Select).widget.v_model = key
    _settle(rc)


def _utm_tooltip(rc):
    """The tooltip attached to the ⌖ button, found by its text."""
    text = t("tiles.process.auto_utm_tooltip")
    hits = [tip for tip in rc.find(vw.Tooltip).widgets if text in list(_leaves(tip))]
    assert len(hits) == 1, f"expected one ⌖ tooltip, got {len(hits)}"
    return hits[0]


def _utm_button(rc):
    """The ⌖ icon button: the tooltip's activator, not the strip's icon."""
    slots = _utm_tooltip(rc).v_slots
    (activator,) = [s for s in slots if s["name"] == "activator"]
    icons = [w for w in activator["children"] if isinstance(w, vw.Icon)]
    assert len(icons) == 1, f"expected one ⌖ icon, got {activator['children']}"
    assert UTM_ICON in list(_leaves(icons[0]))
    return icons[0]


def test_choosing_a_reference_raster_leaves_the_epsg_empty():
    """The raster pre-fills its pixel size, never the analysis CRS."""
    rc = _render(_project())
    try:
        assert not _epsg(rc), "the EPSG field is not empty on open"
        _choose(rc, "fc_2020")
        assert not _epsg(rc), f"choosing a raster filled the EPSG: {_epsg(rc)!r}"
        # Resolution pre-fill is kept: the layer's native pixel size is a
        # sensible default and the issue is about the CRS only.
        assert _field(rc, "tiles.process.resolution_label").v_model == "10"
    finally:
        rc.close()


def test_a_saved_reference_still_shows_its_epsg():
    """Reopening the form of a project with a reference shows what it was set to."""
    rc = _render(_project(with_base=True))
    try:
        assert rc.find(vw.Select).widget.v_model == "fc_2020"
        assert _epsg(rc) == "5490"
        assert _field(rc, "tiles.process.resolution_label").v_model == "30"
    finally:
        rc.close()


def test_the_utm_button_carries_a_tooltip():
    """The ⌖ button says what it does before anyone clicks it."""
    rc = _render(_project())
    try:
        button = _utm_button(rc)
        # The tooltip opens from the button's own hover and focus events.
        assert button.v_on == "tooltip.on"
        # Inside the EPSG field, not beside it: the field's append slot.
        append = [
            s
            for s in _field(rc, "tiles.process.epsg_label").v_slots
            if s["name"] == "append"
        ]
        assert append and _utm_tooltip(rc) in append[0]["children"]
    finally:
        rc.close()


def test_the_utm_button_fills_in_the_suggested_zone():
    """Clicking ⌖ puts the reference raster's UTM zone in the empty field."""
    rc = _render(_project())
    try:
        _choose(rc, "fc_2020")
        _utm_button(rc).click()
        assert _wait_until(lambda: _epsg(rc) == "EPSG:5490"), _epsg(rc)
    finally:
        rc.close()


def test_the_suggestion_is_read_off_the_event_handler(monkeypatch):
    """The click returns while the raster is still being read.

    Widget callbacks run on the session's websocket loop, so a raster open
    inside one freezes every open session until it returns. Now that the
    button is the only way to get the suggestion, it must not block.
    """
    started, release, finished = (threading.Event() for _ in range(3))

    def _slow_utm(path):
        started.set()
        release.wait(5.0)
        finished.set()
        return "EPSG:32721"

    monkeypatch.setattr(process_actions, "auto_utm_epsg", _slow_utm)
    rc = _render(_project())
    try:
        _choose(rc, "fc_2020")
        _utm_button(rc).click()
        returned_while_reading = not finished.is_set()
        assert started.wait(5.0), "the suggestion never started"
        assert returned_while_reading, "the click waited for the raster read"
        release.set()
        assert _wait_until(lambda: _epsg(rc) == "EPSG:32721"), _epsg(rc)
    finally:
        release.set()
        rc.close()


def test_a_suggestion_for_a_raster_switched_away_from_is_dropped(monkeypatch):
    """A slow suggestion must not land on a different raster's form."""
    started, release = threading.Event(), threading.Event()

    def _slow_utm(path):
        started.set()
        release.wait(5.0)
        return "EPSG:32721"

    monkeypatch.setattr(process_actions, "auto_utm_epsg", _slow_utm)
    rc = _render(_project())
    try:
        _choose(rc, "fc_2020")
        _utm_button(rc).click()
        assert started.wait(5.0), "the suggestion never started"
        _choose(rc, "dem_2020")
        release.set()
        _settle(rc)
        time.sleep(0.2)  # a stale suggestion would have landed by now
        assert not _epsg(rc), f"fc's zone landed on dem's form: {_epsg(rc)!r}"
    finally:
        release.set()
        rc.close()


def test_the_utm_button_without_a_reference_says_why(bus):
    """With no raster chosen there is nothing to suggest — and the user hears it."""
    rc = _render(_project())
    try:
        _utm_button(rc).click()
        _settle(rc)
        assert not _epsg(rc)
        shown = [toast.message for toast in bus.toasts.value]
        assert t("tiles.process.error_pick_reference") in shown, shown
        assert all(
            toast.type is ToastType.WARNING for toast in bus.toasts.value
        ), bus.toasts.value
    finally:
        rc.close()


def test_submitting_without_an_epsg_names_the_missing_field():
    """Empty is now the default, so the dialog must refuse it in words."""
    rc = _render(_project())
    try:
        _choose(rc, "fc_2020")
        set_label = t("tiles.process.set_base_button")
        (submit,) = [b for b in rc.find(vw.Btn).widgets if set_label in _leaves(b)]
        submit.click()
        shown = [s for a in rc.find(vw.Alert).widgets for s in _leaves(a)]
        assert t("tiles.process.error_need_epsg") in shown, shown
        time.sleep(0.2)  # a submit that slipped through would have warped by now
        assert SUBMITTED == [], f"an empty EPSG was submitted: {SUBMITTED}"
    finally:
        rc.close()


# The tile is mounted once for the whole session (a plain tab, never keyed by
# project), so the form's state outlives the project it was filled for: every
# project has to arrive at a form that shows its own reference, or nothing.


@pytest.mark.parametrize("via_close", [False, True], ids=["switch", "close-then-new"])
def test_another_project_opens_with_an_empty_form(via_close):
    """A project without a reference never inherits the last project's CRS."""
    project = _reactive(_make("with-reference", crs=5490, res=100.0))
    rc = _render(project)
    try:
        assert _epsg(rc) == "5490"
        if via_close:
            _switch(rc, project, None)
        _switch(rc, project, _make("without-reference"))
        assert not rc.find(vw.Select).widget.v_model, "the last project's layer"
        assert not _epsg(rc), f"the last project's CRS: {_epsg(rc)!r}"
        assert _field(rc, "tiles.process.resolution_label").v_model == "30"
        _choose(rc, "dem_2020")
        assert not _epsg(rc), f"the last project's CRS: {_epsg(rc)!r}"
    finally:
        rc.close()


def test_another_project_shows_its_own_saved_epsg():
    """Two projects referenced on the same layer each show their own CRS."""
    project = _reactive(_make("utm-21s", crs=5490))
    rc = _render(project)
    try:
        assert _epsg(rc) == "5490"
        _switch(rc, project, _make("utm-18n", crs=32618, res=100.0))
        assert rc.find(vw.Select).widget.v_model == "fc_2020"
        assert _epsg(rc) == "32618"
        assert _field(rc, "tiles.process.resolution_label").v_model == "100"
    finally:
        rc.close()


def test_a_republished_project_keeps_what_the_user_typed():
    """Tiles republish the open project as a copy; that is not a switch."""
    project = _project()
    rc = _render(project)
    try:
        _choose(rc, "fc_2020")
        _field(rc, "tiles.process.epsg_label").v_model = "3857"
        _settle(rc)
        project.set(project.value.model_copy())
        _settle(rc)
        assert rc.find(vw.Select).widget.v_model == "fc_2020"
        assert _epsg(rc) == "3857"
    finally:
        rc.close()


def test_losing_the_reference_empties_the_form():
    """Removing the reference's source layer drops the reference from the form."""
    p = _make(crs=5490)
    project = _reactive(p)
    rc = _render(project)
    try:
        assert _epsg(rc) == "5490"
        # What the Variables step does when the reference's source goes.
        del p.raw_variables["fc_2020"]
        p.base_raster = None
        project.set(p.model_copy())
        _settle(rc)
        assert not rc.find(vw.Select).widget.v_model, "a removed layer is chosen"
        assert not _epsg(rc), f"the lost reference's CRS: {_epsg(rc)!r}"
    finally:
        rc.close()


def test_a_suggestion_does_not_follow_the_user_into_another_project(monkeypatch):
    """A zone read for one project's raster never lands in the next project."""
    started, release = threading.Event(), threading.Event()

    def _slow_utm(path):
        started.set()
        release.wait(5.0)
        return "EPSG:32721"

    monkeypatch.setattr(process_actions, "auto_utm_epsg", _slow_utm)
    project = _reactive(_make("first"))
    rc = _render(project)
    try:
        _choose(rc, "fc_2020")
        _utm_button(rc).click()
        assert started.wait(5.0), "the suggestion never started"
        # Same layer key in the next project: only the project tells them apart.
        _switch(rc, project, _make("second"))
        _choose(rc, "fc_2020")
        release.set()
        _settle(rc)
        time.sleep(0.2)  # a stale suggestion would have landed by now
        assert not _epsg(rc), f"the first project's zone landed: {_epsg(rc)!r}"
    finally:
        release.set()
        rc.close()


def test_what_the_user_types_during_a_suggestion_wins(monkeypatch):
    """The field stays editable while ⌖ reads; a late zone must not overwrite."""
    started, release = threading.Event(), threading.Event()

    def _slow_utm(path):
        started.set()
        release.wait(5.0)
        return "EPSG:32721"

    monkeypatch.setattr(process_actions, "auto_utm_epsg", _slow_utm)
    rc = _render(_project())
    try:
        _choose(rc, "fc_2020")
        _utm_button(rc).click()
        assert started.wait(5.0), "the suggestion never started"
        _field(rc, "tiles.process.epsg_label").v_model = "3857"
        release.set()
        _settle(rc)
        time.sleep(0.2)  # a late suggestion would have landed by now
        assert _epsg(rc) == "3857", f"the suggestion overwrote: {_epsg(rc)!r}"
    finally:
        release.set()
        rc.close()
