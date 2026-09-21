"""Failed allocation jobs are editable: reopen the form prefilled and rerun.

A failed run used to be a dead end — the row carried no actions at all, so the
user could neither retry it nor clear it. Now the job dict keeps the
AllocationForm it was launched from, the failed row offers edit and dismiss,
and the allocation form can seed itself from that entry.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.scripts.allocation_runner import (  # noqa: E402
    AllocationForm,
    BordersSelection,
)
from gui.widget.allocation_list import AllocationList  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _icon_button(box, icon):
    for btn in _find(box, vw.Btn):
        if any(icon in str(i.children) for i in _find(btn, vw.Icon)):
            return btn
    return None


def _entry(**kw):
    """The AllocationForm a job was launched with."""
    base = dict(
        name="allocation_1",
        prediction_key="icar_run",
        user_defrate_path=None,
        borders=BordersSelection(method="FILE", file_path="/data/borders.gpkg"),
        mask_file=None,
        defor_juris_ha=20000.0,
        years_forecast=4.0,
        density_extent=None,
    )
    base.update(kw)
    return AllocationForm(**base)


def _job_row(status="failed", entry=None, job_id="j1"):
    return {
        "kind": "job",
        "key": job_id,
        "job_id": job_id,
        "name": "allocation_1",
        "status": status,
        "error": "boom" if status == "failed" else None,
        "entry": _entry() if entry is None else entry,
    }


def _render_list(rows, on_edit=None, on_dismiss=None):
    box, _rc = reacton.render(
        AllocationList(
            rows=rows,
            on_delete=lambda key: None,
            on_edit=on_edit,
            on_dismiss=on_dismiss,
        )
    )
    return box


def test_failed_row_offers_edit_that_hands_back_the_row():
    """The pencil on a failed row hands the whole row (job_id + entry) up."""
    edited = []
    box = _render_list([_job_row()], on_edit=edited.append)
    btn = _icon_button(box, "mdi-pencil-outline")
    assert btn is not None
    btn.fire_event("click", {})
    assert edited and edited[0]["job_id"] == "j1"
    assert edited[0]["entry"].name == "allocation_1"


def test_running_row_offers_neither_action():
    """A run in flight cannot be edited or dismissed out from under its worker."""
    box = _render_list(
        [_job_row(status="running")],
        on_edit=lambda row: None,
        on_dismiss=lambda i: None,
    )
    assert _icon_button(box, "mdi-pencil-outline") is None
    assert _icon_button(box, "mdi-close") is None


def test_failed_row_without_an_entry_offers_dismiss_only():
    """Defensive: a job that never recorded its entry can still be cleared."""
    box = _render_list(
        [_job_row(entry=False)], on_edit=lambda row: None, on_dismiss=lambda i: None
    )
    assert _icon_button(box, "mdi-pencil-outline") is None
    assert _icon_button(box, "mdi-close") is not None


def test_failed_row_dismiss_hands_back_the_job_id():
    """Dismiss passes the job id, not the row."""
    dismissed = []
    box = _render_list([_job_row()], on_dismiss=dismissed.append)
    _icon_button(box, "mdi-close").fire_event("click", {})
    assert dismissed == ["j1"]


# --- the form seeds itself from a prefill entry -------------------------------

import inspect  # noqa: E402
import types  # noqa: E402

from pysepal.sepalwidgets.file_input import FileInput  # noqa: E402
from pysepal.solara.components.inputs import AdminLevelSelector  # noqa: E402

from gui.widget.allocation_form import AllocationFormDialog  # noqa: E402
from gui.widget.borders_picker import BordersPicker  # noqa: E402


def _text_field(box, label):
    return next((f for f in _find(box, vw.TextField) if f.label == label), None)


def _select(box, label):
    return next((s for s in _find(box, vw.Select) if s.label == label), None)


def _file_input(box, label):
    return next((f for f in _find(box, FileInput) if f.label == label), None)


def _prefill_project():
    """A project exposing one prediction and one maskable raster.

    Enough for the risk-map select and the mask select to both have a real
    choice available to seed.
    """
    pred = types.SimpleNamespace(
        model_key="icar", dataset_name="calibration", window=None, path="/tmp/p.tif"
    )
    mask_var = types.SimpleNamespace(path="/tmp/mask.tif")
    return types.SimpleNamespace(
        predictions={"icar_run": pred},
        processed_variables={"forest_mask": mask_var},
        allocations={},
    )


def _render_form(entry, running_names=frozenset(), project=None):
    if project is None:
        project = solara.reactive(_prefill_project())
    box, _rc = reacton.render(
        AllocationFormDialog(
            open_=solara.reactive(True),
            project=project,
            on_launch=lambda form: None,
            on_close=lambda: None,
            prefill=solara.reactive(entry),
            running_names=running_names,
        )
    )
    return box


def test_form_seeds_every_scalar_field_from_the_prefill():
    """Name, risk map, mask, borders, hectares, years and density all come back.

    The seeded name ("retry_me") is deliberately different from the fresh
    form's own suggestion ("allocation_1"): use_artifact_name only marks the
    field dirty (and therefore keeps a typed/seeded value instead of tracking
    the live suggestion) when the value differs from that suggestion, so a
    same-as-suggestion name would pass this assertion whether or not
    `on_name_input` actually ran. See test_form_without_a_prefill_keeps_its_
    defaults below, which shares that "allocation_1" suggestion.
    """
    box = _render_form(
        _entry(
            name="retry_me",
            defor_juris_ha=1234.0,
            years_forecast=7.0,
            mask_file="/tmp/mask.tif",
            density_extent="aoi",
        )
    )
    assert _text_field(box, t("toolbox.allocation.field_name")).v_model == "retry_me"
    assert _select(box, t("toolbox.allocation.field_riskmap")).v_model == "icar_run"
    assert _text_field(box, t("toolbox.allocation.field_juris_ha")).v_model == "1234"
    assert _text_field(box, t("toolbox.allocation.field_years")).v_model == "7"
    assert _select(box, t("toolbox.allocation.field_mask")).v_model == "/tmp/mask.tif"
    # Default _entry() borders is a FILE selection: the picker's own file
    # input (not a bare TextField) is what carries the seeded path.
    assert (
        _file_input(box, t("toolbox.allocation.field_borders_file")).v_model
        == "/data/borders.gpkg"
    )
    assert _select(box, t("toolbox.allocation.field_density")).v_model == "aoi"


def test_form_seeds_a_large_hectare_value_without_truncation():
    """`:g` truncates to 6 significant digits; `.12g` must not.

    A jurisdiction in the tens of millions of hectares is ordinary, and
    `f"{12345678.0:g}"` renders as "1.23457e+07" — silently corrupting the
    value a retry would run with.
    """
    box = _render_form(_entry(defor_juris_ha=12345678.0))
    assert (
        _text_field(box, t("toolbox.allocation.field_juris_ha")).v_model == "12345678"
    )


def test_form_seeds_a_custom_rate_table_as_custom_mode():
    """A run submitted with its own table reopens in custom mode, path shown."""
    box = _render_form(_entry(user_defrate_path="/data/rates.csv"))
    assert _select(box, t("toolbox.allocation.field_defrate")).v_model == "custom"
    assert (
        _file_input(box, t("toolbox.allocation.field_defrate_override")).v_model
        == "/data/rates.csv"
    )


def test_form_maps_a_legacy_density_bool_to_whole_aoi():
    """A pre-rename entry carrying density_map=True seeds the AOI extent."""
    entry = _entry()
    # Simulate an old in-flight entry: the attribute no longer exists on the
    # dataclass, so it is bolted on the instance the way a stale object would
    # still carry it.
    entry.density_map = True
    box = _render_form(entry)
    assert _select(box, t("toolbox.allocation.field_density")).v_model == "aoi"


def test_form_without_a_prefill_keeps_its_defaults():
    """No prefill: the form opens as a fresh New-allocation dialog."""
    box = _render_form(None)
    assert (
        _text_field(box, t("toolbox.allocation.field_name")).v_model == "allocation_1"
    )
    assert _select(box, t("toolbox.allocation.field_riskmap")).v_model is None
    assert _text_field(box, t("toolbox.allocation.field_years")).v_model == "4"


def test_edit_keeps_the_seeded_name_when_the_edited_job_stays_in_running_names():
    """`running_names` must include the job currently being edited.

    See docs/superpowers/specs/2026-08-05-allocation-edit-failed-design.md §4:
    the tile deliberately passes EVERY job in its list, including the failed
    one the form is seeded from. Excluding it looks like a harmless cleanup
    (an earlier external review actually proposed it) but breaks
    `use_artifact_name`: with the edited job's own name removed from
    `running_names`, the live suggestion for an "allocation_1" job becomes
    "allocation_1" itself — identical to the seeded name — so
    `on_name_input(entry.name)` finds nothing to mark dirty. The field then
    silently tracks the live suggestion instead of the seeded name, and flips
    the moment something else takes "allocation_1" out from under it.

    The seeded value looks identical either way at first render (both land on
    "allocation_1"), so this only diverges once the live suggestion changes
    under the field — done here by pushing a *new* `Project` (not a mutation:
    `Project` is a pydantic BaseModel and reacton skips re-render on
    `==`-equal props, see test_summary_tile_reactivity.py) carrying a saved
    run also named "allocation_1".
    """
    from spatialrisk.allocations.record import AllocationRun
    from spatialrisk.project import Project

    project = solara.reactive(Project(project_name="p"), equals=lambda a, b: a is b)
    box = _render_form(
        _entry(name="allocation_1"),
        running_names=frozenset({"allocation_1"}),
        project=project,
    )
    field = _text_field(box, t("toolbox.allocation.field_name"))
    assert field.v_model == "allocation_1"

    updated = Project(project_name="p")
    updated.allocations["allocation_1_r1"] = AllocationRun(
        name="allocation_1",
        run_id="r1",
        borders_file="/b.gpkg",
        defor_juris_ha=20000.0,
        years_forecast=4,
        annual_ha=1.0,
        total_ha=1.0,
        out_dir="/out",
        csv_path="/out/defor.csv",
    )
    project.set(updated)

    field = _text_field(box, t("toolbox.allocation.field_name"))
    assert field.v_model == "allocation_1"


def test_borders_picker_passes_an_admin_restore_seed(monkeypatch):
    """AdminLevelSelector takes its restore seed from `initial`, not `value`.

    pysepal documents `value` as output-only and snapshots `initial` once at
    mount, so this prop is the only way a prefilled admin code can come back.
    A stub stands in for the real selector: the genuine one drives an async
    pygaul/WFS cascade that a browserless render cannot resolve. The
    signature assertion below guards the stub itself — without it, this test
    would still pass even if the real component dropped `initial` entirely.
    """
    assert "initial" in inspect.signature(AdminLevelSelector.f).parameters

    import gui.widget.borders_picker as borders_picker_module

    seen = {}

    @solara.component
    def FakeSelector(method, gee=True, value=None, on_value=None, initial=None):
        seen["method"] = method
        seen["value"] = value
        seen["initial"] = initial
        solara.Text("stub")

    monkeypatch.setattr(borders_picker_module, "AdminLevelSelector", FakeSelector)

    reacton.render(
        BordersPicker(
            value=BordersSelection(method="ADMIN1", admin_code="1234"),
            on_value=lambda sel: None,
        )
    )
    assert seen["method"] == "ADMIN1"
    assert seen["initial"] == "1234"


# --- tile wiring --------------------------------------------------------------


def _failed_job(job_id="j1", name="allocation_1"):
    return {
        "id": job_id,
        "name": name,
        "status": "failed",
        "error": "boom",
        "entry": _entry(name=name),
    }


def _render_tile(jobs):
    """Render ToolboxTile with `jobs` in the module-level reactive.

    Returns (box, restore) — call restore() to put the reactive back, or the
    jobs leak into every later test in the session.
    """
    from gui.tile import toolbox_tile
    from spatialrisk.project import Project

    previous = toolbox_tile.allocation_jobs.value
    toolbox_tile.allocation_jobs.set(jobs)
    box, _rc = reacton.render(
        toolbox_tile.ToolboxTile(project=solara.reactive(Project(project_name="p")))
    )
    return box, lambda: toolbox_tile.allocation_jobs.set(previous)


def test_dismiss_removes_the_job_row():
    """The ✕ drops the job from the tile's job list."""
    from gui.tile import toolbox_tile

    box, restore = _render_tile([_failed_job()])
    try:
        _icon_button(box, "mdi-close").fire_event("click", {})
        assert toolbox_tile.allocation_jobs.value == []
    finally:
        restore()


def test_dismiss_leaves_saved_runs_alone():
    """Dismiss is job-list only: it never touches project.allocations."""
    from gui.tile import toolbox_tile
    from spatialrisk.project import Project

    project = Project(project_name="p")
    previous = toolbox_tile.allocation_jobs.value
    toolbox_tile.allocation_jobs.set([_failed_job()])
    try:
        box, _rc = reacton.render(
            toolbox_tile.ToolboxTile(project=solara.reactive(project))
        )
        _icon_button(box, "mdi-close").fire_event("click", {})
        assert project.allocations == {}
    finally:
        toolbox_tile.allocation_jobs.set(previous)


def test_edit_opens_the_form_seeded_with_the_failed_run():
    """The pencil reopens the dialog carrying the failed job's own name."""
    box, restore = _render_tile([_failed_job(name="reserve_north")])
    try:
        _icon_button(box, "mdi-pencil-outline").fire_event("click", {})
        field = _text_field(box, t("toolbox.allocation.field_name"))
        assert field.v_model == "reserve_north"
    finally:
        restore()


def test_submitting_an_edit_removes_the_old_failed_job_row(monkeypatch, tmp_path):
    """The whole point of the pencil: relaunching a failed run replaces its row.

    The other tests in this file cover dismiss and edit-seeds-the-form in
    isolation; neither exercises pencil -> edit -> Run -> job-list. This is
    the only test that drives that full sequence, so it is the only one that
    guards the `if editing_job_id.current is not None:` replace block at the
    end of `launch` (gui/tile/toolbox_tile.py) — deleting that block leaves
    every other test in this suite green (verified 2026-08-05).
    """
    from gui.tile import toolbox_tile
    from spatialrisk.predictions.prediction import Prediction
    from spatialrisk.project import Project

    # A real project with one registered prediction matching the prefill's
    # prediction_key, and a project-borders file that really exists on disk.
    # validate_form only checks existence for a FILE borders selection (see
    # tests/test_allocation_runner.py::test_validate_form_accepts_a_complete_form,
    # which uses the same empty-file trick), so this drives the dialog's own
    # validate_form() honestly rather than stubbing it out.
    borders_file = tmp_path / "borders.gpkg"
    borders_file.write_text("")
    # The icar run has no ready-made rate table, so the form also requires
    # the forest-at-period-start layer the table would be computed from.
    forest_file = tmp_path / "forest.tif"
    forest_file.write_text("")

    project = Project(project_name="p")
    project.predictions = {
        "icar_run": Prediction(
            path="/tmp/p.tif", model_key="icar", dataset_name="calibration"
        )
    }

    entry = _entry(
        name="reserve_north",
        borders=BordersSelection(method="FILE", file_path=str(borders_file)),
        forest_file=str(forest_file),
    )
    job = _failed_job(name="reserve_north")
    job["entry"] = entry

    # No background worker: the assertion is about the job list the tile
    # manages, not about an allocation actually running.
    monkeypatch.setattr(toolbox_tile, "spawn_in_context", lambda *a, **k: None)

    previous = toolbox_tile.allocation_jobs.value
    toolbox_tile.allocation_jobs.set([job])
    try:
        box, _rc = reacton.render(
            toolbox_tile.ToolboxTile(project=solara.reactive(project))
        )
        _icon_button(box, "mdi-pencil-outline").fire_event("click", {})

        run_label = t("toolbox.allocation.run")
        run_btn = next(b for b in _find(box, vw.Btn) if run_label in b.children)
        run_btn.fire_event("click", {})

        jobs = toolbox_tile.allocation_jobs.value
        assert "j1" not in [j["id"] for j in jobs]
        assert len(jobs) == 1
    finally:
        toolbox_tile.allocation_jobs.set(previous)
