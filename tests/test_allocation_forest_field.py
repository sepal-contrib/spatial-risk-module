"""The allocation form's "forest at period start" select.

Shown only while the automatic rate table would be *computed* (the one case
that needs the layer), seeded from what the prediction recorded, and
overridable from the project's processed rasters.
"""

import types

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
from gui.widget.allocation_form import AllocationFormDialog  # noqa: E402

FOREST_LABEL = "toolbox.allocation.field_forest"


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _select(box, label):
    return next((s for s in _find(box, vw.Select) if s.label == label), None)


def _project(tmp_path, *, persisted=False, mask_layer="geobosques_2010"):
    forest = tmp_path / "geobosques_2010.tif"
    forest.write_bytes(b"")
    csv = tmp_path / "defrate_cat_bm.csv"
    if persisted:
        csv.write_text("cat\n1\n")
    pred = types.SimpleNamespace(
        model_key="jnr_v1" if persisted else "rf_v1",
        dataset_name="dataset_1",
        window=None,
        path=tmp_path / "prob.tif",
        defrate_path=csv if persisted else None,
        run_params={"mask_layer": mask_layer} if mask_layer else {},
        model_snapshot={},
    )
    dataset = types.SimpleNamespace(
        name="dataset_1", target=types.SimpleNamespace(name="x", path="/x"), features=[]
    )
    return types.SimpleNamespace(
        predictions={"run": pred},
        processed_variables={
            "geobosques_2010": types.SimpleNamespace(path=str(forest)),
            "altitude": types.SimpleNamespace(path=str(tmp_path / "alt.tif")),
        },
        datasets={"dataset_1": dataset},
        get_dataset=lambda name: dataset,
        allocations={},
    )


def _render(project, prefill=None, launched=None):
    box, rc = reacton.render(
        AllocationFormDialog(
            open_=solara.reactive(True),
            project=solara.reactive(project),
            on_launch=(launched.append if launched is not None else lambda f: None),
            on_close=lambda: None,
            prefill=solara.reactive(prefill),
        )
    )
    return box, rc


def _pick_prediction(box):
    _select(box, t("toolbox.allocation.field_riskmap")).v_model = "run"


def test_field_appears_seeded_when_the_table_will_be_computed(tmp_path):
    """The select shows for a computed table, seeded from the recorded mask layer."""
    project = _project(tmp_path)
    box, _rc = _render(project)
    assert _select(box, t(FOREST_LABEL)) is None  # no prediction picked yet
    _pick_prediction(box)
    field = _select(box, t(FOREST_LABEL))
    assert field is not None
    assert field.v_model == str(tmp_path / "geobosques_2010.tif")
    # Same choices as the forest mask: every processed raster.
    assert {i["text"] for i in field.items} == {"geobosques_2010", "altitude"}


def test_field_is_hidden_when_the_table_is_ready_made(tmp_path):
    """A persisted table needs no forest layer, so the select stays hidden."""
    project = _project(tmp_path, persisted=True)
    box, _rc = _render(project)
    _pick_prediction(box)
    assert _select(box, t(FOREST_LABEL)) is None


def test_field_is_hidden_in_custom_table_mode(tmp_path):
    """A custom table file bypasses the computation, so the select stays hidden."""
    project = _project(tmp_path)
    box, _rc = _render(project)
    _pick_prediction(box)
    _select(box, t("toolbox.allocation.field_defrate")).v_model = "custom"
    assert _select(box, t(FOREST_LABEL)) is None


def test_unseedable_field_shows_a_hint_and_blocks_the_run(tmp_path):
    """Nothing to seed from: empty select plus the pick-a-layer hint."""
    project = _project(tmp_path, mask_layer=None)
    box, _rc = _render(project)
    _pick_prediction(box)
    field = _select(box, t(FOREST_LABEL))
    assert field is not None and field.v_model is None
    texts = " ".join(
        str(c)
        for h in _find(box, vw.Html)
        for c in (h.children or [])
        if isinstance(c, str)
    )
    assert t("toolbox.allocation.field_forest_none") in texts


def test_prefill_wins_over_the_suggestion(tmp_path):
    """Re-editing a failed run keeps the forest layer it was launched with."""
    project = _project(tmp_path)
    alt = str(tmp_path / "alt.tif")
    entry = AllocationForm(
        name="retry",
        prediction_key="run",
        user_defrate_path=None,
        borders=BordersSelection(method="ADMIN0", admin_code="1"),
        mask_file=None,
        defor_juris_ha=1.0,
        years_forecast=1.0,
        forest_file=alt,
    )
    box, _rc = _render(project, prefill=entry)
    assert _select(box, t(FOREST_LABEL)).v_model == alt


def test_form_payload_carries_the_forest_file(tmp_path):
    """build_form is private; pin it through the module source instead."""
    import inspect

    from gui.widget import allocation_form

    src = inspect.getsource(allocation_form.AllocationFormDialog)
    assert "forest_file=" in src


def test_rate_table_select_carries_the_in_field_help_icon(tmp_path):
    """Same idiom as the Train dialog's model select: icon + click listener."""
    box, _rc = _render(_project(tmp_path))
    sel = _select(box, t("toolbox.allocation.field_defrate"))
    assert sel.prepend_inner_icon == "mdi-information-outline"
    assert "field-info-icon" in (sel.class_ or "")
    assert "click:prepend-inner" in sel._event_handlers_map


def test_rate_table_help_popup_explains_the_forest_field(tmp_path):
    """The popup is what tells the user why the forest field comes and goes."""
    box, _rc = _render(_project(tmp_path))
    dialogs = _find(box, vw.Dialog)
    assert any(
        t("toolbox.allocation.field_forest").split(" (")[0] in str(d.children)
        for d in dialogs
    )
    assert t("toolbox.allocation.field_forest").split(" (")[0] in t(
        "toolbox.allocation.defrate_info_md"
    )
