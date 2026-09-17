"""Failed inference jobs are editable: reopen the dialog prefilled and rerun.

A failed run used to be a dead end — the row could only be dismissed, and the
user had to re-enter every parameter in a fresh dialog. Now the job dict keeps
the submission entry it was launched from, the failed row offers an edit
action, and the Predict dialog can be seeded from that entry so the user fixes
the one wrong parameter and reruns.
"""

import inspect
import types

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.scripts.product_rows import inference_rows  # noqa: E402
from gui.widget.inference_output_list import InferenceOutputList  # noqa: E402
from gui.widget.prediction_form_dialog import (  # noqa: E402
    NO_MASK,
    PredictionFormDialog,
)

MODEL_ENTRY = {
    "kind": "model",
    "model_key": "glm_glm_v1",
    "dataset_key": "calibration",
    "name": "run_a",
    "mask_layer": "forest_gfc_tc75",
}


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


def _select(box, label):
    return next((s for s in _find(box, vw.Select) if s.label == label), None)


def _text_field(box, label):
    return next((f for f in _find(box, vw.TextField) if f.label == label), None)


# --- job rows carry their submission entry -----------------------------------


def test_job_rows_carry_their_submission_entry():
    """inference_rows passes the launch entry through so edit can prefill."""
    jobs = [
        {
            "id": "j1",
            "model_key": "glm_glm_v1",
            "dataset_name": "calibration",
            "pred_name": "run_a",
            "status": "failed",
            "error": "boom",
            "entry": MODEL_ENTRY,
        }
    ]
    rows = inference_rows(None, jobs)
    assert rows[0]["entry"] == MODEL_ENTRY


# --- failed rows offer an edit action ----------------------------------------


def _job(status, entry=MODEL_ENTRY, job_id="j1"):
    return {
        "id": job_id,
        "model_key": "glm_glm_v1",
        "dataset_name": "calibration",
        "pred_name": "run_a",
        "status": status,
        "error": "boom" if status == "failed" else None,
        "entry": entry,
    }


def _render_list(jobs, on_edit):
    project = solara.reactive(types.SimpleNamespace(predictions={}))
    box, _rc = reacton.render(
        InferenceOutputList(
            project=project,
            inference_jobs=solara.reactive(jobs),
            on_dismiss=lambda job_id: None,
            on_edit=on_edit,
        )
    )
    return box


def test_failed_row_offers_edit_that_hands_back_the_row():
    """The pencil action on a failed row hands the row (job + entry) up."""
    edited = []
    box = _render_list([_job("failed")], edited.append)
    btn = _icon_button(box, "mdi-pencil-outline")
    assert btn is not None
    btn.fire_event("click", {})
    assert edited and edited[0]["job_id"] == "j1"
    assert edited[0]["entry"] == MODEL_ENTRY


def test_running_row_offers_no_edit():
    """A run in flight cannot be edited out from under its worker."""
    box = _render_list([_job("running")], lambda row: None)
    assert _icon_button(box, "mdi-pencil-outline") is None


def test_failed_row_without_an_entry_offers_no_edit():
    """Defensive: a job that never recorded its entry cannot be re-edited."""
    box = _render_list([_job("failed", entry=None)], lambda row: None)
    assert _icon_button(box, "mdi-pencil-outline") is None


# --- the dialog seeds itself from a prefill entry ----------------------------


def _project(tmp_path, layer_names):
    variables = {
        n: types.SimpleNamespace(name=n, data_type="raster", path=f"/tmp/{n}.tif")
        for n in layer_names
    }
    dataset = types.SimpleNamespace(name="calibration", features=[])
    return types.SimpleNamespace(
        models={"glm_glm_v1": object(), "mw_calibration_mw": object()},
        datasets={"calibration": dataset},
        processed_variables=variables,
        predictions={},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def _render_dialog(tmp_path, prefill_entry, layer_names=None):
    submitted = []
    box, _rc = reacton.render(
        PredictionFormDialog(
            project=solara.reactive(
                _project(
                    tmp_path, layer_names or ["forest_gfc_tc30", "forest_gfc_tc75"]
                )
            ),
            open_=solara.reactive(True),
            on_submit=submitted.append,
            prefill=solara.reactive(prefill_entry),
        )
    )
    return box, submitted


def test_dialog_prefills_a_model_entry(tmp_path):
    """Model, dataset, mask and name all come back exactly as submitted.

    Two forest layers on purpose: the ambiguous seed suggests nothing, so a
    prefilled mask select proves the entry won over the suggestion.
    """
    box, _ = _render_dialog(tmp_path, MODEL_ENTRY)
    assert _select(box, t("tiles.inference.model_select_label")).v_model == "glm_glm_v1"
    assert (
        _select(box, t("tiles.inference.dataset_select_label")).v_model == "calibration"
    )
    mask = _select(box, t("tiles.inference.mask_layer_label"))
    assert mask.v_model == "forest_gfc_tc75"
    assert _text_field(box, t("tiles.inference.pred_name_label")).v_model == "run_a"


def test_dialog_prefills_no_mask_as_the_explicit_choice(tmp_path):
    """A run submitted with 'no mask' (None) reopens showing the sentinel."""
    entry = dict(MODEL_ENTRY, mask_layer=None)
    box, _ = _render_dialog(tmp_path, entry)
    assert _select(box, t("tiles.inference.mask_layer_label")).v_model == NO_MASK


def test_dialog_without_prefill_behaves_as_before(tmp_path):
    """No prefill: the sole-forest suggestion still seeds the mask select."""
    box, _ = _render_dialog(tmp_path, None, layer_names=["altitude", "forest_gfc_tc30"])
    _select(box, t("tiles.inference.model_select_label")).v_model = "glm_glm_v1"
    mask = _select(box, t("tiles.inference.mask_layer_label"))
    assert mask.v_model == "forest_gfc_tc30"


def test_dialog_prefills_an_import_entry(tmp_path):
    """An import entry reopens in import mode with value scale and name restored."""
    entry = {
        "kind": "import",
        "name": "external_pred",
        "path": "/data/external_pred.tif",
        "value_scale": "probability",
    }
    box, _ = _render_dialog(tmp_path, entry)
    assert _select(box, t("tiles.inference.source_label")).v_model == "import"
    assert (
        _select(box, t("widgets.prediction_import_modal.label_scale")).v_model
        == "probability"
    )
    name = _text_field(box, t("tiles.inference.pred_name_label"))
    assert name.v_model == "external_pred"


# --- tile wiring --------------------------------------------------------------


def test_tile_records_the_entry_on_every_job():
    """Both launchers stamp the submission entry onto their job dict."""
    from gui.tile.inference_tile import InferenceTile

    src = inspect.getsource(InferenceTile)
    assert src.count('"entry": entry') >= 2


def test_tile_wires_edit_to_a_prefilled_dialog_and_replaces_the_job():
    """Edit opens the dialog prefilled; submitting drops the old failed row."""
    from gui.tile.inference_tile import InferenceTile

    src = inspect.getsource(InferenceTile)
    assert "on_edit=" in src
    assert "prefill=" in src
    assert "editing_job_id" in src
