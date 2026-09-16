"""PredictionDetailsDialog: how a registered prediction was produced.

Everything the dialog shows was frozen onto the Prediction when it was written
(model_snapshot / dataset_snapshot / run_params), so a prediction stays
explainable long after the model that made it was retrained or deleted.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_model_form_dialog_render: warm the translator before the first
# render — the first t() *during* a render breaks reacton's widget map.
t("common.cancel")

from gui.widget.prediction_form_dialog import PredictionDetailsDialog  # noqa: E402
from spatialrisk.predictions.prediction import Prediction  # noqa: E402
from spatialrisk.project import Project  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _project_with(*predictions) -> solara.Reactive:
    p = Project(project_name="p")
    for pred in predictions:
        p.predictions[pred.name or pred.model_key] = pred
    return solara.reactive(p)


def _render(project, row_key):
    box, _rc = reacton.render(
        PredictionDetailsDialog(project=project, row_key=row_key, on_close=lambda: None)
    )
    return box


def _values(box):
    return {f.label: f.v_model for f in _find(box, vw.TextField)}


def _glm_prediction(**overrides):
    """A GLM run, as _register_prediction would have frozen it."""
    kwargs = dict(
        name="run_a",
        path="/tmp/preds/run_a/calibration.tif",
        model_key="glm_glm_v1",
        dataset_name="calibration",
        year=2020,
        created_at="2026-08-20T11:30:00",
        model_snapshot={
            "model_type": "glm",
            "name": "glm_v1",
            "formula": "fcc ~ slope + C(pa, levels=[0, 1])",
        },
        dataset_snapshot={
            "name": "calibration",
            "year": 2020,
            "target_name": "forest_loss_2015_2020",
            "target_year": 2020,
            "feature_names": ["slope", "dist_road"],
        },
        run_params={"mask_layer": "forest_gfc_tc30"},
    )
    kwargs.update(overrides)
    return Prediction(**kwargs)


def _imported_prediction():
    """An imported raster: no model ran, so both snapshots are empty."""
    return Prediction(
        name="external_map",
        path="/tmp/imported_predictions/external_map.tif",
        model_key="external_map",
        dataset_name="imported",
        display_palette="stretch",
        created_at="2026-08-21T09:00:00",
    )


# --- open / closed -----------------------------------------------------------


def test_dialog_is_closed_when_no_row_is_selected():
    """No row picked, no dialog."""
    box = _render(_project_with(_glm_prediction()), None)
    assert [d.v_model for d in _find(box, vw.Dialog)] == [False]


def test_dialog_is_open_for_a_registered_prediction():
    """Picking a row opens the dialog on that prediction."""
    box = _render(_project_with(_glm_prediction()), "run_a")
    assert [d.v_model for d in _find(box, vw.Dialog)] == [True]


def test_dialog_stays_closed_for_an_unknown_row_key():
    """A row key with no registered prediction must not open an empty shell."""
    box = _render(_project_with(_glm_prediction()), "no_such_run")
    assert [d.v_model for d in _find(box, vw.Dialog)] == [False]


# --- what produced it --------------------------------------------------------


def test_shows_the_model_that_produced_the_prediction():
    """The model family is resolved to its catalogue label, not its raw type."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert values[t("tiles.inference.model_select_label")] == t("models.glm.label")


def test_shows_the_dataset_and_target_from_the_frozen_snapshot():
    """Dataset identity comes from the snapshot, not from today's registry."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert values[t("tiles.inference.dataset_select_label")] == "calibration"
    assert values[t("tiles.inference.details_target")] == "forest_loss_2015_2020"


def test_shows_the_features_the_model_was_applied_over():
    """The feature list is rendered as the form renders multi-value fields."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert values[t("tiles.inference.details_features")] == "slope, dist_road"


def test_shows_the_mask_layer_the_run_used():
    """The mask is an apply() argument, recoverable only from run_params."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert values[t("tiles.inference.mask_layer_label")] == "forest_gfc_tc30"


def test_shows_no_mask_as_an_explicit_choice_not_a_blank():
    """Predicting everywhere was a decision; a blank would hide that."""
    pred = _glm_prediction(run_params={"mask_layer": None})
    values = _values(_render(_project_with(pred), "run_a"))
    assert values[t("tiles.inference.mask_layer_label")] == t(
        "tiles.inference.mask_layer_none"
    )


def test_omits_the_mask_field_for_a_family_that_never_takes_one():
    """MW resolves its own layers; a blank Mask row would imply it had a choice."""
    pred = _glm_prediction(
        model_key="mw_calibration_mw",
        model_snapshot={"model_type": "mw", "name": "calibration_mw"},
        run_params={"windows": [5, 11]},
    )
    values = _values(_render(_project_with(pred), "run_a"))
    assert t("tiles.inference.mask_layer_label") not in values


def test_shows_the_formula_without_its_fit_time_categorical_levels():
    """levels=[...] is a fit-time safety net, noise to a reader."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert values[t("tiles.train.formula_label")] == "fcc ~ slope + C(pa)"


def test_shows_when_the_prediction_was_produced():
    """The run timestamp frozen at registration is shown."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert "2026-08-20" in values[t("tiles.inference.details_created")]


def test_a_param_missing_from_the_snapshot_reads_as_unknown():
    """A registry default must never be reported as the value a run used.

    A prediction registered before a param joined the registry has no value
    for it; showing today's default would invent provenance.
    """
    pred = _glm_prediction(
        model_snapshot={"model_type": "glm", "name": "glm_v1"}  # no solver/seed
    )
    values = _values(_render(_project_with(pred), "run_a"))
    assert values[t("models.glm.params.solver.label")] == "—"


# --- outputs -----------------------------------------------------------------


def test_shows_the_output_raster_path():
    """The raster this row refers to is named explicitly."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert "/tmp/preds/run_a/calibration.tif" in values.values()


def test_lists_every_output_of_a_multi_window_run():
    """Every window of an MW run is listed.

    One row groups one raster per window, so showing a single file would
    imply the others do not exist.
    """
    w5 = _glm_prediction(
        name="mwrun",
        path="/tmp/rmj_mw/mwrun/prob_mw_5.tif",
        window=5,
        model_key="mw_calibration_mw",
        model_snapshot={"model_type": "mw", "name": "calibration_mw"},
        run_params={"windows": [5, 11]},
    )
    w11 = _glm_prediction(
        name="mwrun",
        path="/tmp/rmj_mw/mwrun/prob_mw_11.tif",
        window=11,
        model_key="mw_calibration_mw",
        model_snapshot={"model_type": "mw", "name": "calibration_mw"},
        run_params={"windows": [5, 11]},
    )
    p = Project(project_name="p")
    p.predictions["mwrun_w5"] = w5
    p.predictions["mwrun_w11"] = w11

    values = _values(_render(solara.reactive(p), "mwrun"))

    assert values[t("tiles.inference.details_output_window", n=5)].endswith(
        "prob_mw_5.tif"
    )
    assert values[t("tiles.inference.details_output_window", n=11)].endswith(
        "prob_mw_11.tif"
    )


# --- imported rasters --------------------------------------------------------


def test_imported_raster_is_labelled_as_imported():
    """An import must never read as a model run."""
    values = _values(_render(_project_with(_imported_prediction()), "external_map"))
    assert values[t("tiles.inference.source_label")] == t(
        "tiles.inference.source_import"
    )


def test_imported_raster_omits_the_model_section_it_has_no_answer_for():
    """Empty snapshots must read as "no model ran", not a wall of em-dashes."""
    values = _values(_render(_project_with(_imported_prediction()), "external_map"))
    assert t("tiles.inference.model_select_label") not in values
    assert t("tiles.inference.details_target") not in values


def test_model_run_is_labelled_as_a_model_run():
    """The converse of the import case, so the Source field is not a constant."""
    values = _values(_render(_project_with(_glm_prediction()), "run_a"))
    assert values[t("tiles.inference.source_label")] == t(
        "tiles.inference.source_model"
    )


def _adapted_import():
    """An import adapted by the value-scale flow: run_params carry the source."""
    return Prediction(
        name="ext",
        path="/tmp/imported_predictions/ext.tif",
        model_key="ext",
        dataset_name="imported",
        display_palette="far",
        created_at="2026-09-16T09:00:00",
        run_params={
            "source_path": "/data/ext.tif",
            "value_scale": "probability",
            "source_crs": "EPSG:4326",
            "source_resolution": [0.00027, 0.00027],
            "source_dtype": "float32",
            "source_range": [0.0, 0.97],
        },
    )


def test_details_show_import_provenance():
    """An adapted import explains the source raster and the scale it declared."""
    values = _values(_render(_project_with(_adapted_import()), "ext"))
    assert values[t("tiles.inference.details_source_crs")] == "EPSG:4326"
    assert values[t("tiles.inference.details_source_dtype")] == "float32"
    assert values[t("tiles.inference.details_source_path")] == "/data/ext.tif"
    assert values[t("tiles.inference.details_value_scale")] == t(
        "widgets.prediction_import_modal.scale_probability"
    )
    # The joins are the only new formatting in this change: %g drops the
    # trailing zero of 0.0 and keeps 0.00027 out of scientific notation.
    res = values[t("tiles.inference.details_source_resolution")]
    assert res == "0.00027 \u00d7 0.00027"
    rng = values[t("tiles.inference.details_source_range")]
    assert rng == "0 \u2013 0.97"


def test_pre_adaptation_import_still_shows_its_display_palette():
    """An import registered before the adaptation flow has only a palette to show."""
    values = _values(_render(_project_with(_imported_prediction()), "external_map"))
    assert values[t("tiles.inference.details_palette_label")] == "stretch"
    assert t("tiles.inference.details_value_scale") not in values
