"""A dataset lacking a variable the model's formula uses fails with a clear message.

Without the up-front check the run dies deep inside patsy with
``NameError: name 'slope' is not defined`` — meaningless to a user who only
picked a dataset in the Predict dialog.
"""

import types

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.scripts.inference_runner import missing_model_variables  # noqa: E402
from gui.tile.inference_tile import _inference_error_message  # noqa: E402
from gui.widget.prediction_form_dialog import (  # noqa: E402
    NO_MASK,
    PredictionFormDialog,
)
from spatialrisk.mlmodels.base import MissingModelVariablesError  # noqa: E402
from spatialrisk.mlmodels.glm_model import GLMModel  # noqa: E402

_FORMULA = (
    "I(loss_2010_2020) + trial ~ scale(roads_dist) + scale(slope)"
    " + C(geobosques, levels=[0, 1])"
)


def _dataset(*features, name="validation"):
    return types.SimpleNamespace(
        name=name, features=[types.SimpleNamespace(name=f) for f in features]
    )


def _model(**kw):
    return GLMModel(name="glm_v1", model_type="glm", formula=_FORMULA, **kw)


def test_required_variables_are_the_formula_rhs_names():
    """Transforms and C()'s levels= keyword are not variables; the target isn't."""
    assert _model().required_variables() == ["geobosques", "roads_dist", "slope"]


def test_required_variables_fall_back_to_feature_names_without_a_formula():
    """No formula to parse: the training feature list is the contract."""
    model = GLMModel(name="glm_v1", model_type="glm", feature_names=["a", "b"])
    assert model.required_variables() == ["a", "b"]


def test_missing_variables_lists_what_the_dataset_lacks():
    """Only the absent formula variables are reported."""
    ds = _dataset("roads_dist", "geobosques")
    assert _model().missing_variables(ds) == ["slope"]


def test_unused_dataset_features_are_not_required():
    """A feature dropped from the formula at training time need not exist."""
    model = _model(feature_names=["roads_dist", "slope", "geobosques", "altitude"])
    assert model.missing_variables(_dataset("roads_dist", "slope", "geobosques")) == []


def test_resolve_dataset_raises_a_clear_error_for_a_foreign_dataset():
    """The error names the dataset, the model and every missing variable."""
    model = _model()
    with pytest.raises(MissingModelVariablesError) as info:
        model._resolve_dataset(_dataset("roads_dist", "geobosques"))
    err = info.value
    assert err.missing == ["slope"]
    assert err.dataset_name == "validation"
    assert err.model_name == "glm_v1"
    assert "slope" in str(err) and "validation" in str(err)


def test_resolve_dataset_checks_even_when_it_is_the_models_own_dataset():
    """run_inference rebinds model.dataset before apply(), so identity is no proof."""
    ds = _dataset("roads_dist")
    model = _model()
    model.dataset = ds
    with pytest.raises(MissingModelVariablesError) as info:
        model._resolve_dataset(ds)
    assert info.value.missing == ["geobosques", "slope"]


def test_resolve_dataset_passes_a_complete_dataset_through():
    """A dataset with every formula variable is returned unchanged."""
    ds = _dataset("roads_dist", "slope", "geobosques")
    assert _model()._resolve_dataset(ds) is ds


# --- GUI: the Predict dialog refuses the run, the worker translates it --------


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _project(tmp_path, *features):
    return types.SimpleNamespace(
        models={"glm_glm_v1": _model()},
        datasets={"validation": _dataset(*features)},
        processed_variables={},
        predictions={},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def _render_and_submit(tmp_path, *features):
    submitted = []
    box, _rc = reacton.render(
        PredictionFormDialog(
            project=solara.reactive(_project(tmp_path, *features)),
            open_=solara.reactive(True),
            on_submit=submitted.append,
        )
    )
    selects = {s.label: s for s in _find(box, vw.Select)}
    selects[t("tiles.inference.model_select_label")].v_model = "glm_glm_v1"
    selects[t("tiles.inference.dataset_select_label")].v_model = "validation"
    mask = next(
        s
        for s in _find(box, vw.Select)
        if s.label == t("tiles.inference.mask_layer_label")
    )
    mask.v_model = NO_MASK
    label = t("tiles.inference.run_button")
    next(b for b in _find(box, vw.Btn) if label in str(b.children)).fire_event(
        "click", {}
    )
    alerts = " ".join(
        str(c)
        for a in _find(box, vw.Alert)
        for c in (a.children or [])
        if isinstance(c, str)
    )
    return submitted, alerts


def test_runner_helper_reports_missing_variables(tmp_path):
    """The dialog-side helper sees the same gap apply() would."""
    proj = _project(tmp_path, "roads_dist")
    assert missing_model_variables(proj, "glm_glm_v1", "validation") == [
        "geobosques",
        "slope",
    ]


def test_runner_helper_ignores_benchmark_families(tmp_path):
    """JNR/MW resolve their own layers: no formula check applies."""
    proj = _project(tmp_path)
    proj.models["mw_calibration_mw"] = proj.models["glm_glm_v1"]
    assert missing_model_variables(proj, "mw_calibration_mw", "validation") == []


def test_dialog_refuses_a_dataset_missing_model_variables(tmp_path):
    """The run is never queued; the alert says what is missing."""
    submitted, alerts = _render_and_submit(tmp_path, "roads_dist", "geobosques")
    assert not submitted
    assert alerts == t(
        "tiles.inference.error_missing_variables",
        dataset="validation",
        model="glm_glm_v1",
        names="slope",
    )


def test_dialog_accepts_a_complete_dataset(tmp_path):
    """A dataset with every formula variable submits as before."""
    submitted, _ = _render_and_submit(tmp_path, "roads_dist", "slope", "geobosques")
    assert submitted and submitted[0]["dataset_key"] == "validation"


def test_worker_error_message_is_translated_for_missing_variables():
    """A run that still fails at apply() shows the same translated text."""
    exc = MissingModelVariablesError(["slope"], "validation", "glm_v1")
    msg = _inference_error_message(exc, "glm_glm_v1", "validation")
    assert msg == t(
        "tiles.inference.error_missing_variables",
        dataset="validation",
        model="glm_glm_v1",
        names="slope",
    )
    assert "NameError" not in msg


def test_worker_error_message_passes_other_errors_through():
    """Unknown failures keep their raw message."""
    assert _inference_error_message(ValueError("boom"), "m", "d") == "boom"
