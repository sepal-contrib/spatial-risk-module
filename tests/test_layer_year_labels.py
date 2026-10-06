"""Layer selects name a year the way the dataset dialog does: ``defor (2015)``.

The sampling raster/mask selects and the Predict dialog's mask select list
processed layers by storage key (``defor_2015``). They keep the key as the
value — samples and predictions still record it — and show the label as text;
the read-only details views label the recorded key the same way.
"""

import types

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.scripts.variable_labels import layer_items, layer_label  # noqa: E402
from gui.widget.prediction_form_dialog import (  # noqa: E402
    NO_MASK,
    PredictionFormDialog,
)
from gui.widget.sample_form_dialog import (  # noqa: E402
    SampleDetailsDialog,
    SampleFormDialog,
)
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.sample import Sample  # noqa: E402

Project._ensure_model_schemas()


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _select(box, label):
    return next(s for s in _find(box, vw.Select) if s.label == label)


def _var(name, year=None):
    return types.SimpleNamespace(
        name=name, year=year, data_type="raster", path=f"/tmp/{name}.tif"
    )


_VARS = {
    "altitude": _var("altitude"),
    "forest_gfc_2015": _var("forest_gfc", 2015),
    "forest_gfc_2020": _var("forest_gfc", 2020),
}


def _project():
    p = Project(project_name="labels")
    p.processed_variables.update(_VARS)
    return p


def test_layer_label_names_the_year_of_a_temporal_layer():
    """A temporal layer reads name (year); a static one its bare name."""
    p = _project()
    assert layer_label(p, "forest_gfc_2015") == "forest_gfc (2015)"
    assert layer_label(p, "altitude") == "altitude"


def test_layer_label_falls_back_to_the_key_of_a_removed_layer():
    """A key the project no longer has is shown as is."""
    p = _project()
    assert layer_label(p, "deleted_2019") == "deleted_2019"
    assert layer_label(p, None) is None


def test_layer_items_keep_the_storage_key_as_value():
    """Only the text changes; the select value is still the storage key."""
    assert layer_items(_project(), ["forest_gfc_2020"]) == [
        {"text": "forest_gfc (2020)", "value": "forest_gfc_2020"}
    ]


def test_sampling_raster_and_mask_selects_show_years():
    """The New-sample raster and mask selects label layers by year."""
    box, rc = reacton.render(
        SampleFormDialog(
            project=solara.reactive(_project()),
            open_=solara.reactive(True),
            existing_names=frozenset(),
            running_names=frozenset(),
            on_submit=lambda entry: None,
        ),
        handle_error=False,
    )
    try:
        raster = _select(box, t("tiles.sampling.raster_variable_label_area"))
        mask = _select(box, t("tiles.sampling.mask_variable_label"))
        expected = {"text": "forest_gfc (2015)", "value": "forest_gfc_2015"}
        assert expected in raster.items
        assert expected in mask.items
        assert {"text": "altitude", "value": "altitude"} in raster.items
    finally:
        rc.close()


def test_sample_details_label_the_recorded_layers():
    """The sample details label the recorded raster and mask keys."""
    p = _project()
    p.samples["s1"] = Sample(
        name="s1",
        raster_var_name="forest_gfc_2020",
        mask_var_name="forest_gfc_2015",
        strategy="random",
        n_samples=10,
    )
    box, rc = reacton.render(
        SampleDetailsDialog(
            project=solara.reactive(p), sample_key="s1", on_close=lambda: None
        ),
        handle_error=False,
    )
    try:
        values = {f.label: f.v_model for f in _find(box, vw.TextField)}
        assert (
            values[t("tiles.sampling.raster_variable_label_area")]
            == "forest_gfc (2020)"
        )
        assert values[t("tiles.sampling.mask_variable_label")] == "forest_gfc (2015)"
    finally:
        rc.close()


def test_predict_mask_select_shows_years(tmp_path):
    """The Predict mask select labels layers by year after No mask."""
    project = types.SimpleNamespace(
        models={"glm_glm_v1": object()},
        datasets={
            "calibration": types.SimpleNamespace(name="calibration", features=[])
        },
        processed_variables=dict(_VARS),
        predictions={},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )
    box, rc = reacton.render(
        PredictionFormDialog(
            project=solara.reactive(project),
            open_=solara.reactive(True),
            on_submit=lambda entry: None,
        ),
        handle_error=False,
    )
    try:
        _select(box, t("tiles.inference.model_select_label")).v_model = "glm_glm_v1"
        items = _select(box, t("tiles.inference.mask_layer_label")).items
        assert items[0]["value"] == NO_MASK
        assert {"text": "forest_gfc (2015)", "value": "forest_gfc_2015"} in items
        assert {"text": "altitude", "value": "altitude"} in items
    finally:
        rc.close()
