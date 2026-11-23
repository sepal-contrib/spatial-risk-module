"""The Predict dialog lets the user assign which layer masks an ML run.

The mask is generic — any processed raster of the project can be assigned,
whether or not it is a feature of the selected dataset, and "no mask" is a
valid explicit choice — so the algorithms carry no Hansen-specific assumption.
The deforisk convention (mask to forest at period start) survives only as a
*seed*: a sole ``forest_gfc``-derived layer is preselected as a suggestion the
user can override. The field is only meaningful for the ML families
(GLM/RF/ICAR); JNR and MW resolve their own layers and must never see it.
"""

import types

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.widget.prediction_form_dialog import (  # noqa: E402
    NO_MASK,
    PredictionFormDialog,
)


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _project(tmp_path, layer_names):
    """Fake project whose processed rasters are ``layer_names``.

    The dataset's own features stay empty on purpose: the mask select must
    list the *project's* rasters, independent of dataset membership.
    """
    variables = {
        n: types.SimpleNamespace(name=n, data_type="raster", path=f"/tmp/{n}.tif")
        for n in layer_names
    }
    dataset = types.SimpleNamespace(name="calibration", features=[])
    return types.SimpleNamespace(
        models={
            "glm_glm_v1": object(),
            # A realistic MW entry: the Predict dialog reads its trained
            # windows to build the window-sizes field.
            "mw_calibration_mw": types.SimpleNamespace(
                ldefrate_files={"5": "a.tif", "11": "b.tif"}, win_size_list=[5, 11]
            ),
        },
        datasets={"calibration": dataset},
        processed_variables=variables,
        predictions={},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def _render(tmp_path, layer_names):
    submitted = []
    box, _rc = reacton.render(
        PredictionFormDialog(
            project=solara.reactive(_project(tmp_path, layer_names)),
            open_=solara.reactive(True),
            on_submit=submitted.append,
        )
    )
    return box, submitted


def _select(box, label):
    return next((s for s in _find(box, vw.Select) if s.label == label), None)


def _mask_select(box):
    return _select(box, t("tiles.inference.mask_layer_label"))


def _pick(box, model_key, dataset_key="calibration"):
    """Drive the model select (and optionally the dataset select)."""
    _select(box, t("tiles.inference.model_select_label")).v_model = model_key
    if dataset_key is not None:
        _select(box, t("tiles.inference.dataset_select_label")).v_model = dataset_key


def _create(box):
    label = t("common.add")
    btn = next(b for b in _find(box, vw.Btn) if label in str(b.children))
    btn.fire_event("click", {})


def _item_values(select):
    return [i["value"] for i in select.items]


def test_mask_select_is_hidden_until_an_ml_model_is_chosen(tmp_path):
    """No model yet — the field's relevance is unknown, so it stays hidden."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30"])
    assert _mask_select(box) is None


def test_mask_select_appears_without_a_dataset(tmp_path):
    """The mask is dataset-independent: an ML model alone reveals the field."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30"])
    _pick(box, "glm_glm_v1", dataset_key=None)
    assert _mask_select(box) is not None


def test_mask_select_offers_every_project_raster_plus_no_mask(tmp_path):
    """Any processed raster can be the mask, and 'no mask' is always on offer.

    None of these layers are features of the selected dataset — the candidate
    list is the project's raster catalogue, not the dataset's feature list.
    """
    box, _ = _render(tmp_path, ["my_forest_mask", "altitude", "forest_gfc_tc30"])
    _pick(box, "glm_glm_v1")
    sel = _mask_select(box)
    assert sel is not None
    assert _item_values(sel) == [
        NO_MASK,
        "altitude",
        "forest_gfc_tc30",
        "my_forest_mask",
    ]


def test_sole_forest_layer_is_seeded_as_the_suggestion(tmp_path):
    """One Hansen-derived layer: seed it, the deforisk-style default."""
    box, _ = _render(tmp_path, ["altitude", "forest_gfc_tc30"])
    _pick(box, "glm_glm_v1")
    assert _mask_select(box).v_model == "forest_gfc_tc30"


def test_two_forest_layers_start_unset(tmp_path):
    """Two forest definitions: the app must not pick one on the user's behalf."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30", "forest_gfc_tc75"])
    _pick(box, "glm_glm_v1")
    assert _mask_select(box).v_model in ("", None)


def test_no_forest_layer_starts_unset(tmp_path):
    """Without a forest layer nothing is seeded — not even 'no mask'."""
    box, _ = _render(tmp_path, ["altitude", "forest_tmf"])
    _pick(box, "glm_glm_v1")
    assert _mask_select(box).v_model in ("", None)


def test_benchmark_family_never_shows_the_field(tmp_path):
    """JNR/MW resolve their own layers — the field would be meaningless."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30"])
    _pick(box, "mw_calibration_mw")
    assert _mask_select(box) is None


def test_submitted_entry_carries_the_chosen_mask_layer(tmp_path):
    """The pick reaches the tile as entry["mask_layer"]."""
    box, submitted = _render(tmp_path, ["forest_gfc_tc30", "forest_gfc_tc75"])
    _pick(box, "glm_glm_v1")
    _mask_select(box).v_model = "forest_gfc_tc75"
    _create(box)
    assert submitted and submitted[0]["mask_layer"] == "forest_gfc_tc75"
    assert submitted[0]["kind"] == "model"


def test_no_mask_choice_submits_none(tmp_path):
    """The 'No mask' choice reaches the runner as None: predict everywhere."""
    box, submitted = _render(tmp_path, ["altitude", "forest_tmf"])
    _pick(box, "glm_glm_v1")
    _mask_select(box).v_model = NO_MASK
    _create(box)
    assert submitted and submitted[0]["mask_layer"] is None


def test_unset_mask_blocks_the_run(tmp_path):
    """An ML run cannot launch until the user assigns a mask (or 'no mask')."""
    box, submitted = _render(tmp_path, ["forest_gfc_tc30", "forest_gfc_tc75"])
    _pick(box, "glm_glm_v1")
    _create(box)
    assert not submitted


def test_benchmark_entry_does_not_carry_the_key(tmp_path):
    """A benchmark entry stays exactly as it was before this feature."""
    box, submitted = _render(tmp_path, ["forest_gfc_tc30"])
    _pick(box, "mw_calibration_mw")
    _create(box)
    assert submitted and "mask_layer" not in submitted[0]


def test_dialog_and_runner_share_one_candidate_source(tmp_path):
    """Candidates and the forest seed live next to run_inference, not here."""
    import inspect

    import gui.widget.prediction_form_dialog as mod

    src = inspect.getsource(mod)
    assert "mask_layer_candidates" in src
    assert "suggested_mask_layer" in src
    # the resolve_predefined(...) == "forest_gfc" test must not be duplicated here
    assert "resolve_predefined" not in src


# --- the mask belongs in the suggested name ----------------------------------
#
# The mask changes the output raster, so two runs of one model over one dataset
# that differ only by mask are different products. Without the mask in the
# suggested name they both defaulted to "<model>__<dataset>", and the second
# read as an overwrite of the first rather than a new map.


def _name_field(box):
    label = t("tiles.inference.pred_name_label")
    return next(f for f in _find(box, vw.TextField) if f.label == label)


def _suggested_name(box, mask_value):
    _pick(box, "glm_glm_v1")
    _mask_select(box).v_model = mask_value
    return _name_field(box).v_model


def test_two_masks_suggest_two_different_names(tmp_path):
    """The reported bug: same model, same dataset, different mask, one name."""
    box_a, _ = _render(tmp_path, ["forest_gfc_tc30", "forest_2015"])
    box_b, _ = _render(tmp_path, ["forest_gfc_tc30", "forest_2015"])

    name_a = _suggested_name(box_a, "forest_gfc_tc30")
    name_b = _suggested_name(box_b, "forest_2015")

    assert name_a != name_b


def test_the_suggested_name_carries_the_mask_layer(tmp_path):
    """The mask layer key is the discriminator, appended after the dataset."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30"])
    assert (
        _suggested_name(box, "forest_gfc_tc30")
        == "glm_glm_v1__calibration__forest_gfc_tc30"
    )


def test_predicting_everywhere_gets_its_own_token(tmp_path):
    """An unmasked run must not collide with a masked one either."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30"])
    assert _suggested_name(box, NO_MASK) == "glm_glm_v1__calibration__nomask"


def test_the_same_mask_still_suggests_the_same_name(tmp_path):
    """Re-running one configuration still lands on its existing name.

    Refreshing a stale map stays an overwrite rather than a pile-up.
    """
    box_a, _ = _render(tmp_path, ["forest_gfc_tc30"])
    box_b, _ = _render(tmp_path, ["forest_gfc_tc30"])
    assert _suggested_name(box_a, "forest_gfc_tc30") == _suggested_name(
        box_b, "forest_gfc_tc30"
    )


def test_a_family_without_a_mask_keeps_the_two_part_name(tmp_path):
    """MW resolves its own layers, so its name gains no mask token."""
    box, _ = _render(tmp_path, ["forest_gfc_tc30"])
    _pick(box, "mw_calibration_mw")
    assert _name_field(box).v_model == "mw_calibration_mw__calibration"


def test_an_edited_name_is_not_overwritten_when_the_mask_changes(tmp_path):
    """A typed name survives a later mask change.

    The suggestion only tracks a field the user has not touched.
    """
    box, _ = _render(tmp_path, ["forest_gfc_tc30", "forest_2015"])
    _pick(box, "glm_glm_v1")
    _mask_select(box).v_model = "forest_gfc_tc30"
    _name_field(box).v_model = "my_run"
    _mask_select(box).v_model = "forest_2015"
    assert _name_field(box).v_model == "my_run"
