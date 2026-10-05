"""The dataset dialog picks each temporal variable together with its year (#40).

The dialog used to have one "temporal alignment" year for the whole dataset.
Leaving it empty passed the dialog, which closed before the tile's
``on_submit`` failed in ``Dataset.set_target`` / ``set_features``, so the error
showed up in the step panel instead of the modal. Now a temporal variable is
offered once per year ("defor (2015)"): picking it picks its year, each
variable may use a different year, and there is no year left to forget. What
the dialog still checks is caught in the modal; what only the builder can
catch (a layer missing on disk) is a toast, never a step-panel alert.
"""

import ipyvuetify as vw
import pytest
import reacton
import solara

import spatialrisk.project as project_module
from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.scripts.dataset_validation import (  # noqa: E402
    choice_value,
    dataset_form_error,
    variable_choices,
)
from gui.tile import dataset_tile  # noqa: E402
from gui.widget.dataset_form_dialog import DatasetFormDialog  # noqa: E402
from spatialrisk.dataset import Dataset  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402
from spatialrisk.variables.models import RasterType  # noqa: E402

Project._ensure_model_schemas()

# name -> years (None = one static layer)
_LAYERS = {
    "forest_loss": None,
    "altitude": None,
    "defor": (2015, 2020),
    "forest_gfc": (2015, 2020),
    "roads": (2016, 2021),
}


def _project(tmp_path, name="issue-40"):
    """Project with static and temporal layers whose files exist on disk."""
    p = Project(project_name=name)
    for var_name, years in _LAYERS.items():
        for year in years or (None,):
            key = f"{var_name}_{year}" if year else var_name
            path = tmp_path / f"{key}.tif"
            path.touch()
            var = LocalRasterVar(
                name=var_name,
                year=year,
                path=path,
                raster_type=RasterType.continuous,
            )
            var.project = p
            p.processed_variables[key] = var
    return p


# --- the choices -------------------------------------------------------------


def test_temporal_variables_are_offered_once_per_year(tmp_path):
    """A static layer is one choice; a temporal one is one choice per year."""
    choices = variable_choices(_project(tmp_path))
    texts = [c["text"] for c in choices]
    assert "altitude" in texts
    assert {"defor (2015)", "defor (2020)", "roads (2016)", "roads (2021)"} <= set(
        texts
    )
    assert "defor" not in texts
    by_text = {c["text"]: c for c in choices}
    assert (by_text["roads (2021)"]["name"], by_text["roads (2021)"]["year"]) == (
        "roads",
        2021,
    )
    assert by_text["altitude"]["value"] == choice_value("altitude", None)


def test_missing_or_removed_target_is_refused(tmp_path):
    """No target, or one no longer in the project, reads as none selected."""
    choices = variable_choices(_project(tmp_path))
    feats = ["altitude"]
    assert (
        dataset_form_error(choices, "", feats) == "tiles.dataset.error_target_required"
    )
    assert (
        dataset_form_error(choices, "gone@2015", feats)
        == "tiles.dataset.error_target_required"
    )


def test_no_features_is_refused(tmp_path):
    """A target alone cannot be registered."""
    choices = variable_choices(_project(tmp_path))
    assert (
        dataset_form_error(choices, "defor@2015", [])
        == "tiles.dataset.error_features_required"
    )


def test_a_complete_selection_passes(tmp_path):
    """A target and a feature, each at its own year, pass."""
    choices = variable_choices(_project(tmp_path))
    assert dataset_form_error(choices, "defor@2015", ["roads@2021"]) is None


# --- the Dataset builder: one year per temporal variable ---------------------


def test_each_temporal_feature_uses_its_own_year(tmp_path):
    """Features with no year in common, which one shared year could not do."""
    p = _project(tmp_path)
    ds = Dataset(project=p, name="d")
    ds.set_target("defor", year=2020)
    ds.set_features(
        ["forest_gfc", "roads", "altitude"], years={"forest_gfc": 2015, "roads": 2016}
    )
    ds.validate()
    assert ds.target.year == 2020
    assert [(f.name, f.year) for f in ds.features] == [
        ("forest_gfc", 2015),
        ("roads", 2016),
        ("altitude", None),
    ]
    assert ds.year == 2020  # the target's year stays the dataset's period


def test_a_temporal_feature_without_its_year_falls_back_to_the_dataset_year(tmp_path):
    """The notebook API is unchanged: set_target's year covers the features."""
    p = _project(tmp_path)
    ds = Dataset(project=p, name="d")
    ds.set_target("defor", year=2015)
    ds.set_features(["forest_gfc"])
    assert ds.features[0].year == 2015


def test_a_shared_feature_year_becomes_the_dataset_year(tmp_path):
    """Static target: the one year its features share is the period, as before."""
    p = _project(tmp_path)
    ds = Dataset(project=p, name="d")
    ds.set_target("forest_loss")
    ds.set_features(["forest_gfc", "defor"], years={"forest_gfc": 2020, "defor": 2020})
    assert ds.year == 2020


def test_a_temporal_feature_with_no_year_at_all_is_refused(tmp_path):
    """No entry in years and no dataset year: the feature is named."""
    p = _project(tmp_path)
    ds = Dataset(project=p, name="d")
    ds.set_target("forest_loss")
    with pytest.raises(ValueError, match="roads"):
        ds.set_features(["roads"])


def test_a_year_the_feature_lacks_is_refused(tmp_path):
    """A year the variable has no layer for is refused."""
    p = _project(tmp_path)
    ds = Dataset(project=p, name="d")
    ds.set_target("forest_loss")
    with pytest.raises(ValueError, match="roads 2015"):
        ds.set_features(["roads"], years={"roads": 2015})


def test_a_year_for_a_static_feature_is_refused(tmp_path):
    """A static feature takes no year."""
    p = _project(tmp_path)
    ds = Dataset(project=p, name="d")
    ds.set_target("forest_loss")
    with pytest.raises(ValueError, match="altitude"):
        ds.set_features(["altitude"], years={"altitude": 2015})


def test_feature_years_survive_save_and_load(tmp_path, monkeypatch):
    """Each feature's year is persisted, not re-derived from the dataset year."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    p = _project(tmp_path, name="ds_years")
    ds = Dataset(project=p, name="calib")
    ds.set_target("defor", year=2020)
    ds.set_features(["forest_gfc", "roads"], years={"forest_gfc": 2015, "roads": 2021})
    p.add_dataset(ds, key="calib", auto_save=True)

    restored = Project.load("ds_years").get_dataset("calib")
    assert (restored.target.name, restored.target.year) == ("defor", 2020)
    assert [(f.name, f.year) for f in restored.features] == [
        ("forest_gfc", 2015),
        ("roads", 2021),
    ]


# --- the dialog --------------------------------------------------------------


def _find(widget, cls, out=None):
    """Every widget of type ``cls`` under ``widget``."""
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _select(box, label_key):
    """The select labelled with the translation of ``label_key``."""
    label = t(label_key)
    return next(s for s in _find(box, vw.Select) if s.label == label)


def _click(box, label):
    """Click the button whose text contains ``label``."""
    next(b for b in _find(box, vw.Btn) if label in str(b.children)).click()


def _alert_text(widget):
    """Concatenated text of every alert under ``widget``."""
    return " ".join(
        str(c)
        for a in _find(widget, vw.Alert)
        for c in (a.children or [])
        if isinstance(c, str)
    )


def _dialog(box):
    """The dataset form's dialog widget."""
    title = t("tiles.dataset.dialog_title_new")
    return next(
        d
        for d in _find(box, vw.Dialog)
        if any(h.children == [title] for h in _find(d, vw.Html))
    )


def _fill(box, target, features):
    """Pick the target and the features (select values)."""
    _select(box, "tiles.dataset.target_variable_label").v_model = target
    _select(box, "tiles.dataset.feature_variables_label").v_model = features


def _render_dialog(p, **kwargs):
    """Render an open DatasetFormDialog; returns (box, rc, submitted, open_)."""
    submitted = []
    open_ = solara.reactive(True)
    box, rc = reacton.render(
        DatasetFormDialog(
            project=solara.reactive(p),
            open_=open_,
            on_submit=lambda entry, key: submitted.append((entry, key)),
            **kwargs,
        ),
        handle_error=False,
    )
    return box, rc, submitted, open_


def test_dialog_has_no_year_select(tmp_path):
    """The year comes with the variable: target and features are the selects."""
    box, rc, _, _ = _render_dialog(_project(tmp_path))
    try:
        labels = {s.label for s in _find(box, vw.Select)}
        assert labels == {
            t("tiles.dataset.target_variable_label"),
            t("tiles.dataset.feature_variables_label"),
        }
    finally:
        rc.close()


def test_dialog_submits_each_variable_with_its_year(tmp_path):
    """Different years per variable reach the tile as target_year/feature_years."""
    box, rc, submitted, open_ = _render_dialog(_project(tmp_path))
    try:
        _fill(box, "defor@2020", ["forest_gfc@2015", "roads@2021", "altitude"])
        _click(box, t("tiles.dataset.register_button"))
        assert open_.value is False
        ((entry, _key),) = submitted
        assert (entry["target"], entry["target_year"]) == ("defor", 2020)
        assert entry["features"] == ["forest_gfc", "roads", "altitude"]
        assert entry["feature_years"] == {"forest_gfc": 2015, "roads": 2021}
    finally:
        rc.close()


def test_dialog_greys_out_other_years_of_a_picked_feature(tmp_path):
    """One year per variable: the formula names each feature once."""
    box, rc, _, _ = _render_dialog(_project(tmp_path))
    try:
        features = _select(box, "tiles.dataset.feature_variables_label")
        features.v_model = ["roads@2016"]
        disabled = {i["value"]: i["disabled"] for i in features.items}
        assert disabled["roads@2016"] is False
        assert disabled["roads@2021"] is True
        assert disabled["defor@2015"] is False
    finally:
        rc.close()


def test_dialog_keeps_the_modal_open_without_features(tmp_path):
    """A form error stays in the modal and nothing launches."""
    box, rc, submitted, open_ = _render_dialog(_project(tmp_path))
    try:
        _fill(box, "defor@2015", [])
        _click(box, t("tiles.dataset.register_button"))
        assert not submitted
        assert open_.value is True
        assert _alert_text(box) == t("tiles.dataset.error_features_required")
    finally:
        rc.close()


def test_dialog_never_submits_the_target_as_a_feature(tmp_path):
    """A feature whose variable is later picked as the target leaves the entry."""
    box, rc, submitted, _ = _render_dialog(_project(tmp_path))
    try:
        _select(box, "tiles.dataset.feature_variables_label").v_model = [
            "altitude",
            "defor@2015",
        ]
        _select(box, "tiles.dataset.target_variable_label").v_model = "defor@2020"
        _click(box, t("tiles.dataset.register_button"))
        ((entry, _key),) = submitted
        assert (entry["target"], entry["features"]) == ("defor", ["altitude"])
    finally:
        rc.close()


def test_editing_skips_features_removed_from_the_project(tmp_path):
    """A removed feature has no chip to delete, so it must not block the save."""
    initial = {
        "name": "ds1",
        "target": "forest_loss",
        "features": ["altitude", "deleted_layer", "roads@2019"],
    }
    box, rc, submitted, open_ = _render_dialog(
        _project(tmp_path), editing_key="ds1", initial=initial
    )
    try:
        _click(box, t("common.save"))
        assert open_.value is False
        ((entry, key),) = submitted
        assert key == "ds1"
        assert entry["features"] == ["altitude"]
    finally:
        rc.close()


# --- the tile ----------------------------------------------------------------


class _Notifier:
    """Records error toasts (the real bus replaces errors, so spy instead)."""

    def __init__(self):
        self.errors = []

    def error(self, message, **kwargs):
        self.errors.append(message)


@pytest.fixture
def tile(tmp_path, monkeypatch):
    """A rendered DatasetTile; returns (box, project, notifier, open_new)."""
    monkeypatch.setattr(Project, "save", lambda self, filename=None: None)
    notifier = _Notifier()
    monkeypatch.setattr(dataset_tile, "use_notifications", lambda: notifier)
    captured = {}

    @solara.component
    def _StubList(**kwargs):
        captured.update(kwargs)
        solara.Text("list")

    monkeypatch.setattr(dataset_tile, "DatasetList", _StubList)
    project = solara.reactive(_project(tmp_path))
    box, rc = reacton.render(
        dataset_tile.DatasetTile(project=project), handle_error=False
    )
    _click(box, t("tiles.dataset.new_button"))
    yield box, project, notifier, captured
    rc.close()


def test_tile_registers_variables_with_different_years(tile):
    """The dataset is built with each variable at the year picked with it."""
    box, project, notifier, _ = tile
    _fill(box, "defor@2015", ["forest_gfc@2020", "roads@2016"])
    _click(box, t("tiles.dataset.register_button"))

    assert _dialog(box).v_model is False
    assert not _alert_text(box)
    assert not notifier.errors
    ds = project.value.datasets["dataset_1"]
    assert ds.target.year == 2015
    assert [(f.name, f.year) for f in ds.features] == [
        ("forest_gfc", 2020),
        ("roads", 2016),
    ]


def test_tile_edit_prefills_each_variable_with_its_year(tile):
    """Edit reopens with the saved (variable, year) picks selected."""
    box, project, _, captured = tile
    _fill(box, "defor@2015", ["roads@2021"])
    _click(box, t("tiles.dataset.register_button"))

    captured["on_edit"]("dataset_1")
    assert _select(box, "tiles.dataset.target_variable_label").v_model == "defor@2015"
    assert _select(box, "tiles.dataset.feature_variables_label").v_model == [
        "roads@2021"
    ]


def _add_single_year_layer(project, tmp_path, name="cover", year=2010):
    """Register a layer that exists for one year only (``geobosque_2010``)."""
    p = project.value
    path = tmp_path / f"{name}_{year}.tif"
    path.touch()
    var = LocalRasterVar(
        name=name, year=year, path=path, raster_type=RasterType.continuous
    )
    var.project = p
    p.processed_variables[f"{name}_{year}"] = var
    project.set(p.model_copy())


def test_tile_edit_prefills_a_single_year_feature(tile, tmp_path):
    """A one-year layer is offered bare, so edit must select it bare too.

    Its instance still carries the year, but with one year it is not temporal:
    the select offers ``cover``, not ``cover@2010``, and a prefilled value with
    no item renders no chip -- the dataset reopened with fewer features than it
    was saved with, and saving it dropped them.
    """
    box, project, _, captured = tile
    _add_single_year_layer(project, tmp_path)
    _fill(box, "forest_loss", ["cover", "altitude"])
    _click(box, t("tiles.dataset.register_button"))

    captured["on_edit"]("dataset_1")
    assert _select(box, "tiles.dataset.feature_variables_label").v_model == [
        "cover",
        "altitude",
    ]


def test_tile_edit_prefills_a_single_year_target(tile, tmp_path):
    """Same for the target: a one-year target reopens selected."""
    box, project, _, captured = tile
    _add_single_year_layer(project, tmp_path)
    _fill(box, "cover", ["altitude"])
    _click(box, t("tiles.dataset.register_button"))

    captured["on_edit"]("dataset_1")
    assert _select(box, "tiles.dataset.target_variable_label").v_model == "cover"


def test_tile_reports_a_builder_failure_as_a_toast(tile, tmp_path):
    """A layer gone from disk passes the dialog; the failure is a toast."""
    box, project, notifier, _ = tile
    (tmp_path / "altitude.tif").unlink()
    _fill(box, "forest_loss", ["altitude"])
    _click(box, t("tiles.dataset.register_button"))

    assert _dialog(box).v_model is False
    assert not _alert_text(box)
    assert len(notifier.errors) == 1
    assert not project.value.datasets
