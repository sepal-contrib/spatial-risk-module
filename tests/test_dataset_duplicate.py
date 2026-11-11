"""Duplicate a dataset: a pre-filled *New* dataset dialog, not a silent clone.

The point is to make a near-copy -- typically the same layers minus a
categorical that makes iCAR training slow -- so the copy opens in the dialog
with the source's target and features already picked and a free "_copy" name.
"""

import types

import ipyvuetify as vw
import solara
from _notification_host import render_under_notifications

from gui.tile import dataset_tile
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

Project._ensure_model_schemas()


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _texts(widget, out=None):
    out = [] if out is None else out
    for child in getattr(widget, "children", []) or []:
        if isinstance(child, str):
            out.append(child)
        else:
            _texts(child, out)
    return out


def _raster_var(project, name, raster_type=RasterType.continuous):
    return LocalRasterVar.model_construct(
        name=name,
        year=None,
        path=f"/tmp/{name}.tif",
        project=project,
        data_type=DataType.raster,
        raster_type=raster_type,
        active=True,
    )


class _StubDataset:
    """Stands in for spatialrisk.dataset.Dataset: no disk, no spatial checks."""

    def __init__(self, project, name=None, year=None):
        self.project, self.name, self.year = project, name, year
        self.target, self.features = None, []

    def set_target(self, name, year=None):
        self.target = types.SimpleNamespace(name=name)

    def set_features(self, names, years=None):
        self.features = [types.SimpleNamespace(name=n) for n in names]

    def validate(self):
        return True


def _project():
    p = Project(project_name="dup")
    p.processed_variables["altitude"] = _raster_var(p, "altitude")
    p.processed_variables["forest_loss"] = _raster_var(p, "forest_loss")
    p.processed_variables["subj"] = _raster_var(p, "subj", RasterType.categorical)
    p.datasets["dataset_1"] = types.SimpleNamespace(
        name="dataset_1",
        target=types.SimpleNamespace(name="forest_loss"),
        features=[
            types.SimpleNamespace(name="altitude"),
            types.SimpleNamespace(name="subj"),
        ],
        year=None,
    )
    return p


def _render(monkeypatch):
    monkeypatch.setattr(dataset_tile, "Dataset", _StubDataset)
    monkeypatch.setattr(Project, "save", lambda self, filename=None: None)
    project = solara.reactive(_project())
    box, rc = render_under_notifications(
        lambda: dataset_tile.DatasetTile(project=project), handle_error=False
    )
    return project, box, rc


def _duplicate_buttons(box):
    return [b for b in _find(box, vw.Btn) if "mdi-content-copy" in _texts(b)]


def _text_field(box, label):
    return next(f for f in _find(box, vw.TextField) if f.label == label)


def _select(box, label):
    return next(s for s in _find(box, vw.Select) if s.label == label)


def test_each_dataset_row_offers_a_duplicate_action(monkeypatch):
    """The dataset list shows a copy action per row."""
    _project_, box, rc = _render(monkeypatch)
    try:
        assert len(_duplicate_buttons(box)) == 1
    finally:
        rc.close()


def test_duplicate_opens_a_prefilled_new_dataset_dialog(monkeypatch):
    """Duplicate opens New mode with the source's fields and a _copy name."""
    _project_, box, rc = _render(monkeypatch)
    try:
        _duplicate_buttons(box)[0].click()

        assert "Duplicate dataset 'dataset_1'" in _texts(box)
        name = _text_field(box, "Dataset name")
        assert name.v_model == "dataset_1_copy"
        assert not name.disabled  # a new dataset: the name is the user's to change
        assert _select(box, "Target variable").v_model == "forest_loss"
        assert _select(box, "Feature variables").v_model == ["altitude", "subj"]
    finally:
        rc.close()


def test_saving_the_duplicate_adds_a_copy_and_keeps_the_original(monkeypatch):
    """Saving after dropping a layer adds the copy and leaves the source alone."""
    project, box, rc = _render(monkeypatch)
    try:
        _duplicate_buttons(box)[0].click()
        _select(box, "Feature variables").v_model = ["altitude"]  # drop subj

        next(b for b in _find(box, vw.Btn) if "Create dataset" in _texts(b)).click()

        datasets = project.value.datasets
        assert [f.name for f in datasets["dataset_1_copy"].features] == ["altitude"]
        assert datasets["dataset_1_copy"].target.name == "forest_loss"
        assert [f.name for f in datasets["dataset_1"].features] == ["altitude", "subj"]
    finally:
        rc.close()
