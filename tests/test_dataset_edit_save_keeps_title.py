"""Saving an edited dataset must not flash the "New dataset" dialog.

``DatasetTile.on_submit`` used to clear ``editing_key``/``initial`` before the
CreationDialog frame had closed itself, so the still-open dialog re-rendered
with the create title and then faded out as "New dataset". The tile now leaves
the edit state alone on submit — ``on_new``/``on_edit`` set it on every open —
so the dialog keeps its edit title through the close.
"""

import types

import ipyvuetify as vw
import reacton
import solara

from gui.tile import dataset_tile
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

Project._ensure_model_schemas()

EDIT_TITLE = "Edit dataset 'dataset_1'"
NEW_TITLE = "New dataset"


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _raster_var(project, name):
    return LocalRasterVar.model_construct(
        name=name,
        year=None,
        path=f"/tmp/{name}.tif",
        project=project,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        active=True,
    )


class _StubDataset:
    """Stands in for spatialrisk.dataset.Dataset: no disk, no spatial checks."""

    def __init__(self, project, name=None, year=None):
        self.project, self.name, self.year = project, name, year
        self.target, self.features = None, []

    def set_target(self, name, year=None):
        self.target = types.SimpleNamespace(name=name)

    def set_features(self, names):
        self.features = [types.SimpleNamespace(name=n) for n in names]

    def validate(self):
        return True


def _project():
    p = Project(project_name="edit-title")
    for name in ("altitude", "forest_loss"):
        p.processed_variables[name] = _raster_var(p, name)
    p.datasets["dataset_1"] = types.SimpleNamespace(
        name="dataset_1",
        target=types.SimpleNamespace(name="forest_loss"),
        features=[types.SimpleNamespace(name="altitude")],
        year=None,
    )
    return p


def _title_widget(box):
    return next(
        h for h in _find(box, vw.Html) if h.children and h.children[0] == EDIT_TITLE
    )


def _save_button(box):
    return next(b for b in _find(box, vw.Btn) if "Save" in str(b.children))


def test_edit_save_never_shows_new_dataset_title(monkeypatch):
    """Save on an edit closes the dialog with its edit title, never the create one."""
    monkeypatch.setattr(dataset_tile, "Dataset", _StubDataset)
    monkeypatch.setattr(Project, "save", lambda self, filename=None: None)
    captured = {}

    @solara.component
    def _StubList(**kwargs):
        captured.update(kwargs)
        solara.Text("list")

    monkeypatch.setattr(dataset_tile, "DatasetList", _StubList)

    project = solara.reactive(_project())
    box, rc = reacton.render(
        dataset_tile.DatasetTile(project=project), handle_error=False
    )
    captured["on_edit"]("dataset_1")

    title = _title_widget(box)
    dialog = next(d for d in _find(box, vw.Dialog) if title in _find(d, vw.Html))
    assert dialog.v_model is True

    seen = []

    def snapshot(*_):
        seen.append((dialog.v_model, tuple(title.children)))

    dialog.observe(snapshot, "v_model")
    title.observe(snapshot, "children")

    _save_button(box).click()

    assert dialog.v_model is False
    assert (True, (NEW_TITLE,)) not in seen, seen
    assert title.children == [EDIT_TITLE], seen
    assert project.value.datasets["dataset_1"].features[0].name == "altitude"
    rc.close()
