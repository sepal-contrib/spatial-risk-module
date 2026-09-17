"""The Harmonization and Derived-layers dialogs carry the same file checkbox.

The shared helper is proved in test_derived_delete_files; what these add is that
each tile actually renders the offer and forwards the tick — driven through the
real dialog, because a prop that never reaches the widget is invisible to a
source check.
"""

from pathlib import Path

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.scripts import process_actions  # noqa: E402
from gui.tile import postprocess_tile, process_tile  # noqa: E402
from spatialrisk import project as project_module  # noqa: E402
from spatialrisk.harmonization import HarmonizationStatus  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402

Project._ensure_model_schemas()


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """Both tiles read rasters on mount; the file question needs none of it."""
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=[], current=list(p.raw_variables)),
    )
    monkeypatch.setattr(process_actions, "auto_utm_epsg", lambda path: "EPSG:5490")
    monkeypatch.setattr(process_actions, "base_raster_resolution", lambda var: 30.0)


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _remove_button(box):
    hits = [b for b in _find(box, vw.Btn) if b.children == [t("common.remove")]]
    assert hits, "no Remove button"
    return hits[0]


def _project(tmp_path, monkeypatch):
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    data = tmp_path / "proj" / "data"
    data.mkdir(parents=True)
    path = data / "dist_edge_forest_2020.tif"
    path.write_bytes(b"\0" * 4096)

    p = Project(project_name="proj")
    # Step 3 renders nothing at all without a source variable to harmonize.
    p.raw_variables["forest_2020"] = LocalRasterVar.model_construct(
        name="forest_2020",
        path=tmp_path / "proj" / "data_raw" / "forest_2020.tif",
        project=p,
        data_type="raster",
        raster_type="continuous",
        active=True,
    )
    p.processed_variables["dist_edge_forest_2020"] = LocalRasterVar.model_construct(
        name="dist_edge_forest_2020",
        path=path,
        project=p,
        data_type="raster",
        raster_type="continuous",
        active=True,
    )
    return p, path


def _capture_remove(box, captured):
    assert "on_remove" in captured, "the list was never handed an on_remove"
    return captured["on_remove"]


def _mount(monkeypatch, tile, project_reactive, **props):
    captured = {}

    @solara.component
    def _StubList(**kwargs):
        captured.update(kwargs)
        solara.Text("list")

    for module, name in (
        (process_tile, "HarmonizationVariableList"),
        (postprocess_tile, "DerivedVariableList"),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _StubList)
    box, _rc = reacton.render(
        tile(project=project_reactive, **props), handle_error=False
    )
    return box, captured


def _delete_through(box, captured, key):
    _capture_remove(box, captured)(key)
    boxes = _find(box, vw.Checkbox)
    assert boxes, "no 'also delete the file' checkbox in the dialog"
    boxes[0].v_model = True
    _remove_button(box).fire_event("click", {})


def test_harmonization_dialog_deletes_the_output_when_asked(monkeypatch, tmp_path):
    """Step 3's dialog carries the offer and honours it."""
    p, path = _project(tmp_path, monkeypatch)
    project = solara.reactive(p, equals=lambda a, b: a is b)
    box, captured = _mount(
        monkeypatch,
        process_tile.ProcessTile,
        project,
        processing=solara.reactive(False),
    )

    _delete_through(box, captured, "dist_edge_forest_2020")

    assert not path.exists()
    assert "dist_edge_forest_2020" not in project.value.processed_variables


def test_derived_dialog_deletes_the_raster_when_asked(monkeypatch, tmp_path):
    """Step 4's dialog carries the offer and honours it."""
    p, path = _project(tmp_path, monkeypatch)
    project = solara.reactive(p, equals=lambda a, b: a is b)
    box, captured = _mount(monkeypatch, postprocess_tile.PostProcessTile, project)

    _delete_through(box, captured, "dist_edge_forest_2020")

    assert not path.exists()
    assert "dist_edge_forest_2020" not in project.value.processed_variables


def test_derived_dialog_keeps_the_raster_by_default(monkeypatch, tmp_path):
    """An untouched checkbox keeps the file, as before."""
    p, path = _project(tmp_path, monkeypatch)
    project = solara.reactive(p, equals=lambda a, b: a is b)
    box, captured = _mount(monkeypatch, postprocess_tile.PostProcessTile, project)

    _capture_remove(box, captured)("dist_edge_forest_2020")
    _remove_button(box).fire_event("click", {})

    assert path.exists()
    assert "dist_edge_forest_2020" not in project.value.processed_variables


def test_the_listed_path_is_shown_relative_to_the_project(monkeypatch, tmp_path):
    """The dialog names the file, so "delete" is never a blind tick."""
    p, _path = _project(tmp_path, monkeypatch)
    project = solara.reactive(p, equals=lambda a, b: a is b)
    box, captured = _mount(monkeypatch, postprocess_tile.PostProcessTile, project)

    _capture_remove(box, captured)("dist_edge_forest_2020")

    rendered = " ".join(
        "".join(str(c) for c in (w.children or [])) for w in _find(box, vw.Html)
    )
    assert str(Path("data") / "dist_edge_forest_2020.tif") in rendered
