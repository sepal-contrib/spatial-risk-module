"""A harmonized layer shares its storage key with the raw layer it came from.

``add_as_processed`` stores the output of raw ``forest_tmf_2016`` under the
same key, ``forest_tmf_2016``. The file planner looked the key up raw-first,
so "also delete the file" in the Harmonization step deleted the *raw source*
and left the harmonized raster on disk. In the Variables step, once the raw
entry was gone, the still-open dialog briefly offered the harmonized raster.
Each tile now asks about the registry it removes from.

Removals are also saved straight away, like dataset, model, sample and
prediction removals: a file deleted from disk while the project file still
listed its layer came back on the next open pointing at nothing.
"""

import json

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.scripts import process_actions  # noqa: E402
from gui.tile import postprocess_tile, process_tile, variables_tile  # noqa: E402
from spatialrisk import project as project_module  # noqa: E402
from spatialrisk.harmonization import HarmonizationStatus  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.file_cleanup import plan_variable_files  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402
from spatialrisk.variables.models import DataType, RasterType  # noqa: E402

Project._ensure_model_schemas()

KEY = "forest_tmf_2016"


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """The tiles read rasters on mount; the file question needs none of it."""
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


def _texts(widget, out=None):
    out = [] if out is None else out
    for child in getattr(widget, "children", []) or []:
        if isinstance(child, str):
            out.append(child)
        else:
            _texts(child, out)
    return out


def _raster(project, path):
    return LocalRasterVar.model_construct(
        name="forest_tmf",
        year=2016,
        path=path,
        project=project,
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        active=True,
    )


def _project(tmp_path, monkeypatch):
    """Raw forest_tmf 2016 and its harmonized output, both on disk."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    folder = tmp_path / "proj"
    raw = folder / "data_raw" / "forest_tmf_2016.tif"
    out = folder / "data" / "forest_tmf_reprojected_matched_2016.tif"
    for path in (raw, out):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 1024)
    p = Project(project_name="proj")
    p.raw_variables[KEY] = _raster(p, raw)
    p.processed_variables[KEY] = _raster(p, out)
    return p, raw, out


def _saved(tmp_path, registry):
    data = json.loads((tmp_path / "proj" / "proj_project.json").read_text())
    return set(data.get(registry) or {})


# --- the planner -------------------------------------------------------------


def test_the_planner_reads_the_registry_it_is_asked_about(tmp_path, monkeypatch):
    """Asked about the harmonized output, it plans the output, not the source."""
    p, raw, out = _project(tmp_path, monkeypatch)

    assert plan_variable_files(p, KEY, registry="processed").files == (out,)
    assert plan_variable_files(p, KEY, registry="raw").files == (raw,)


def test_a_raw_layer_already_removed_plans_nothing(tmp_path, monkeypatch):
    """It must not fall through to the harmonized output under the same key."""
    p, _raw, _out = _project(tmp_path, monkeypatch)
    del p.raw_variables[KEY]

    assert plan_variable_files(p, KEY, registry="raw").files == ()


def test_a_file_both_registries_use_under_one_key_is_kept(tmp_path, monkeypatch):
    """Same key is not the same layer: a raster both still use stays."""
    p, raw, _out = _project(tmp_path, monkeypatch)
    p.processed_variables[KEY] = _raster(p, raw)

    plan = plan_variable_files(p, KEY, registry="raw")

    assert plan.files == () and plan.blocked == "shared"


# --- the tiles ---------------------------------------------------------------

_OPEN_CONTEXTS = []


@pytest.fixture(autouse=True)
def _close_mounted_tiles():
    """Close every mounted tile (see test_derived_delete_files_wiring)."""
    yield
    while _OPEN_CONTEXTS:
        _OPEN_CONTEXTS.pop().close()


def _mount(monkeypatch, tile, project_reactive, **props):
    captured = {}

    @solara.component
    def _Stub(**kwargs):
        captured.update(kwargs)
        solara.Text("stub")

    monkeypatch.setattr(process_tile, "HarmonizationVariableList", _Stub)
    monkeypatch.setattr(postprocess_tile, "DerivedVariableList", _Stub)
    monkeypatch.setattr(variables_tile, "SourceVariableList", _Stub)
    monkeypatch.setattr(variables_tile, "VariableModal", _Stub)
    box, rc = reacton.render(
        tile(project=project_reactive, **props), handle_error=False
    )
    _OPEN_CONTEXTS.append(rc)
    return box, captured


def _remove(box, captured, key, delete_file):
    captured["on_remove"](key)
    if delete_file:
        _find(box, vw.Checkbox)[0].v_model = True
    label = t("common.remove")
    next(b for b in _find(box, vw.Btn) if b.children == [label]).fire_event("click", {})


def _harmonization(monkeypatch, project):
    return _mount(
        monkeypatch,
        process_tile.ProcessTile,
        project,
        processing=solara.reactive(False),
    )


def test_harmonization_offers_the_harmonized_raster(tmp_path, monkeypatch):
    """The dialog lists the output's path, not the raw source's."""
    p, _raw, _out = _project(tmp_path, monkeypatch)
    box, captured = _harmonization(monkeypatch, solara.reactive(p))

    captured["on_remove"](KEY)

    shown = " ".join(_texts(box))
    assert "forest_tmf_reprojected_matched_2016.tif" in shown
    assert "data_raw" not in shown


def test_harmonization_deletes_the_output_and_keeps_the_source(tmp_path, monkeypatch):
    """Ticking the box in Step 3 removes the harmonized raster only."""
    p, raw, out = _project(tmp_path, monkeypatch)
    box, captured = _harmonization(monkeypatch, solara.reactive(p))

    _remove(box, captured, KEY, delete_file=True)

    assert not out.exists()
    assert raw.exists()


def test_variables_deletes_the_source_and_keeps_the_output(tmp_path, monkeypatch):
    """Ticking the box in Step 2 removes the raw file only."""
    p, raw, out = _project(tmp_path, monkeypatch)
    box, captured = _mount(
        monkeypatch, variables_tile.VariablesTile, solara.reactive(p)
    )

    _remove(box, captured, KEY, delete_file=True)

    assert not raw.exists()
    assert out.exists()


def test_removing_a_source_layer_is_saved(tmp_path, monkeypatch):
    """Reopening the project must not bring the layer back."""
    p, _raw, _out = _project(tmp_path, monkeypatch)
    box, captured = _mount(
        monkeypatch, variables_tile.VariablesTile, solara.reactive(p)
    )

    _remove(box, captured, KEY, delete_file=False)

    assert KEY not in _saved(tmp_path, "raw_variables")


def test_removing_a_harmonized_layer_is_saved(tmp_path, monkeypatch):
    """Same for an output removed in the Harmonization step."""
    p, _raw, _out = _project(tmp_path, monkeypatch)
    box, captured = _harmonization(monkeypatch, solara.reactive(p))

    _remove(box, captured, KEY, delete_file=False)

    assert KEY not in _saved(tmp_path, "processed_variables")
    assert KEY in _saved(tmp_path, "raw_variables")


def test_removing_a_derived_layer_is_saved(tmp_path, monkeypatch):
    """And for one removed in the Derived layers list."""
    p, _raw, _out = _project(tmp_path, monkeypatch)
    project = solara.reactive(p, equals=lambda a, b: a is b)
    box, captured = _mount(monkeypatch, postprocess_tile.PostProcessTile, project)

    _remove(box, captured, KEY, delete_file=False)

    assert KEY not in _saved(tmp_path, "processed_variables")
