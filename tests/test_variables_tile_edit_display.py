"""A display-only edit restyles in place: no rebuild, download or re-harmonize.

Changing only a layer's colours must keep the downloaded file and its
harmonized copy; any other edit keeps the rebuild path that
``test_variables_tile_edit_reharmonizes.py`` pins.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import solara
from _notification_host import render_under_notifications

from gui.scripts import process_actions
from gui.tile import derived_map, variables_tile
from gui.tile.variables_tile import _variable_to_entry, same_layer
from spatialrisk.project import Project
from spatialrisk.variables.gee_var import GEEVar, download_path
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

Project._ensure_model_schemas()

NAME = "forest_gfc_tc30"
YEARS = (2018, 2019, 2020)
RED = {"palette": ["000000", "ff0000"], "min": 0, "max": 1}


@pytest.fixture(autouse=True)
def _clean_map_state():
    """The on-map sets are module-level; leave them as each test found them."""
    yield
    variables_tile.vars_on_map.set(set())
    derived_map.derived_on_map.set(set())


def _project(tmp_path, monkeypatch, years=YEARS):
    """Downloaded catalogue layers, one per year, each with a harmonized copy."""
    monkeypatch.setenv("SPATIAL_RISK_DATA_DIR", str(tmp_path / "data"))
    p = Project(project_name="edit-display")
    for year in years:
        key = f"{NAME}_{year}"
        common = dict(
            name=NAME,
            year=year,
            project=p,
            raster_type=RasterType.categorical,
            params={"tree_cover_threshold": 30},
        )
        p.raw_variables[key] = LocalRasterVar(
            path=download_path(p.folders.data_raw_folder, NAME, year), **common
        )
        p.processed_variables[key] = LocalRasterVar(
            path=tmp_path / f"{key}_matched.tif", **common
        )
    return p


def _modal_entry(year=2020, threshold=30, vis_params=RED):
    """What ``VariableModal._submit_predefined`` hands ``on_save``."""
    return {
        "source": "predefined",
        "type": "GEEVar",
        "name": f"forest_gfc_tc{threshold}",
        "year": year,
        "data_type": DataType.raster,
        "raster_type": RasterType.categorical,
        "predefined_key": "forest_gfc",
        "params": {"tree_cover_threshold": threshold},
        "vis_params": vis_params,
    }


def _capture_on_save(monkeypatch, project, map_=None):
    """Mount the real VariablesTile and return its on_save closure plus context."""
    captured = {}

    @solara.component
    def _StubModal(**kwargs):
        captured.update(kwargs)
        solara.Text("modal")

    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    _, rc = render_under_notifications(
        lambda: variables_tile.VariablesTile(project=project, map_=map_),
        handle_error=False,
    )
    return captured["on_save"], rc


def _save(monkeypatch, p, key, entry, map_=None):
    project = solara.reactive(p, equals=lambda a, b: a is b)
    on_save, rc = _capture_on_save(monkeypatch, project, map_)
    try:
        on_save(key, entry)
    finally:
        rc.close()
    return project


def _forbid_rebuild(monkeypatch):
    """Any rebuild, download or harmonize drop fails the test."""

    def _boom(*_a, **_k):
        raise AssertionError("a display edit must not rebuild")

    monkeypatch.setattr(variables_tile, "_build_variable", _boom)
    monkeypatch.setattr(process_actions, "materialize_raw_layers", _boom)
    monkeypatch.setattr(process_actions, "remove_processed_variable", _boom)


def test_colour_edit_keeps_the_downloaded_file_and_the_harmonized_copy(
    monkeypatch, tmp_path
):
    """The same objects stay registered and both carry the override."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    key = f"{NAME}_2020"
    raw, processed = p.raw_variables[key], p.processed_variables[key]
    _forbid_rebuild(monkeypatch)

    project = _save(monkeypatch, p, key, _modal_entry())

    assert p.raw_variables[key] is raw
    assert p.processed_variables[key] is processed
    assert raw.vis_params == RED and processed.vis_params == RED
    assert project.value is not p  # published
    # Saved: a reopen keeps the new look.
    assert Project.load("edit-display").raw_variables[key].vis_params == RED


def test_unchanged_save_leaves_the_layer_as_it_is(monkeypatch, tmp_path):
    """Save with nothing changed: no rebuild, so the download is not lost."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    key = f"{NAME}_2020"
    raw, processed = p.raw_variables[key], p.processed_variables[key]
    _forbid_rebuild(monkeypatch)

    project = _save(monkeypatch, p, key, _modal_entry(vis_params=None))

    assert p.raw_variables[key] is raw
    assert p.processed_variables[key] is processed
    assert raw.vis_params is None and processed.vis_params is None
    assert project.value is p  # nothing published


def test_reset_clears_the_override_without_a_rebuild(monkeypatch, tmp_path):
    """Reset submits ``vis_params`` None: raw and harmonized follow the default."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    key = f"{NAME}_2020"
    raw, processed = p.raw_variables[key], p.processed_variables[key]
    raw.vis_params = processed.vis_params = RED
    _forbid_rebuild(monkeypatch)

    project = _save(monkeypatch, p, key, _modal_entry(vis_params=None))

    assert p.raw_variables[key] is raw and p.processed_variables[key] is processed
    assert raw.vis_params is None and processed.vis_params is None
    assert project.value is not p  # published


def test_failed_save_still_publishes_the_restyle(monkeypatch, tmp_path):
    """The UI shows the in-memory style even when writing the project fails."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    _forbid_rebuild(monkeypatch)

    def _fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(Project, "save", _fail)

    project = _save(monkeypatch, p, f"{NAME}_2020", _modal_entry())

    assert project.value is not p
    assert project.value.raw_variables[f"{NAME}_2020"].vis_params == RED


def test_display_edit_applies_to_every_year(monkeypatch, tmp_path):
    """Editing one year restyles every year of the layer."""
    p = _project(tmp_path, monkeypatch)
    _forbid_rebuild(monkeypatch)

    _save(monkeypatch, p, f"{NAME}_2019", _modal_entry(year=2019))

    for year in YEARS:
        assert p.raw_variables[f"{NAME}_{year}"].vis_params == RED
        assert p.processed_variables[f"{NAME}_{year}"].vis_params == RED


def test_apply_display_leaves_other_layers_alone(monkeypatch, tmp_path):
    """Only vars sharing the edited name are touched; None resets to the default."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    other = LocalRasterVar(
        name="forest_gfc_tc45",
        year=2020,
        path=tmp_path / "x.tif",
        raster_type=RasterType.categorical,
        vis_params=RED,
    )
    p.raw_variables["forest_gfc_tc45_2020"] = other
    p.raw_variables[f"{NAME}_2020"].vis_params = RED

    touched = process_actions.apply_display(p, f"{NAME}_2020", None)

    assert touched == [f"{NAME}_2020"]
    assert p.raw_variables[f"{NAME}_2020"].vis_params is None
    assert other.vis_params == RED


def test_param_edit_still_rebuilds_and_reharmonizes(monkeypatch, tmp_path):
    """A threshold change is a different layer: today's rebuild path."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    built = []

    def _build(entry, project):
        built.append(entry)
        return GEEVar(
            name=entry["name"],
            year=entry["year"],
            data_type=DataType.raster,
            gee_images=["dummy"],
        )

    monkeypatch.setattr(variables_tile, "_build_variable", _build)

    _save(monkeypatch, p, f"{NAME}_2020", _modal_entry(threshold=45))

    assert len(built) == 1
    assert f"{NAME}_2020" not in p.raw_variables
    assert f"{NAME}_2020" not in p.processed_variables
    assert "forest_gfc_tc45_2020" in p.raw_variables


def test_shown_layer_is_redrawn(monkeypatch, tmp_path):
    """A layer on the map, raw or harmonized, comes back in its new style."""
    p = _project(tmp_path, monkeypatch, years=(2019, 2020))
    removed, raw_toggles, derived_toggles = [], [], []
    map_ = SimpleNamespace(
        remove_layer=lambda key, none_ok=False: removed.append(key),
        gee_interface=None,
    )

    def _toggle(key, *_args):
        raw_toggles.append(key)
        variables_tile.vars_inflight.release(key)

    monkeypatch.setattr(variables_tile, "_toggle_var_on_map", _toggle)
    monkeypatch.setattr(
        variables_tile, "spawn_in_context", lambda fn, args=(): fn(*args)
    )
    monkeypatch.setattr(
        variables_tile,
        "use_derived_map_toggle",
        lambda *a, **k: derived_toggles.append,
    )
    variables_tile.vars_on_map.set({f"{NAME}_2019", f"{NAME}_2020"})
    derived_map.derived_on_map.set({f"{NAME}_2020"})

    _save(monkeypatch, p, f"{NAME}_2020", _modal_entry(), map_=map_)

    assert sorted(raw_toggles) == [f"{NAME}_2019", f"{NAME}_2020"]
    assert derived_toggles == [f"{NAME}_2020"]
    assert "derived_forest_gfc_tc30_2020" in removed
    # Dropped first, so each toggle re-adds instead of removing.
    assert not derived_map.derived_on_map.value


def test_same_layer_reads_layer_identity(monkeypatch, tmp_path):
    """The reopened entry against the modal's own submit shape."""
    p = _project(tmp_path, monkeypatch, years=(2020,))
    old = _variable_to_entry(f"{NAME}_2020", p.raw_variables[f"{NAME}_2020"], p)

    assert same_layer(old, _modal_entry())
    assert same_layer(old, _modal_entry(vis_params=None))  # identical
    assert not same_layer(old, _modal_entry(threshold=45))
    assert not same_layer(old, _modal_entry(year=2019))
    assert not same_layer(old, {**_modal_entry(), "source": "custom"})


def test_same_layer_fills_catalogue_defaults():
    """A bare catalogue name reopens with {} params; the modal sends them all."""
    var = GEEVar(name="rivers", data_type=DataType.raster, gee_images=["dummy"])
    old = _variable_to_entry("rivers", var, None)
    new = {
        "source": "predefined",
        "type": "GEEVar",
        "name": "rivers",
        "year": None,
        "predefined_key": "rivers",
        "params": {"classes": ["large", "major", "canal", "stream"]},
        "vis_params": RED,
    }

    assert same_layer(old, new)


def test_same_layer_custom_layer(tmp_path):
    """Custom: every field but the style, in the modal's types (Path, enums)."""
    var = LocalRasterVar(
        name="my_dem",
        year=2020,
        path=tmp_path / "my_dem.tif",
        raster_type=RasterType.continuous,
    )
    old = _variable_to_entry("my_dem_2020", var, None)
    new = {
        "source": "custom",
        "type": "LocalRasterVar",
        "name": "my_dem",
        "year": 2020,
        "path": Path(tmp_path / "my_dem.tif"),
        "band": None,
        "raster_type": RasterType.continuous,
        "data_type": DataType.raster,
        "vis_params": RED,
    }

    assert same_layer(old, new)
    assert not same_layer(old, {**new, "path": tmp_path / "other.tif"})
    assert not same_layer(old, {**new, "raster_type": RasterType.categorical})
