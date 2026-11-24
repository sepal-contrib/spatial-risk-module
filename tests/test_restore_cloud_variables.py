"""Not-downloaded Earth Engine layers come back after a save and a reload."""

import ee
import pytest

import gui.scripts.predefined_variables as pv
import spatialrisk.project as project_module
from gui.tile.variables_tile import restore_cloud_variables
from spatialrisk import Project
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.models import DataType, RasterType


class _Reactive:
    value = object()


class _FakeState:
    aoi_result = _Reactive()


@pytest.fixture
def seen(tmp_path, monkeypatch):
    """Projects in tmp_path, a fake AOI, and a recording forest_gfc builder."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    calls = {}

    def _fake_get_image(aoi, year=None, **params):
        calls.update(year=year, params=params)
        return "IMAGE"

    monkeypatch.setitem(
        pv.PREDEFINED_CATALOGUE,
        "forest_gfc",
        {**pv.PREDEFINED_CATALOGUE["forest_gfc"], "get_image": _fake_get_image},
    )
    # An uninitialised ee.Geometry passes Variable.aoi's isinstance check.
    fake_aoi = ee.Geometry.__new__(ee.Geometry)
    monkeypatch.setattr(pv, "resolve_aoi_ee", lambda _result: fake_aoi)
    monkeypatch.setattr("gui.store.state_manager.app_state", _FakeState)
    return calls


def _save_and_reload(*variables) -> Project:
    Project._ensure_model_schemas()
    project = Project(project_name="cloud")
    for key, var in variables:
        var.project = project
        project.raw_variables[key] = var
    project.save()
    return Project.load("cloud")


def test_a_catalogue_layer_is_rebuilt_with_its_params_year_and_style(seen):
    """forest_gfc_tc45 for 2020 comes back as a GEEVar with its own style."""
    loaded = _save_and_reload(
        (
            "forest_gfc_tc45_2020",
            GEEVar(
                name="forest_gfc_tc45",
                year=2020,
                data_type=DataType.raster,
                raster_type=RasterType.categorical,
                params={"tree_cover_threshold": 45},
                vis_params={"palette": ["000000", "00ff00"], "min": 0, "max": 1},
                gee_images=["IMAGE"],
            ),
        )
    )

    assert restore_cloud_variables(loaded) == []

    var = loaded.raw_variables["forest_gfc_tc45_2020"]
    assert isinstance(var, GEEVar) and var.gee_images == ["IMAGE"]
    assert seen == {"year": 2020, "params": {"tree_cover_threshold": 45}}
    assert var.vis_params["palette"] == ["000000", "00ff00"]
    assert var.project is loaded
    assert loaded.cloud_variables == {}


def test_a_custom_asset_is_rebuilt_from_its_asset_id(seen):
    """A custom asset layer keeps its asset id and band."""
    loaded = _save_and_reload(
        (
            "roads",
            GEEVar(
                name="roads",
                path="projects/x/assets/roads",
                band="b1",
                data_type=DataType.raster,
                raster_type=RasterType.continuous,
            ),
        )
    )

    assert restore_cloud_variables(loaded) == []
    var = loaded.raw_variables["roads"]
    assert (var.path, var.band) == ("projects/x/assets/roads", "b1")


def test_a_layer_that_cannot_be_rebuilt_is_kept_for_the_next_save(seen, monkeypatch):
    """Without an AOI the layer is reported, not listed, and still saved."""
    loaded = _save_and_reload(
        (
            "forest_gfc_tc45_2020",
            GEEVar(
                name="forest_gfc_tc45",
                year=2020,
                data_type=DataType.raster,
                params={"tree_cover_threshold": 45},
                gee_images=["IMAGE"],
            ),
        )
    )

    def _no_aoi(_result):
        raise ValueError("No AOI selected")

    monkeypatch.setattr(pv, "resolve_aoi_ee", _no_aoi)

    assert restore_cloud_variables(loaded) == ["forest_gfc_tc45_2020"]
    assert "forest_gfc_tc45_2020" not in loaded.raw_variables
    loaded.save()
    assert set(Project.load("cloud").cloud_variables) == {"forest_gfc_tc45_2020"}
