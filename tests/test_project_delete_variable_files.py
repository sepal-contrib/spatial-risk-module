"""Project.delete_variable_files: the destructive half of a variable removal.

Sibling of ``delete_model(delete_files=...)`` and ``delete_prediction``: it
unlinks through ``_safe_unlink`` and acts only on what ``plan_variable_files``
says is safe, so the refusals proved in test_variable_file_cleanup hold here
too — a shared raster or a file outside the project folder survives.
"""

from pathlib import Path

from spatialrisk import project as project_module
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

LocalRasterVar.model_rebuild()


def _touch(path: Path, size: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _raster(path: Path, name: str) -> LocalRasterVar:
    return LocalRasterVar(
        name=name,
        path=path,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
    )


def _project(tmp_path, monkeypatch) -> Project:
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    (tmp_path / "p" / "data_raw").mkdir(parents=True)
    return Project(project_name="p")


def test_deletes_the_raster_and_its_overview(tmp_path, monkeypatch):
    """The raster and its sidecar go together."""
    project = _project(tmp_path, monkeypatch)
    main = _touch(tmp_path / "p" / "data_raw" / "slope_2020.tif")
    ovr = _touch(tmp_path / "p" / "data_raw" / "slope_2020.tif.ovr")
    project.raw_variables["slope_2020"] = _raster(main, "slope")

    removed = project.delete_variable_files("slope_2020")

    assert set(removed) == {main, ovr}
    assert not main.exists() and not ovr.exists()


def test_keeps_a_raster_another_variable_still_uses(tmp_path, monkeypatch):
    """A shared raster survives — the other variable still needs it."""
    project = _project(tmp_path, monkeypatch)
    main = _touch(tmp_path / "p" / "data_raw" / "forest_2020.tif")
    project.raw_variables["forest_2020"] = _raster(main, "forest_2020")
    project.processed_variables["forest_2020_matched"] = _raster(main, "matched")

    assert project.delete_variable_files("forest_2020") == []
    assert main.exists()


def test_keeps_a_file_outside_the_project_folder(tmp_path, monkeypatch):
    """Files the app did not write are never deleted."""
    project = _project(tmp_path, monkeypatch)
    outside = _touch(tmp_path / "downloads" / "my_dem.tif")
    project.raw_variables["my_dem"] = _raster(outside, "my_dem")

    assert project.delete_variable_files("my_dem") == []
    assert outside.exists()


def test_unknown_key_removes_nothing(tmp_path, monkeypatch):
    """An unknown key removes nothing."""
    project = _project(tmp_path, monkeypatch)
    assert project.delete_variable_files("nope") == []


def test_the_registry_entry_is_left_to_the_caller(tmp_path, monkeypatch):
    """Files only: the tiles unregister themselves (base-raster reset, map layer)."""
    project = _project(tmp_path, monkeypatch)
    main = _touch(tmp_path / "p" / "data_raw" / "slope_2020.tif")
    project.raw_variables["slope_2020"] = _raster(main, "slope")

    project.delete_variable_files("slope_2020")

    assert "slope_2020" in project.raw_variables
