"""A download that would land on an existing file is now a decision, not a skip.

``GEEVar._download`` has always defaulted to ``overwrite=False`` and *silently*
reused whatever was on disk (a bare ``print`` the GUI never shows), so a
half-written or stale raster could never be replaced from the app. The Variables
tile now asks instead, which needs two things this module pins: the path a
download will write, knowable before starting it, and an ``overwrite`` flag that
travels from the tile down to the export.
"""

from pathlib import Path

import pytest

from gui.scripts import process_actions
from spatialrisk import project as project_module
from spatialrisk.project import Project
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

GEEVar.model_rebuild()
LocalRasterVar.model_rebuild()


@pytest.fixture()
def project(tmp_path, monkeypatch) -> Project:
    """A project rooted in a temporary folder."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    return Project(project_name="p")


def _geevar(project, name="slope", year=2020, data_type=DataType.raster) -> GEEVar:
    return GEEVar(
        name=name,
        year=year,
        data_type=data_type,
        raster_type=RasterType.continuous,
        gee_images=["img"],
        project=project,
    )


def test_expected_path_is_named_after_the_variable_and_year(project):
    """The raster lands under '<name>_<year>.tif' in data_raw."""
    var = _geevar(project)
    assert var.expected_local_path == project.folders.data_raw_folder / "slope_2020.tif"


def test_expected_path_of_a_vector_is_a_shapefile(project):
    """A vector layer lands as a shapefile."""
    var = _geevar(project, name="roads", year=None, data_type=DataType.vector)
    assert var.expected_local_path == project.folders.data_raw_folder / "roads.shp"


def test_expected_path_is_where_the_download_actually_writes(project, monkeypatch):
    """The tile's conflict check and the export must agree by construction."""
    var = _geevar(project)
    written = []

    def _fake_export(image, path, **kw):
        written.append(Path(path))
        Path(path).write_bytes(b"raster")  # _download verifies the file landed

    monkeypatch.setattr("spatialrisk.variables.gee_var.download_ee_image", _fake_export)

    var._download()

    assert written == [var.expected_local_path]


def test_an_existing_file_is_skipped_without_overwrite(project, monkeypatch):
    """The default still reuses what is on disk — now a deliberate choice."""
    var = _geevar(project)
    var.expected_local_path.parent.mkdir(parents=True, exist_ok=True)
    var.expected_local_path.write_bytes(b"old")
    calls = []
    monkeypatch.setattr(
        "spatialrisk.variables.gee_var.download_ee_image",
        lambda image, path, **kw: calls.append(path),
    )

    var._download()

    assert calls == []
    assert var.expected_local_path.read_bytes() == b"old"


def test_overwrite_re_downloads_over_an_existing_file(project, monkeypatch):
    """overwrite=True replaces the file that was there."""
    var = _geevar(project)
    var.expected_local_path.parent.mkdir(parents=True, exist_ok=True)
    var.expected_local_path.write_bytes(b"old")
    monkeypatch.setattr(
        "spatialrisk.variables.gee_var.download_ee_image",
        lambda image, path, **kw: Path(path).write_bytes(b"new"),
    )

    var._download(overwrite=True)

    assert var.expected_local_path.read_bytes() == b"new"


def test_materialize_passes_overwrite_down_to_the_export(project, monkeypatch):
    """The tile's "Re-download" choice has to survive the whole call chain."""
    project.raw_variables["slope_2020"] = _geevar(project)
    seen = {}

    def _fake_to_local_raster(self, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here — only the kwargs matter")

    monkeypatch.setattr(GEEVar, "to_local_raster", _fake_to_local_raster)

    with pytest.raises(RuntimeError):
        process_actions.materialize_raw_layers(project, overwrite=True)

    assert seen["overwrite"] is True


def test_materialize_keeps_existing_files_by_default(project, monkeypatch):
    """Without a choice, the export reuses what is on disk."""
    project.raw_variables["slope_2020"] = _geevar(project)
    seen = {}

    def _fake_to_local_raster(self, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here — only the kwargs matter")

    monkeypatch.setattr(GEEVar, "to_local_raster", _fake_to_local_raster)

    with pytest.raises(RuntimeError):
        process_actions.materialize_raw_layers(project)

    assert seen["overwrite"] is False


def test_existing_download_targets_reports_only_files_already_on_disk(project):
    """Only layers whose file exists are reported as conflicts."""
    project.raw_variables["slope_2020"] = _geevar(project)
    project.raw_variables["roads"] = _geevar(
        project, name="roads", year=None, data_type=DataType.vector
    )
    on_disk = project.raw_variables["slope_2020"].expected_local_path
    on_disk.parent.mkdir(parents=True, exist_ok=True)
    on_disk.write_bytes(b"x")

    assert process_actions.existing_download_targets(project) == [
        ("slope_2020", on_disk)
    ]


def test_existing_download_targets_honours_the_key_filter(project):
    """A per-row download only asks about its own layer."""
    project.raw_variables["slope_2020"] = _geevar(project)
    path = project.raw_variables["slope_2020"].expected_local_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    assert process_actions.existing_download_targets(project, ["other"]) == []


def test_existing_download_targets_ignores_local_variables(project, tmp_path):
    """A downloaded variable is no longer a GEEVar — it is not a conflict."""
    path = tmp_path / "p" / "data_raw" / "done.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    project.raw_variables["done"] = LocalRasterVar(
        name="done",
        path=path,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
    )

    assert process_actions.existing_download_targets(project) == []
