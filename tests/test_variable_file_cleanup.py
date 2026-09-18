"""Which files a variable owns on disk, and which of them may be deleted.

Removing a variable has always been registry-only: the raster stayed on disk
for good. The Remove dialog now offers to delete the files too, so it needs to
know — before anything is unlinked — exactly what would go, how much space that
frees, and when the answer must be "nothing" (a file the user picked from
outside the project, or one another registered variable still points at).
"""

from pathlib import Path

import pytest

from spatialrisk.project import Project  # noqa: F401 — resolves forward refs
from spatialrisk.variables.file_cleanup import (
    plan_variable_files,
    variable_files,
)
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.local_vector_var import LocalVectorVar
from spatialrisk.variables.models import DataType, RasterizationMethod, RasterType

for _model in (GEEVar, LocalRasterVar, LocalVectorVar):
    _model.model_rebuild()


class _Project:
    """Stands in for Project: the planner only reads these three things."""

    def __init__(self, folder: Path):
        self._folder = folder
        self.raw_variables = {}
        self.processed_variables = {}
        self.base_raster = None

    def _project_dir(self) -> Path:
        return self._folder


def _touch(path: Path, size: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _raster(path: Path, name="slope") -> LocalRasterVar:
    return LocalRasterVar(
        name=name,
        path=path,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
    )


def _vector(path: Path, name="roads") -> LocalVectorVar:
    return LocalVectorVar(
        name=name,
        path=path,
        data_type=DataType.vector,
        rasterization_method=RasterizationMethod.binary,
    )


def test_raster_files_include_the_overview_sidecar(tmp_path):
    """A tiled raster's .ovr pyramid goes with it.

    A leftover .ovr would be served for the next raster written under the same
    name.
    """
    main = _touch(tmp_path / "data_raw" / "slope_2020.tif", 10)
    ovr = _touch(tmp_path / "data_raw" / "slope_2020.tif.ovr", 5)
    _touch(tmp_path / "data_raw" / "slope_2021.tif", 10)  # a different variable

    assert set(variable_files(_raster(main))) == {main, ovr}


def test_vector_files_include_the_shapefile_companions(tmp_path):
    """A .shp alone is unreadable; its companions are part of the same file."""
    main = _touch(tmp_path / "data_raw" / "roads.shp")
    companions = {
        _touch(tmp_path / "data_raw" / f"roads{ext}")
        for ext in (".shx", ".dbf", ".prj", ".cpg")
    }
    _touch(tmp_path / "data_raw" / "roads_buffered.shp")  # not a companion

    assert set(variable_files(_vector(main))) == {main} | companions


def test_a_cloud_variable_owns_no_files(tmp_path):
    """GEEVar.path is an Earth Engine asset id, never a filesystem path."""
    gv = GEEVar(
        name="forest",
        path="projects/foo/assets/bar",
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
    )
    assert variable_files(gv) == []


def test_plan_totals_the_bytes_it_would_free(tmp_path):
    """The dialog can say how much space the tick frees."""
    project = _Project(tmp_path)
    main = _touch(tmp_path / "data_raw" / "slope_2020.tif", 300)
    _touch(tmp_path / "data_raw" / "slope_2020.tif.ovr", 40)
    project.raw_variables["slope_2020"] = _raster(main)

    plan = plan_variable_files(project, "slope_2020")

    assert plan.deletable is True
    assert plan.blocked is None
    assert len(plan.files) == 2
    assert plan.total_bytes == 340


def test_plan_refuses_a_file_outside_the_project_folder(tmp_path):
    """A raster the user picked from their own folder is never ours to delete."""
    project = _Project(tmp_path / "project")
    (tmp_path / "project").mkdir()
    outside = _touch(tmp_path / "downloads" / "my_dem.tif", 10)
    project.raw_variables["my_dem"] = _raster(outside, name="my_dem")

    plan = plan_variable_files(project, "my_dem")

    assert plan.deletable is False
    assert plan.blocked == "outside_project"
    assert plan.files == ()


def test_plan_refuses_a_file_another_variable_still_uses(tmp_path):
    """Two registry entries can share one raster — the survivor keeps it."""
    project = _Project(tmp_path)
    main = _touch(tmp_path / "data_raw" / "forest_2020.tif", 10)
    project.raw_variables["forest_2020"] = _raster(main, name="forest_2020")
    project.processed_variables["forest_2020_matched"] = _raster(
        main, name="forest_2020_matched"
    )

    plan = plan_variable_files(project, "forest_2020")

    assert plan.deletable is False
    assert plan.blocked == "shared"
    assert plan.blocked_by == "forest_2020_matched"


def test_plan_refuses_the_file_the_base_raster_points_at(tmp_path):
    """The reference raster is a variable too, just stored in its own field."""
    project = _Project(tmp_path)
    main = _touch(tmp_path / "data_raw" / "base.tif", 10)
    project.raw_variables["base"] = _raster(main, name="base")
    project.base_raster = _raster(main, name="base_reference")

    plan = plan_variable_files(project, "base")

    assert plan.blocked == "shared"
    assert plan.blocked_by == "base_reference"


def test_plan_is_empty_when_nothing_is_on_disk(tmp_path):
    """A cloud variable has nothing to offer — no checkbox, no warning."""
    project = _Project(tmp_path)
    project.raw_variables["forest"] = GEEVar(
        name="forest",
        path="projects/foo/assets/bar",
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
    )

    plan = plan_variable_files(project, "forest")

    assert plan.files == ()
    assert plan.blocked is None
    assert plan.deletable is False


def test_plan_finds_a_processed_variable_by_key(tmp_path):
    """Derived layers live in their own registry; the planner reads both."""
    project = _Project(tmp_path)
    main = _touch(tmp_path / "data" / "dist_edge.tif", 7)
    project.processed_variables["dist_edge"] = _raster(main, name="dist_edge")

    plan = plan_variable_files(project, "dist_edge")

    assert plan.files == (main,)


def test_plan_of_an_unknown_key_is_empty(tmp_path):
    """An unknown key plans nothing."""
    assert plan_variable_files(_Project(tmp_path), "nope").files == ()


@pytest.mark.parametrize("missing", ["raw", "processed"])
def test_planner_tolerates_a_project_without_a_registry(tmp_path, missing):
    """Fakes in other tests carry only the registry they exercise."""
    project = _Project(tmp_path)
    delattr(project, "raw_variables" if missing == "raw" else "processed_variables")

    assert plan_variable_files(project, "anything").files == ()
