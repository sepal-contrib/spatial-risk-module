"""Harmonized and derived layers get the same "also delete the file" offer.

Both lists route through ``process_actions.remove_processed_variable``, so the
file half is proved once here and the two tiles only have to pass the flag on.
"""

from pathlib import Path

import pytest

from gui.scripts import process_actions
from spatialrisk import project as project_module
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

LocalRasterVar.model_rebuild()


@pytest.fixture()
def project(tmp_path, monkeypatch) -> Project:
    """A project rooted in a temporary folder."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    (tmp_path / "p" / "data").mkdir(parents=True)
    return Project(project_name="p")


def _registered(project, key="dist_edge") -> Path:
    path = Path(project._project_dir()) / "data" / f"{key}.tif"
    path.write_bytes(b"\0" * 64)
    project.processed_variables[key] = LocalRasterVar.model_construct(
        name=key,
        path=path,
        project=project,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        active=True,
    )
    return path


def test_the_file_survives_a_plain_removal(project):
    """The default is unchanged: unregister only."""
    path = _registered(project)

    assert process_actions.remove_processed_variable(project, "dist_edge") is True

    assert "dist_edge" not in project.processed_variables
    assert path.exists()


def test_delete_file_removes_the_raster_too(project):
    """The opt-in removes the layer's raster as well."""
    path = _registered(project)

    process_actions.remove_processed_variable(project, "dist_edge", delete_file=True)

    assert "dist_edge" not in project.processed_variables
    assert not path.exists()


def test_a_failed_unlink_still_unregisters_the_layer(project, monkeypatch):
    """A file we cannot remove is no reason to keep a layer the user dropped."""
    _registered(project)
    monkeypatch.setattr(
        Project,
        "delete_variable_files",
        lambda self, key: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    with pytest.raises(OSError):
        process_actions.remove_processed_variable(
            project, "dist_edge", delete_file=True
        )

    assert "dist_edge" not in project.processed_variables


def test_an_unknown_key_deletes_nothing(project):
    """An unknown key is a no-op, files included."""
    path = _registered(project)

    assert (
        process_actions.remove_processed_variable(project, "missing", delete_file=True)
        is False
    )
    assert path.exists()
