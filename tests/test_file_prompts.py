"""The copy both file dialogs put on screen, decided away from any tile.

Three tiles ask the delete question (Variables, Harmonization, Derived layers)
and two buttons ask the overwrite one, so the wording — and the decision of
whether to offer the checkbox at all — lives here and is tested once.
"""

from pathlib import Path

import pytest

from gui.i18n import t
from gui.scripts.file_prompts import delete_prompt, overwrite_prompt
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
    (tmp_path / "p" / "data_raw").mkdir(parents=True)
    return Project(project_name="p")


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


def _geevar(project, name="slope", year=2020) -> GEEVar:
    return GEEVar(
        name=name,
        year=year,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=["img"],
        project=project,
    )


def test_delete_prompt_offers_the_files_and_what_they_cost(project, tmp_path):
    """The checkbox says how many files go and how much that frees."""
    main = _touch(tmp_path / "p" / "data_raw" / "slope_2020.tif", 2 * 1024 * 1024)
    _touch(tmp_path / "p" / "data_raw" / "slope_2020.tif.ovr", 1024)
    project.raw_variables["slope_2020"] = _raster(main, "slope")

    prompt = delete_prompt(project, "slope_2020")

    assert prompt.checkbox_label is not None
    assert "2" in prompt.checkbox_label  # two files
    assert "MB" in prompt.checkbox_label  # and what they free
    assert prompt.note is None
    assert set(prompt.details) == {
        "data_raw/slope_2020.tif",
        "data_raw/slope_2020.tif.ovr",
    }


def test_delete_prompt_uses_the_singular_for_one_file(project, tmp_path):
    """One file reads as one file."""
    main = _touch(tmp_path / "p" / "data_raw" / "dist_edge.tif")
    project.processed_variables["dist_edge"] = _raster(main, "dist_edge")

    label = delete_prompt(project, "dist_edge").checkbox_label

    assert label == t("widgets.file_cleanup.checkbox_one", size="1 B")


def test_delete_prompt_explains_a_file_outside_the_project(project, tmp_path):
    """No offer, and the path it refuses to touch is named."""
    outside = _touch(tmp_path / "downloads" / "my_dem.tif")
    project.raw_variables["my_dem"] = _raster(outside, "my_dem")

    prompt = delete_prompt(project, "my_dem")

    assert prompt.checkbox_label is None
    assert prompt.note is not None
    assert "my_dem.tif" in prompt.note


def test_delete_prompt_names_the_variable_that_keeps_the_file(project, tmp_path):
    """No offer, and the variable still using the file is named."""
    main = _touch(tmp_path / "p" / "data_raw" / "forest_2020.tif")
    project.raw_variables["forest_2020"] = _raster(main, "forest_2020")
    project.processed_variables["forest_2020_matched"] = _raster(main, "matched")

    prompt = delete_prompt(project, "forest_2020")

    assert prompt.checkbox_label is None
    assert "forest_2020_matched" in prompt.note


def test_delete_prompt_says_nothing_when_there_is_no_file(project):
    """A cloud variable leaves the dialog untouched."""
    project.raw_variables["forest"] = _geevar(project, name="forest")

    prompt = delete_prompt(project, "forest")

    assert prompt.checkbox_label is None
    assert prompt.note is None
    assert prompt.details == ()


def test_no_overwrite_prompt_when_no_file_is_in_the_way(project):
    """Nothing on disk means no question to ask."""
    project.raw_variables["slope_2020"] = _geevar(project)

    assert overwrite_prompt(project, None) is None


def test_overwrite_prompt_lists_the_file_with_its_size_and_date(project):
    """Size and date are what tell a stale file from a good one."""
    var = _geevar(project)
    project.raw_variables["slope_2020"] = var
    _touch(var.expected_local_path, 3 * 1024)

    prompt = overwrite_prompt(project, ["slope_2020"])

    assert prompt.keys == ("slope_2020",)
    assert len(prompt.details) == 1
    assert "slope_2020.tif" in prompt.details[0]
    assert "3.0 KB" in prompt.details[0]
    assert "-" in prompt.details[0]  # the ISO date it was written


def test_overwrite_prompt_of_one_layer_reads_in_the_singular(project):
    """A per-row download speaks about that one layer."""
    var = _geevar(project)
    project.raw_variables["slope_2020"] = var
    _touch(var.expected_local_path)

    prompt = overwrite_prompt(project, ["slope_2020"])

    assert prompt.title == t("tiles.variables.confirm_overwrite_title_one")
    assert "slope_2020" in prompt.message


def test_overwrite_prompt_says_how_many_layers_would_still_download(project):
    """Keeping the files still leaves work to do, and says how much."""
    for name in ("slope", "altitude", "roads"):
        project.raw_variables[f"{name}_2020"] = _geevar(project, name=name)
    _touch(project.raw_variables["slope_2020"].expected_local_path)

    prompt = overwrite_prompt(project, None)

    assert prompt.title == t("tiles.variables.confirm_overwrite_title")
    assert "3" in prompt.message  # 1 of the 3 layers is already there
    assert "2" in prompt.note  # keeping it still downloads the other 2


def test_overwrite_prompt_ignores_layers_outside_the_requested_keys(project):
    """A per-row download never reports another row's file."""
    for name in ("slope", "altitude"):
        var = _geevar(project, name=name)
        project.raw_variables[f"{name}_2020"] = var
        _touch(var.expected_local_path)

    prompt = overwrite_prompt(project, ["slope_2020"])

    assert prompt.keys == ("slope_2020",)
