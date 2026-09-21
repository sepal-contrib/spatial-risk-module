"""Editing a source variable must always re-harmonize it.

The freshness check in ``spatialrisk/harmonization.py`` compares mtimes one way
only — an output OLDER than its source is stale — so re-pointing a variable at
a file that is older than the existing output reads as "already harmonized":
the registry key is unchanged (name + year are unchanged), the output still
sits on the base geobox, and the ``raster_type`` still matches. The user is
told every layer is done, presses Run, nothing happens, and the next step
trains on a raster derived from the file they just replaced.

``variables_tile.on_save`` therefore drops the processed entry on every edit.
These tests drive the real ``on_save`` closure out of a mounted VariablesTile.
"""

import numpy as np
import rasterio
import reacton
import solara
from rasterio.transform import from_origin

from gui.tile import variables_tile
from spatialrisk.harmonization import harmonization_status_from_disk
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

Project._ensure_model_schemas()

# Three rasters on one grid, with hand-set mtimes telling the story: the source
# was downloaded in January, harmonized in March, and the file the user picks in
# the edit was uploaded in February — older than the output, newer than nothing.
JANUARY, FEBRUARY, MARCH = 1_000, 2_000, 3_000


def _write(path, shape=(8, 12)):
    """Write a 1-band float32 raster on the shared test grid."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32618",
        transform=from_origin(0, 200, 10, 10),
    ) as dst:
        dst.write(np.zeros(shape, "float32"), 1)
    return path


def _raster_var(project, name, path, year=None):
    """A registered-shaped raw/processed raster variable."""
    return LocalRasterVar.model_construct(
        name=name,
        year=year,
        path=path,
        project=project,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        active=True,
    )


def _harmonized_project(tmp_path):
    """A project whose single layer reads as already harmonized."""
    import os

    base = _write(tmp_path / "base.tif")
    src = _write(tmp_path / "srtm_global.tif")
    out = _write(tmp_path / "altitude_reprojected_matched.tif")
    os.utime(src, (JANUARY, JANUARY))
    os.utime(out, (MARCH, MARCH))

    p = Project(project_name="edit-reharmonize")
    # The base raster is a reprojected copy, registered only as base_raster —
    # leaving it out of raw_variables keeps "altitude" the only candidate, so
    # the status lists below are exactly about the edited layer.
    p.base_raster = _raster_var(p, "base", base)
    p.raw_variables["altitude"] = _raster_var(p, "altitude", src)
    p.processed_variables["altitude"] = _raster_var(p, "altitude", out)
    return p, out


def _capture_on_save(monkeypatch, project):
    """Mount the real VariablesTile and return its on_save closure plus the context."""
    captured = {}

    @solara.component
    def _StubModal(**kwargs):
        # The modal itself needs the GEE stack and a SEPAL client; only the
        # callbacks it is handed matter here.
        captured.update(kwargs)
        solara.Text("modal")

    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    _, rc = reacton.render(
        variables_tile.VariablesTile(project=project), handle_error=False
    )
    return captured["on_save"], rc


def _capture_on_add(monkeypatch, project):
    """Mount the real VariablesTile and return its on_add closure plus the context."""
    captured = {}

    @solara.component
    def _StubModal(**kwargs):
        # The modal itself needs the GEE stack and a SEPAL client; only the
        # callbacks it is handed matter here.
        captured.update(kwargs)
        solara.Text("modal")

    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    _, rc = reacton.render(
        variables_tile.VariablesTile(project=project), handle_error=False
    )
    return captured["on_add"], rc


def _edit_entry(path):
    """The modal entry an edit that only swaps the file produces."""
    return {
        "source": "custom",
        "type": "LocalRasterVar",
        "name": "altitude",
        # The modal emits ``int(year) or None`` — never the raw text field.
        "year": None,
        "path": str(path),
        "data_type": DataType.raster,
        "raster_type": RasterType.continuous,
    }


def test_editing_a_variable_onto_an_older_file_makes_it_pending(monkeypatch, tmp_path):
    """The scenario the mtime check cannot see: a NEWER output, a changed source."""
    p, out = _harmonized_project(tmp_path)
    new_src = _write(tmp_path / "my_dem.tif")
    import os

    os.utime(new_src, (FEBRUARY, FEBRUARY))
    # The trap, pinned: the replacement is older than the harmonized output, so
    # ``is_current``'s mtime condition stays False and cannot help.
    assert out.stat().st_mtime > new_src.stat().st_mtime

    assert harmonization_status_from_disk(p).current == ["altitude"]

    project = solara.reactive(p, equals=lambda a, b: a is b)
    on_save, rc = _capture_on_save(monkeypatch, project)
    try:
        on_save("altitude", _edit_entry(new_src))
    finally:
        rc.close()

    # The edit landed (on_save swallows and toasts its own failures).
    assert p.raw_variables["altitude"].path == new_src
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_editing_a_variable_unregisters_its_harmonized_output(monkeypatch, tmp_path):
    """The mechanism: the processed entry is dropped, so condition one trips."""
    p, _ = _harmonized_project(tmp_path)
    new_src = _write(tmp_path / "my_dem.tif")

    project = solara.reactive(p, equals=lambda a, b: a is b)
    on_save, rc = _capture_on_save(monkeypatch, project)
    try:
        on_save("altitude", _edit_entry(new_src))
    finally:
        rc.close()

    assert "altitude" not in p.processed_variables


def test_renaming_a_variable_drops_the_output_registered_under_the_old_key(
    monkeypatch, tmp_path
):
    """An edit may move the key; the output under the OLD key must go too."""
    p, _ = _harmonized_project(tmp_path)
    new_src = _write(tmp_path / "my_dem.tif")
    entry = {**_edit_entry(new_src), "name": "elevation"}

    project = solara.reactive(p, equals=lambda a, b: a is b)
    on_save, rc = _capture_on_save(monkeypatch, project)
    try:
        on_save("altitude", entry)
    finally:
        rc.close()

    assert "altitude" not in p.raw_variables
    assert "altitude" not in p.processed_variables
    assert p.raw_variables["elevation"].path == new_src


def test_readding_a_removed_variable_onto_an_older_file_makes_it_pending(
    monkeypatch, tmp_path
):
    """The add route's own version of the gap on_save closes for edits."""
    p, out = _harmonized_project(tmp_path)
    assert harmonization_status_from_disk(p).current == ["altitude"]

    # "Remove the source variable": _do_remove never touches
    # processed_variables, so the stale output stays registered exactly like
    # this in production.
    del p.raw_variables["altitude"]

    new_src = _write(tmp_path / "my_dem.tif")
    import os

    os.utime(new_src, (FEBRUARY, FEBRUARY))
    # The trap, pinned: the replacement is older than the harmonized output, so
    # the mtime condition stays False and cannot save it on its own.
    assert out.stat().st_mtime > new_src.stat().st_mtime

    project = solara.reactive(p, equals=lambda a, b: a is b)
    on_add, rc = _capture_on_add(monkeypatch, project)
    try:
        # _edit_entry's shape is exactly what the modal emits for add too.
        on_add(_edit_entry(new_src))
    finally:
        rc.close()

    # The add landed (on_add/_do_add swallow and toast their own failures).
    assert p.raw_variables["altitude"].path == new_src
    assert "altitude" not in p.processed_variables
    assert harmonization_status_from_disk(p).pending == ["altitude"]
