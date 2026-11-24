"""Regression: ``Project.save()`` must not choke on un-downloaded GEEVars.

A GEEVar created from an ``ee.Image`` holds the live image object in
``gee_images``. Those objects are not JSON-serializable, and ``load()`` only
ever reconstructs ``Local*Var`` — GEEVars are session-only. Downloading a
single variable triggers a full ``save()`` that used to serialize the *other*
still-pending GEEVars and crash with::

    PydanticSerializationError: Unable to serialize unknown type:
    <class 'ee.image.Image'>

save() keeps a GEEVar as its recipe (``cloud_variables``) without the image,
so a layer added but not downloaded survives a reload; the GUI rebuilds it.
"""

from pathlib import Path

import spatialrisk.project as project_module
from spatialrisk import Project
from spatialrisk.variables import LocalRasterVar
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.models import DataType, RasterType


class _Unserializable:
    """Stand-in for an ``ee.Image``: pydantic cannot dump it in json mode."""


def _project_with_mixed_raw(name: str) -> Project:
    Project._ensure_model_schemas()
    project = Project(project_name=name)

    # A downloaded, local raster var — this one must persist.
    local = LocalRasterVar(
        name="altitude",
        path=Path("/tmp/altitude.tif"),
        raster_type=RasterType.continuous,
    )
    local.project = project
    project.raw_variables["altitude"] = local

    # A still-pending GEEVar holding a non-serializable image object.
    gee = GEEVar(
        name="slope",
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=[_Unserializable()],
    )
    gee.project = project
    project.raw_variables["slope"] = gee

    return project


def test_save_skips_geevars_and_persists_local(tmp_path, monkeypatch):
    """Saving does not choke on the image; the GEEVar reloads as a recipe."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)

    project = _project_with_mixed_raw("gee_skip")

    # Must not raise PydanticSerializationError on the pending GEEVar.
    project.save()

    loaded = Project.load("gee_skip")
    # The local var survives; the GEEVar comes back as a recipe to rebuild.
    assert set(loaded.raw_variables) == {"altitude"}
    recipe = loaded.cloud_variables["slope"]
    assert recipe["name"] == "slope" and "gee_images" not in recipe


def test_unrebuilt_recipe_survives_another_save(tmp_path, monkeypatch):
    """A recipe the GUI could not rebuild is written back, not dropped."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    _project_with_mixed_raw("gee_keep").save()

    Project.load("gee_keep").save()

    assert set(Project.load("gee_keep").cloud_variables) == {"slope"}
