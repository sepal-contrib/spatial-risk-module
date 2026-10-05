"""Every copy of an open project sees the same reference raster and AOI.

The GUI republishes the open project as a shallow ``model_copy()`` after
almost every action, and each background job keeps the copy it started with,
then saves and republishes *that copy* when it finishes. The registries
(variables, models, samples, …) are dicts the copies share, so a job's copy
never misses what happened meanwhile — but the reference raster and the AOI
used to be plain per-copy attributes. A job that started before the user set
the reference, and finished after, wrote its reference-less copy over the
manifest and the app: the reference and its CRS were gone on the next load.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import spatialrisk.project as project_module
import spatialrisk.variables.local_raster_var as local_raster_var
from gui.scripts import process_actions
from gui.scripts.solara_threads import publish_if_current
from spatialrisk.project import Project
from spatialrisk.variables import LocalRasterVar

Project._ensure_model_schemas()

AOI = {"method": "ADMIN1", "name": "PRY_Amambay", "gee": True, "admin": "2184"}


class _Reactive:
    """The two members ``publish_if_current`` uses of a solara reactive."""

    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A saved-and-reloaded project with one small geographic raster."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    src = tmp_path / "forest_2020.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(10.0, 1.0, 0.001, 0.001),
    ) as ds:
        ds.write(np.ones((1, 20, 20), "uint8"))
    p = Project(project_name="shared")
    var = LocalRasterVar(name="forest", year=2020, path=src, raster_type="categorical")
    var.project = p
    p.raw_variables["forest_2020"] = var
    p.save()
    return Project.load("shared")


def _set_reference(reactive):
    """What the Process tile's reference worker does with the open project."""
    live = reactive.value
    process_actions.set_base_raster(
        live, "forest_2020", "EPSG:32633", 30.0, auto_save=False
    )
    if publish_if_current(reactive, live):
        live.save()


def _saved_reference():
    base = Project.load("shared").base_raster
    if base is None:
        return None
    return (base.name, base.year, base.default_crs, base.default_resolution)


REFERENCE = ("forest", 2020, "EPSG:32633", 30.0)


def test_a_job_started_before_the_reference_keeps_it_when_it_finishes(project):
    """A download started in Step 2 finishing after Step 3 keeps the reference."""
    reactive = _Reactive(project)
    job_copy = reactive.value  # e.g. a download started in Step 2
    reactive.set(job_copy.model_copy())  # any tile republishing meanwhile
    _set_reference(reactive)
    assert _saved_reference() == REFERENCE

    job_copy.save()  # the job finishes: save, then republish its copy
    publish_if_current(reactive, job_copy)

    assert _saved_reference() == REFERENCE
    assert reactive.value.base_raster is not None


def test_a_republish_during_the_reference_warp_keeps_the_reference(
    project, monkeypatch
):
    """``reproject`` reads ``self.project`` after the warp: the newest copy."""
    reactive = _Reactive(project)
    warp = local_raster_var.reproject_raster_gdal_warp

    def warp_then_republish(*args, **kwargs):
        out = warp(*args, **kwargs)
        reactive.set(reactive.value.model_copy())
        return out

    monkeypatch.setattr(
        local_raster_var, "reproject_raster_gdal_warp", warp_then_republish
    )
    _set_reference(reactive)

    assert _saved_reference() == REFERENCE
    assert reactive.value.base_raster is not None


def test_clearing_the_reference_reaches_every_copy(project):
    """Removing the reference's layer must not be undone by an older copy."""
    reactive = _Reactive(project)
    _set_reference(reactive)
    job_copy = reactive.value.model_copy()

    reactive.value.base_raster = None  # variables_tile: its source was removed
    job_copy.save()

    assert job_copy.base_raster is None
    assert _saved_reference() is None


def test_an_aoi_attached_after_a_job_started_is_kept(project):
    """An older copy saving later still writes the AOI attached meanwhile."""
    job_copy = project.model_copy()
    live = job_copy.model_copy()
    live.aoi = dict(AOI)

    job_copy.save()

    assert Project.load("shared").aoi == AOI


def test_a_deep_copy_is_independent(project):
    """Only shallow copies share; a deep copy is a separate project."""
    other = project.model_copy(deep=True)
    other.aoi = dict(AOI)
    assert project.aoi is None


def test_an_update_applies_to_that_copy_only(project):
    """``model_copy(update=...)`` overrides on the copy and detaches it."""
    detached = project.model_copy(update={"aoi": dict(AOI)})
    assert detached.aoi == AOI
    assert project.aoi is None
    project.aoi = {"method": "DRAW"}
    assert detached.aoi == AOI


def test_separately_loaded_projects_do_not_share():
    """Two projects never share a reference or an AOI."""
    a = Project(project_name="a", aoi=dict(AOI))
    b = Project(project_name="b")
    assert a.aoi == AOI
    assert b.aoi is None
    b.aoi = {"method": "DRAW"}
    assert a.aoi == AOI


def test_the_constructor_and_model_construct_accept_both_fields():
    """``base_raster``/``aoi`` still work as constructor arguments."""
    base = LocalRasterVar.model_construct(name="forest", year=2020)
    p = Project(project_name="c", base_raster=base, aoi=dict(AOI))
    assert p.base_raster is base and p.aoi == AOI
    q = Project.model_construct(project_name="d")
    assert q.base_raster is None and q.aoi is None


def test_equal_content_still_compares_equal():
    """Reacton skips re-renders on ``==`` props; equality must not change."""
    a = Project(project_name="e", aoi=dict(AOI))
    assert a == a.model_copy()
    assert a == Project(project_name="e", aoi=dict(AOI))
    assert a != Project(project_name="e")
