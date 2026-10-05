"""A categorical layer's categories are read once, when it is registered.

Counting the distinct values of a sub-jurisdiction raster over a big AOI takes
10-20 s. The New model dialog needs the count to warn about slow iCAR training
and fit() needs the values to declare each C() term's full domain, so the scan
runs once in ``add_as_processed`` -- inside the processing job -- and the
result is saved with the variable as ``categorical_levels``.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import spatialrisk.project as project_module
from spatialrisk.far_helpers import MissingCategoricalLevels, get_categorical_levels
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import RasterType

Project._ensure_model_schemas()


def _write(path, values, dtype="uint8", nodata=255):
    data = np.array(values, dtype=dtype).reshape(2, 2)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=dtype,
        crs="EPSG:32631",
        transform=from_origin(0, 0, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _var(project, path, raster_type=RasterType.categorical, name="subj"):
    var = LocalRasterVar(name=name, path=path, raster_type=raster_type)
    var.project = project
    return var


def test_registering_a_categorical_layer_stores_its_categories(tmp_path):
    """Sorted distinct values, nodata left out."""
    p = Project(project_name="stamp")
    var = _var(p, _write(tmp_path / "subj.tif", [7, 3, 3, 255]))

    var.add_as_processed(auto_save=False)

    assert p.processed_variables["subj"].categorical_levels == [3, 7]


def test_float_categories_drop_nan_and_keep_integral_values_as_int(tmp_path):
    """A float raster's NaN is not a category; 2.0 reads as 2."""
    p = Project(project_name="stamp")
    path = _write(tmp_path / "pa.tif", [2.0, np.nan, 1.0, 2.0], "float32", None)

    _var(p, path, name="pa").add_as_processed(auto_save=False)

    assert p.processed_variables["pa"].categorical_levels == [1, 2]


def test_a_continuous_layer_gets_no_categories(tmp_path):
    """Only categorical layers are scanned."""
    p = Project(project_name="stamp")
    var = _var(p, _write(tmp_path / "alt.tif", [1, 2, 3, 4]), RasterType.continuous)

    var.add_as_processed(auto_save=False)

    assert p.processed_variables["subj"].categorical_levels is None


def test_categories_survive_save_and_load(tmp_path, monkeypatch):
    """They are saved with the variable, so a reopened project has them."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    p = Project(project_name="stamp_roundtrip")
    _var(p, _write(tmp_path / "subj.tif", [5, 6, 6, 255])).add_as_processed()

    restored = Project.load("stamp_roundtrip")

    assert restored.processed_variables["subj"].categorical_levels == [5, 6]


def test_get_categorical_levels_reads_the_stored_categories(tmp_path, monkeypatch):
    """Training uses the stored list and never rescans the raster."""
    p = Project(project_name="stamp")
    var = _var(p, _write(tmp_path / "subj.tif", [1, 2, 2, 255]))
    var.add_as_processed(auto_save=False)

    def _boom(*args, **kwargs):
        raise AssertionError("the raster must not be read again")

    monkeypatch.setattr(rasterio, "open", _boom)

    assert get_categorical_levels(var) == [1, 2]


def test_a_layer_harmonized_before_categories_were_stored_says_so(tmp_path):
    """A clear error naming the layer and the fix, not a silent bare C(x)."""
    var = _var(None, _write(tmp_path / "subj.tif", [1, 2, 2, 255]))

    with pytest.raises(MissingCategoricalLevels, match="subj.*[Hh]armoniz"):
        get_categorical_levels(var)


def test_re_harmonizing_updates_the_datasets_that_use_the_layer(tmp_path):
    """A dataset holds layer objects; a re-registered layer must replace them.

    Re-harmonizing registers a *new* object under the same key. Datasets kept
    the old one -- without categories -- so the iCAR warning stayed silent and
    training refused the layer until the project was reopened.
    """
    import types

    p = Project(project_name="stamp")
    path = _write(tmp_path / "subj.tif", [1, 2, 2, 255])
    old = LocalRasterVar.model_construct(
        name="subj", path=path, raster_type="categorical", project=p, year=None
    )
    target = _var(p, _write(tmp_path / "loss.tif", [0, 1, 1, 255]), name="loss")
    p.processed_variables["subj"] = old
    p.datasets["calib"] = types.SimpleNamespace(
        name="calib", target=target, features=[old]
    )

    _var(p, path).add_as_processed(auto_save=False)

    (feature,) = p.datasets["calib"].features
    assert feature is p.processed_variables["subj"]
    assert feature.categorical_levels == [1, 2]
    assert p.datasets["calib"].target is target  # untouched
