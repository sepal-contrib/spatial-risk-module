"""Importing a local raster registers it as a first-class Prediction.

It shows in the Inference outputs list and is selectable in Step 8 — Evaluation.
"""

import numpy as np
import odc.geo.xr  # noqa: F401  # registers the .odc accessor
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

from gui.scripts.prediction_import import import_prediction
from spatialrisk.predictions.import_raster import ImportRasterError


class _FakeFolders:
    def __init__(self, root):
        self.project_folder = root


class _FakeProject:
    """Minimal Project stand-in: just the bits import_prediction touches."""

    def __init__(self, root):
        self.folders = _FakeFolders(root)
        self.predictions = {}
        self.saves = 0

    def add_prediction(self, pred, key=None, auto_save=True):
        storage_key = key or pred.storage_key()
        pred.project = self
        self.predictions[storage_key] = pred
        if auto_save:
            self.save()

    def save(self):
        self.saves += 1


class _FakeBase:
    """Stand-in for project.base_raster: only get_base_geobox is used."""

    def get_base_geobox(self):
        return GeoBox.from_bbox(
            (700000.0, 7330000.0, 720000.0, 7350000.0),
            crs="EPSG:32721",
            resolution=1000.0,
        )


def _project(tmp_path, with_base=True):
    proj = _FakeProject(tmp_path / "proj")
    proj.base_raster = _FakeBase() if with_base else None
    return proj


def _src_raster(tmp_path, value=0.5, dtype=np.float32, nodata=None):
    src = tmp_path / "source" / "my map.tif"
    src.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((60, 60), value, dtype=dtype)
    data[:, 30:] = value / 2  # not constant
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        height=60,
        width=60,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(-55.0, -24.0, 0.002, 0.002),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return src


def test_import_adapts_file_and_registers_prediction(tmp_path):
    """import_prediction warps the source onto the base grid and registers it."""
    proj = _project(tmp_path)
    src = _src_raster(tmp_path)

    pred = import_prediction(proj, str(src), name="my map", value_scale="probability")

    # Adapted copy inside the project, always a .tif on the base grid.
    assert pred.path.exists()
    assert pred.path.parent == (tmp_path / "proj" / "imported_predictions")
    assert pred.path.suffix == ".tif"
    with rasterio.open(pred.path) as ds:
        assert ds.dtypes[0] == "uint16"
        assert ds.nodata == 0
        assert ds.crs.to_epsg() == 32721
        assert (ds.width, ds.height) == (20, 20)

    assert pred.name == "my map"
    assert pred.model_key == "my-map"
    assert pred.dataset_name == "imported"
    assert pred.display_palette == "far"
    assert pred.run_params["value_scale"] == "probability"
    assert pred.run_params["source_path"] == str(src)
    assert pred.run_params["source_crs"] == "EPSG:4326"
    assert pred.run_params["source_dtype"] == "float32"
    assert len(pred.run_params["source_range"]) == 2
    assert pred.storage_key() in proj.predictions
    assert proj.saves >= 1


def test_import_risk_scale_keeps_values(tmp_path):
    """A risk-scale import keeps raw whole-number values instead of rescaling them."""
    proj = _project(tmp_path)
    src = _src_raster(tmp_path, value=40000, dtype=np.uint16, nodata=0)

    pred = import_prediction(proj, str(src), name="risk", value_scale="risk")

    with rasterio.open(pred.path) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert values <= {0, 40000, 20000}
    assert 40000 in values
    assert 20000 in values


def test_import_without_base_raster_raises_before_writing(tmp_path):
    """Importing without a processed base raster fails before touching disk."""
    proj = _project(tmp_path, with_base=False)
    src = _src_raster(tmp_path)

    with pytest.raises(ImportRasterError, match="base raster"):
        import_prediction(proj, str(src), name="x", value_scale="probability")
    assert not (tmp_path / "proj" / "imported_predictions").exists()
    assert proj.predictions == {}


def test_import_rejects_range_contradicting_scale(tmp_path):
    """A value range that contradicts the declared scale is rejected before writing."""
    proj = _project(tmp_path)
    src = _src_raster(tmp_path, value=100.0)  # 0..100, declared as 0..1

    with pytest.raises(ImportRasterError, match=r"0\.\.1"):
        import_prediction(proj, str(src), name="x", value_scale="probability")
    assert not list((tmp_path / "proj").rglob("*.tif"))
    assert proj.predictions == {}


def test_import_disambiguates_duplicate_names(tmp_path):
    """Importing the same name twice yields two distinct registry entries and files."""
    proj = _project(tmp_path)
    src = _src_raster(tmp_path)

    p1 = import_prediction(proj, str(src), name="map", value_scale="probability")
    p2 = import_prediction(proj, str(src), name="map", value_scale="probability")

    assert p1.storage_key() != p2.storage_key()
    assert p1.path != p2.path
    assert len(proj.predictions) == 2


def test_import_missing_file_raises(tmp_path):
    """Importing a nonexistent source path raises FileNotFoundError."""
    proj = _project(tmp_path)
    with pytest.raises(FileNotFoundError):
        import_prediction(
            proj, str(tmp_path / "nope.tif"), name="x", value_scale="probability"
        )


def test_sanitize_import_name_is_public():
    """sanitize_import_name is importable and sanitizes free-text names."""
    from gui.scripts.prediction_import import sanitize_import_name

    assert sanitize_import_name("my map") == "my-map"
    assert sanitize_import_name("  a  b  ") == "a-b"
    assert sanitize_import_name("weird/&chars") == "weirdchars"
    assert sanitize_import_name("***") == "imported"  # fallback token


def test_resolve_import_key_free_name(tmp_path):
    """resolve_import_key returns the sanitized name when it is not yet taken."""
    from gui.scripts.prediction_import import resolve_import_key

    proj = _FakeProject(tmp_path / "proj")
    assert resolve_import_key(proj, "my map") == "my-map"


def test_resolve_import_key_previews_import_disambiguation(tmp_path):
    """resolve_import_key previews the key import_prediction will actually assign."""
    from gui.scripts.prediction_import import resolve_import_key

    proj = _project(tmp_path)
    src = _src_raster(tmp_path)

    # Before any import the name is free; after, the preview shows the suffix
    # import_prediction would actually assign.
    assert resolve_import_key(proj, "map", src_suffix=src.suffix) == "map"
    p1 = import_prediction(proj, str(src), name="map", value_scale="probability")
    assert p1.model_key == "map"
    assert resolve_import_key(proj, "map", src_suffix=src.suffix) == "map-2"
    p2 = import_prediction(proj, str(src), name="map", value_scale="probability")
    assert p2.model_key == "map-2"
