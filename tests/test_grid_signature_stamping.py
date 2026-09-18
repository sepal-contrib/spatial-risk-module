"""Harmonized outputs carry the signature of the grid they were matched to."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from spatialrisk.harmonization import geobox_signature
from spatialrisk.project import Project
from spatialrisk.variables import local_raster_var as mod
from spatialrisk.variables import local_vector_var as vector_mod
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.local_vector_var import LocalVectorVar
from spatialrisk.variables.models import (
    DataType,
    RasterizationMethod,
    RasterType,
)

Project._ensure_model_schemas()

CRS = "EPSG:32618"


class _Shape:
    def __init__(self, rows, cols):
        self.yx = (rows, cols)


class _CRS:
    def to_epsg(self):
        return 32618

    def __str__(self):
        return "EPSG:32618"


class _Res:
    x = 30.0


class _Geobox:
    """Duck-typed stand-in for an odc-geo GeoBox."""

    crs = _CRS()
    resolution = _Res()
    transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
    shape = _Shape(100, 200)


class _Folders:
    def __init__(self, folder):
        self.processed_data_folder = folder
        self.data_raw_folder = folder


class _Project:
    project_name = "stamping"

    def __init__(self, folder):
        self.folders = _Folders(folder)


class _Base:
    """Minimal stand-in for the base raster ``rasterize`` takes."""

    data_type = DataType.raster

    def get_base_geobox(self):
        return _Geobox()


def _raster(path, transform, shape, fill=1.0):
    """Write a small single-band raster with real (non-nodata) values."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(np.full(shape, fill, "float32"), 1)
    return path


def test_reproject_and_match_stamps_the_target_geobox(monkeypatch, tmp_path):
    """The output records the grid it was matched to, without reopening it."""
    source = tmp_path / "layer.tif"
    source.write_bytes(b"")

    # The warp and its all-nodata guard are the only disk work in the method;
    # stub both so the test exercises the stamping, not GDAL.
    monkeypatch.setattr(mod, "xr_reproject", lambda **kwargs: None)
    monkeypatch.setattr(mod, "raster_is_all_nodata", lambda _path: False)

    var = LocalRasterVar.model_construct(
        name="layer",
        path=source,
        raster_type=RasterType.continuous,
        processing_history=[],
        project=_Project(tmp_path),
        year=None,
        tags=[],
        active=True,
    )

    out = var.reproject_and_match(_Geobox())

    assert out.grid_signature == geobox_signature(_Geobox())


def test_signature_moves_when_the_target_grid_moves(monkeypatch, tmp_path):
    """A different reference grid must produce a different stamp."""
    source = tmp_path / "layer.tif"
    source.write_bytes(b"")
    monkeypatch.setattr(mod, "xr_reproject", lambda **kwargs: None)
    monkeypatch.setattr(mod, "raster_is_all_nodata", lambda _path: False)

    var = LocalRasterVar.model_construct(
        name="layer",
        path=source,
        raster_type=RasterType.continuous,
        processing_history=[],
        project=_Project(tmp_path),
        year=None,
        tags=[],
        active=True,
    )

    wide = _Geobox()
    wide.shape = _Shape(100, 999)
    assert var.reproject_and_match(_Geobox()).grid_signature != (
        var.reproject_and_match(wide).grid_signature
    )


def test_rasterize_stamps_the_base_geobox(monkeypatch, tmp_path):
    """A rasterized vector is written onto the base grid, so it records it."""
    source = tmp_path / "roads.gpkg"
    source.write_bytes(b"")
    monkeypatch.setattr(vector_mod, "xr_rasterize", lambda **kwargs: None)

    var = LocalVectorVar.model_construct(
        name="roads",
        path=source,
        rasterization_method=RasterizationMethod.binary,
        project=_Project(tmp_path),
        year=None,
        tags=[],
        active=True,
    )

    out = var.rasterize(_Base())

    assert out.grid_signature == geobox_signature(_Geobox())


def test_use_as_base_raster_stamps_the_grid_on_disk(tmp_path):
    """The base is stamped from the file itself — its grid is only known there."""
    path = _raster(
        tmp_path / "base.tif", from_origin(500_000, 4_500_000, 100, 100), (20, 20)
    )
    var = LocalRasterVar.model_construct(
        name="base",
        path=path,
        raster_type=RasterType.continuous,
        data_type=DataType.raster,
        processing_history=[],
        project=_Project(tmp_path),
        year=None,
        tags=[],
        active=True,
    )

    var.use_as_base_raster(auto_save=False)

    # A geobox read back off a file carries its CRS in WKT form; the stamp must
    # still collapse to the EPSG token, or it could never equal an output's.
    assert var.grid_signature.startswith("EPSG:32618|")
    assert var.grid_signature.endswith("|20x20")
    assert var.grid_signature == geobox_signature(var.get_base_geobox())


def test_real_outputs_are_stamped_with_the_grid_they_landed_on(tmp_path, monkeypatch):
    """On real files: one identical stamp, and each describes the bytes on disk.

    Both stub tests above take the stamp on trust — the write is patched out.
    Here the warp and the burn actually run, so a stamp taken from the *target*
    geobox is checked against the geobox read back off the file it produced.
    A stamp that describes intent rather than the file would be worse than no
    stamp at all, because the status check trusts it.

    It also exercises the CRS canonicalization end to end: the base's geobox is
    read off disk (WKT form) while the outputs are stamped from the in-memory
    target, so the three strings only agree because ``geobox_signature``
    collapses both spellings onto the EPSG token.
    """
    import spatialrisk.project as project_module

    monkeypatch.setattr(project_module, "downloads_folder", tmp_path / "out")

    src = tmp_path / "src"
    src.mkdir()
    base_path = _raster(
        src / "base.tif", from_origin(500_000, 4_500_000, 100, 100), (20, 20)
    )
    dem_path = _raster(
        src / "dem.tif", from_origin(499_950, 4_500_050, 50, 50), (44, 44), 42.0
    )
    roads_path = src / "roads.gpkg"
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(500_400, 4_498_600, 501_600, 4_499_400)], crs=CRS
    ).to_file(roads_path, driver="GPKG")

    p = Project(project_name="stamping-e2e")
    base = LocalRasterVar(
        name="base",
        path=base_path,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        project=p,
    )
    base.use_as_base_raster(auto_save=False)
    dem = LocalRasterVar(
        name="altitude",
        path=dem_path,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        project=p,
    )
    roads = LocalVectorVar(
        name="roads",
        path=roads_path,
        data_type=DataType.vector,
        rasterization_method=RasterizationMethod.binary,
        project=p,
    )

    geobox = p.base_raster.get_base_geobox()
    warped = dem.reproject_and_match(geobox)
    burned = roads.rasterize(base=p.base_raster)

    # Pinned against the grid itself, not merely against each other: two
    # unstamped variables also compare equal, and that agreement means nothing.
    assert p.base_raster.grid_signature == geobox_signature(geobox)
    assert warped.grid_signature == p.base_raster.grid_signature
    assert burned.grid_signature == p.base_raster.grid_signature

    # ...and each stamp must match the file that was actually written.
    assert geobox_signature(warped.get_base_geobox()) == warped.grid_signature
    assert geobox_signature(burned.get_base_geobox()) == burned.grid_signature
