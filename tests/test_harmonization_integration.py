"""End-to-end: what harmonization writes is what ``harmonization_status`` skips.

Every other test on this branch stubs one side of that mirror — the status
predicate against fabricated outputs, or the bulk methods against MagicMock
variables. The load-bearing assumption is that the *real*
``reproject_and_match`` / ``rasterize`` produce, under the *real*
``add_as_processed`` key, a file the *real* status check then reports as
current. Nothing pins that, and it has already drifted once: ``rasterize``
wrote ``{name}.tif`` while ``add_as_processed`` registered ``{name}_{year}``,
so two years of one vector shared a file (fixed on this branch; only a hand
audit caught it).

So: harmonize one real raster and two real years of one vector through
``run_processing``, then run it again and assert the second run does nothing at
all — no keys processed, no bytes rewritten. (Simulating the pre-F5 output path
fails both tests here, which is what makes them worth their runtime.)
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from gui.scripts import process_actions
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.local_vector_var import LocalVectorVar
from spatialrisk.variables.models import DataType, RasterizationMethod, RasterType

Project._ensure_model_schemas()

CRS = "EPSG:32618"
KEYS = ["altitude", "roads_2000", "roads_2020"]


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


def _project(tmp_path, monkeypatch):
    """A project on disk with one raw raster and one raw vector to harmonize."""
    import spatialrisk.project as project_module

    monkeypatch.setattr(project_module, "downloads_folder", tmp_path / "out")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    # Base grid: 20x20 @ 100 m. The source raster is finer and slightly offset,
    # so reprojection genuinely resamples rather than copying.
    base = _raster(
        src_dir / "base.tif", from_origin(500_000, 4_500_000, 100, 100), (20, 20)
    )
    dem = _raster(
        src_dir / "dem.tif", from_origin(499_950, 4_500_050, 50, 50), (44, 44), 42.0
    )

    roads = src_dir / "roads.gpkg"
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(500_400, 4_498_600, 501_600, 4_499_400)], crs=CRS
    ).to_file(roads, driver="GPKG")

    p = Project(project_name="harmonization-integration")
    p.base_raster = LocalRasterVar(
        name="base",
        path=base,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        project=p,
    )
    p.raw_variables["altitude"] = LocalRasterVar(
        name="altitude",
        path=dem,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        project=p,
    )
    # Two years of ONE vector on purpose: that is where the output path and the
    # add_as_processed key diverged (both years wrote {name}.tif while holding
    # two registry entries), which is the drift this test exists to catch.
    for year in (2000, 2020):
        p.raw_variables[f"roads_{year}"] = LocalVectorVar(
            name="roads",
            year=year,
            path=roads,
            data_type=DataType.vector,
            rasterization_method=RasterizationMethod.binary,
            project=p,
        )
    return p


def test_a_second_run_reharmonizes_nothing(tmp_path, monkeypatch):
    """Harmonize for real, then ask for it again: every layer must be skipped."""
    p = _project(tmp_path, monkeypatch)

    first = process_actions.run_processing(p)
    assert sorted(first["processed"]) == KEYS

    # add_as_processed must have registered every output under the raw key —
    # the mapping harmonization_status.output_key mirrors — and each at its own
    # path, or the shared-output guard would hold them pending forever.
    outputs = {k: p.processed_variables[k].path for k in KEYS}
    assert all(path.exists() for path in outputs.values())
    assert len({str(path) for path in outputs.values()}) == len(KEYS)
    before = {k: path.stat().st_mtime_ns for k, path in outputs.items()}

    second = process_actions.run_processing(p)

    assert second == {"processed": [], "skipped": KEYS}
    after = {k: path.stat().st_mtime_ns for k, path in outputs.items()}
    assert after == before, "the second run rewrote outputs it reported as skipped"


def test_a_changed_reference_grid_makes_every_layer_pending_again(
    tmp_path, monkeypatch
):
    """The skip is a claim about the grid, so moving the grid must undo it."""
    p = _project(tmp_path, monkeypatch)
    process_actions.run_processing(p)
    assert process_actions.run_processing(p)["processed"] == []

    # A different reference raster: same CRS, coarser pixels — every existing
    # output is now on the wrong grid.
    coarse = _raster(
        tmp_path / "src" / "base_200m.tif",
        from_origin(500_000, 4_500_000, 200, 200),
        (10, 10),
    )
    p.base_raster = LocalRasterVar(
        name="base",
        path=coarse,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        project=p,
    )

    third = process_actions.run_processing(p)
    assert sorted(third["processed"]) == KEYS
    assert third["skipped"] == []
