"""``xr_rasterize`` must burn and write the grid in row bands, never whole.

Measured 2026-09-18 on the Bolivia grid (44661 x 49363, uint8): the
whole-grid path peaked at 8.4 GB over baseline for a 2.2 GB output -- the
``rasterize`` array, its ``wrap_xr`` DataArray and the writer's encode copy,
all resident at once. Peru at 30 m is larger still, and a vector layer runs
inside the same harmonize-all as the raster warps on the same 16 GB budget.
See tests/test_xr_reproject_streaming.py for the raster side of that budget.

Pinned here: peak traced allocation stays well below the output's size, and
the burn is correct at the band seams.
"""

import tracemalloc

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from spatialrisk.processing import xr_rasterize

SIDE = 12000  # 144 MB uint8 output


def _geobox():
    return GeoBox.from_bbox(
        (500000, 8000000, 500000 + SIDE * 30, 8000000 + SIDE * 30),
        crs="EPSG:32720",
        resolution=30,
    )


def _write_vector(path, geobox):
    b = geobox.boundingbox
    # Three squares: one at the top-left corner, one straddling the middle of
    # the grid (so it crosses any band seam), one at the bottom-right.
    side = (b.right - b.left) / 5
    mid_x, mid_y = (b.left + b.right) / 2, (b.bottom + b.top) / 2
    polys = [
        box(b.left, b.top - side, b.left + side, b.top),
        box(mid_x - side / 2, mid_y - side / 2, mid_x + side / 2, mid_y + side / 2),
        box(b.right - side, b.bottom, b.right, b.bottom + side),
    ]
    gpd.GeoDataFrame({"geometry": polys}, crs=geobox.crs.to_wkt()).to_file(
        path, driver="GPKG"
    )


def test_rasterize_peak_memory_stays_below_output(tmp_path):
    """Three squares on a 144 MB grid burn with a fraction of that resident."""
    geobox = _geobox()
    vec = tmp_path / "squares.gpkg"
    _write_vector(vec, geobox)
    out = tmp_path / "out.tif"
    out_bytes = geobox.shape[0] * geobox.shape[1]

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        xr_rasterize(shapefile_path=str(vec), geobox=geobox, output_path=str(out))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 0.5 * out_bytes, (
        f"peak {peak / 1e6:.0f} MB for a {out_bytes / 1e6:.0f} MB output: "
        "the grid was rasterized whole instead of in row bands"
    )

    with rasterio.open(out) as ds:
        assert ds.shape == geobox.shape
        assert ds.dtypes == ("uint8",)
        data = ds.read(1)
    n = SIDE // 5
    assert data[:n, :n].all()  # top-left square
    assert data[n : n + 10, n : n + 10].sum() == 0  # just outside it
    mid = SIDE // 2
    assert data[mid - n // 2 : mid + n // 2, mid - n // 2 : mid + n // 2].all()
    assert data[-n:, -n:].all()  # bottom-right square
    # Bit-identical to the whole-grid burn: rasterize() decides per pixel
    # centre, so banding the rows must not move a single seam pixel.
    gdf = gpd.read_file(vec).to_crs(geobox.crs)
    reference = rasterio.features.rasterize(
        list(gdf.geometry),
        out_shape=geobox.shape,
        transform=geobox.transform,
        dtype="uint8",
    )
    assert np.array_equal(data, reference)


def test_unique_mode_matches_whole_grid_burn(tmp_path):
    """Overlapping features: band filtering must keep the whole-grid burn order."""
    geobox = GeoBox.from_bbox(
        (500000, 8000000, 500000 + 3000 * 30, 8000000 + 3000 * 30),
        crs="EPSG:32720",
        resolution=30,
    )
    b = geobox.boundingbox
    w = (b.right - b.left) / 3
    # Three overlapping squares whose union straddles the band seam at row 2048.
    polys = [
        box(b.left, b.top - 2 * w, b.left + 2 * w, b.top),
        box(b.left + w, b.top - 3 * w, b.left + 3 * w, b.top - w),
        box(b.left + w / 2, b.top - 2.5 * w, b.left + 1.5 * w, b.top - 1.5 * w),
    ]
    vec = tmp_path / "overlap.gpkg"
    gpd.GeoDataFrame({"geometry": polys}, crs=geobox.crs.to_wkt()).to_file(
        vec, driver="GPKG"
    )
    out = tmp_path / "out.tif"
    xr_rasterize(
        shapefile_path=str(vec), geobox=geobox, output_path=str(out), mode="unique"
    )
    with rasterio.open(out) as ds:
        data = ds.read(1)
    gdf = gpd.read_file(vec).to_crs(geobox.crs)
    reference = rasterio.features.rasterize(
        list(zip(gdf.geometry, range(1, len(gdf) + 1))),
        out_shape=geobox.shape,
        transform=geobox.transform,
        dtype="uint8",
    )
    assert set(np.unique(data)) == {0, 1, 2, 3}
    assert np.array_equal(data, reference)
