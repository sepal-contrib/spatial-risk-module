"""Raster scans the app runs on a whole AOI must read in row strips.

Measured 2026-10-05 on synthetic 20000 x 20000 rasters: showing an
AOI-extent density map read the Float64 band whole and copied it twice
(9.7 GB peak, ~24 bytes a pixel), and an allocation run counted the cropped
risk map's categories from whole-band reads (4.2 GB for uint16). Either
kills a 16 GB kernel at a big AOI's ~1 Gpx.

As in tests/test_change_layer_streaming.py the observable is that peak
allocation stays flat when the raster grows -- here 4x taller at the same
width, so one row strip costs the same at both sizes -- with the raster's own
size as an absolute bound.
"""

import tracemalloc

import numpy as np
from osgeo import gdal

from gui.scripts.density_map import density_value_range
from spatialrisk.allocation import DENSITY_NODATA, _count_categories

WIDTH = 2000


def _write(path, arr, gdal_type, nodata=None):
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path),
        arr.shape[1],
        arr.shape[0],
        1,
        gdal_type,
        ["TILED=YES", "COMPRESS=DEFLATE"],
    )
    ds.SetGeoTransform((0, 30, 0, arr.shape[0] * 30, 0, -30))
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds = None
    return path


def _density(path, height):
    yy, xx = np.mgrid[0:height, 0:WIDTH]
    arr = ((xx // 50 + yy // 70) % 30) * 0.01
    arr[:, :100] = DENSITY_NODATA
    arr[height // 2, 500] = 7.5
    return _write(path, arr, gdal.GDT_Float64, DENSITY_NODATA)


def _risk(path, height):
    yy, xx = np.mgrid[0:height, 0:WIDTH]
    return _write(path, ((xx // 50 + yy // 70) % 30).astype(np.uint16), gdal.GDT_UInt16)


def _peak(fn):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_density_range_matches_a_whole_band_read(tmp_path):
    """Strips give the same (vmin, vmax) a whole read would, nodata excluded."""
    path = _density(tmp_path / "d.tif", 3000)
    ds = gdal.Open(str(path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    valid = arr[arr != DENSITY_NODATA]

    assert density_value_range(path) == (float(valid.min()), float(valid.max()))


def test_density_range_peak_memory_does_not_grow_with_raster(tmp_path):
    """A 4x taller density raster leaves the range scan's peak flat."""
    small = _density(tmp_path / "small.tif", 2000)
    large = _density(tmp_path / "large.tif", 8000)

    _, peak_small = _peak(lambda: density_value_range(small))
    _, peak_large = _peak(lambda: density_value_range(large))

    assert peak_large < 1.5 * peak_small, (
        f"peak grew {peak_small / 1e6:.0f} MB -> {peak_large / 1e6:.0f} MB for a "
        "4x taller raster: the band is read whole"
    )
    assert peak_large < 8000 * WIDTH * 8  # the Float64 band itself


def test_category_counts_match_a_whole_band_count(tmp_path):
    """Strip counts equal np.unique over the whole band, mask applied."""
    risk_path = _risk(tmp_path / "risk.tif", 3000)
    mask = np.zeros((3000, WIDTH), dtype=np.uint8)
    mask[::3, ::2] = 1
    mask_path = _write(tmp_path / "mask.tif", mask, gdal.GDT_Byte)

    counts = _count_categories(risk_path, mask_path, tmp_path, [], blk_rows=128)

    ds = gdal.Open(str(risk_path))
    risk = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    eligible = (risk != 0) & (mask == 1)
    values, expected = np.unique(risk[eligible], return_counts=True)
    assert counts["cat"].tolist() == values.tolist()
    assert counts["counts"].tolist() == expected.tolist()


def test_category_count_peak_memory_does_not_grow_with_raster(tmp_path):
    """A 4x taller risk map leaves the category count's peak flat."""
    small = _risk(tmp_path / "small.tif", 2000)
    large = _risk(tmp_path / "large.tif", 8000)

    _, peak_small = _peak(
        lambda: _count_categories(small, None, tmp_path, [], blk_rows=128)
    )
    _, peak_large = _peak(
        lambda: _count_categories(large, None, tmp_path, [], blk_rows=128)
    )

    assert peak_large < 1.5 * peak_small, (
        f"peak grew {peak_small / 1e6:.0f} MB -> {peak_large / 1e6:.0f} MB for a "
        "4x taller risk map: the band is read whole"
    )
    assert peak_large < 8000 * WIDTH * 2  # the UInt16 band itself
