"""``process_change_xarray`` must stream the change layer, never hold it whole.

A loss layer over a big AOI killed the kernel. Measured 2026-10-05 on two
synthetic 32000 x 32000 uint8 masks (16 threads): 16.2 GB peak RSS, 0.8 GB
after the fix, identical output. Three sinks added up:

* ``chunks="auto"`` read uint8 masks in ~11k x 11k (127 Mpx) tiles.
* ``xr.where(..., 1, ..., 255)`` promoted every tile to int64 -- 1 GB a tile
  before the boolean temporaries -- and the cast back to uint8 came last.
* dask's threaded scheduler held one such tile per core.

As in tests/test_xr_reproject_streaming.py, the observable pinned is that
peak allocation does not grow with the layer: the same masks at two sizes,
the peak flat when the output quadruples (before the fix it grew ~4x, past
the size of the output), with an absolute bound as a second guard.
"""

import tracemalloc

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from spatialrisk.processing import process_change_xarray


def _masks(side):
    """Two presence masks (1/0) with a nodata strip and every transition."""
    yy, xx = np.mgrid[0:side, 0:side]
    t1 = (((xx // 97) + (yy // 89)) % 3 != 0).astype(np.uint8)
    t2 = t1.copy()
    t2[(t1 == 1) & (((xx // 13) * 7 + (yy // 11) * 3) % 100 < 15)] = 0
    t2[(t1 == 0) & (((xx // 17) + (yy // 7)) % 10 == 0)] = 1
    t1[:, : side // 20] = 255
    t2[side // 20 : side // 10, :] = 255
    return t1, t2


def _write(path, data):
    side = data.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=side,
        width=side,
        count=1,
        dtype="uint8",
        crs="EPSG:32719",
        transform=from_origin(0, side * 30, 30, 30),
        nodata=255,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
    ) as dst:
        dst.write(data, 1)


def _expected(t1, t2, op):
    valid = (t1 != 255) & (t2 != 255)
    if op == "loss":
        event, stable = valid & (t1 == 1) & (t2 == 0), valid & (t1 == 1) & (t2 == 1)
    else:
        event, stable = valid & (t1 == 0) & (t2 == 1), valid & (t1 == 0) & (t2 == 0)
    return np.where(event, 1, np.where(stable, 0, 255)).astype(np.uint8)


@pytest.mark.parametrize("op", ["loss", "gain"])
def test_change_layer_values(tmp_path, op):
    """1 = event, 0 = stable, 255 = nodata or out of scope, on a multi-chunk grid."""
    t1, t2 = _masks(5000)  # > one 2048 chunk each way
    _write(tmp_path / "t1.tif", t1)
    _write(tmp_path / "t2.tif", t2)
    out = tmp_path / "out.tif"

    process_change_xarray(
        str(tmp_path / "t1.tif"), str(tmp_path / "t2.tif"), str(out), op=op
    )

    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "uint8"
        assert ds.nodata == 255
        assert ds.crs.to_epsg() == 32719
        np.testing.assert_array_equal(ds.read(1), _expected(t1, t2, op))


def _traced_peak(tmp_path, side):
    t1, t2 = _masks(side)
    _write(tmp_path / f"t1_{side}.tif", t1)
    _write(tmp_path / f"t2_{side}.tif", t2)
    del t1, t2
    out = tmp_path / f"out{side}.tif"

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        process_change_xarray(
            str(tmp_path / f"t1_{side}.tif"),
            str(tmp_path / f"t2_{side}.tif"),
            str(out),
            op="loss",
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, side * side  # uint8 output bytes


def test_change_peak_memory_does_not_grow_with_output(tmp_path, monkeypatch):
    """Quadrupling the layer leaves the change computation's peak flat."""
    # Pool and GDAL threads pinned for a clean ratio; see the note in
    # tests/test_xr_reproject_streaming.py on why GDAL_NUM_THREADS goes
    # through rasterio.Env rather than setenv.
    monkeypatch.setenv("SPATIAL_RISK_NUM_THREADS", "2")
    with rasterio.Env(GDAL_NUM_THREADS=1):
        peak_small, _ = _traced_peak(tmp_path, 6000)
        peak_large, out_large = _traced_peak(tmp_path, 12000)

    assert peak_large < 1.5 * peak_small, (
        f"peak grew {peak_small / 1e6:.0f} MB -> {peak_large / 1e6:.0f} MB for a "
        "4x larger layer: the change is materialising the layer, not streaming it"
    )
    assert (
        peak_large < out_large
    ), f"peak {peak_large / 1e6:.0f} MB exceeds the {out_large / 1e6:.0f} MB output"
