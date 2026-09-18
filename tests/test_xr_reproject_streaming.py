"""``xr_reproject`` must stream the warp to disk, never hold the output whole.

Harmonize-all on a country-scale project (Peru, ~2.7 Gpx at 30 m) killed the
kernel on an 8-core / 16 GB machine. Measured 2026-09-18 on the Bolivia
project (45k x 49k source, 8 workers): 10.2 GB peak RSS for one *uint8*
layer, 1.8 GB after the fix, identical output. Three sinks added up:

* ``DataArray.rio.to_raster`` was called with rioxarray's default
  ``lock=None``, which takes the *non-dask* branch of its writer and
  evaluates the whole dask graph into one numpy array before the first
  ``write`` -- the entire output layer in RAM, copies on top. A float32
  layer at Peru scale is >10 GB before the copies.
* dask's threaded scheduler used every core, and each worker held the
  concatenation of the 128 MiB "auto" source chunks its tile overlapped.
* The GDAL block cache grew to its 5 %-of-RAM default.

The observable this test pins is the one that matters for a kernel on a
fixed memory budget: peak allocation does not grow with the output. It
warps the same synthetic source at two sizes and requires the peak to stay
flat when the output quadruples (before the fix it grew ~4x, 731 MB for a
35 MB output), with an absolute bound as a second guard.
"""

import tracemalloc

import numpy as np
import odc.geo.xr  # noqa: F401  # registers the .odc accessor
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

from spatialrisk.geo_utils import xr_reproject


def _write_source(path, side):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=(side, side), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=side,
        width=side,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(-66.0, -14.0, 0.001, 0.001),
        nodata=255,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
    ) as dst:
        dst.write(data, 1)


def _traced_peak(tmp_path, side):
    """Peak traced (numpy) allocation while warping a ``side``-pixel source.

    The target is the same footprint on a UTM grid at ~111 m, so the source
    is rotated onto the target and every destination tile pulls from several
    source tiles -- the shape of a real harmonization, not an identity copy.
    """
    src = tmp_path / f"src{side}.tif"
    _write_source(src, side)
    deg = side * 0.001
    geobox = GeoBox.from_bbox(
        (-66.0, -14.0 - deg, -66.0 + deg, -14.0), crs="EPSG:4326", resolution=0.001
    ).to_crs("EPSG:32720", resolution=111)
    out = tmp_path / f"out{side}.tif"

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        xr_reproject(
            raster_path=str(src),
            geobox=geobox,
            resampling_method="nearest",
            output_path=str(out),
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    with rasterio.open(out) as ds:
        assert ds.shape == geobox.shape
        # Centre window: the rotated footprint's corners are nodata.
        r, c = ds.height // 2, ds.width // 2
        assert ds.read(1, window=((r, r + 256), (c, c + 256))).max() <= 1
    return peak, geobox.shape[0] * geobox.shape[1]  # uint8 output bytes


def test_warp_peak_memory_does_not_grow_with_output(tmp_path, monkeypatch):
    """Quadrupling the output leaves the warp's peak allocation flat."""
    # Two workers: the invariant is about the write path, and the pool policy
    # (parallel.worker_threads) honours this override. Per-worker working
    # set is a constant; with the pool pinned the ratio below is clean.
    monkeypatch.setenv("SPATIAL_RISK_NUM_THREADS", "2")

    peak_small, _ = _traced_peak(tmp_path, 6000)
    peak_large, out_large = _traced_peak(tmp_path, 12000)

    # Measured with the fix: 111 MB -> 118 MB for 35 MB -> 143 MB outputs.
    assert peak_large < 1.5 * peak_small, (
        f"peak grew {peak_small / 1e6:.0f} MB -> {peak_large / 1e6:.0f} MB for a "
        "4x larger output: the warp is materialising the layer, not streaming it"
    )
    assert (
        peak_large < out_large
    ), f"peak {peak_large / 1e6:.0f} MB exceeds the {out_large / 1e6:.0f} MB output"
