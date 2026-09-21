"""Regression tests for the windowed read path of ``Dataset.extract_at_points``.

The reference implementation below is the pre-optimisation algorithm (full
``src.read(1)`` per layer) copied verbatim. The optimised method must return a
frame that is value-identical to it in every case the old code handled:
out-of-bounds points, nodata hits, NaN hits, layers on different CRSs and
grids, layers without a nodata value, and points sharing a tile.
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_bounds, from_origin  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from shapely.geometry import Point  # noqa: E402


class _Var:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.year = None


def _reference_extract(ds, points, *, drop_nodata=True):
    """The original full-read algorithm, inlined as the oracle."""
    all_vars = [ds.target] + ds.features
    n_pts = len(points)
    df_data = {}
    valid = np.ones(n_pts, dtype=bool)
    target_cell = None

    for i, var in enumerate(all_vars):
        with rasterio.open(var.path) as src:
            arr = src.read(1)
            nodata = src.nodata
            vcrs = src.crs
            vtransform = src.transform
            vheight, vwidth = src.height, src.width

        vpts = points.to_crs(vcrs) if points.crs != vcrs else points
        r, c = rasterio.transform.rowcol(
            vtransform, vpts.geometry.x.to_numpy(), vpts.geometry.y.to_numpy()
        )
        r = np.asarray(r, dtype=int)
        c = np.asarray(c, dtype=int)
        in_bounds = (r >= 0) & (r < vheight) & (c >= 0) & (c < vwidth)

        rc = np.clip(r, 0, vheight - 1)
        cc = np.clip(c, 0, vwidth - 1)
        vals = arr[rc, cc]

        layer_valid = in_bounds.copy()
        if np.issubdtype(vals.dtype, np.floating):
            layer_valid &= ~np.isnan(vals)
        if nodata is not None:
            layer_valid &= vals != nodata
        valid &= layer_valid
        df_data[var.name] = vals

        if i == 0:
            target_cell = r * vwidth + c

    df_data["cell_id"] = target_cell
    df_data["trial"] = 1
    df = pd.DataFrame(df_data)
    if drop_nodata:
        df = df[valid].reset_index(drop=True)
    return df


def _write(path, array, *, transform, crs, nodata, tiled):
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype.name,
        nodata=nodata,
        crs=crs,
        transform=transform,
    )
    if tiled:
        profile.update(tiled=True, blockxsize=64, blockysize=64)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


@pytest.fixture
def stack(tmp_path):
    """Target (uint8, tiled, EPSG:3857) + two features on other grids/CRSs."""
    from spatialrisk.dataset import Dataset

    rng = np.random.default_rng(7)
    h, w = 300, 350
    t_transform = from_origin(1_000_000, 2_000_300, 1, 1)
    t_crs = "EPSG:3857"
    target = rng.integers(0, 2, size=(h, w)).astype("uint8")
    target[rng.random((h, w)) < 0.05] = 255  # nodata holes
    tpath = tmp_path / "target.tif"
    _write(tpath, target, transform=t_transform, crs=t_crs, nodata=255, tiled=True)

    # Float feature in EPSG:4326 over the same footprint, coarser grid, NaN holes.
    b = transform_bounds(
        t_crs, "EPSG:4326", *rasterio.transform.array_bounds(h, w, t_transform)
    )
    fh, fw = 120, 140
    feat = rng.random((fh, fw)).astype("float32") * 100
    feat[rng.random((fh, fw)) < 0.05] = np.nan
    feat[rng.random((fh, fw)) < 0.03] = -9999.0
    fpath = tmp_path / "feat.tif"
    _write(
        fpath,
        feat,
        transform=from_bounds(*b, fw, fh),
        crs="EPSG:4326",
        nodata=-9999.0,
        tiled=False,
    )

    # Integer feature, same CRS as the target but shifted/finer grid, no nodata.
    ih, iw = 500, 500
    ifeat = rng.integers(0, 1000, size=(ih, iw)).astype("int32")
    ipath = tmp_path / "ifeat.tif"
    _write(
        ipath,
        ifeat,
        transform=from_origin(1_000_020, 2_000_280, 0.5, 0.5),
        crs=t_crs,
        nodata=None,
        tiled=True,
    )

    ds = Dataset(project=None, name="ds")
    ds.target = _Var("target", tpath)
    ds.features = [_Var("elev", fpath), _Var("dist", ipath)]
    return ds


def _points(n, seed=3):
    """Points in EPSG:3857: mostly inside the target, some outside on every side."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(1_000_000 - 40, 1_000_350 + 40, n)
    ys = rng.uniform(2_000_000 - 40, 2_000_300 + 40, n)
    # A few exact duplicates so tiles/pixels are shared between points.
    xs[:5] = xs[5:10]
    ys[:5] = ys[5:10]
    return gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:3857"
    )


@pytest.mark.parametrize("drop_nodata", [True, False])
def test_windowed_extract_matches_reference(stack, drop_nodata):
    """Windowed reads return the same frame as the full-band oracle."""
    pts = _points(2000)
    got = stack.extract_at_points(pts, drop_nodata=drop_nodata)
    want = _reference_extract(stack, pts, drop_nodata=drop_nodata)
    assert list(got.columns) == ["target", "elev", "dist", "cell_id", "trial"]
    pd.testing.assert_frame_equal(got, want)
    # Sanity: the fixture really exercises every branch.
    full = _reference_extract(stack, pts, drop_nodata=False)
    assert (full["target"] == 255).any()
    assert full["elev"].isna().any()
    assert (full["elev"] == -9999.0).any()
    assert len(want) < len(full) if drop_nodata else len(want) == len(full)


def test_windowed_extract_points_in_other_crs(stack):
    """Points supplied in a CRS that matches none of the layers."""
    pts = _points(500).to_crs("EPSG:4326")
    pd.testing.assert_frame_equal(
        stack.extract_at_points(pts), _reference_extract(stack, pts)
    )


def test_windowed_extract_all_points_out_of_bounds(stack):
    """No touched block at all still yields the oracle's (empty) frame."""
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(-5, 9)], crs="EPSG:3857")
    got = stack.extract_at_points(pts)
    want = _reference_extract(stack, pts)
    assert len(got) == 0
    pd.testing.assert_frame_equal(got, want)
    pd.testing.assert_frame_equal(
        stack.extract_at_points(pts, drop_nodata=False),
        _reference_extract(stack, pts, drop_nodata=False),
    )


def test_windowed_extract_empty_points(stack):
    """An empty point set returns an empty frame with the full schema."""
    pts = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    got = stack.extract_at_points(pts)
    assert len(got) == 0
    assert list(got.columns) == ["target", "elev", "dist", "cell_id", "trial"]


def test_extract_at_points_never_reads_a_full_band():
    """Guard against regressing to ``src.read(1)`` without a window."""
    from spatialrisk.dataset import Dataset

    src = inspect.getsource(Dataset.extract_at_points)
    for m in re.finditer(r"\.read\(\s*1\b([^)]*)\)", src):
        assert "window=" in m.group(
            1
        ), f"unwindowed read in extract_at_points: {m.group(0)}"


@pytest.mark.parametrize("threads", ["1", "4"])
def test_windowed_extract_threaded_matches_reference(stack, monkeypatch, threads):
    """The thread pool must be invisible in the output, whatever its size."""
    monkeypatch.setenv("SPATIAL_RISK_NUM_THREADS", threads)
    pts = _points(3000, seed=11)
    for drop in (True, False):
        pd.testing.assert_frame_equal(
            stack.extract_at_points(pts, drop_nodata=drop),
            _reference_extract(stack, pts, drop_nodata=drop),
        )
