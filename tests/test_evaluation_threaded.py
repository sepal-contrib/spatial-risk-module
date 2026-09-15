"""Regression tests for the band-wise, threaded evaluation tallies.

``compute_validation`` and ``defrate_per_cat`` used to read one coarse square
(or one row block) at a time and count risk categories with a 65535-category
``pd.Categorical``. The reference functions below are those loops copied
verbatim; the rewritten code must reproduce their frames exactly, at any
thread count, including a nodata value outside the category set, a remainder
column, a remainder row band and a zero-forest region that gets dropped.
"""

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
gdal = pytest.importorskip("osgeo.gdal")
from rasterio.transform import from_origin  # noqa: E402

from spatialrisk.evaluation import compute_validation, make_square  # noqa: E402
from spatialrisk.rmj.deforrate import defrate_per_cat  # noqa: E402


def _reference_validation_points(
    defor_file, forest_file, riskmap_file, tab_file_defor, time_interval, csize
):
    """The original per-square loop of ``compute_validation`` (tally only)."""
    defor_ds = gdal.Open(str(defor_file))
    defor_band = defor_ds.GetRasterBand(1)
    forest_ds = gdal.Open(str(forest_file))
    forest_band = forest_ds.GetRasterBand(1)
    risk_ds = gdal.Open(str(riskmap_file))
    risk_band = risk_ds.GetRasterBand(1)
    defor_dens_per_cat = pd.read_csv(tab_file_defor)
    cat = defor_dens_per_cat["cat"].values
    defor_dens_period = defor_dens_per_cat["defor_dens"].values * time_interval
    gt = defor_ds.GetGeoTransform()
    pix_area = gt[1] * (-gt[5])
    nsquare, nsquare_x, _, x, y, nx, ny = make_square(defor_file, csize)
    df = pd.DataFrame(
        {
            "cell": list(range(nsquare)),
            "nfor_obs": 0,
            "ndefor_obs": 0,
            "nfor_obs_ha": 0.0,
            "ndefor_obs_ha": 0.0,
            "ndefor_pred_ha": 0.0,
        }
    )
    for s in range(nsquare):
        px, py = s % nsquare_x, s // nsquare_x
        defor_data = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        forest_data = forest_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        defor_mask = defor_data == 1
        forest_start = (forest_data == 1) | defor_mask
        df.loc[s, "nfor_obs"] = int(forest_start.sum())
        df.loc[s, "ndefor_obs"] = int(defor_mask.sum())
        risk_data = risk_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        risk_cat = pd.Categorical(risk_data.flatten(), categories=cat)
        risk_count = risk_cat.value_counts().values
        df.loc[s, "ndefor_pred_ha"] = np.nansum(risk_count * defor_dens_period)
    del defor_ds, forest_ds, risk_ds
    df = df[df["nfor_obs"] > 0]
    df["nfor_obs_ha"] = df["nfor_obs"] * pix_area / 10000
    df["ndefor_obs_ha"] = df["ndefor_obs"] * pix_area / 10000
    return df


def _reference_defrate_counts(defor_file, forest_file, riskmap_file, blk_rows=128):
    """The original per-block Categorical tally of ``defrate_per_cat``."""
    from riskmapjnr.misc import makeblock

    defor_ds, forest_ds, cat_ds = (
        gdal.Open(str(defor_file)),
        gdal.Open(str(forest_file)),
        gdal.Open(str(riskmap_file)),
    )
    defor_band, forest_band, cat_band = (
        defor_ds.GetRasterBand(1),
        forest_ds.GetRasterBand(1),
        cat_ds.GetRasterBand(1),
    )
    nblock, nblock_x, _, x, y, nx, ny = makeblock(str(defor_file), blk_rows=blk_rows)[
        :7
    ]
    cat = [c + 1 for c in range(65535)]
    df = pd.DataFrame({"cat": cat, "nfor": 0, "ndefor": 0})
    for b in range(nblock):
        px, py = b % nblock_x, b // nblock_x
        defor_arr = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        forest_arr = forest_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        cat_arr = cat_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        defor_mask = defor_arr == 1
        forest_start = (forest_arr == 1) | defor_mask
        cat_for = pd.Categorical(cat_arr[forest_start].flatten(), categories=cat)
        df["nfor"] += cat_for.value_counts().values
        cat_defor = pd.Categorical(cat_arr[defor_mask].flatten(), categories=cat)
        df["ndefor"] += cat_defor.value_counts().values
    return df


def _write(path, array, dtype, nodata=0, pixel=30.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:3857",
        nodata=nodata,
        transform=from_origin(0, array.shape[0] * pixel, pixel, pixel),
        tiled=True,
        blockxsize=64,
        blockysize=64,
    ) as dst:
        dst.write(array.astype(dtype), 1)
    return str(path)


@pytest.fixture(params=["uint16", "int32"])
def layers(tmp_path, request):
    """Irregular 410 x 530 stack: remainders on both axes, holes, a dead zone."""
    rng = np.random.default_rng(5)
    h, w = 410, 530
    forest = (rng.random((h, w)) < 0.7).astype("int64")
    defor = ((rng.random((h, w)) < 0.05) & (forest == 1)).astype("int64")
    defor[rng.random((h, w)) < 0.01] = 1  # deforested where forest says 0
    forest[300:, :] = 0  # bottom band: no forest ...
    defor[300:, :] = 0  # ... and no loss -> cells there must be dropped
    risk = rng.integers(1, 40, size=(h, w)).astype("int64")
    risk[rng.random((h, w)) < 0.2] = 0  # nodata, NOT in the category table
    risk[rng.random((h, w)) < 0.02] = 65535  # top category present
    risk[rng.random((h, w)) < 0.02] = 999  # a category absent from the CSV
    dt = request.param
    cats = np.array([1, 2, 3, 5, 8, 13, 21, 34, 65535])
    tab = tmp_path / "defrate.csv"
    pd.DataFrame(
        {"cat": cats, "defor_dens": np.linspace(0.0001, 0.0009, len(cats))}
    ).to_csv(tab, index=False)
    return dict(
        defor_file=_write(tmp_path / "defor.tif", defor, dt),
        forest_file=_write(tmp_path / "forest.tif", forest, dt),
        riskmap_file=_write(tmp_path / "risk.tif", risk, dt),
        tab_file_defor=str(tab),
        time_interval=5,
    )


@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("csize", [64, 100, 300])
def test_compute_validation_matches_square_loop(layers, monkeypatch, threads, csize):
    """Band-wise tallies equal the per-square oracle at every cell size."""
    monkeypatch.setenv("SPATIAL_RISK_NUM_THREADS", threads)
    got = compute_validation(**layers, csize_coarse_grid=csize).plot_data.points
    want = _reference_validation_points(csize=csize, **layers)
    assert len(want) > 0
    assert (want["cell"].max() >= 5) if csize == 64 else True
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        want.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,  # frozen numerics: the CSV bytes must not move
    )
    # The frozen numerics must still see the same dtypes as before.
    assert list(got.columns) == list(want.columns)
    assert got["nfor_obs"].dtype.kind == "i" and got["ndefor_obs"].dtype.kind == "i"
    assert got["ndefor_pred_ha"].dtype.kind == "f"


@pytest.mark.parametrize("threads", ["1", "4"])
def test_defrate_per_cat_matches_block_loop(layers, monkeypatch, threads):
    """Per-category counts equal the per-block Categorical oracle."""
    monkeypatch.setenv("SPATIAL_RISK_NUM_THREADS", threads)
    got = defrate_per_cat(
        layers["defor_file"], layers["forest_file"], layers["riskmap_file"], 5.0
    )
    want = _reference_defrate_counts(
        layers["defor_file"], layers["forest_file"], layers["riskmap_file"]
    )
    assert got["ndefor"].sum() > 0 and got.loc[got["cat"] == 65535, "nfor"].item() > 0
    pd.testing.assert_series_equal(
        got["nfor"], want["nfor"], check_dtype=False, check_exact=True
    )
    pd.testing.assert_series_equal(
        got["ndefor"], want["ndefor"], check_dtype=False, check_exact=True
    )
    assert list(got.columns[:3]) == ["cat", "nfor", "ndefor"]
