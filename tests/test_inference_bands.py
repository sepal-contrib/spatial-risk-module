"""The ML predictors stream tile-aligned bands with BLAS/OpenMP on one thread.

The three ``apply`` bodies share the same block loop, so the wiring is
checked once for real on the cheapest model (GLM) and by a source guard on
all three: each must ask forestatrisk for ``PREDICT_BAND_ROWS`` bands and
run its loop inside ``single_thread_math``.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

import spatialrisk.mlmodels as mlmodels  # noqa: E402

PREDICTORS = ("glm_model.py", "rf_model.py", "icar_model.py")


def _tiles(path, data, nodata):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype.name,
        crs="EPSG:32631",
        nodata=nodata,
        transform=from_origin(500000.0, 5000000.0, 30.0, 30.0),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data, 1)
    return path


def _glm(tmp_path, rng, h=600, w=300):
    from spatialrisk.mlmodels.glm_model import GLMModel

    target = (rng.random((h, w)) < 0.5).astype("uint8")
    alt = rng.integers(1, 200, size=(h, w)).astype("uint8")
    alt[10:20, :] = 255
    tpath = _tiles(tmp_path / "target.tif", target, nodata=255)
    apath = _tiles(tmp_path / "alt.tif", alt, nodata=255)

    class _Var:
        def __init__(self, name, path):
            self.name, self.path, self.raster_type = name, path, "continuous"

    class _Dataset:
        name, year = "calibration", 2020
        target = _Var("target", tpath)
        features = [_Var("altitude", apath)]

        def extract_at_points(self, points, *, drop_nodata=True):
            n = 60
            x = rng.integers(1, 200, n).astype(float)
            return pd.DataFrame(
                {
                    "target": (x / 200 + rng.random(n) * 0.5 > 0.6).astype(int),
                    "trial": 1,
                    "altitude": x,
                    "cell_id": np.arange(n),
                }
            )

    class _Sample:
        name = "s1"

        def load_points(self):
            return object()

    model = GLMModel(name="m")
    model.dataset = _Dataset()
    model.sample = _Sample()
    model.formula = "target + trial ~ altitude"
    model.fit(folder=tmp_path / "out")
    return model


def test_glm_apply_streams_tile_aligned_bands_under_single_thread_math(
    tmp_path, monkeypatch
):
    """Every band is PREDICT_BAND_ROWS tall and predicted with BLAS on one thread."""
    import forestatrisk as far
    from threadpoolctl import threadpool_info

    from spatialrisk.parallel import PREDICT_BAND_ROWS

    rng = np.random.default_rng(7)
    model = _glm(tmp_path, rng)

    seen = {}
    real_makeblock = far.misc.makeblock

    def spy_makeblock(rasterfile, blk_rows=128):
        seen["blk_rows"] = blk_rows
        return real_makeblock(rasterfile, blk_rows)

    monkeypatch.setattr(far.misc, "makeblock", spy_makeblock)
    blas_threads = []
    real_predict = model._ml_model.predict_proba

    def spy_predict(x):
        blas_threads.extend(i["num_threads"] for i in threadpool_info())
        return real_predict(x)

    monkeypatch.setattr(model._ml_model, "predict_proba", spy_predict)

    pred = model.apply(output_file=tmp_path / "out" / "pred.tif")

    assert seen["blk_rows"] == PREDICT_BAND_ROWS
    if blas_threads:
        assert set(blas_threads) == {1}
    with rasterio.open(pred) as src:
        out = src.read(1)
    assert out.shape == (600, 300)
    assert (out[10:20] == 0).all()  # nodata rows stay nodata
    assert (out[30:] > 0).all()


def test_glm_apply_output_is_identical_to_the_128_row_unpinned_loop(tmp_path):
    """Band height and BLAS thread count must not change a single pixel."""
    import forestatrisk as far
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(11)
    model = _glm(tmp_path, rng, h=700, w=257)
    new = model.apply(output_file=tmp_path / "out" / "new.tif")
    with rasterio.open(new) as src:
        new_arr = src.read(1)

    real_makeblock = far.misc.makeblock
    try:
        far.misc.makeblock = lambda rasterfile, blk_rows=128: real_makeblock(
            rasterfile, 128
        )
        with threadpool_limits(limits=None):
            old = model.apply(output_file=tmp_path / "out" / "old.tif")
    finally:
        far.misc.makeblock = real_makeblock
    with rasterio.open(old) as src:
        old_arr = src.read(1)
    np.testing.assert_array_equal(new_arr, old_arr)


@pytest.mark.parametrize("fname", PREDICTORS)
def test_every_predictor_is_wired_to_the_shared_band_policy(fname):
    """Source guard: PREDICT_BAND_ROWS bands, loop under single_thread_math."""
    src = (Path(mlmodels.__file__).parent / fname).read_text()
    calls = []
    for m in re.finditer(r"makeblock\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        calls.append(src[m.start() : i])
    assert calls, fname
    assert all("blk_rows=PREDICT_BAND_ROWS" in c for c in calls), calls
    assert "with single_thread_math()" in src, fname
