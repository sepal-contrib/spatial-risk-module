"""The ML predictors stream tile-aligned stripes with BLAS/OpenMP on one thread.

The block loop the three ``apply`` bodies used to share now lives in
:mod:`spatialrisk.mlmodels.windowed_predict`, which also owns the stripe
plan, the thread pinning and the writing. So the wiring is checked once for
real on the cheapest model (GLM) -- default stripes are ``PREDICT_BAND_ROWS``
tall, the run is pinned to one math thread, and the worker count changes no
pixel -- and by a source guard on all three predictors: each must call
``predict_windowed`` and must contain no loop of its own, i.e. no
``makeblock``, no ``single_thread_math`` and no ``dst.write``.
"""

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


def test_glm_apply_streams_tile_aligned_stripes_under_single_thread_math(
    tmp_path, monkeypatch
):
    """Stripes default to PREDICT_BAND_ROWS and the run is pinned to one BLAS thread."""
    from threadpoolctl import threadpool_info

    from spatialrisk.mlmodels import windowed_predict as wp
    from spatialrisk.mlmodels.linear_predictor import LinearPredictor
    from spatialrisk.parallel import PREDICT_BAND_ROWS

    rng = np.random.default_rng(7)
    model = _glm(tmp_path, rng)

    seen = {}
    real_stripes = wp._stripes

    def spy_stripes(target_path, rows):
        seen["rows"] = rows
        return real_stripes(target_path, rows)

    monkeypatch.setattr(wp, "_stripes", spy_stripes)
    blas_threads = []
    real_eta = LinearPredictor.eta

    def spy_eta(self, block_df):
        blas_threads.extend(i["num_threads"] for i in threadpool_info())
        return real_eta(self, block_df)

    monkeypatch.setattr(LinearPredictor, "eta", spy_eta)

    pred = model.apply(output_file=tmp_path / "out" / "pred.tif", workers=1)

    assert seen["rows"] == PREDICT_BAND_ROWS
    assert blas_threads  # the spy must have fired
    assert set(blas_threads) == {1}
    with rasterio.open(pred) as src:
        out = src.read(1)
    assert out.shape == (600, 300)
    assert (out[10:20] == 0).all()  # nodata rows stay nodata
    assert (out[30:] > 0).all()


def test_glm_apply_output_is_identical_for_any_worker_count(tmp_path):
    """Worker count must not change a single pixel."""
    rng = np.random.default_rng(11)
    model = _glm(tmp_path, rng, h=700, w=257)
    out_a = model.apply(output_file=tmp_path / "out" / "a.tif", workers=1)
    with rasterio.open(out_a) as src:
        a_arr = src.read(1)

    out_b = model.apply(output_file=tmp_path / "out" / "b.tif", workers=4)
    with rasterio.open(out_b) as src:
        b_arr = src.read(1)
    np.testing.assert_array_equal(a_arr, b_arr)


@pytest.mark.parametrize("fname", PREDICTORS)
def test_every_predictor_delegates_to_the_stripe_engine(fname):
    """Source guard: apply() calls predict_windowed and owns no block loop."""
    src = (Path(mlmodels.__file__).parent / fname).read_text()
    assert "predict_windowed(" in src, fname
    assert "makeblock(" not in src, fname
    assert "single_thread_math" not in src, fname  # the engine pins, not the model
    assert "dst.write(" not in src, fname
