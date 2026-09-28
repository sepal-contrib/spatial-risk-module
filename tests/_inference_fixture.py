"""Deterministic synthetic dataset + fitted models for the inference engine tests.

Not a test module (leading underscore) so pytest does not collect it; imported
by test_inference_golden.py and test_windowed_predict.py. Seeded so the same
rasters and the same fitted coefficients come out on every run, which is what
lets the golden .npy files be regenerated and compared.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

GOLDEN_DIR = Path(__file__).parent / "data" / "inference_golden"
H, W = 700, 300  # 3 stripes of 256 rows; the last one is 188 rows tall
ORIGIN = (500000.0, 5000000.0)


def write_tiles(path, data, nodata, res=30.0):
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
        transform=from_origin(ORIGIN[0], ORIGIN[1], res, res),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data, 1)
    return path


class Var:
    def __init__(self, name, path, raster_type="continuous"):
        self.name, self.path, self.raster_type = name, path, raster_type


class Dataset:
    name, year = "calibration", 2020

    def __init__(self, tmp_path, rng):
        target = (rng.random((H, W)) < 0.5).astype("uint8")
        target[0:5, :] = 255  # nodata rows in the target itself
        alt = rng.integers(1, 200, size=(H, W)).astype("uint8")
        alt[10:20, :] = 255  # nodata band inside stripe 0
        alt[300:310, 40:60] = 255  # nodata patch inside stripe 1
        pa = (rng.random((H, W)) < 0.3).astype("uint8")  # categorical 0/1
        unused = rng.random((H, W)).astype("float32") * 10
        unused[600:650, :] = -9999.0  # nodata in a feature the formula ignores
        mask = np.ones((H, W), dtype="uint8")
        mask[100:130, :] = 0  # suppressed rows
        mask[400:420, 100:200] = 255  # mask nodata
        # rho at half the cell size (15 m): 1400 x 600, smooth gradient
        yy, xx = np.mgrid[0 : 2 * H, 0 : 2 * W]
        rho = ((yy / (2 * H)) - 0.5 + (xx / (2 * W)) * 0.2).astype("float32")

        self.target = Var("target", write_tiles(tmp_path / "target.tif", target, 255))
        self.features = [
            Var("alt", write_tiles(tmp_path / "alt.tif", alt, 255)),
            Var("pa", write_tiles(tmp_path / "pa.tif", pa, 255), "categorical"),
            Var("unused", write_tiles(tmp_path / "unused.tif", unused, -9999.0)),
        ]
        self.mask_path = write_tiles(tmp_path / "mask.tif", mask, 255)
        self.rho_path = write_tiles(tmp_path / "rho.tif", rho, None, res=15.0)
        self._rng = rng

    def extract_at_points(self, points, *, drop_nodata=True):
        rng = np.random.default_rng(11)
        n = 120
        alt = rng.integers(1, 200, n).astype(float)
        pa = rng.integers(0, 2, n)
        return pd.DataFrame(
            {
                "target": (alt / 200 + pa * 0.3 + rng.random(n) * 0.5 > 0.7).astype(
                    int
                ),
                "trial": 1,
                "alt": alt,
                "pa": pa,
                "unused": rng.random(n) * 10,
                "cell_id": np.arange(n),
            }
        )


class Sample:
    name = "s1"

    def load_points(self):
        return object()


FORMULA = "target + trial ~ scale(alt) + C(pa, levels=[0, 1])"


def build_dataset(tmp_path):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Dataset(tmp_path, np.random.default_rng(7))


def _finish(model, ds, tmp_path):
    model.dataset = ds
    model.sample = Sample()
    model.formula = FORMULA
    model.fit(folder=Path(tmp_path) / "out")
    model._register_prediction = lambda *a, **k: None
    return model


def build_glm(tmp_path, ds):
    from spatialrisk.mlmodels.glm_model import GLMModel

    return _finish(GLMModel(name="g", random_seed=7), ds, tmp_path)


def build_rf(tmp_path, ds):
    from spatialrisk.mlmodels.rf_model import RFModel

    model = _finish(RFModel(name="r", n_trees=20, random_seed=7), ds, tmp_path)
    model._ml_model.n_jobs = 1  # deterministic vote accumulation for the golden
    return model


def build_icar(tmp_path, ds):
    """An iCAR model with hand-set betas and the fixture rho: no MCMC needed.

    ICARModel.apply only needs ``_ml_model["betas"]``, ``_x_design_info`` and
    ``rho_path``; the design info is rebuilt from the formula and a sample frame
    the same way apply() would from samples_path.
    """
    from patsy import dmatrices

    from spatialrisk.mlmodels.icar_model import ICARModel

    model = ICARModel(name="i")
    model.dataset = ds
    model.sample = Sample()
    model.formula = FORMULA
    _, x = dmatrices(FORMULA, ds.extract_at_points(None), NA_action="drop")
    model._x_design_info = x.design_info
    # patsy column order (verified): Intercept, C(pa, levels=[0, 1])[T.1], scale(alt)
    model._ml_model = {
        "betas": np.array([-1.0, 0.8, 0.5])
    }  # intercept, pa[1], scale(alt)
    model.rho_path = ds.rho_path
    model.trained = True
    model._register_prediction = lambda *a, **k: None
    return model


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), {
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }
