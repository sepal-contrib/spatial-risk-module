"""ICARModel.fit() must hand its own settings to forestatrisk.

Two settings used to stop at the model:

* ``csize`` went to ``far.cellneigh`` positionally, where the second slot is
  ``region`` -- so the neighbourhood graph was always built on forestatrisk's
  default 10 km grid while ``compute_cell_indices`` and ``interpolate_rho``
  used the model's value. Any csize other than 10 put the observations, the
  graph and the rho raster on three different grids.
* ``beta_start`` was a model field and a GUI parameter that never reached the
  sampler, which then always started the coefficients at 0.
"""

import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from spatialrisk.mlmodels.icar_model import ICARModel


def _write_raster(path):
    """8 x 8 pixels of 30 m in a metric CRS: a 240 m square.

    ``far.cellneigh`` reads ``csize`` in kilometres against the raster's own
    units, so a 0.06 km cell splits this square into a 4 x 4 grid.
    """
    data = np.zeros((8, 8), dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype=data.dtype,
        crs="EPSG:32631",
        transform=from_origin(500000.0, 5000000.0, 30.0, 30.0),
        nodata=255,
    ) as dst:
        dst.write(data, 1)
    return path


class _Var:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.raster_type = "continuous"


class _StubDataset:
    def __init__(self, raster):
        self.name = "calibration"
        self.year = 2020
        self.target = _Var("target", raster)
        self.features = [_Var("altitude", raster)]

    def extract_at_points(self, points, *, drop_nodata=True):
        # cell_id is a flat pixel index (row * 8 + col); these four points sit
        # in the four corners of the 240 m square.
        return pd.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "trial": [1, 1, 1, 1],
                "altitude": [1.0, 2.0, 3.0, 4.0],
                "cell_id": [0, 7, 56, 63],
            }
        )


class _StubSample:
    name = "s1"

    def load_points(self):
        return object()


@pytest.fixture()
def captured_mcmc(monkeypatch):
    """Record what fit() hands the sampler; answer with right-sized posteriors."""
    import spatialrisk.mlmodels.icar_model as icar_module

    captured = {}

    def _stub(formula, data, n_neighbors, neighbors, **kwargs):
        captured.update(data=data, n_neighbors=n_neighbors, **kwargs)
        return {
            "betas": np.array([0.1, 0.2]),
            "rho": np.zeros(len(n_neighbors)),
            "Vrho": 1.0,
            "deviance": 1.0,
            "worker_pid": os.getpid(),
        }

    monkeypatch.setattr(icar_module, "run_icar_mcmc", _stub)
    return captured


def _model(tmp_path, **fields):
    model = ICARModel(name="m", **fields)
    model.dataset = _StubDataset(_write_raster(tmp_path / "ref.tif"))
    model.sample = _StubSample()
    model.formula = "target + trial ~ altitude"
    return model


def test_neighbourhood_graph_uses_the_models_cell_size(tmp_path, captured_mcmc):
    """csize=0.06 km -> a 4 x 4 graph, and every observation's cell is in it."""
    model = _model(tmp_path, csize=0.06, csize_interpolate=0.03)

    # interpolate_rho runs for real: it reshapes rho onto the csize grid, so it
    # only succeeds when the graph the sampler saw is that same grid.
    model.fit(folder=tmp_path / "out")

    n_neighbors = captured_mcmc["n_neighbors"]
    assert len(n_neighbors) == 16
    # Corner cells of a 4 x 4 king's-move grid have 3 neighbours, inner ones 8.
    assert n_neighbors[0] == 3 and n_neighbors[5] == 8
    assert captured_mcmc["data"]["cell"].tolist() == [0, 3, 12, 15]
    with rasterio.open(model.rho_path) as rho:
        assert rho.shape == (8, 8)  # the 240 m square resampled to 30 m


def test_fit_hands_beta_start_to_the_sampler(tmp_path, captured_mcmc):
    """The GUI's beta_start must arrive at run_icar_mcmc unchanged."""
    model = _model(tmp_path, beta_start=0.5)

    model.fit(folder=tmp_path / "out")

    assert captured_mcmc["beta_start"] == 0.5
