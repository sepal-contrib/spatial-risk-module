"""Every prediction-shaped raster the package writes comes out canonical.

Covers the three writer kinds: a rasterio profile (ML predictors), GDAL
creation options (MW local rate), and the re-encode step behind the two
riskmapjnr wrappers, whose upstream write is stubbed to produce the LZW row
strips the real library produces.
"""

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from spatialrisk import raster_profile as rp  # noqa: E402


def _assert_canonical(path, dtype):
    with rasterio.open(path) as src:
        assert src.is_tiled and src.block_shapes[0] == (256, 256), path
        assert src.profile["compress"] == rp.compression("gdal"), path
        assert src.dtypes[0] == dtype


def _strips(path, data, nodata=0):
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
        tiled=False,
        compress="lzw",
        predictor=2,
    ) as dst:
        dst.write(data, 1)
    return path


def test_set_defor_cat_zero_wrapper_reencodes_upstream_strips(tmp_path, monkeypatch):
    """The MW probability map comes back tiled with its values intact."""
    import riskmapjnr

    from spatialrisk.rmj import set_defor_cat_zero

    data = np.arange(300 * 300, dtype="uint16").reshape(300, 300) % 65535 + 1

    def fake_upstream(**kw):
        _strips(kw["ldefrate_with_zero_file"], data)

    monkeypatch.setattr(riskmapjnr, "set_defor_cat_zero", fake_upstream)
    out = tmp_path / "prob_mw.tif"
    set_defor_cat_zero(
        ldefrate_file="a.tif",
        forest_edge_file="b.tif",
        dist_thresh=1.0,
        output_file=out,
    )
    _assert_canonical(out, "uint16")
    with rasterio.open(out) as src:
        np.testing.assert_array_equal(src.read(1), data)


def test_vulnerability_map_wrapper_reencodes_upstream_strips(tmp_path, monkeypatch):
    """The JNR vulnerability map comes back tiled."""
    import riskmapjnr

    from spatialrisk.rmj import vulnerability_map

    data = np.full((300, 300), 1001, dtype="uint16")

    def fake_upstream(**kw):
        _strips(kw["output_file"], data)

    monkeypatch.setattr(riskmapjnr.benchmark, "vulnerability_map", fake_upstream)
    out = tmp_path / "vuln.tif"
    vulnerability_map(
        forest_file="f.tif",
        forest_edge_file="e.tif",
        dist_bins=[1, 2],
        subj_file="s.tif",
        output_file=out,
    )
    _assert_canonical(out, "uint16")


def test_wrappers_tolerate_an_upstream_that_wrote_nothing(tmp_path, monkeypatch):
    """The stubbed-upstream tests elsewhere rely on this: no file, no error."""
    import riskmapjnr

    from spatialrisk.rmj import set_defor_cat_zero

    monkeypatch.setattr(riskmapjnr, "set_defor_cat_zero", lambda **kw: None)
    set_defor_cat_zero(
        ldefrate_file="a.tif",
        forest_edge_file="b.tif",
        dist_thresh=1.0,
        output_file=tmp_path / "missing.tif",
    )
    assert not (tmp_path / "missing.tif").exists()


def test_local_defor_rate_writes_canonical_tiles(tmp_path):
    """The GDAL-written MW rate raster uses the shared creation options."""
    from spatialrisk.rmj.deforrate import local_defor_rate

    rng = np.random.default_rng(0)
    forest = (rng.random((300, 300)) < 0.7).astype("uint8")
    defor = ((rng.random((300, 300)) < 0.1) & (forest == 1)).astype("uint8")
    fpath = _strips(tmp_path / "forest.tif", forest, nodata=255)
    dpath = _strips(tmp_path / "defor.tif", defor, nodata=255)
    out = tmp_path / "ldefrate.tif"
    local_defor_rate(
        defor_file=dpath,
        forest_file=fpath,
        ldefrate_file=out,
        win_size=5,
        time_interval=5,
        blk_rows=128,
    )
    _assert_canonical(out, "uint16")


def test_glm_prediction_raster_is_canonical(tmp_path):
    """A real (tiny) GLM fit + apply: the prediction ignores the target layout."""
    from spatialrisk.mlmodels.glm_model import GLMModel

    rng = np.random.default_rng(3)
    target = (rng.random((40, 40)) < 0.5).astype("uint8")
    alt = rng.integers(1, 200, size=(40, 40)).astype("uint8")
    tpath = _strips(tmp_path / "target.tif", target, nodata=255)
    apath = _strips(tmp_path / "alt.tif", alt, nodata=255)

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
    pred = tmp_path / "out" / "pred.tif"
    model.apply(output_file=pred)
    _assert_canonical(pred, "uint16")
