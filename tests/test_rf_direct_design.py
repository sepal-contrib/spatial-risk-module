# tests/test_rf_direct_design.py
"""RFModel.apply(): the float32 chunk builder's charge, output and failures."""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")

from _inference_fixture import (  # noqa: E402
    GOLDEN_DIR,
    build_dataset,
    build_rf,
    read_raster,
    write_tiles,
)

from spatialrisk.mlmodels import design_matrix  # noqa: E402


def _golden_rf():
    """The committed RF golden raster (pre-engine serial loop, n_jobs=1)."""
    return np.load(GOLDEN_DIR / "rf.npy")


def _capture_predict_windowed(monkeypatch):
    """Replace the engine with a recorder of the closure and its kwargs."""
    captured = {}

    def fake_predict_windowed(target, features, block, out, **kw):
        captured.update(kw, block=block)
        return out

    monkeypatch.setattr(
        "spatialrisk.mlmodels.windowed_predict.predict_windowed", fake_predict_windowed
    )
    return captured


def test_rf_apply_charges_one_column_and_predicts_what_patsy_plus_sklearn_did(
    tmp_path, monkeypatch
):
    """A 106-level categorical: charge 1, probabilities identical to the old closure."""
    from patsy import dmatrices
    from patsy.highlevel import build_design_matrices
    from sklearn.ensemble import RandomForestClassifier

    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    df = ds.extract_at_points(None)  # the fixture's training frame
    levels = list(range(0, 106))  # 'pa' is 0/1, inside the domain
    formula = f"target + trial ~ scale(alt) + C(pa, levels={levels})"
    _, x = dmatrices(formula, df, NA_action="drop")
    assert len(x.design_info.column_names) == 107
    model._x_design_info = x.design_info
    model._ml_model = RandomForestClassifier(
        n_estimators=10, max_depth=8, n_jobs=1, random_state=0
    ).fit(np.asarray(x, dtype=np.float32), df["target"].to_numpy())

    captured = _capture_predict_windowed(monkeypatch)
    model.apply(tmp_path / "out" / "r.tif", ds, ds.mask_path, 0)
    assert captured["n_design_cols"] == 1

    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "alt": rng.integers(1, 200, 5000).astype(float),
            "pa": rng.integers(0, 2, 5000).astype(float),
            "unused": rng.random(5000) * 10,
        }
    )
    (whole,) = build_design_matrices([x.design_info], frame, NA_action="drop")
    expected = model._ml_model.predict_proba(np.asarray(whole))[:, 1]
    assert np.array_equal(captured["block"](frame, {}), expected)


@pytest.mark.parametrize("workers", [1, 2])
def test_rf_matches_its_golden_across_many_small_chunks(tmp_path, monkeypatch, workers):
    """1000-row chunks, stripe tails included, reproduce the golden raster exactly."""
    monkeypatch.setattr(design_matrix, "_chunk_rows", lambda n_columns: 1000)
    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    seen = []
    real = model._ml_model.predict_proba

    def spy(x):
        seen.append(x.shape[0])
        return real(x)

    model._ml_model.predict_proba = spy
    out = tmp_path / "out" / "rf.tif"
    model.apply(out, ds, ds.mask_path, 0, workers=workers)
    arr, _ = read_raster(out)
    np.testing.assert_array_equal(arr, _golden_rf())
    assert max(seen) <= 1000
    assert len(seen) > 3


def test_rf_matches_its_golden_with_the_design_rebuilt_from_samples(tmp_path):
    """A model loaded from its pickle rebuilds its design info from the samples CSV."""
    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    model._x_design_info = None
    out = tmp_path / "out" / "rf.tif"
    model.apply(out, ds, ds.mask_path, 0)
    arr, _ = read_raster(out)
    np.testing.assert_array_equal(arr, _golden_rf())


def test_rf_logs_its_design_and_charges_one_working_col(tmp_path, caplog):
    """The plan line says 1 working col; the RF design line carries the real counts."""
    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        model.apply(tmp_path / "out" / "rf.tif", ds, ds.mask_path, 0)
    messages = [r.getMessage() for r in caplog.records]
    plan = [m for m in messages if "rows/stripe" in m]
    assert plan and re.search(r"\b1 working cols\b", plan[0]), plan
    assert (
        "RF design: 1 one-hot term(s) by lookup, 1 materialised col(s), "
        "3 col(s) in 262144-row float32 chunks"
    ) in messages


def test_an_unseen_categorical_value_fails_the_run_cleanly(tmp_path):
    """A 'pa' pixel outside levels=[0, 1] raises, and no output or .part is left."""
    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    pa_var = next(v for v in ds.features if v.name == "pa")
    with rasterio.open(pa_var.path) as src:
        pa = src.read(1)
    pa[300, 5] = 2  # stripe 1, outside every nodata band and the mask
    pa_var.path = write_tiles(Path(tmp_path) / "pa_bad.tif", pa, 255)
    out = tmp_path / "out" / "rf.tif"
    with pytest.raises(ValueError, match=r"C\(pa, levels=\[0, 1\]\).*\[2\.0\]"):
        model.apply(out, ds, ds.mask_path, 0)
    assert not out.exists()
    assert not out.with_name(out.stem + ".part.tif").exists()


def test_a_value_too_large_for_float32_is_rejected_as_before(tmp_path, monkeypatch):
    """scale(alt) past float32's range becomes inf, which sklearn still rejects."""
    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    captured = _capture_predict_windowed(monkeypatch)
    model.apply(tmp_path / "out" / "rf.tif", ds, ds.mask_path, 0)
    frame = pd.DataFrame({"alt": [1.0, 1e41], "pa": [0.0, 1.0], "unused": [1.0, 1.0]})
    with pytest.raises(ValueError, match="infinity"):
        captured["block"](frame, {})


def test_rf_model_no_longer_builds_patsys_whole_matrix():
    """Source guard: the RF closure must go through the chunk builder."""
    import spatialrisk.mlmodels.rf_model as rf_model

    source = Path(rf_model.__file__).read_text()
    assert "build_design_matrices" not in source
    assert "compile_design_builder" in source
