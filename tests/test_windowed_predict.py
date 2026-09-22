# tests/test_windowed_predict.py
"""The shared stripe engine behind GLM/RF/iCAR apply()."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from _inference_fixture import (  # noqa: E402
    GOLDEN_DIR,
    build_dataset,
    build_glm,
    build_icar,
    read_raster,
)

from spatialrisk.gdal_env import INFERENCE_LEDGER  # noqa: E402


def _design_info(model):
    """The model's patsy design info, rebuilt from its samples CSV as apply() does.

    ``GLMModel.fit`` deliberately does not keep the design info (it is not
    picklable with the model), so every caller reconstructs it the same way.
    """
    import pandas as pd
    from patsy import dmatrices

    if model._x_design_info is None:
        df = pd.read_csv(model.samples_path).dropna()
        _, x = dmatrices(model.formula, df, NA_action="drop")
        model._x_design_info = x.design_info
    return model._x_design_info


def _glm_closure(model):
    """A ``predict_block`` closure over a fitted GLM, as Task 5's apply() will build."""
    from patsy.highlevel import build_design_matrices

    design_info = _design_info(model)

    def predict_block(block_df, extras):
        (x,) = build_design_matrices([design_info], block_df, NA_action="drop")
        return model._ml_model.predict_proba(np.asarray(x))[:, 1]

    return predict_block


def _run_glm(tmp_path, ds, model, out, **kw):
    """Run the engine over the fixture dataset with the GLM closure and the mask."""
    from spatialrisk.mlmodels.windowed_predict import predict_windowed

    return predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        _glm_closure(model),
        out,
        mask=ds.mask_path,
        mask_values=(0,),
        n_design_cols=len(_design_info(model).column_names),
        **kw,
    )


@pytest.fixture(scope="module")
def golden():
    """The committed golden arrays and their metadata, keyed by model kind."""
    if not (GOLDEN_DIR / "meta.json").exists():
        pytest.skip("goldens missing: run tests/test_inference_golden.py")
    meta = json.loads((GOLDEN_DIR / "meta.json").read_text())
    return {k: (np.load(GOLDEN_DIR / f"{k}.npy"), meta[k]) for k in meta}


def test_serial_engine_reproduces_the_glm_golden(tmp_path, golden):
    """The serial engine writes the GLM serial loop's raster, pixel for pixel."""
    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = _run_glm(tmp_path, ds, model, tmp_path / "p.tif", workers=1)
    arr, meta = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])
    assert meta == golden["glm"][1]


def test_serial_engine_reproduces_the_icar_golden(tmp_path, golden):
    """Extra layers and a bounds-read mask reproduce the iCAR serial loop."""
    from patsy.highlevel import build_design_matrices

    from spatialrisk.mlmodels.windowed_predict import ExtraLayer, predict_windowed

    ds = build_dataset(tmp_path)
    model = build_icar(tmp_path, ds)
    betas = model._ml_model["betas"]

    def predict_block(block_df, extras):
        (x,) = build_design_matrices([model._x_design_info], block_df, NA_action="drop")
        x = np.asarray(x)
        lp = x @ betas[: x.shape[1]] + extras["rho"]
        return 1.0 / (1.0 + np.exp(-lp))

    out = predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        predict_block,
        tmp_path / "i.tif",
        mask=ds.mask_path,
        mask_by_bounds=True,
        extra_layers={"rho": ExtraLayer(ds.rho_path, "bilinear")},
        workers=1,
        n_design_cols=3,
    )
    arr, meta = read_raster(out)
    np.testing.assert_array_equal(arr, golden["icar"][0])
    assert meta == golden["icar"][1]


def test_no_mask_predicts_the_suppressed_rows_too(tmp_path, golden):
    """Without a mask only feature nodata suppresses pixels; the rest is unchanged."""
    from spatialrisk.mlmodels.windowed_predict import predict_windowed

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        _glm_closure(model),
        tmp_path / "nomask.tif",
        workers=1,
        n_design_cols=3,
    )
    arr, _ = read_raster(out)
    g = golden["glm"][0]
    assert (arr[100:130] > 0).all()  # mask-suppressed in the golden, predicted here
    np.testing.assert_array_equal(arr[200:250], g[200:250])  # untouched elsewhere
    assert (arr[10:20] == 0).all()  # feature nodata still wins


def test_explicit_short_stripes_change_nothing(tmp_path, golden):
    """A stripe height that is not a multiple of the tile height changes no pixel."""
    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = _run_glm(
        tmp_path, ds, model, tmp_path / "s.tif", workers=1, rows_per_stripe=97
    )
    arr, _ = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])


def test_an_all_nodata_stripe_writes_zeros(tmp_path):
    """A stripe with no valid pixel writes zeros and never calls predict_block."""
    from _inference_fixture import write_tiles

    from spatialrisk.mlmodels.windowed_predict import predict_windowed

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    alt = np.full((700, 300), 255, dtype="uint8")
    alt[300:] = 50
    dead = write_tiles(tmp_path / "dead.tif", alt, 255)
    feats = {v.name: v.path for v in ds.features}
    feats["alt"] = dead
    calls = []

    def spy(block_df, extras):
        calls.append(len(block_df))
        return _glm_closure(model)(block_df, extras)

    out = predict_windowed(
        ds.target.path,
        feats,
        spy,
        tmp_path / "z.tif",
        workers=1,
        rows_per_stripe=256,
        n_design_cols=3,
    )
    arr, _ = read_raster(out)
    assert (arr[:256] == 0).all()
    # Rows 300+ are predicted, except the `unused` feature's own nodata band:
    # a feature the formula never references still invalidates its pixels.
    assert (arr[300:600] > 0).all()
    assert (arr[600:650] == 0).all()
    assert (arr[650:] > 0).all()
    assert len(calls) == 2  # stripe 0 never reaches predict_block


def test_output_is_atomic_and_a_failure_keeps_the_old_file(tmp_path):
    """A failing prediction leaves the previous output intact and no partial behind."""
    from spatialrisk.mlmodels.windowed_predict import predict_windowed

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = tmp_path / "keep.tif"
    _run_glm(tmp_path, ds, model, out, workers=1)
    before, _ = read_raster(out)

    def poisoned(block_df, extras):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        predict_windowed(
            ds.target.path,
            {v.name: v.path for v in ds.features},
            poisoned,
            out,
            workers=1,
            n_design_cols=3,
        )
    after, _ = read_raster(out)
    np.testing.assert_array_equal(after, before)
    assert not list(tmp_path.glob("*.part.tif"))
    assert INFERENCE_LEDGER.snapshot() == []


def test_the_ledger_is_released_on_every_path(tmp_path, monkeypatch):
    """A run hands its reservation back whether it finishes, fails mid-walk or early.

    A leaked reservation is unrecoverable: nothing but ``release`` removes it
    from the ledger, so it subtracts from every later plan's budget for the
    life of the process and can park the next run in ``_wait_until_fits``.
    """
    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    assert INFERENCE_LEDGER.snapshot() == []

    _run_glm(tmp_path, ds, model, tmp_path / "ok.tif", workers=1)
    assert INFERENCE_LEDGER.outstanding_bytes == 0
    assert INFERENCE_LEDGER.outstanding_workers == 0
    assert INFERENCE_LEDGER.snapshot() == []

    # A failure between the reservation and the stripe walk — what a read-only
    # or quota-bound output directory does to the setup steps.
    def boom(*a, **k):
        raise OSError("read-only output directory")

    monkeypatch.setattr(wp, "_stripes", boom)
    with pytest.raises(OSError, match="read-only"):
        _run_glm(tmp_path, ds, model, tmp_path / "early.tif", workers=1)
    assert INFERENCE_LEDGER.outstanding_bytes == 0
    assert INFERENCE_LEDGER.snapshot() == []


def test_every_opened_handle_is_closed(tmp_path, monkeypatch):
    """Every dataset the engine opens for reading is closed, success or failure."""
    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    opened = []
    real_open = wp.rasterio.open

    def spy_open(*a, **k):
        h = real_open(*a, **k)
        if k.get("mode", a[1] if len(a) > 1 else "r") == "r":
            opened.append(h)
        return h

    monkeypatch.setattr(wp.rasterio, "open", spy_open)
    _run_glm(tmp_path, ds, model, tmp_path / "h.tif", workers=1)
    assert opened and all(h.closed for h in opened)

    def poisoned(block_df, extras):
        raise RuntimeError("boom")

    opened.clear()
    with pytest.raises(RuntimeError):
        wp.predict_windowed(
            ds.target.path,
            {v.name: v.path for v in ds.features},
            poisoned,
            tmp_path / "h2.tif",
            workers=1,
            n_design_cols=3,
        )
    assert opened and all(h.closed for h in opened)


def test_the_run_is_pinned_to_one_blas_thread(tmp_path):
    """predict_block runs inside single_thread_math()."""
    from threadpoolctl import threadpool_info

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    seen = []

    def spy(block_df, extras):
        seen.extend(i["num_threads"] for i in threadpool_info())
        return _glm_closure(model)(block_df, extras)

    from spatialrisk.mlmodels.windowed_predict import predict_windowed

    predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        spy,
        tmp_path / "b.tif",
        workers=1,
        n_design_cols=3,
    )
    if seen:
        assert set(seen) == {1}
