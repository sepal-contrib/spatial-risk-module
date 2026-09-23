# tests/test_windowed_predict.py
"""The shared stripe engine behind GLM/RF/iCAR apply()."""
import json
import logging
import threading

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


#: (dtype, nodata) per feature: a Peru-like stack (7 float32, int16, uint8)
#: and 15 uint8 layers, the two shapes the read phase was measured on.
_READ_PHASE_STACKS = {
    "peru_like": [("float32", -9999.0)] * 7 + [("int16", -32768), ("uint8", 255)],
    "uint8_x15": [("uint8", 255)] * 15,
}


@pytest.mark.parametrize("stack", sorted(_READ_PHASE_STACKS))
def test_read_phase_peak_fits_the_charge(tmp_path, stack):
    """Reading a dense stripe peaks within the smallest charge any model makes.

    GLM and iCAR are charged one working column, so the engine's own read
    phase is their stripe's peak. Measured with tracemalloc on a stripe where
    every pixel is valid (the worst case: the filtered frame is as large as
    the full-stripe columns), against
    :func:`spatialrisk.gdal_env.inference_working_set` with a mask and
    ``n_design_cols=1``. Holding the full-stripe columns, a dict of filtered
    copies and pandas' consolidated block at once (24 B per pixel per
    feature) overran it on both stacks.
    """
    import tracemalloc

    from _inference_fixture import write_tiles
    from rasterio.windows import Window

    from spatialrisk.gdal_env import inference_working_set
    from spatialrisk.mlmodels.windowed_predict import _read_stripe

    rows, width = 256, 1024
    rng = np.random.default_rng(3)
    paths, itemsizes = {}, []
    for i, (dtype, nodata) in enumerate(_READ_PHASE_STACKS[stack]):
        data = (rng.random((rows, width)) * 100).astype(dtype)  # no nodata pixel
        paths[f"f{i}"] = write_tiles(tmp_path / f"f{i}.tif", data, nodata)
        itemsizes.append(np.dtype(dtype).itemsize)
    mask = write_tiles(tmp_path / "mask.tif", np.ones((rows, width), "uint8"), 255)

    features = {name: rasterio.open(p) for name, p in paths.items()}
    mask_src = rasterio.open(mask)
    try:
        transform = mask_src.transform

        def read():
            return _read_stripe(
                (features, mask_src, {}),
                Window(0, 0, width, rows),
                transform,
                list(paths),
                (0,),
                False,
                None,
            )

        read()  # warm GDAL's block cache outside the measurement
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            valid, block_df, _ = read()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    finally:
        for src in [*features.values(), mask_src]:
            src.close()

    assert valid.all() and len(block_df) == rows * width
    assert list(block_df.columns) == list(paths)
    assert (block_df.dtypes == np.float64).all()
    charge = inference_working_set(
        width,
        rows,
        n_features=len(itemsizes),
        feature_itemsizes=itemsizes,
        n_design_cols=1,
        with_mask=True,
        with_extra=False,
    )
    assert peak - base <= charge, (
        f"read phase {(peak - base) / (rows * width):.1f} B/px > "
        f"charge {charge / (rows * width):.1f} B/px"
    )


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


# --------------------------------------------------------------------------- #
# the pool: worker threads, one bounded writer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [2, 4])
def test_pooled_engine_reproduces_the_glm_golden(tmp_path, golden, workers):
    """Every worker count writes the serial engine's raster, pixel for pixel."""
    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = _run_glm(
        tmp_path,
        ds,
        model,
        tmp_path / f"p{workers}.tif",
        workers=workers,
        rows_per_stripe=64,  # 11 stripes: more stripes than workers, real contention
    )
    arr, meta = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])
    assert meta == golden["glm"][1]


def test_pooled_engine_reproduces_the_icar_golden(tmp_path, golden):
    """Extra layers and a bounds-read mask are stripe-local, so the pool is safe."""
    from patsy.highlevel import build_design_matrices

    from spatialrisk.mlmodels.windowed_predict import ExtraLayer, predict_windowed

    ds = build_dataset(tmp_path)
    model = build_icar(tmp_path, ds)
    betas = model._ml_model["betas"]

    def predict_block(block_df, extras):
        (x,) = build_design_matrices([model._x_design_info], block_df, NA_action="drop")
        x = np.asarray(x)
        return 1.0 / (1.0 + np.exp(-(x @ betas[: x.shape[1]] + extras["rho"])))

    out = predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        predict_block,
        tmp_path / "i4.tif",
        mask=ds.mask_path,
        mask_by_bounds=True,
        extra_layers={"rho": ExtraLayer(ds.rho_path, "bilinear")},
        workers=4,
        rows_per_stripe=64,
        n_design_cols=3,
    )
    arr, meta = read_raster(out)
    np.testing.assert_array_equal(arr, golden["icar"][0])
    assert meta == golden["icar"][1]


def test_pool_runs_predict_block_on_worker_threads_and_closes_their_handles(
    tmp_path, monkeypatch
):
    """Stripes are predicted on named worker threads, each with its own handles.

    The first stripe holds its worker until a second thread shows up, so a run
    that quietly stayed on one thread cannot pass by scheduling everything
    before the first task returns.
    """
    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    predict = _glm_closure(model)
    lock = threading.Lock()
    names, started, opened = set(), [], []
    second_thread = threading.Event()
    real_open = wp.rasterio.open

    def spy_open(*a, **k):
        handle = real_open(*a, **k)
        if k.get("mode", a[1] if len(a) > 1 else "r") == "r":
            with lock:
                opened.append((threading.current_thread().name, handle))
        return handle

    monkeypatch.setattr(wp.rasterio, "open", spy_open)

    def spy(block_df, extras):
        with lock:
            names.add(threading.current_thread().name)
            first = not started
            started.append(1)
            if len(names) > 1:
                second_thread.set()
        if first:
            second_thread.wait(10)  # hold worker 1: stripe 2 needs another thread
        return predict(block_df, extras)

    wp.predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        spy,
        tmp_path / "t.tif",
        mask=ds.mask_path,
        workers=3,
        rows_per_stripe=64,
        n_design_cols=3,
    )
    assert len(names) > 1 and all(n.startswith("predict") for n in names)
    worker_threads = {n for n, _ in opened if n.startswith("predict")}
    worker_handles = [h for n, h in opened if n.startswith("predict")]
    assert worker_handles and all(h.closed for h in worker_handles)
    # 4 handles (3 features + the mask) opened once per worker thread that ran
    assert len(worker_handles) == 4 * len(worker_threads)


def test_a_worker_failure_cancels_the_run_and_cleans_up(tmp_path, monkeypatch):
    """One failed stripe ends the run: the queued stripes are cancelled, never run.

    The very first stripe fails, before any real work, while the others take
    milliseconds of raster reading; the writer therefore meets the failure
    while stripes it looked ahead to are still queued, and those must be
    cancelled instead of quietly finishing the raster the run is abandoning.
    """
    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = tmp_path / "fail.tif"
    _run_glm(tmp_path, ds, model, out, workers=1)
    before, _ = read_raster(out)
    futures, started = [], []
    lock = threading.Lock()
    real_predict_stripe = wp._predict_stripe

    class RecordingPool(wp.ThreadPoolExecutor):
        """A pool that keeps every future it hands back to the writer."""

        def submit(self, fn, *args, **kwargs):
            """Record the future so the test can read its final state."""
            future = super().submit(fn, *args, **kwargs)
            with lock:
                futures.append(future)
            return future

    def spy_stripe(handles, window, *args, **kwargs):
        with lock:
            started.append(int(window.row_off))
        if window.row_off == 0:
            raise RuntimeError("stripe zero died")
        return real_predict_stripe(handles, window, *args, **kwargs)

    monkeypatch.setattr(wp, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(wp, "_predict_stripe", spy_stripe)

    with pytest.raises(RuntimeError, match="stripe zero died"):
        wp.predict_windowed(
            ds.target.path,
            {v.name: v.path for v in ds.features},
            _glm_closure(model),
            out,
            workers=2,
            rows_per_stripe=64,  # 11 stripes, 4 of them submitted up front
            n_design_cols=3,
        )
    cancelled = [f for f in futures if f.cancelled()]
    assert cancelled  # the stripes still queued behind the failure
    assert len(started) < 11  # and they never ran
    after, _ = read_raster(out)
    np.testing.assert_array_equal(after, before)
    assert not list(tmp_path.glob("*.part.tif"))
    assert INFERENCE_LEDGER.snapshot() == []


def test_two_concurrent_runs_share_one_budget_and_the_second_waits(
    tmp_path, golden, monkeypatch
):
    """The 2026-09-22 SEPAL incident: GLM + RF at once must queue, not die.

    The budget is faked so that exactly one run's reservation fits. The second
    run must not read a stripe until the first has released, and both outputs
    must still be golden.
    """
    from spatialrisk import gdal_env
    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    predict = _glm_closure(model)
    # One 256-row stripe of the 300 px wide fixture (3 features of 1, 1 and 4
    # bytes, a mask, no extras): a two-worker run reserves two of these, and
    # the budget below holds two and a half of them — one run, not two.
    stripe = gdal_env.inference_working_set(
        300,
        256,
        n_features=3,
        feature_itemsizes=[1, 1, 4],
        n_design_cols=3,
        with_mask=True,
        with_extra=False,
    )
    held = []
    ledger = gdal_env.ResourceLedger(
        budget_fn=lambda: int(2.5 * stripe) - held[0].outstanding_bytes
    )
    held.append(ledger)
    monkeypatch.setattr(wp, "INFERENCE_LEDGER", ledger)

    parked = threading.Event()

    class _Parked(logging.Handler):
        """Fires ``parked`` when the ledger reports a run waiting for memory."""

        def emit(self, record):
            """Watch one log line."""
            if "waiting for memory" in record.getMessage():
                parked.set()

    second_log = logging.Logger("second-run")  # unregistered: nothing else sees it
    second_log.addHandler(_Parked())

    events, errors = [], []
    ev_lock = threading.Lock()
    first_in_body = threading.Event()
    release_first = threading.Event()

    def slow_first(block_df, extras):
        with ev_lock:
            events.append("first-block")
        first_in_body.set()
        assert release_first.wait(30)
        return predict(block_df, extras)

    def second(block_df, extras):
        with ev_lock:
            events.append("second-block")
        return predict(block_df, extras)

    def run(*args, **kwargs):
        try:
            wp.predict_windowed(*args, **kwargs)
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)

    feats = {v.name: v.path for v in ds.features}
    out1, out2 = tmp_path / "one.tif", tmp_path / "two.tif"
    common = dict(mask=ds.mask_path, workers=2, rows_per_stripe=256, n_design_cols=3)
    t1 = threading.Thread(
        target=run,
        args=(ds.target.path, feats, slow_first, out1),
        kwargs=dict(label="one", **common),
    )
    t2 = threading.Thread(
        target=run,
        args=(ds.target.path, feats, second, out2),
        kwargs=dict(label="two", log=second_log, **common),
    )
    t1.start()
    assert first_in_body.wait(30)
    t2.start()
    assert parked.wait(30)  # the second run is queued on the ledger, not running
    with ev_lock:
        assert "second-block" not in events
    assert [r.label for r in ledger.snapshot()] == ["one"]
    release_first.set()
    t1.join(60)
    t2.join(60)
    assert not t1.is_alive() and not t2.is_alive()
    assert errors == []
    assert events.index("second-block") > events.index("first-block")
    assert ledger.snapshot() == []
    for out in (out1, out2):
        arr, _ = read_raster(out)
        np.testing.assert_array_equal(arr, golden["glm"][0])


def test_look_ahead_is_bounded_to_twice_the_workers(tmp_path, golden, monkeypatch):
    """The writer keeps at most 2 x workers stripes submitted ahead of itself."""
    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    predict = _glm_closure(model)
    workers = 2
    lock = threading.Lock()
    submits, warm_ups = [0], [0]
    submitted = threading.Semaphore(0)
    entered = threading.Semaphore(0)
    release = threading.Event()
    in_flight, peak, starts = [0], [0], [0]

    class CountingPool(wp.ThreadPoolExecutor):
        """A pool that counts the stripes the writer submits, warm-up apart."""

        def submit(self, fn, *args, **kwargs):
            """Count the submission before it reaches the pool."""
            if args:  # a stripe carries its index; a warm-up task takes none
                with lock:
                    submits[0] += 1
                submitted.release()
            else:
                with lock:
                    warm_ups[0] += 1
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(wp, "ThreadPoolExecutor", CountingPool)

    def blocking(block_df, extras):
        with lock:
            nth = starts[0]
            starts[0] += 1
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        if nth < workers:  # park every worker; the writer can make no progress
            entered.release()
            assert release.wait(30)
        with lock:
            in_flight[0] -= 1
        return predict(block_df, extras)

    out = tmp_path / "la.tif"
    errors = []

    def run():
        try:
            wp.predict_windowed(
                ds.target.path,
                {v.name: v.path for v in ds.features},
                blocking,
                out,
                mask=ds.mask_path,
                workers=workers,
                rows_per_stripe=64,
                n_design_cols=3,
            )
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)

    runner = threading.Thread(target=run)
    runner.start()
    try:
        for _ in range(2 * workers):
            assert submitted.acquire(timeout=30)  # the look-ahead fills up
        for _ in range(workers):
            assert entered.acquire(timeout=30)  # every worker is parked
        with lock:
            # Nothing can progress now, so this is the whole look-ahead.
            assert submits[0] == 2 * workers
            assert warm_ups[0] == workers  # one pin per worker, before any stripe
    finally:
        release.set()
        runner.join(60)
    assert not runner.is_alive()
    assert errors == []
    assert peak[0] <= workers  # never more predictions at once than workers
    arr, _ = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])


def test_pooled_workers_predict_with_one_math_thread_each(tmp_path):
    """Every worker thread runs predict_block with BLAS and OpenMP pinned to one.

    ``single_thread_math()`` on the calling thread does not carry over to the
    pool: libgomp keeps its thread count per thread, so an unpinned worker
    would run scikit-learn's OpenMP regions on every core — ``workers`` times
    over, the oversubscription the pin exists to prevent.
    """
    from threadpoolctl import threadpool_info

    from spatialrisk.mlmodels.windowed_predict import predict_windowed

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    predict = _glm_closure(model)
    lock = threading.Lock()
    seen = set()

    def spy(block_df, extras):
        with lock:
            seen.update((i["user_api"], i["num_threads"]) for i in threadpool_info())
        return predict(block_df, extras)

    predict_windowed(
        ds.target.path,
        {v.name: v.path for v in ds.features},
        spy,
        tmp_path / "pin.tif",
        mask=ds.mask_path,
        workers=2,
        rows_per_stripe=64,
        n_design_cols=3,
    )
    assert seen and {n for _, n in seen} == {1}


def test_rf_apply_pins_n_jobs_per_worker_and_restores_it(tmp_path):
    """RF.apply() pins the estimator to one joblib worker per pooled run.

    ``workers=None`` is the serial path for a forest, not the resource
    policy's choice, so the estimator keeps its own ``n_jobs``: on a SEPAL c8
    the policy's 2 workers made prediction 77 % slower than serial
    (2026-09-22, see the comment in ``RFModel.apply``).
    """
    from _inference_fixture import build_rf

    ds = build_dataset(tmp_path)
    model = build_rf(tmp_path, ds)
    model._ml_model.n_jobs = -1
    seen = []
    real = model._ml_model.predict_proba

    def spy(x):
        seen.append(model._ml_model.n_jobs)
        return real(x)

    model._ml_model.predict_proba = spy
    model.apply(tmp_path / "rf4.tif", ds, ds.mask_path, 0, workers=4)
    assert set(seen) == {1}
    assert model._ml_model.n_jobs == -1
    seen.clear()
    model.apply(tmp_path / "rf1.tif", ds, ds.mask_path, 0, workers=1)
    assert set(seen) == {-1}
    seen.clear()
    model.apply(tmp_path / "rfdefault.tif", ds, ds.mask_path, 0)
    assert set(seen) == {-1}


# --------------------------------------------------------------------------- #
# pool start-up: nothing is imported on a worker, every worker is pinned first
# --------------------------------------------------------------------------- #
def test_the_engine_imports_nothing_at_call_time():
    """Source guard: no function in the engine imports, so no worker dlopens.

    The 2026-09-22 inference bench deadlocked a pooled run at pool start-up,
    roughly one attempt in three at 16 workers and also at the 8 the policy
    picks on a 16-core box. One worker sat in ``_predict_stripe``'s per-stripe
    ``import forestatrisk``, holding CPython's import lock and the GIL while
    ``dlopen`` waited for glibc's loader lock; two more held that loader lock
    inside ``threadpoolctl``'s ``dl_iterate_phdr`` walk and waited for the GIL
    from its ctypes callback; the rest queued on the import lock. Every import
    the engine needs is therefore resolved once, at module import, on the
    calling thread that has no pool yet.

    Scope: the engine module only. A ``predict_block`` closure runs on the
    same pool threads and is under the same rule, but its source is not
    checked here -- each model's ``apply`` imports what its closure needs
    before calling in (``predict_windowed``'s docstring states the contract).
    """
    import ast
    import inspect

    from spatialrisk.mlmodels import windowed_predict as wp

    tree = ast.parse(inspect.getsource(wp))
    lazy = [
        f"{fn.name}:{node.lineno}"
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(fn)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert lazy == [], f"call-time imports in the stripe engine: {lazy}"
    top_level = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {
        a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names
    }
    assert any(m and m.split(".")[0] == "forestatrisk" for m in top_level)


def test_every_pool_thread_is_pinned_before_the_first_stripe(
    tmp_path, golden, monkeypatch
):
    """The pool is warmed up: every worker is pinned and idle before any stripe.

    A stripe running on a thread whose OpenMP pool is still the machine's is
    the oversubscription the pin exists to prevent, and the executor
    ``initializer`` -- started lazily, one thread per submit -- cannot rule it
    out: it pins the last worker while the first is already predicting.
    Submitting the pins as barriered warm-up tasks and waiting for them does.
    """
    from threadpoolctl import threadpool_info

    from spatialrisk.mlmodels import windowed_predict as wp

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    workers = 4
    lock = threading.Lock()
    events = []  # ("pin" | "stripe", thread name), in the order they happened
    math_threads = {}  # thread name -> the math pool sizes it saw in its first stripe
    all_in = threading.Barrier(workers, timeout=60)
    real_pin, real_stripe = wp._pin_worker_math, wp._predict_stripe

    def spy_pin(controller):
        real_pin(controller)
        with lock:
            events.append(("pin", threading.current_thread().name))

    def spy_stripe(*args, **kwargs):
        name = threading.current_thread().name
        with lock:
            events.append(("stripe", name))
            first = name not in math_threads
            if first:
                math_threads[name] = {i["num_threads"] for i in threadpool_info()}
        if first:
            all_in.wait()  # hold every worker until all of them have a stripe
        return real_stripe(*args, **kwargs)

    monkeypatch.setattr(wp, "_pin_worker_math", spy_pin)
    monkeypatch.setattr(wp, "_predict_stripe", spy_stripe)

    out = _run_glm(
        tmp_path,
        ds,
        model,
        tmp_path / "warm.tif",
        workers=workers,
        rows_per_stripe=64,  # 11 stripes, look-ahead 8: all four workers run
    )

    kinds = [kind for kind, _ in events]
    # The pins are the first events there are: the last one happened before
    # the first stripe body, on every thread the pool will ever run a stripe on.
    assert kinds.count("pin") == workers
    assert kinds.index("stripe") == workers
    pinned = {n for kind, n in events if kind == "pin"}
    ran = {n for kind, n in events if kind == "stripe"}
    assert len(pinned) == workers and pinned == ran
    assert all(n.startswith("predict") for n in pinned)
    assert {n for sizes in math_threads.values() for n in sizes} == {1}
    arr, _ = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])


def test_no_pool_thread_builds_a_threadpool_controller(tmp_path, golden, monkeypatch):
    """The run's one threadpoolctl library scan happens on the calling thread.

    Building a ``ThreadpoolController`` is the ``dl_iterate_phdr`` walk over
    every shared object in the process: glibc's loader lock taken while a
    ctypes callback re-enters Python for the GIL. On a pool thread that
    inverts against anything a stripe ``dlopen``s -- in this run or in a
    concurrent one, since the ledger queues predictions by memory, not
    exclusivity. The workers pin themselves through the controller the caller
    built, which only sets thread counts on handles it already resolved.
    """
    import threadpoolctl

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    lock = threading.Lock()
    built = []
    real_init = threadpoolctl.ThreadpoolController.__init__

    def spy_init(self, *args, **kwargs):
        with lock:
            built.append(threading.current_thread().name)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(threadpoolctl.ThreadpoolController, "__init__", spy_init)

    out = _run_glm(
        tmp_path,
        ds,
        model,
        tmp_path / "scan.tif",
        workers=4,
        rows_per_stripe=64,
    )

    assert built, "the run builds no controller at all: the pin cannot work"
    assert [n for n in built if n.startswith("predict")] == []
    arr, _ = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])


# --------------------------------------------------------------------------- #
# the plan line: stripe size, working width, and the over-budget warning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [1, 2])
def test_sub_tile_stripes_reproduce_the_glm_golden(tmp_path, golden, workers):
    """64-row stripes on 256-row tiles write the same raster as the golden."""
    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    out = _run_glm(
        tmp_path,
        ds,
        model,
        tmp_path / f"sub{workers}.tif",
        workers=workers,
        rows_per_stripe=64,
    )
    arr, meta = read_raster(out)
    np.testing.assert_array_equal(arr, golden["glm"][0])
    assert meta == golden["glm"][1]


def test_plan_line_reports_stripe_size_and_working_width(tmp_path, caplog):
    """The INFO plan line carries the per-stripe MiB and the charged working width.

    The field reads "working cols": for GLM and iCAR it is the charged
    per-pixel width (1), not the design's column count, which their own
    "design: ..." line reports.
    """
    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        model.apply(tmp_path / "out" / "g.tif", ds, ds.mask_path, 0, workers=1)
    line = next(
        r.getMessage() for r in caplog.records if "rows/stripe" in r.getMessage()
    )
    assert "MiB each, 1 working cols)" in line
    assert "design cols" not in line


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("kind", ["glm", "icar"])
def test_over_budget_plan_logs_a_warning(
    tmp_path, caplog, monkeypatch, golden, kind, workers
):
    """A stripe the budget cannot hold is announced, and still predicted exactly.

    With 1 MiB free nothing fits, so the plan halves down to
    ``INFERENCE_MIN_STRIPE_ROWS`` and warns. The run goes through the real
    ``apply()``, i.e. the production compiled-predictor closures, on those
    sub-tile stripes, serially and pooled, and must still write the golden
    raster -- the only test that runs GLM's and iCAR's own closures on
    planner-chosen sub-tile stripes (the 64-row test above uses a patsy
    closure of its own).
    """
    from spatialrisk import gdal_env
    from spatialrisk.gdal_env import INFERENCE_MIN_STRIPE_ROWS

    real = gdal_env.plan_inference
    plans = []

    def cramped(**kw):
        kw["free_bytes"] = 1 << 20  # 1 MiB free: nothing fits
        plan = real(**kw)
        plans.append(plan)
        return plan

    monkeypatch.setattr("spatialrisk.mlmodels.windowed_predict.plan_inference", cramped)
    ds = build_dataset(tmp_path)
    model = {"glm": build_glm, "icar": build_icar}[kind](tmp_path, ds)
    out = tmp_path / "out" / f"{kind}.tif"
    with caplog.at_level(logging.WARNING, logger="spatial_risk"):
        model.apply(out, ds, ds.mask_path, 0, workers=workers)

    assert [(p.rows_per_stripe, p.workers) for p in plans] == [
        (INFERENCE_MIN_STRIPE_ROWS, workers)
    ]
    assert plans[0].over_budget_bytes > 0
    assert any(
        r.levelno == logging.WARNING and "exceeds the memory budget" in r.getMessage()
        for r in caplog.records
    )
    arr, meta = read_raster(out)
    np.testing.assert_array_equal(arr, golden[kind][0])
    assert meta == golden[kind][1]
