"""The stripe-level thread pool and its resource policy (sampling track E, pool).

Pass 1 and pass 2 of ``spatialrisk.sampling.blocked`` walk the stripes through
a pool of worker threads that each own a dataset handle. The contract is that
the output is bit-identical to the serial walk for every worker count, that
pass 2 only reads the stripes that hold a wanted rank, and that the policy in
``spatialrisk.gdal_env.plan_sampling`` picks workers from cores and memory
without ever going below one worker at one tile row per stripe.
"""

import numpy as np
import pytest
import test_sampling_blocked as tb

MIB = 1024 * 1024


@pytest.fixture(scope="module")
def pool_raster(tmp_path_factory):
    """A 700 x 300 four-class raster with nodata holes and a partial mask."""
    d = tmp_path_factory.mktemp("pool")
    arr = tb._class_array(700, 300, 4, seed=11)
    mask = np.ones((700, 300), dtype="uint8")
    mask[:50, :] = 0
    mask[:, 220:] = 0
    rpath, mpath = d / "strata.tif", d / "mask.tif"
    tb._write(rpath, arr)
    tb._write(mpath, mask, nodata=0)
    return rpath, mpath


# --------------------------------------------------------------------------- #
# identity across worker counts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [1, 2, 3, 8])
@pytest.mark.parametrize("stripe", [64, 97, 256])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strategy="stratified", n_samples=150, allocation="equal"),
        dict(strategy="stratified", n_samples=5000, allocation="proportional"),
        dict(strategy="stratified", n_samples=100_000, allocation="deforisk"),
        dict(strategy="random", n_samples=150),
        dict(strategy="random", n_samples=5000),
        dict(strategy="systematic", n_samples=300),
        dict(strategy="systematic", n_samples=None, spacing_m=17.0),
    ],
    ids=lambda k: f"{k['strategy']}-{k.get('n_samples')}-{k.get('spacing_m', '')}",
)
def test_pooled_output_is_identical_to_the_reference(
    pool_raster, workers, stripe, kwargs
):
    """Every worker count reproduces the pre-E2 algorithm exactly."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = pool_raster
    new = generate_points(
        rpath, mpath, seed=7, rows_per_stripe=stripe, workers=workers, **kwargs
    )
    ref = tb._reference_generate_points(rpath, mpath, seed=7, **kwargs)
    tb._assert_same(new, ref)


def test_take_all_is_identical_with_a_pool(pool_raster):
    """The output-sized 'take all' collector keeps global row-major order."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = pool_raster
    kwargs = dict(strategy="random", n_samples=10_000_000)
    new = generate_points(rpath, mpath, rows_per_stripe=64, workers=4, **kwargs)
    ref = tb._reference_generate_points(rpath, mpath, **kwargs)
    tb._assert_same(new, ref)


# --------------------------------------------------------------------------- #
# pass 2 reads only what it needs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [1, 3])
def test_pass2_reads_only_stripes_holding_wanted_ranks(pool_raster, workers):
    """A rank in the first stripe never makes pass 2 decode the others."""
    from spatialrisk.sampling import blocked

    rpath, mpath = pool_raster
    with blocked.RasterScan(rpath, mpath, rows_per_stripe=64, workers=workers) as scan:
        raw_counts, _ = blocked.count_values(scan)
        n_stripes = len(scan.windows)
        assert scan.stripes_read == n_stripes
        # class 0's first valid pixel: rank 0 lives in the first stripe that has it
        first_stripe = next(
            i for i, counts in enumerate(scan.stripe_counts) if counts.get(0, 0) > 0
        )
        got = blocked.collect_ranks(scan, {0: np.array([0])})
        assert scan.stripes_read == n_stripes + 1
        rows, cols, vals = got[0]
        assert vals.tolist() == [0]
        assert int(rows[0]) // 64 == first_stripe
        # a rank in the last stripe reads every stripe up to it and none twice
        last_stripe = max(
            i for i, counts in enumerate(scan.stripe_counts) if counts.get(0, 0) > 0
        )
        before = scan.stripes_read
        blocked.collect_ranks(scan, {0: np.array([raw_counts[0] - 1])})
        assert scan.stripes_read - before == 1
        assert last_stripe >= first_stripe


def test_collect_ranks_without_pass1_falls_back_to_a_full_walk(pool_raster):
    """No per-stripe counts (a caller that skipped pass 1) still works."""
    from spatialrisk.sampling import blocked

    rpath, mpath = pool_raster
    with blocked.RasterScan(rpath, mpath, rows_per_stripe=64, workers=2) as scan:
        got = blocked.collect_ranks(scan, {None: np.array([0, 5, 6])})
        assert scan.stripes_read >= 1
        rows, cols, vals = got[None]
        assert rows.size == 3
    with blocked.RasterScan(rpath, mpath, rows_per_stripe=64, workers=1) as ref_scan:
        ref = blocked.collect_ranks(ref_scan, {None: np.array([0, 5, 6])})[None]
    for a, b in zip(got[None], ref):
        np.testing.assert_array_equal(a, b)


def test_worker_handles_are_closed_with_the_scan(pool_raster):
    """Each worker thread opens its own datasets; close() closes them all."""
    from spatialrisk.sampling import blocked

    rpath, mpath = pool_raster
    scan = blocked.RasterScan(rpath, mpath, rows_per_stripe=64, workers=3)
    blocked.count_valid(scan)
    handles = list(scan._worker_handles)
    assert handles, "the pool opened no per-thread datasets"
    scan.close()
    assert all(ds.closed for ds in handles)
    assert scan._worker_handles == []


# --------------------------------------------------------------------------- #
# the resource policy
# --------------------------------------------------------------------------- #
def _plan(**overrides):
    from spatialrisk.gdal_env import plan_sampling

    kw = dict(
        width=40_000,
        tile_rows=256,
        itemsize=1,
        with_mask=True,
        cores=8,
        free_bytes=8 * 1024 * MIB,
        gdal_cache_bytes=512 * MIB,
    )
    kw.update(overrides)
    return plan_sampling(**kw)


def test_policy_is_half_the_cores_when_memory_is_plentiful(monkeypatch):
    """Half the affinity cores, 512-row stripes, one decode thread per worker."""
    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    plan = _plan()
    assert plan.workers == 4
    assert plan.rows_per_stripe == 512
    assert plan.gdal_threads == 1
    assert plan.cachemax_bytes >= 512 * MIB


def test_policy_never_exceeds_the_memory_budget(monkeypatch):
    """Half the free memory, minus the GDAL cache, divided by one stripe."""
    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    plan = _plan(free_bytes=1536 * MIB)  # 768 MiB budget - 512 MiB cache = 2 stripes
    assert 1 <= plan.workers < 4
    assert plan.workers == plan.by_memory
    assert plan.workers * plan.stripe_bytes <= plan.memory_budget_bytes


def test_policy_shrinks_the_stripe_before_going_below_one_worker(monkeypatch):
    """Tight memory shrinks the stripe in tile rows, floor = one worker x one tile."""
    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    plan = _plan(free_bytes=1100 * MIB)  # 550 MiB budget - 512 MiB cache = 38 MiB
    assert plan.workers == 1
    assert plan.rows_per_stripe == 256
    assert plan.rows_per_stripe % 256 == 0
    # and never below one tile row even when nothing fits
    plan = _plan(free_bytes=0)
    assert plan.workers == 1
    assert plan.rows_per_stripe == 256


def test_policy_honours_an_explicit_stripe_height(monkeypatch):
    """A requested stripe height is never shrunk, even when memory is tight."""
    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    plan = _plan(rows_per_stripe=97, free_bytes=0)
    assert plan.rows_per_stripe == 97
    assert plan.workers == 1


def test_policy_env_override_wins(monkeypatch):
    """SPATIAL_RISK_SAMPLING_WORKERS replaces the computed count (min 1)."""
    monkeypatch.setenv("SPATIAL_RISK_SAMPLING_WORKERS", "6")
    plan = _plan()
    assert plan.workers == 6
    monkeypatch.setenv("SPATIAL_RISK_SAMPLING_WORKERS", "0")
    assert _plan().workers == 1


def test_serial_plan_keeps_multithreaded_decode(monkeypatch):
    """One worker keeps GDAL's multi-threaded decode (the faster serial pass)."""
    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    plan = _plan(cores=2)
    assert plan.workers == 1
    assert plan.gdal_threads >= 1


def test_cache_grows_with_in_flight_stripes(monkeypatch):
    """The block cache holds every in-flight raster and mask stripe."""
    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    plan = _plan(cores=32, width=100_000, free_bytes=64 * 1024 * MIB)
    assert plan.workers == 16
    decoded = plan.workers * 2 * plan.rows_per_stripe * plan.width_bytes_per_row
    assert plan.cachemax_bytes >= decoded


def test_scan_default_uses_the_policy(pool_raster, monkeypatch):
    """RasterScan(workers=None) plans from cores and memory."""
    from spatialrisk import gdal_env
    from spatialrisk.sampling import blocked

    monkeypatch.delenv("SPATIAL_RISK_SAMPLING_WORKERS", raising=False)
    monkeypatch.setattr(gdal_env, "_available_cores", lambda: 6)
    monkeypatch.setattr(gdal_env, "free_memory_bytes", lambda: (4 * 1024 * MIB, "test"))
    rpath, mpath = pool_raster
    with blocked.RasterScan(rpath, mpath) as scan:
        assert scan.workers == 3
        assert scan.plan.memory_source == "test"


def test_free_memory_prefers_the_tighter_reading(monkeypatch):
    """The cgroup limit wins when it is below the host's available memory."""
    from spatialrisk import parallel

    monkeypatch.setattr(
        parallel, "cgroup_memory", lambda: {"version": 2, "limit": 1000, "usage": 400}
    )
    monkeypatch.setattr(
        parallel, "host_memory", lambda: {"total": 5000, "available": 900}
    )
    assert parallel.free_memory_bytes() == (600, "cgroup v2 limit - usage")
    monkeypatch.setattr(
        parallel,
        "cgroup_memory",
        lambda: {"version": None, "limit": None, "usage": None},
    )
    assert parallel.free_memory_bytes() == (900, "psutil.available")


def test_parse_cgroup_bytes():
    """``max`` and the v1 sentinel mean no limit."""
    from spatialrisk.parallel import parse_cgroup_bytes

    assert parse_cgroup_bytes("max\n") is None
    assert parse_cgroup_bytes(str(2**63 - 4096)) is None
    assert parse_cgroup_bytes("1048576") == 1048576


# --------------------------------------------------------------------------- #
# memory: the policy's per-worker estimate is an upper bound
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_pool_peak_memory_grows_at_most_one_estimated_stripe_per_worker(tmp_path):
    """peak(4 workers) - peak(1 worker) <= 3 x the plan's stripe working set.

    The policy sizes the pool as ``budget // stripe_bytes``, so that estimate
    has to be at least what one extra worker really adds; measured in fresh
    processes (VmHWM) on a 9 Mpx raster, like the E2 memory test.
    """
    from spatialrisk.gdal_env import plan_sampling

    rpath, mpath = tmp_path / "s.tif", tmp_path / "m.tif"
    tb._write(rpath, tb._class_array(3000, 3000, 2, seed=41, nodata_frac=0.05))
    tb._write(mpath, np.ones((3000, 3000), dtype="uint8"), nodata=0)
    one_kib, _ = tb._probe("new", rpath, mpath, workers=1)
    four_kib, _ = tb._probe("new", rpath, mpath, workers=4)
    plan = plan_sampling(width=3000, tile_rows=1, itemsize=1, with_mask=True, cores=8)
    extra_kib = (four_kib - one_kib) * 1024 / 1024
    assert extra_kib * 1024 <= 3 * plan.stripe_bytes, (
        f"one={one_kib} KiB four={four_kib} KiB "
        f"stripe={plan.stripe_bytes / 1024:.0f} KiB"
    )
