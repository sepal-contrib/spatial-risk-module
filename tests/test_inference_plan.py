# tests/test_inference_plan.py
"""The inference stripe plan: workers from cores and memory, stripe shrink floor."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from spatialrisk import gdal_env
from spatialrisk.gdal_env import (
    INFERENCE_WORKERS_ENV,
    ResourceLedger,
    inference_working_set,
    plan_inference,
)

GiB = 1 << 30


def _plan(**kw):
    """Build a plausible plan_inference() call, overridden by kw."""
    base = dict(
        width=4000,
        tile_rows=256,
        n_features=3,
        feature_itemsizes=[1, 1, 4],
        n_design_cols=3,
        with_mask=True,
        with_extra=False,
        cores=8,
        free_bytes=16 * GiB,
        gdal_cache_bytes=64 << 20,
    )
    base.update(kw)
    return plan_inference(**base)


def test_working_set_counts_every_stripe_temporary():
    """Every per-pixel temporary the prediction body allocates is counted."""
    # per px: raw(1+1+4) + float64 col(3x8) + frame(3x8) + design(3x8)
    # + 2x8 + 2 + mask 1
    per_px = 6 + 24 + 24 + 24 + 16 + 2 + 1
    assert (
        inference_working_set(
            100,
            10,
            n_features=3,
            feature_itemsizes=[1, 1, 4],
            n_design_cols=3,
            with_mask=True,
            with_extra=False,
        )
        == 100 * 10 * per_px
    )


def test_extra_layer_adds_eight_bytes_per_pixel():
    """The iCAR extra layer (float64 rho) costs 8 bytes per pixel."""
    a = inference_working_set(
        10,
        10,
        n_features=1,
        feature_itemsizes=[1],
        n_design_cols=1,
        with_mask=False,
        with_extra=False,
    )
    b = inference_working_set(
        10,
        10,
        n_features=1,
        feature_itemsizes=[1],
        n_design_cols=1,
        with_mask=False,
        with_extra=True,
    )
    assert b - a == 100 * 8


def test_policy_is_half_the_cores_when_memory_is_plentiful(monkeypatch):
    """With ample memory, workers default to half the affinity cores."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    plan = _plan()
    assert plan.workers == 4
    assert plan.rows_per_stripe == 256
    assert plan.by_cores == 4


def test_policy_never_exceeds_the_memory_budget(monkeypatch):
    """Workers are capped so in-flight stripes never exceed the memory budget."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    one_stripe = inference_working_set(
        4000,
        256,
        n_features=3,
        feature_itemsizes=[1, 1, 4],
        n_design_cols=3,
        with_mask=True,
        with_extra=False,
    )
    # budget = 50 % of free - cache; make room for exactly two stripes
    free = int((2 * one_stripe + (64 << 20)) / 0.5) + 1
    plan = _plan(free_bytes=free)
    assert plan.workers == 2
    assert plan.by_memory == 2


def test_policy_shrinks_the_stripe_before_going_below_one_worker(monkeypatch):
    """A stripe that cannot fit even one worker shrinks but never below one tile row."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    plan = _plan(free_bytes=1 * GiB, width=50000, tile_rows=256, gdal_cache_bytes=0)
    assert plan.workers == 1
    assert plan.rows_per_stripe % 256 == 0
    assert plan.rows_per_stripe < 256 * 1 or plan.rows_per_stripe == 256
    # a 50k-wide 256-row stripe is > 500 MB at ~100 B/px, so it cannot shrink
    # below one tile row: the floor holds and the serial path is still allowed
    assert plan.rows_per_stripe == 256
    assert plan.stripe_bytes > plan.memory_budget_bytes


def test_policy_shrinks_a_tall_stripe_in_tile_rows(monkeypatch):
    """A multi-tile-row stripe shrinks in whole tile rows, not arbitrary amounts."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    tall = _plan(rows_per_stripe=None, tile_rows=64, free_bytes=16 * GiB)
    assert tall.rows_per_stripe == 256  # default height, 4 tile rows of 64
    one_stripe = tall.stripe_bytes
    free = int((one_stripe * 0.6 + (64 << 20)) / 0.5)  # fits ~2.4 tile rows
    plan = _plan(tile_rows=64, free_bytes=free)
    assert plan.workers == 1
    assert plan.rows_per_stripe == 128


def test_policy_honours_an_explicit_stripe_height(monkeypatch):
    """An explicit rows_per_stripe is honoured as given and never shrunk."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    plan = _plan(rows_per_stripe=97, free_bytes=1 * GiB)
    assert plan.rows_per_stripe == 97


def test_policy_env_override_wins(monkeypatch):
    """SPATIAL_RISK_INFERENCE_WORKERS overrides the computed worker count."""
    monkeypatch.setenv(INFERENCE_WORKERS_ENV, "3")
    assert _plan().workers == 3
    monkeypatch.setenv(INFERENCE_WORKERS_ENV, "0")
    assert _plan().workers == 1


def test_serial_plan_keeps_multithreaded_decode(monkeypatch):
    """One worker keeps sampling_num_threads(); a pool gives each worker one thread."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    assert _plan(cores=1).gdal_threads == gdal_env.sampling_num_threads()
    assert _plan(cores=8).gdal_threads == 1


def test_reserved_workers_come_off_the_core_budget(monkeypatch):
    """reserved_workers reduces the core budget but never below one worker."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    assert _plan(reserved_workers=3).workers == 1
    assert _plan(reserved_workers=4).workers == 1  # floor, never zero


def test_reserved_bytes_come_off_the_memory_budget(monkeypatch):
    """reserved_bytes reduces the memory budget available to this plan."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    one_stripe = _plan().stripe_bytes
    free = int((3 * one_stripe + (64 << 20)) / 0.5) + 1  # room for three
    assert _plan(free_bytes=free).workers == 3
    assert _plan(free_bytes=free, reserved_bytes=one_stripe).workers == 2
    assert _plan(free_bytes=free, reserved_bytes=3 * one_stripe).workers == 1


def test_cache_holds_every_in_flight_input_stripe(monkeypatch):
    """The GDAL cache budget covers every worker's in-flight raster and mask stripe."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    plan = _plan()
    raw_row = 4000 * (1 + 1 + 4 + 1)  # features + mask, bytes per row
    assert plan.cachemax_bytes >= plan.workers * 2 * plan.rows_per_stripe * raw_row
    assert plan.cachemax_bytes >= 64 << 20


def test_reservations_add_up_and_release():
    """Live reservations sum per resource and disappear on release."""
    led = ResourceLedger(budget_fn=lambda: 10 * GiB)
    a = led.reserve(bytes_=GiB, workers=2, label="a", minimum_bytes=GiB)
    b = led.reserve(bytes_=2 * GiB, workers=1, label="b", minimum_bytes=GiB)
    assert led.outstanding_bytes == 3 * GiB
    assert led.outstanding_workers == 3
    led.release(a)
    assert led.outstanding_bytes == 2 * GiB
    assert [r.label for r in led.snapshot()] == ["b"]
    led.release(b)
    assert led.outstanding_bytes == 0 and led.outstanding_workers == 0


def test_minimum_proceeds_when_nothing_is_outstanding():
    """A lone job on a starved machine runs anyway: nothing else holds memory."""
    led = ResourceLedger(budget_fn=lambda: 0)
    r = led.reserve(bytes_=GiB, workers=1, label="lonely", minimum_bytes=GiB)
    assert led.outstanding_bytes == GiB
    led.release(r)


def test_reserve_waits_for_a_release_when_the_minimum_does_not_fit():
    """A second job blocks while the first holds the budget; a release wakes it."""
    budget = 3 * GiB
    led = ResourceLedger(
        budget_fn=lambda: budget - led.outstanding_bytes, sleep_log_every_s=0.05
    )
    first = led.reserve(bytes_=3 * GiB, workers=2, label="first", minimum_bytes=GiB)
    started, got = threading.Event(), {}

    def second():
        started.set()
        got["r"] = led.reserve(bytes_=GiB, workers=1, label="second", minimum_bytes=GiB)
        got["t"] = time.monotonic()

    t = threading.Thread(target=second)
    t.start()
    started.wait()
    time.sleep(0.2)
    assert "r" not in got  # still waiting
    t_release = time.monotonic()
    led.release(first)
    t.join(5)
    assert not t.is_alive()
    assert got["t"] >= t_release
    assert led.outstanding_bytes == GiB
    led.release(got["r"])


def test_plan_and_reserve_replans_against_what_is_left(monkeypatch):
    """The second run plans against the remainder, not against the whole machine."""
    monkeypatch.delenv(INFERENCE_WORKERS_ENV, raising=False)
    budget = 6 * GiB
    led = ResourceLedger(budget_fn=lambda: budget - led.outstanding_bytes)
    seen = []

    def plan_fn(reserved_bytes, reserved_workers):
        seen.append((reserved_bytes, reserved_workers))
        return _plan(
            free_bytes=int((budget - reserved_bytes) / 0.5),
            gdal_cache_bytes=0,
            reserved_workers=reserved_workers,
        )

    p1, r1 = led.plan_and_reserve(plan_fn, minimum_bytes=1, label="one")
    assert seen[-1] == (0, 0)
    p2, r2 = led.plan_and_reserve(plan_fn, minimum_bytes=1, label="two")
    assert seen[-1] == (r1.bytes_, r1.workers)
    assert p2.workers <= max(1, 4 - p1.workers)
    assert led.outstanding_bytes == r1.bytes_ + r2.bytes_
    led.release(r1)
    led.release(r2)


# --------------------------------------------------------------------------- #
# Memory probe — pins the working-set model against a real GLM apply()
# --------------------------------------------------------------------------- #
def _spawn_env():
    """Pin PYTHONPATH to this checkout so the probe never imports a different one."""
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.mark.slow
def test_pool_peak_memory_grows_at_most_one_estimated_stripe_per_worker(tmp_path):
    """peak(4 workers) - peak(1 worker) <= 3 x the plan's stripe working set.

    The policy sizes the pool as ``budget // stripe_bytes``, so the estimate
    must be at least what one extra worker really adds. Measured in fresh
    processes (VmHWM) on a 4 x 3000x3000 float32 stack, mirroring
    ``tests/test_sampling_pool.py``'s sampling-side probe.
    """
    probe = Path(__file__).resolve().parents[1] / "benchmarks" / "_inference_probe.py"

    def run(workers):
        d = tmp_path / f"w{workers}"
        d.mkdir()
        out = subprocess.run(
            [sys.executable, str(probe), str(d), str(workers)],
            check=True,
            capture_output=True,
            text=True,
            env=_spawn_env(),
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    one, four = run(1), run(4)
    plan = plan_inference(
        width=3000,
        tile_rows=256,
        n_features=4,
        feature_itemsizes=[4, 4, 4, 4],
        n_design_cols=5,
        with_mask=False,
        with_extra=False,
        cores=8,
        free_bytes=64 * GiB,
        rows_per_stripe=256,
    )
    extra_bytes = (four["peak_kib"] - one["peak_kib"]) * 1024
    assert extra_bytes <= 3 * plan.stripe_bytes, (
        f"one={one['peak_kib']} KiB four={four['peak_kib']} KiB "
        f"stripe={plan.stripe_bytes / 1024:.0f} KiB"
    )
