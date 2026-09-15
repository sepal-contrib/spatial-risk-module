"""Two sampling jobs must never perform their raster reads concurrently.

Sampling is memory-heavy (tens of GiB peak on a country-scale raster), so
letting distinct sample names run at the same time in one process multiplies
peak memory and can OOM the whole voila process. `gui/tile/sampling_tile.py`
serializes the raster-read portion of each job through a single-slot queue
(a bounded semaphore); this test proves that serialization by patching
`generate_points` to record its own enter/exit window and asserting no two
windows overlap when two jobs are driven concurrently.
"""

import threading
import time

import pytest

gpd = pytest.importorskip("geopandas")


class _Var:
    def __init__(self, path):
        self.path = path


class _StubProject:
    """Minimal duck-typed stand-in for `spatialrisk.project.Project`."""

    def __init__(self, folders_dir, project_name="proj"):
        self.project_name = project_name

        class _Folders:
            samples_folder = folders_dir

        self.folders = _Folders()
        self._vars = {
            "target": _Var(folders_dir / "target.tif"),
            "mask": _Var(folders_dir / "mask.tif"),
        }
        self.samples = {}

    def get_variable(self, name, year=None):
        return self._vars[name]

    def add_sample(self, sample, key=None, auto_save=True):
        self.samples[key or sample.name] = sample

    def model_copy(self):
        return self


class _StubReactive:
    """Duck-typed `solara.Reactive`: just `.value` / `.set()`."""

    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value


def _fake_generate_points_factory(intervals, lock, hold_s=0.08):
    """Build a `generate_points` stand-in that records its own active window."""

    def _fake_generate_points(raster_path, mask_path, **kwargs):
        start = time.monotonic()
        time.sleep(hold_s)
        end = time.monotonic()
        with lock:
            intervals.append((start, end))
        return gpd.GeoDataFrame(
            {"strata": [0, 1]},
            geometry=gpd.points_from_xy([0, 1], [0, 1]),
            crs="EPSG:3857",
        )

    return _fake_generate_points


def _overlaps(intervals):
    ordered = sorted(intervals)
    for (_, end_a), (start_b, _) in zip(ordered, ordered[1:]):
        if start_b < end_a:
            return True
    return False


def test_two_sampling_jobs_never_read_concurrently(tmp_path, monkeypatch):
    """Two jobs started at (roughly) the same time still run reads one at a time."""
    import spatialrisk.sampling as sampling_pkg
    from gui.tile import sampling_tile as st

    intervals = []
    lock = threading.Lock()
    monkeypatch.setattr(
        sampling_pkg,
        "generate_points",
        _fake_generate_points_factory(intervals, lock),
    )

    project = _StubProject(tmp_path)
    project_reactive = _StubReactive(project)

    threads = []
    for i in range(2):
        t = threading.Thread(
            target=st._run_sampling,
            args=(
                f"job-{i}",
                f"sample-{i}",
                "target",
                "mask",
                "random",
                None,
                False,
                10,
                None,
                1,
                project_reactive,
                None,
                None,
            ),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    st.sampling_jobs.set([])  # module-level reactive: reset for other tests

    assert len(intervals) == 2, "both jobs must have run generate_points"
    assert not _overlaps(intervals), f"raster reads overlapped: {intervals}"
    assert set(project.samples) == {"sample-0", "sample-1"}


def test_sampling_semaphore_is_single_slot():
    """The module exposes a single-slot serialization primitive for jobs."""
    from gui.tile import sampling_tile as st

    sem = getattr(st, "_sampling_slot", None)
    assert sem is not None, "expected a module-level semaphore/lock for sampling jobs"
    # threading.Semaphore/Lock/BoundedSemaphore all support the context-manager
    # protocol; assert it behaves like a single-slot primitive.
    acquired_twice = sem.acquire(blocking=False)
    try:
        assert acquired_twice is True
        assert sem.acquire(blocking=False) is False
    finally:
        sem.release()


def test_sampling_gdal_env_is_process_wide_not_per_thread():
    """Pin the real scope of the sampling cache budget.

    An earlier docstring claimed `rasterio.Env` stores GDAL config per-thread,
    so the budget could not affect other threads. That is false, and the
    difference matters: while a sampling job holds this context, concurrent
    training/processing/map-tile reads also run with the reduced cache. This
    test exists so nobody reinstates the per-thread claim without measuring.
    """
    import threading

    from rasterio.env import get_gdal_config

    from spatialrisk.gdal_env import sampling_gdal_env

    seen = {}

    def worker():
        seen["other_thread"] = get_gdal_config("GDAL_CACHEMAX")

    default = get_gdal_config("GDAL_CACHEMAX")
    with sampling_gdal_env(cachemax_bytes=64):
        seen["main"] = get_gdal_config("GDAL_CACHEMAX")
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert seen["main"] == 64
    # The point of the test: the budget IS visible from another thread.
    assert seen["other_thread"] == 64, (
        "rasterio.Env turned out to be per-thread after all — if this fails, "
        "update the sampling_gdal_env docstring, it documents process-wide scope"
    )
    # ...and it is restored on exit, which is what keeps that acceptable.
    assert get_gdal_config("GDAL_CACHEMAX") == default


def test_sampling_gdal_env_sets_num_threads_to_half_the_affinity_mask(monkeypatch):
    """Decoding gets half the cores this process may run on, never zero.

    Uses the affinity mask, not ``os.cpu_count()``: on SEPAL the process is a
    container whose cgroup quota is smaller than the host core count.
    """
    from rasterio.env import get_gdal_config

    import spatialrisk.gdal_env as ge

    monkeypatch.delenv(ge._SAMPLING_NUM_THREADS_ENV, raising=False)
    monkeypatch.setattr(ge, "_available_cores", lambda: 16)
    assert ge.sampling_num_threads() == 8
    with ge.sampling_gdal_env():
        assert get_gdal_config("GDAL_NUM_THREADS") == 8

    monkeypatch.setattr(ge, "_available_cores", lambda: 1)
    assert ge.sampling_num_threads() == 1


def test_sampling_gdal_env_num_threads_env_override(monkeypatch):
    """``SPATIAL_RISK_SAMPLING_NUM_THREADS`` wins over the half-cores default."""
    from rasterio.env import get_gdal_config

    import spatialrisk.gdal_env as ge

    monkeypatch.setattr(ge, "_available_cores", lambda: 16)
    monkeypatch.setenv(ge._SAMPLING_NUM_THREADS_ENV, "3")
    assert ge.sampling_num_threads() == 3
    with ge.sampling_gdal_env():
        assert get_gdal_config("GDAL_NUM_THREADS") == 3

    with ge.sampling_gdal_env(num_threads=5):
        assert get_gdal_config("GDAL_NUM_THREADS") == 5
