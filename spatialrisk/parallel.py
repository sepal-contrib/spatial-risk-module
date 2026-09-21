"""Shared policy for multi-threaded raster scans.

Several library paths (point extraction, evaluation tallies) walk large
rasters block by block. They share two facts worth stating once:

* **GDAL releases the GIL** during ``read``, and per-block decoding is
  CPU-bound, so a pool of Python threads that each own a dataset handle
  scales almost linearly. ``GDAL_NUM_THREADS`` is *not* a substitute: it
  only parallelises inside one read call.
* **GDAL's block cache buys nothing** for a scan that reads every block
  exactly once, yet by default it grows to ~5% of physical RAM and holds
  every decoded tile. Capping it is what actually bounds process RSS.
"""

import contextlib
import os

import rasterio
from threadpoolctl import threadpool_limits

NUM_THREADS_ENV = "SPATIAL_RISK_NUM_THREADS"
"""Environment override for :func:`worker_threads` (integer, min 1)."""

PREDICT_BAND_ROWS = 256
"""Rows per band when a predictor streams a raster.

Equal to the output tile height (:data:`spatialrisk.raster_profile.BLOCK_SIZE`)
so every band write covers whole tiles: GDAL never has to hold a half-written
tile in its cache, nor re-read and re-compress one when the cache is small.
forestatrisk's default of 128 rows produced exactly that for every tile.
"""

SCAN_CACHEMAX_BYTES = 64 * 1024 * 1024
"""``GDAL_CACHEMAX`` for single-pass scans.

Measured 2026-09-15 on a 1.6 Gpx stack: the default cache held 1.5 GiB of
tiles read once; 64 MiB gave identical output and wall clock. Even blocks far
larger than the cap (39 MiB full-width strips, 8 threads) read correctly, since
a block in flight is held regardless of the budget.
"""


def available_cores() -> int:
    """Cores this process may actually run on.

    The affinity mask rather than ``os.cpu_count()``: SEPAL runs the app in a
    container whose cgroup quota is usually smaller than the host's core
    count, and ``cpu_count`` reports the host. Falls back to ``cpu_count``
    where affinity is unsupported (macOS), then to 1.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def worker_threads(env_var: str = None, cores: int = None) -> int:
    """Threads for a raster scan: half the cores, min 1.

    Half leaves room for the Solara server and any concurrent job on the same
    instance. Overrides, most specific first: ``env_var`` (a job-specific
    variable such as ``SPATIAL_RISK_SAMPLING_NUM_THREADS``, if the caller has
    one), then ``SPATIAL_RISK_NUM_THREADS``, then half of ``cores`` (defaults
    to :func:`available_cores`).
    """
    for name in (env_var, NUM_THREADS_ENV):
        env_val = os.environ.get(name) if name else None
        if env_val:
            return max(1, int(env_val))
    if cores is None:
        cores = available_cores()
    return max(1, cores // 2)


def scan_env() -> rasterio.Env:
    """A ``rasterio.Env`` with the block cache capped for a single-pass scan."""
    return rasterio.Env(GDAL_CACHEMAX=SCAN_CACHEMAX_BYTES)


@contextlib.contextmanager
def single_thread_math():
    """Run the body with the BLAS and OpenMP pools pinned to one thread each.

    The predictors call ``predict_proba`` on a few hundred thousand rows by
    a handful of columns at a time. OpenBLAS parallelises that tiny
    matrix-vector product across every core and its workers spin-wait, and
    scikit-learn's OpenMP pool (libgomp) spins the same way around it:
    measured 2026-09-15 on a 256 Mpx GLM prediction, 69 s of CPU for 8 s of
    wall clock on 16 cores. Pinned, the wall clock and the probabilities are
    unchanged and CPU drops to the wall clock, leaving the cores to
    scikit-learn's own tree workers (joblib threads, unaffected) and to the
    Solara server. Limiting BLAS alone is not enough: the OpenMP workers
    still burned 7x the wall clock.
    """
    with threadpool_limits(limits=1):
        yield


# --------------------------------------------------------------------------- #
# memory a job may still take
# --------------------------------------------------------------------------- #
def parse_cgroup_bytes(text: str):
    """Parse a cgroup memory file: ``max`` (v2) or the v1 no-limit sentinel -> None."""
    text = text.strip()
    if not text or text == "max":
        return None
    value = int(text)
    # cgroup v1 reports "unlimited" as a huge page-aligned number (2**63 - 4096).
    if value >= (1 << 62):
        return None
    return value


def _read_cgroup_bytes(path: str):
    try:
        with open(path, encoding="ascii") as fh:
            return parse_cgroup_bytes(fh.read())
    except (OSError, ValueError):
        return None


def cgroup_memory() -> dict:
    """The container's memory limit and current usage, whichever cgroup version.

    ``limit`` is None when the cgroup sets no limit (SEPAL's sandbox reports
    ``memory.max = max``, so the host reading is the one that counts there).
    """
    limit = _read_cgroup_bytes("/sys/fs/cgroup/memory.max")
    if limit is not None or os.path.exists("/sys/fs/cgroup/memory.current"):
        return {
            "version": 2,
            "limit": limit,
            "usage": _read_cgroup_bytes("/sys/fs/cgroup/memory.current"),
        }
    limit = _read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None:
        return {
            "version": 1,
            "limit": limit,
            "usage": _read_cgroup_bytes("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        }
    return {"version": None, "limit": None, "usage": None}


def host_memory() -> dict:
    """Total and available bytes as psutil sees the host."""
    import psutil

    vm = psutil.virtual_memory()
    return {"total": int(vm.total), "available": int(vm.available)}


def free_memory_bytes():
    """Bytes a job may still allocate, and which reading bounded it.

    In a container the cgroup limit is the wall the OOM killer enforces, and
    it can be far below what psutil reports for the host; outside one only the
    host reading exists. The tighter of the two wins.
    """
    cg = cgroup_memory()
    free = host_memory()["available"]
    source = "psutil.available"
    if cg.get("limit") is not None:
        cg_free = cg["limit"] - (cg.get("usage") or 0)
        if cg_free < free:
            free, source = cg_free, f"cgroup v{cg['version']} limit - usage"
    return max(0, int(free)), source
