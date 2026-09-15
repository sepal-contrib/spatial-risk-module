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

import os

import rasterio

NUM_THREADS_ENV = "SPATIAL_RISK_NUM_THREADS"
"""Environment override for :func:`worker_threads` (integer, min 1)."""

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


def worker_threads() -> int:
    """Reader threads for a raster scan: half the cores, min 1.

    Half leaves room for the Solara server and any concurrent job on the same
    instance. ``SPATIAL_RISK_NUM_THREADS`` overrides the default.
    """
    env_val = os.environ.get(NUM_THREADS_ENV)
    if env_val:
        return max(1, int(env_val))
    return max(1, available_cores() // 2)


def scan_env() -> rasterio.Env:
    """A ``rasterio.Env`` with the block cache capped for a single-pass scan."""
    return rasterio.Env(GDAL_CACHEMAX=SCAN_CACHEMAX_BYTES)
