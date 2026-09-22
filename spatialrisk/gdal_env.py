"""Point GDAL's temp-file allocator at a directory we know is writable.

``gdal.ComputeProximity()`` computes in Float32. When the destination band is a
different type (ours is ``GDT_UInt32``) GDAL allocates a full-size Float32
working band in a temporary GeoTIFF named by ``CPLGenerateTempFilename()``,
which resolves its directory from the ``CPL_TMPDIR`` / ``TMPDIR`` / ``TEMP``
config options and falls back to ``"."`` -- the process CWD -- when none is
set. On SEPAL the app runs with its CWD on the read-only shared module mount,
so the call dies with::

    Attempt to create new tiff file `./proximity_997_3' failed:
    ./proximity_997_3: Read-only file system

The same defect lives in third-party code we cannot patch (notably
``riskmapjnr.dist_edge_threshold``), so the fix has to be a process-wide GDAL
config rather than a change to any one destination dtype.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import rasterio
from osgeo import gdal

from spatialrisk.parallel import available_cores, free_memory_bytes, worker_threads

logger = logging.getLogger("spatial_risk")

DEFAULT_SAMPLING_CACHEMAX_BYTES = 512 * 1024 * 1024
"""Default ``GDAL_CACHEMAX`` budget (bytes) for the sampling raster read path.

GDAL's block cache defaults to ~5% of physical RAM, so it scales with the host
machine rather than with anything we control -- bounding the numpy working set
of one row-stripe (see ``spatialrisk/sampling``) does **not** bound process
RSS, because GDAL's own native cache lives outside those arrays. 512 MiB is
comfortably above one stripe's decoded size for the raster widths this module
targets while still capping the unbounded default.
"""

_SAMPLING_CACHEMAX_ENV = "SPATIAL_RISK_SAMPLING_CACHEMAX_BYTES"
_SAMPLING_NUM_THREADS_ENV = "SPATIAL_RISK_SAMPLING_NUM_THREADS"
SAMPLING_WORKERS_ENV = "SPATIAL_RISK_SAMPLING_WORKERS"
"""Override for the stripe-pool worker count chosen by :func:`plan_sampling`."""

SAMPLING_MEMORY_FRACTION = 0.5
"""Share of the free memory one sampling job may take for its in-flight stripes."""

#: Rows a stripe aims for before snapping to whole tile rows (see blocked.py).
SAMPLING_TARGET_STRIPE_ROWS = 512

#: Bytes per stripe pixel one worker holds besides the decoded band itself:
#: the validity bool and the compacted valid values ``arr[valid]``. The band
#: and the mask stripe are counted from their own itemsizes.
_STRIPE_EXTRA_BYTES_PER_PIXEL = 1
#: ``np.bincount`` upcasts its input to intp; blocked._COUNT_CHUNK bounds that.
_BINCOUNT_CHUNK_BYTES = (1 << 22) * 8


# Kept as a module attribute (not a bare re-export) so tests can monkeypatch
# the core count seen by sampling_num_threads(); the policy lives in
# spatialrisk.parallel, shared with point extraction and evaluation.
_available_cores = available_cores


def sampling_num_threads() -> int:
    """``GDAL_NUM_THREADS`` for the sampling read path: half the cores, min 1.

    Multi-threaded DEFLATE/LZW tile decoding is where the sampling scan is
    bound once the numpy work is amortised -- measured on the 2.2 Gpx Bolivia
    loss raster plus mask, one stripe pass decodes in 3.2 s single-threaded and
    0.9 s with all 16 cores. Same half-the-cores policy as every other raster
    scan (:func:`spatialrisk.parallel.worker_threads`);
    ``SPATIAL_RISK_SAMPLING_NUM_THREADS`` overrides it for sampling alone and
    ``SPATIAL_RISK_NUM_THREADS`` for all scans.
    """
    return worker_threads(_SAMPLING_NUM_THREADS_ENV, cores=_available_cores())


def _is_usable(d: Path) -> bool:
    """Create ``d`` and prove we can write a file in it.

    A real write probe rather than ``os.access()``: the whole bug this module
    exists for was a directory that looked fine until GDAL created a file in it.
    """
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".probe_{os.getpid()}"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False


def _candidates() -> list:
    """Scratch directories to try, best first."""
    out = []

    env = os.environ.get("CPL_TMPDIR")
    if env:
        out.append(Path(env))

    # ``tempfile.gettempdir()`` honours TMPDIR/TEMP/TMP and falls back to /tmp,
    # which on SEPAL is writable and wiped when the instance closes -- GDAL
    # unlinks its own scratch file as soon as it is done, so nothing
    # accumulates there. Its candidate list ends with the CWD, though, so a
    # machine where every system temp dir is unusable would hand us back the
    # read-only module mount: skip it in that case rather than reintroduce the
    # bug (or litter the checkout in dev, where the CWD *is* writable).
    system_tmp = Path(tempfile.gettempdir())
    if system_tmp.resolve() != Path.cwd().resolve():
        # The uid suffix keeps us off a directory of the same name created by
        # another user on a shared /tmp: that one would not be writable by us
        # and would reproduce the very bug this module exists to prevent.
        out.append(system_tmp / f"spatial_risk_gdal_{os.getuid()}")

    # Last resort: the module's own output root. SEPAL guarantees it is
    # writable, because every app output goes there. Resolved independently of
    # ``project.DATA_DIR`` to avoid an import cycle -- project.py imports this
    # module -- so keep it in step with ``project._resolve_data_dir``.
    data_dir = os.environ.get("SPATIAL_RISK_DATA_DIR")
    base = (
        Path(data_dir)
        if data_dir
        else Path.home() / "module_results" / "spatial_risk_module"
    )
    out.append(base / ".gdal_tmp")

    return out


def scratch_dir() -> Path:
    """Return the app's writable scratch directory, creating it if needed.

    An explicit ``CPL_TMPDIR`` wins when it is genuinely writable, then the
    system temp dir, then the module's output root. Raises ``OSError`` if none
    of them can hold a file, because every caller's alternative is writing into
    the CWD.
    """
    tried = _candidates()
    for candidate in tried:
        if _is_usable(candidate):
            return candidate
    raise OSError(
        "No writable scratch directory found; tried " + ", ".join(str(c) for c in tried)
    )


LOCAL_SCRATCH_ENV = "SPATIAL_RISK_LOCAL_SCRATCH"
"""Override for :func:`local_scratch_dir` (a directory on local disk)."""

#: Roots tried for local-disk scratch, best first. On SEPAL ``/tmp`` is an NFS
#: mount of the user's home (as is ``~/module_results``), and the container's
#: overlay root is the only local disk: ``/var/tmp`` lives there.
_LOCAL_SCRATCH_ROOTS = (Path("/var/tmp"),)


def local_scratch_dir() -> Path:
    """A scratch directory on *local* disk for small, seek-heavy writes.

    :func:`scratch_dir` is the right place for GDAL's large sequential temp
    files (on SEPAL it lands on the NFS home, which has the space). It is the
    wrong place for a writer that seeks and rewrites constantly: tippecanoe
    building a 10 000-point PMTiles archive took 57-70 s with its output on
    NFS and 1.0 s on local disk (SEPAL c8, 2026-09-21). Callers build such
    files here and move the finished result into place.

    ``SPATIAL_RISK_LOCAL_SCRATCH`` wins when writable, then a per-user
    directory under each of :data:`_LOCAL_SCRATCH_ROOTS`, then
    :func:`scratch_dir` so nothing breaks on a box without a usable local
    root. Keep only small files here: the local disk is the container's
    overlay and is far smaller than the NFS home.
    """
    env = os.environ.get(LOCAL_SCRATCH_ENV)
    if env and _is_usable(Path(env)):
        return Path(env)
    for root in _LOCAL_SCRATCH_ROOTS:
        candidate = Path(root) / f"spatial_risk_{os.getuid()}"
        if _is_usable(candidate):
            return candidate
    return scratch_dir()


def configure_gdal_tmpdir() -> Optional[Path]:
    """Point GDAL's temp-file allocator at :func:`scratch_dir`. Idempotent.

    Sets both the environment variable -- inherited by spawned children, such
    as the iCAR MCMC worker started through ``multiprocessing`` in "spawn"
    mode -- and the GDAL config option, which covers the library already
    initialised in this process.

    Returns the configured directory, or ``None`` if configuration failed.
    Never raises: this runs at import time, so an exception here would break
    every import of the package.
    """
    try:
        d = scratch_dir()
        os.environ["CPL_TMPDIR"] = str(d)
        gdal.SetConfigOption("CPL_TMPDIR", str(d))
        return d
    except Exception as e:  # pragma: no cover - defensive, import-time safety
        logger.warning(
            "Could not configure a GDAL temp directory (%s). GDAL will fall back "
            "to the working directory, so raster steps will fail wherever that is "
            "read-only -- set CPL_TMPDIR to somewhere writable.",
            e,
        )
        return None


def sampling_gdal_env(
    cachemax_bytes: Optional[int] = None, num_threads: Optional[int] = None
) -> rasterio.Env:
    """A ``rasterio.Env`` that budgets GDAL's block cache and decode threads.

    Use as ``with sampling_gdal_env(): ...`` around the raster reads done for
    sample generation.

    IMPORTANT -- this budget is **process-wide, not per-thread.** A
    ``rasterio.Env`` sets GDAL config through GDAL's global (not thread-local)
    store, so while this context is open *every* thread sees the reduced
    ``GDAL_CACHEMAX``, including unrelated raster work (training, processing,
    map tiles). Measured directly: with ``rasterio.Env(GDAL_CACHEMAX=64)`` open
    on the main thread, a second thread read back 64, not the default. Do not
    reintroduce a per-thread claim here without re-measuring.

    Two things keep that acceptable rather than harmful: the sampling jobs are
    serialized (see ``gui/tile/sampling_tile.py``), so at most one such budget
    is ever live, and the previous value is restored when the context exits. The
    cost is that concurrent non-sampling raster work may re-decode more blocks
    while a sampling job runs. If that ever shows up as a slowdown, the fix is
    to move the read into a subprocess, which is the only way to get a genuinely
    isolated GDAL cache.

    This bounds the *cache* footprint of a single read; it does not limit how
    many sampling jobs can run at once. That is a separate concern (see the
    job serialization in ``gui/tile/sampling_tile.py``) -- running jobs one at
    a time caps how many such budgets are live simultaneously, it does not by
    itself reduce what any one job needs.

    Parameters
    ----------
    cachemax_bytes : int, optional
        Explicit ``GDAL_CACHEMAX`` value in bytes. Defaults to
        ``SPATIAL_RISK_SAMPLING_CACHEMAX_BYTES`` if set, else
        :data:`DEFAULT_SAMPLING_CACHEMAX_BYTES`.
    num_threads : int, optional
        Explicit ``GDAL_NUM_THREADS`` for tile decoding. Defaults to
        :func:`sampling_num_threads` (half the available cores). The same
        process-wide caveat as the cache budget applies.
    """
    if cachemax_bytes is None:
        env_val = os.environ.get(_SAMPLING_CACHEMAX_ENV)
        cachemax_bytes = int(env_val) if env_val else DEFAULT_SAMPLING_CACHEMAX_BYTES
    if num_threads is None:
        num_threads = sampling_num_threads()
    return rasterio.Env(GDAL_CACHEMAX=cachemax_bytes, GDAL_NUM_THREADS=num_threads)


# --------------------------------------------------------------------------- #
# stripe-pool policy
# --------------------------------------------------------------------------- #
class SamplingPlan:
    """What one sample-generation job may use: stripe height, workers, GDAL budget.

    Attributes are plain ints so the plan can be logged or written to a bench
    record as is. ``memory_source`` names the reading that bounded the memory
    budget (``psutil.available`` or the cgroup limit).
    """

    __slots__ = (
        "rows_per_stripe",
        "workers",
        "gdal_threads",
        "cachemax_bytes",
        "by_cores",
        "by_memory",
        "stripe_bytes",
        "memory_budget_bytes",
        "free_bytes",
        "memory_source",
        "width_bytes_per_row",
    )

    def __init__(self, **kw):
        """Take every slot as a keyword; all of them are required."""
        for name in self.__slots__:
            setattr(self, name, kw[name])

    def __repr__(self):
        """Every field, for logs and bench records."""
        fields = ", ".join(f"{n}={getattr(self, n)!r}" for n in self.__slots__)
        return f"SamplingPlan({fields})"


def _stripe_working_set(width: int, rows: int, itemsize: int, with_mask: bool) -> int:
    per_px = 2 * itemsize + _STRIPE_EXTRA_BYTES_PER_PIXEL + (1 if with_mask else 0)
    return width * rows * per_px + _BINCOUNT_CHUNK_BYTES


def plan_sampling(
    *,
    width: int,
    tile_rows: int,
    itemsize: int,
    with_mask: bool,
    cores: Optional[int] = None,
    free_bytes: Optional[int] = None,
    rows_per_stripe: Optional[int] = None,
    gdal_cache_bytes: Optional[int] = None,
) -> SamplingPlan:
    """Choose stripe height, worker count and GDAL budget for one sampling job.

    Policy (measured 2026-09-21 on the 2.2 Gpx Bolivia raster locally and on
    a SEPAL c8 sandbox, ``tests/test_sampling_bench_sepal.py``):

    * **workers = min(half the affinity cores, memory budget / one stripe)**.
      Pass 1 scaled 5.2 s -> 2.3 s at 4 workers on the c8 and 2.0 s at 8;
      half the cores keeps the last 0.3 s for the app server and costs
      ~190 MiB less per worker not started.
    * the memory budget is :data:`SAMPLING_MEMORY_FRACTION` of the free memory
      (cgroup limit minus usage inside a limited container, else psutil's
      available) minus GDAL's block cache. If even one worker does not fit,
      the stripe shrinks in whole tile rows first, then the job runs one
      worker at one tile row per stripe -- never less, the serial path must
      always be allowed to run.
    * with more than one worker GDAL's own decode threads go to 1 per worker so
      the total stays at the core budget; a single worker keeps
      :func:`sampling_num_threads` (multi-threaded decode is what makes the
      serial pass faster than a one-worker pool).
    * the block cache grows to hold every in-flight stripe of raster and mask
      (``workers x 2 x stripe``) so concurrent readers never evict each
      other's tiles mid-read.

    ``SPATIAL_RISK_SAMPLING_WORKERS`` overrides the worker count outright (min
    1) for experiments; the stripe height and cache still follow the policy.
    An explicit ``rows_per_stripe`` is honoured as given (tests pin awkward
    heights on purpose) and is never shrunk.
    """
    if cores is None:
        cores = _available_cores()
    if free_bytes is None:
        free_bytes, memory_source = free_memory_bytes()
    else:
        memory_source = "given"
    if gdal_cache_bytes is None:
        env_val = os.environ.get(_SAMPLING_CACHEMAX_ENV)
        gdal_cache_bytes = int(env_val) if env_val else DEFAULT_SAMPLING_CACHEMAX_BYTES

    tile_rows = max(1, int(tile_rows))
    if rows_per_stripe is not None:
        rows = int(rows_per_stripe)
        if rows < 1:
            raise ValueError("rows_per_stripe must be >= 1.")
        fixed_rows = True
    else:
        rows = (
            tile_rows
            if tile_rows >= SAMPLING_TARGET_STRIPE_ROWS
            else (SAMPLING_TARGET_STRIPE_ROWS // tile_rows) * tile_rows
        )
        fixed_rows = False

    by_cores = max(1, int(cores) // 2)
    budget = max(0, int(free_bytes * SAMPLING_MEMORY_FRACTION) - gdal_cache_bytes)
    stripe_bytes = _stripe_working_set(width, rows, itemsize, with_mask)
    by_memory = budget // stripe_bytes
    if by_memory < 1 and not fixed_rows:
        # Shrink the stripe in whole tile rows until one worker fits, but never
        # below one tile row.
        while (
            rows > tile_rows
            and _stripe_working_set(width, rows, itemsize, with_mask) > budget
        ):
            rows = max(tile_rows, ((rows // tile_rows) - 1) * tile_rows)
        stripe_bytes = _stripe_working_set(width, rows, itemsize, with_mask)
        by_memory = budget // stripe_bytes
    workers = max(1, min(by_cores, by_memory))

    env_val = os.environ.get(SAMPLING_WORKERS_ENV)
    if env_val:
        workers = max(1, int(env_val))

    width_bytes_per_row = width * (itemsize + (1 if with_mask else 0))
    cachemax = max(gdal_cache_bytes, workers * 2 * rows * width_bytes_per_row)
    gdal_threads = sampling_num_threads() if workers == 1 else 1
    return SamplingPlan(
        rows_per_stripe=rows,
        workers=workers,
        gdal_threads=gdal_threads,
        cachemax_bytes=int(cachemax),
        by_cores=by_cores,
        by_memory=int(max(0, by_memory)),
        stripe_bytes=int(stripe_bytes),
        memory_budget_bytes=int(budget),
        free_bytes=int(free_bytes),
        memory_source=memory_source,
        width_bytes_per_row=int(width_bytes_per_row),
    )


# --------------------------------------------------------------------------- #
# inference stripe plan
# --------------------------------------------------------------------------- #
INFERENCE_WORKERS_ENV = "SPATIAL_RISK_INFERENCE_WORKERS"
"""Override for the worker count chosen by :func:`plan_inference` (min 1)."""

INFERENCE_MEMORY_FRACTION = SAMPLING_MEMORY_FRACTION
INFERENCE_TARGET_STRIPE_ROWS = 256
"""Default stripe height for prediction = the output tile height
(:data:`spatialrisk.parallel.PREDICT_BAND_ROWS`), so a stripe write covers
whole output tiles."""


class InferencePlan:
    """What :func:`plan_inference` decided, and the readings it decided from."""

    __slots__ = (
        "rows_per_stripe",
        "workers",
        "gdal_threads",
        "cachemax_bytes",
        "by_cores",
        "by_memory",
        "stripe_bytes",
        "memory_budget_bytes",
        "free_bytes",
        "memory_source",
    )

    def __init__(self, **kw):
        """Take every slot as a keyword; all of them are required."""
        for name in self.__slots__:
            setattr(self, name, kw[name])

    def __repr__(self):
        """Every field, for logs and bench records."""
        fields = ", ".join(f"{n}={getattr(self, n)!r}" for n in self.__slots__)
        return f"InferencePlan({fields})"


def inference_working_set(
    width: int,
    rows: int,
    *,
    n_features: int,
    feature_itemsizes,
    n_design_cols: int,
    with_mask: bool,
    with_extra: bool,
) -> int:
    """Bytes one prediction stripe holds at its peak, per the engine's body.

    Per pixel: each feature's decoded band plus its float64 column, the one
    DataFrame of float64 columns, the patsy design matrix, the probability
    vector and its rescaled copy, the uint16 output stripe, the mask byte and
    the float64 extra layer (iCAR rho). See the spec §4; pinned by the memory
    probe in ``tests/test_inference_plan.py``.
    """
    per_px = (
        sum(int(s) for s in feature_itemsizes)
        + n_features * 8
        + n_features * 8
        + n_design_cols * 8
        + 2 * 8
        + 2
        + (1 if with_mask else 0)
        + (8 if with_extra else 0)
    )
    return int(width) * int(rows) * per_px


def plan_inference(
    *,
    width: int,
    tile_rows: int,
    n_features: int,
    feature_itemsizes,
    n_design_cols: int,
    with_mask: bool,
    with_extra: bool,
    cores: Optional[int] = None,
    free_bytes: Optional[int] = None,
    rows_per_stripe: Optional[int] = None,
    gdal_cache_bytes: Optional[int] = None,
    reserved_bytes: int = 0,
    reserved_workers: int = 0,
) -> InferencePlan:
    """Choose stripe height, worker count and GDAL budget for one prediction.

    Same policy as :func:`plan_sampling`, with the inference working set.
    ``reserved_bytes`` / ``reserved_workers`` are what other runs in this
    process already hold (:class:`ResourceLedger`); they come off the memory
    and core budgets first, so a second concurrent prediction plans for what
    is left instead of for the whole machine (the 2026-09-22 SEPAL kernel
    death: GLM + RF launched together).

    * ``workers = min(half the affinity cores - reserved_workers,
      budget // stripe_bytes)``, min 1;
    * the budget is :data:`INFERENCE_MEMORY_FRACTION` of the free memory minus
      GDAL's block cache; if one worker does not fit the stripe shrinks in
      whole tile rows, never below one, and the job runs serially;
    * pooled workers get one GDAL decode thread each, a single worker keeps
      :func:`sampling_num_threads`;
    * the cache grows to hold every in-flight stripe of every input.

    ``SPATIAL_RISK_INFERENCE_WORKERS`` overrides the worker count (min 1). An
    explicit ``rows_per_stripe`` is honoured as given and never shrunk.
    """
    if cores is None:
        cores = _available_cores()
    if free_bytes is None:
        free_bytes, memory_source = free_memory_bytes()
    else:
        memory_source = "given"
    if gdal_cache_bytes is None:
        gdal_cache_bytes = DEFAULT_SAMPLING_CACHEMAX_BYTES

    tile_rows = max(1, int(tile_rows))
    if rows_per_stripe is not None:
        rows = int(rows_per_stripe)
        if rows < 1:
            raise ValueError("rows_per_stripe must be >= 1.")
        fixed_rows = True
    else:
        rows = (
            tile_rows
            if tile_rows >= INFERENCE_TARGET_STRIPE_ROWS
            else (INFERENCE_TARGET_STRIPE_ROWS // tile_rows) * tile_rows
        )
        fixed_rows = False

    def _ws(r):
        return inference_working_set(
            width,
            r,
            n_features=n_features,
            feature_itemsizes=feature_itemsizes,
            n_design_cols=n_design_cols,
            with_mask=with_mask,
            with_extra=with_extra,
        )

    by_cores = max(1, int(cores) // 2 - int(reserved_workers))
    budget = max(
        0,
        int(free_bytes * INFERENCE_MEMORY_FRACTION)
        - gdal_cache_bytes
        - int(reserved_bytes),
    )
    stripe_bytes = _ws(rows)
    by_memory = budget // stripe_bytes
    if by_memory < 1 and not fixed_rows:
        while rows > tile_rows and _ws(rows) > budget:
            rows = max(tile_rows, ((rows // tile_rows) - 1) * tile_rows)
        stripe_bytes = _ws(rows)
        by_memory = budget // stripe_bytes
    workers = max(1, min(by_cores, by_memory))

    env_val = os.environ.get(INFERENCE_WORKERS_ENV)
    if env_val:
        workers = max(1, int(env_val))

    raw_row = int(width) * (
        sum(int(s) for s in feature_itemsizes)
        + (1 if with_mask else 0)
        + (4 if with_extra else 0)
    )
    cachemax = max(gdal_cache_bytes, workers * 2 * rows * raw_row)
    gdal_threads = sampling_num_threads() if workers == 1 else 1
    return InferencePlan(
        rows_per_stripe=rows,
        workers=workers,
        gdal_threads=gdal_threads,
        cachemax_bytes=int(cachemax),
        by_cores=by_cores,
        by_memory=int(max(0, by_memory)),
        stripe_bytes=int(stripe_bytes),
        memory_budget_bytes=int(budget),
        free_bytes=int(free_bytes),
        memory_source=memory_source,
    )
