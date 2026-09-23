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
import threading
import time
from dataclasses import dataclass
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

INFERENCE_MIN_STRIPE_ROWS = 32
"""Floor for the second shrink stage of :func:`plan_inference`: a stripe that
cannot fit one worker at one tile row halves (256 -> 128 -> 64 -> 32) and stops
here. Below this, output tiles are rewritten too many times to be worth it."""


class InferencePlan:
    """What :func:`plan_inference` decided, and the readings it decided from.

    Every field but ``cachemax_bytes`` describes memory this run will really
    take and really reserves on the :class:`ResourceLedger`
    (``workers x stripe_bytes``). ``cachemax_bytes`` is neither: it is the
    ``GDAL_CACHEMAX`` the run asks for, unreserved and -- GDAL latching the
    value on first use -- honoured only if this is the process's first job
    (see :func:`plan_inference`).

    ``over_budget_bytes`` is how far one stripe at the final height exceeds
    the memory budget, 0 when it fits; non-zero means the run proceeds
    anyway (with one worker unless an explicit override set more).
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
        "over_budget_bytes",
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
    DataFrame of float64 columns, ``n_design_cols`` float64 working columns
    for the model's linear predictor, the probability vector and its
    rescaled copy, the uint16 output stripe, the mask byte and the float64
    extra layer (iCAR rho).

    ``n_design_cols`` is the per-pixel float64 width the model closure is
    charged for, not necessarily the fitted formula's column count: GLM and
    iCAR pass
    :attr:`spatialrisk.mlmodels.linear_predictor.LinearPredictor.working_set_columns`
    (1 -- its ``eta()`` walks the stripe in row chunks bounded by its own
    ``chunk_scratch_bytes``, which this budget leaves to headroom instead of
    charging per pixel; see that module's docstring and
    ``tests/test_linear_predictor.py``), while RF passes
    ``len(design_info.column_names)`` because its trees consume the full
    one-hot matrix at once. See the spec §4; pinned by the memory probe in
    ``tests/test_inference_plan.py``.
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
    workers_override: Optional[int] = None,
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
      GDAL's block cache; if one worker does not fit, the stripe shrinks in
      two stages -- first in whole tile rows down to one tile row, then by
      halving below the tile height (256 -> 128 -> 64 -> 32) down to
      :data:`INFERENCE_MIN_STRIPE_ROWS` -- and the job runs serially;
    * pooled workers get one GDAL decode thread each, a single worker keeps
      :func:`sampling_num_threads`;
    * the cache grows to hold every in-flight stripe of every input.

    :data:`INFERENCE_TARGET_STRIPE_ROWS` equals the
    :data:`spatialrisk.raster_profile.BLOCK_SIZE` passed as ``tile_rows``, so
    the starting height is already one tile row and stage 1 (``while rows >
    tile_rows``) never runs; only stage 2's halving applies. A caller with
    ``tile_rows`` below the target starts taller, so stage 1 shrinks it in
    whole tile rows first -- keeping stripes tile-aligned as long as possible
    -- before stage 2 ever halves below one tile row. If a stripe at
    :data:`INFERENCE_MIN_STRIPE_ROWS` still overruns the budget, the plan
    proceeds anyway (one worker) and ``over_budget_bytes`` reports by how
    much; the engine warns about it.

    ``cachemax_bytes`` is budgeted for only as the flat
    :data:`DEFAULT_SAMPLING_CACHEMAX_BYTES` subtracted from the memory budget
    above, never as the larger figure this returns, and it is not reserved on
    the :class:`ResourceLedger` at all: a second concurrent run plans as if
    one nominal cache existed. It is a request rather than a size in any
    case -- GDAL reads ``GDAL_CACHEMAX`` the first time it needs the block
    cache and latches it, so in the long-lived app process the first job of
    the session fixes the cache every later one runs with, whatever their
    plans say.

    ``SPATIAL_RISK_INFERENCE_WORKERS`` overrides the worker count (min 1), and
    a caller's own ``workers_override`` (min 1) wins over both the policy and
    that variable. An override goes in here rather than onto the returned plan
    because the count also sizes ``gdal_threads`` and ``cachemax_bytes``;
    ``by_cores`` and ``by_memory`` still report what the policy would have
    chosen. An explicit ``rows_per_stripe`` is honoured as given and never
    shrunk.
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
        # Stage 1: whole tile rows, down to one tile row.
        while rows > tile_rows and _ws(rows) > budget:
            rows = max(tile_rows, ((rows // tile_rows) - 1) * tile_rows)
        # Stage 2: below the tile height, halving so stripes still divide a
        # tile row (256 -> 128 -> 64 -> 32); stops at INFERENCE_MIN_STRIPE_ROWS.
        while rows > INFERENCE_MIN_STRIPE_ROWS and _ws(rows) > budget:
            rows = max(INFERENCE_MIN_STRIPE_ROWS, rows // 2)
        stripe_bytes = _ws(rows)
        by_memory = budget // stripe_bytes
    workers = max(1, min(by_cores, by_memory))

    env_val = os.environ.get(INFERENCE_WORKERS_ENV)
    if env_val:
        workers = max(1, int(env_val))
    if workers_override is not None:
        workers = max(1, int(workers_override))

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
        over_budget_bytes=int(max(0, stripe_bytes - budget)),
        free_bytes=int(free_bytes),
        memory_source=memory_source,
    )


# --------------------------------------------------------------------------- #
# cross-run budget: one ledger per process
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Reservation:
    """What one running job holds: planned bytes and worker threads."""

    bytes_: int
    workers: int
    label: str
    id: int


class ResourceLedger:
    """Live reservations of memory and workers across the jobs of this process.

    A per-run plan cannot see the other runs; when two predictions were
    launched together on SEPAL each sized itself against the whole machine
    and the kernel died (2026-09-22). Every engine run reserves what its plan
    will use and releases it when done, and the next plan is made against
    the remainder (:func:`plan_inference` ``reserved_bytes`` /
    ``reserved_workers``).

    ``reserve`` blocks only when *another* reservation is outstanding and
    even ``minimum_bytes`` (one worker at one tile row) does not fit the
    budget; a release wakes it. With nothing outstanding the minimum always
    proceeds, so a lone job on a starved machine still runs (the same rule
    as :func:`plan_sampling`). The reading is conservative on purpose: a
    running job's allocations are already gone from the free-memory reading
    *and* still counted here, which only makes the later job smaller.

    What is tracked is the stripe working sets only. A plan's
    ``cachemax_bytes`` is never reserved here: it is allowed for once, flatly,
    as the :data:`DEFAULT_SAMPLING_CACHEMAX_BYTES` that
    :func:`plan_inference` takes off the budget, and the GDAL block cache is
    process-wide and latched at its first use anyway, so it is not a
    per-reservation quantity to begin with.
    """

    def __init__(self, budget_fn=None, clock=time.monotonic, sleep_log_every_s=30.0):
        """Take the budget reading, the clock and how often a wait is logged."""
        self._budget_fn = budget_fn or self._default_budget
        self._clock = clock
        self._log_every = float(sleep_log_every_s)
        self._cond = threading.Condition()
        self._live = {}
        self._next_id = 0

    def _default_budget(self) -> int:
        """Free memory this process may still claim, net of what is outstanding."""
        free, _ = free_memory_bytes()
        return max(0, int(free * INFERENCE_MEMORY_FRACTION) - self.outstanding_bytes)

    @property
    def outstanding_bytes(self) -> int:
        """Bytes every live reservation holds.

        Read without taking the lock, and over a snapshot. Not for fear of
        deadlock: ``threading.Condition`` owns an ``RLock``, so the callers
        that reach this from inside ``with self._cond`` (a ``budget_fn`` in
        the wait loop, :meth:`plan_and_reserve`) could re-enter it freely.
        The lock is skipped because those callers already hold it and the sum
        is the only thing it would protect; the snapshot is what makes the
        property safe for a reader that does *not* hold it, since iterating
        ``self._live`` directly can raise "dictionary changed size during
        iteration" the moment another thread records or releases.
        """
        return sum(r.bytes_ for r in list(self._live.values()))

    @property
    def outstanding_workers(self) -> int:
        """Worker threads every live reservation holds (snapshot, see above)."""
        return sum(r.workers for r in list(self._live.values()))

    def snapshot(self) -> list:
        """The live reservations, as a list, taken under the lock."""
        with self._cond:
            return list(self._live.values())

    def _wait_until_fits(self, minimum_bytes, label, log):
        """Under the lock: block while others hold memory and the minimum cannot fit."""
        last_log = None
        while self._live and self._budget_fn() < minimum_bytes:
            now = self._clock()
            if last_log is None or now - last_log >= self._log_every:
                holders = ", ".join(
                    f"{r.label} {r.bytes_ / 2**20:.0f} MiB" for r in self._live.values()
                )
                (log or logger).info(
                    "%s: waiting for memory (needs %.0f MiB; held by %s)",
                    label,
                    minimum_bytes / 2**20,
                    holders,
                )
                last_log = now
            self._cond.wait(timeout=self._log_every)

    def _record(self, bytes_, workers, label) -> Reservation:
        """Add one reservation to the live set and return it."""
        self._next_id += 1
        r = Reservation(int(bytes_), int(workers), str(label), self._next_id)
        self._live[r.id] = r
        return r

    def reserve(
        self, *, bytes_, workers, label, minimum_bytes, log=None
    ) -> Reservation:
        """Record a reservation, waiting first if the minimum cannot fit."""
        with self._cond:
            self._wait_until_fits(minimum_bytes, label, log)
            return self._record(bytes_, workers, label)

    def plan_and_reserve(self, plan_fn, *, minimum_bytes, label, log=None):
        """Plan against what is left, wait if needed, re-plan, reserve. Atomic.

        ``plan_fn(reserved_bytes, reserved_workers) -> plan`` with
        ``plan.workers`` and ``plan.stripe_bytes``. Returns ``(plan, reservation)``.
        """
        with self._cond:
            self._wait_until_fits(minimum_bytes, label, log)
            plan = plan_fn(self.outstanding_bytes, self.outstanding_workers)
            r = self._record(plan.workers * plan.stripe_bytes, plan.workers, label)
            return plan, r

    def release(self, reservation) -> None:
        """Drop a reservation and wake every thread waiting for room."""
        with self._cond:
            self._live.pop(reservation.id, None)
            self._cond.notify_all()


INFERENCE_LEDGER = ResourceLedger()
"""The process-wide ledger every ``predict_windowed`` run reserves on."""
