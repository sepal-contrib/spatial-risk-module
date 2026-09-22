"""The stripe engine shared by the GLM, RF and iCAR predictors.

One prediction walks the target grid in full-width stripes (forestatrisk's
``makeblock`` geometry: ``rows`` tall, the last one shorter), reads every
feature of the dataset for the stripe, keeps the pixels where no feature is
nodata and the mask does not suppress them, hands that frame to the model's
``predict_block`` closure and writes the rescaled uint16 stripe. The stripe
grid is part of the output: iCAR resamples its rho and mask per stripe from
geographic bounds, so stripes never switch to native TIFF blocks. The default
height is the output tile height (:data:`spatialrisk.parallel.PREDICT_BAND_ROWS`
= :data:`spatialrisk.raster_profile.BLOCK_SIZE`), which is what
:func:`spatialrisk.gdal_env.plan_inference` starts from and the only unit it
shrinks in.

Memory per stripe is what :func:`spatialrisk.gdal_env.plan_inference`
budgets (see its docstring); the body below is written to match that model,
so a change here must be mirrored there and re-pinned by the memory probe.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from spatialrisk.gdal_env import (
    INFERENCE_LEDGER,
    inference_working_set,
    plan_inference,
)
from spatialrisk.parallel import single_thread_math
from spatialrisk.raster_profile import BLOCK_SIZE, rasterio_profile

logger = logging.getLogger("spatial_risk")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ExtraLayer:
    """A co-registered raster read per stripe by geographic bounds (iCAR rho)."""

    path: Path
    resampling: str = "bilinear"


def _stripes(target_path: PathLike, rows: int):
    """``(index, Window)`` for every full-width stripe, top to bottom.

    The same geometry as ``far.misc.makeblock(target, blk_rows=rows)``; computed
    here so the engine does not reopen the target through GDAL for it.
    """
    with rasterio.open(target_path) as src:
        height, width = src.height, src.width
    out = []
    for i, r in enumerate(range(0, height, rows)):
        out.append((i, Window(0, r, width, min(rows, height - r))))
    return out


class _Handles:
    """Dataset handles for one thread, opened on first use, all closed at the end."""

    def __init__(self, feature_paths, mask, extra_layers):
        """Remember what to open; nothing is opened until a thread asks."""
        self._feature_paths = feature_paths
        self._mask = mask
        self._extra = extra_layers or {}
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all = []

    def get(self):
        """This thread's ``(features, mask, extras)`` handles, opening them once.

        Every handle joins the close list as it is opened, so an open that
        fails halfway still leaves the ones before it closable; the thread's
        slots are filled only once all of them are open, so a half-opened set
        is never handed out.
        """
        loc = self._local
        if not hasattr(loc, "features"):
            opened = []

            def _open(path):
                handle = rasterio.open(path)
                opened.append(handle)
                return handle

            try:
                features = {n: _open(p) for n, p in self._feature_paths.items()}
                mask = _open(self._mask) if self._mask is not None else None
                extra = {n: _open(e.path) for n, e in self._extra.items()}
            finally:
                with self._lock:
                    self._all.extend(opened)
            loc.features, loc.mask, loc.extra = features, mask, extra
        return loc.features, loc.mask, loc.extra

    def close_all(self):
        """Close every handle any thread opened. Called once, in the run's finally."""
        with self._lock:
            handles, self._all = self._all, []
        for h in handles:
            try:
                h.close()
            except Exception:  # closing is best effort on the failure path
                pass


def _read_stripe(
    handles,
    window: Window,
    target_transform,
    feature_names: Sequence[str],
    mask_values,
    mask_by_bounds: bool,
    extra_layers: Optional[Dict[str, ExtraLayer]],
):
    """Read one stripe -> (valid bool[n], block_df of valid rows, extras at valid px).

    Feature columns are float64 with nodata -> NaN; ``valid`` is computed on
    the arrays and the DataFrame is built once from the already-filtered
    columns, which keeps the same rows in the same order as building the full
    frame and boolean-indexing it (what the serial loops did) with one
    full-stripe frame fewer.
    """
    features, mask_src, extra_src = handles
    n_rows, n_cols = int(window.height), int(window.width)
    n = n_rows * n_cols
    bounds = rasterio.windows.bounds(window, target_transform)

    valid = np.ones(n, dtype=bool)
    if mask_src is not None:
        if mask_by_bounds:
            mwin = rasterio.windows.from_bounds(*bounds, mask_src.transform)
            mblock = mask_src.read(
                1,
                window=mwin,
                out_shape=(n_rows, n_cols),
                resampling=Resampling.nearest,
            )
        else:
            mblock = mask_src.read(1, window=window)
        mflat = mblock.ravel()
        invalid = np.isin(mflat, list(mask_values))
        if mask_src.nodata is not None:
            invalid |= mflat == mask_src.nodata
        valid &= ~invalid

    columns = {}
    for name in feature_names:
        src = features[name]
        arr = src.read(1, window=window).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        col = arr.ravel()
        valid &= ~np.isnan(col)
        columns[name] = col

    extras = {}
    for name, layer in (extra_layers or {}).items():
        src = extra_src[name]
        ewin = rasterio.windows.from_bounds(*bounds, src.transform)
        block = src.read(
            1,
            window=ewin,
            out_shape=(n_rows, n_cols),
            resampling=Resampling[layer.resampling],
        )
        extras[name] = block.astype(float).ravel()[valid]

    block_df = pd.DataFrame({name: col[valid] for name, col in columns.items()})
    return valid, block_df, extras


def _predict_stripe(
    handles,
    window,
    target_transform,
    feature_names,
    mask_values,
    mask_by_bounds,
    extra_layers,
    predict_block,
):
    """One stripe's uint16 output (0 where invalid or empty)."""
    import forestatrisk as far

    valid, block_df, extras = _read_stripe(
        handles,
        window,
        target_transform,
        feature_names,
        mask_values,
        mask_by_bounds,
        extra_layers,
    )
    out = np.zeros(int(window.height) * int(window.width), dtype=np.uint16)
    if not block_df.empty:
        proba = np.asarray(predict_block(block_df, extras), dtype=float)
        out[valid] = far.misc.rescale(proba).astype(np.uint16)
    return out.reshape(int(window.height), int(window.width))


def _plan_and_reserve(
    target_path,
    feature_paths,
    mask,
    extra_layers,
    n_design_cols,
    workers,
    rows_per_stripe,
    label,
    log,
):
    """Plan this run against what other runs leave, and reserve it on the ledger.

    Returns ``(plan, reservation)``; the caller releases the reservation in
    its ``finally``. The minimum a run may need is one worker at one tile
    row: if even that does not fit while another run holds memory, the
    ledger blocks here until that run releases (spec §4b).
    """
    with rasterio.open(target_path) as src:
        width = src.width
    itemsizes = []
    for p in feature_paths.values():
        with rasterio.open(p) as src:
            itemsizes.append(np.dtype(src.dtypes[0]).itemsize)
    common = dict(
        width=width,
        tile_rows=BLOCK_SIZE,
        n_features=len(feature_paths),
        feature_itemsizes=itemsizes,
        n_design_cols=(
            n_design_cols if n_design_cols is not None else len(feature_paths) + 1
        ),
        with_mask=mask is not None,
        with_extra=bool(extra_layers),
    )
    minimum = inference_working_set(
        width,
        BLOCK_SIZE,
        **{k: v for k, v in common.items() if k not in ("width", "tile_rows")},
    )

    def plan_fn(reserved_bytes, reserved_workers):
        plan = plan_inference(
            rows_per_stripe=rows_per_stripe,
            reserved_bytes=reserved_bytes,
            reserved_workers=reserved_workers,
            **common,
        )
        if workers is not None:
            plan.workers = max(1, int(workers))
        return plan

    return INFERENCE_LEDGER.plan_and_reserve(
        plan_fn, minimum_bytes=minimum, label=label, log=log
    )


def predict_windowed(
    target_path: PathLike,
    feature_paths: Dict[str, PathLike],
    predict_block: Callable[[pd.DataFrame, Dict[str, np.ndarray]], np.ndarray],
    output_file: PathLike,
    *,
    mask: Optional[PathLike] = None,
    mask_values: Sequence = (0,),
    mask_by_bounds: bool = False,
    extra_layers: Optional[Dict[str, ExtraLayer]] = None,
    workers: Optional[int] = None,
    rows_per_stripe: Optional[int] = None,
    log: Optional[logging.Logger] = None,
    n_design_cols: Optional[int] = None,
    label: Optional[str] = None,
) -> Path:
    """Predict a uint16 probability raster over ``target_path``'s grid.

    Every run reserves its planned memory and workers on the process-wide
    :data:`spatialrisk.gdal_env.INFERENCE_LEDGER` (released when it ends), so
    concurrent predictions share one budget instead of each sizing itself
    against the whole machine; a run that cannot fit its minimum while
    another holds memory waits for that run and logs why. ``label`` names
    the run in those log lines (default: the output file name).

    ``predict_block(block_df, extras)`` receives the valid pixels of one stripe
    (float64 feature columns named as in ``feature_paths``) and the extra
    layers' values at those pixels, and returns probabilities in [0, 1] in the
    same order. Output: uint16, 1..65535 (``far.misc.rescale``), 0 = nodata,
    tiled + compressed per :func:`spatialrisk.raster_profile.rasterio_profile`.

    ``workers=None`` lets :func:`spatialrisk.gdal_env.plan_inference` choose
    the pool size and stripe height; ``workers=1`` runs on the calling thread.
    The file is written as ``<output>.part.tif`` and renamed on success; on
    failure the partial file is removed and a previous output is untouched.
    """
    log = log or logger
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    part = output_file.with_name(output_file.stem + ".part.tif")
    feature_paths = {k: Path(v) for k, v in feature_paths.items()}
    feature_names = list(feature_paths)

    label = label or output_file.name
    plan, reservation = _plan_and_reserve(
        target_path,
        feature_paths,
        mask,
        extra_layers,
        n_design_cols,
        workers,
        rows_per_stripe,
        label,
        log,
    )
    log.info(
        "%s: %d worker(s), %d rows/stripe, budget %.0f MiB (%s), reserved %.0f MiB",
        label,
        plan.workers,
        plan.rows_per_stripe,
        plan.memory_budget_bytes / 2**20,
        plan.memory_source,
        reservation.bytes_ / 2**20,
    )

    with rasterio.open(target_path) as ref:
        profile = ref.profile.copy()
        target_transform = ref.transform
    profile.update(dtype="uint16", count=1, nodata=0)
    profile.update(rasterio_profile("uint16"))

    stripes = _stripes(target_path, plan.rows_per_stripe)
    milestones = {max(1, round(len(stripes) * q)) for q in (0.25, 0.5, 0.75, 1.0)}
    handles = _Handles(feature_paths, mask, extra_layers)
    env = rasterio.Env(
        GDAL_CACHEMAX=plan.cachemax_bytes,
        GDAL_NUM_THREADS=str(plan.gdal_threads),
    )

    def one(i):
        return _predict_stripe(
            handles.get(),
            stripes[i][1],
            target_transform,
            feature_names,
            mask_values,
            mask_by_bounds,
            extra_layers,
            predict_block,
        )

    part.unlink(missing_ok=True)
    try:
        with single_thread_math(), env, rasterio.open(part, "w", **profile) as dst:
            done = 0
            for i, window in stripes:
                dst.write(one(i), 1, window=window)
                done += 1
                if done in milestones:
                    log.info(
                        "Prediction %d%% (%d/%d stripes)",
                        round(100 * done / len(stripes)),
                        done,
                        len(stripes),
                    )
        os.replace(part, output_file)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        handles.close_all()
        INFERENCE_LEDGER.release(reservation)
    return output_file
