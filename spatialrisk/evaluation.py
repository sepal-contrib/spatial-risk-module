"""Quantitative model evaluation (udef-arp accuracy indices).

Promoted verbatim from notebooks/6.models_evaluation.ipynb so the GUI and the
notebook share one implementation. Native two-explicit-layer port of
forestatrisk.validation_udef_arp — no forestatrisk dependency.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.windows import Window

from spatialrisk.parallel import scan_env, worker_threads

FAMILY = {"glm": "GLM", "rf": "RF", "icar": "ICAR", "mw": "MW", "jnr": "JNR"}
FOREST_VAR = "forest_gfc"  # dataset feature used as 'forest at period start'

# Axis titles of the predicted-vs-observed scatter. Deliberately NOT i18n'd:
# spatialrisk/ must never import gui/, and these strings are baked into the
# archived PNG. A localized chart supplies its own labels at render time.
PRED_OBS_X_LABEL = "Observed deforestation (ha)"
PRED_OBS_Y_LABEL = "Predicted deforestation (ha)"

_OBS_COL = "ndefor_obs_ha"
_PRED_COL = "ndefor_pred_ha"

# The columns any predicted-vs-observed renderer needs: the two coordinates,
# plus the cell id and forest area both charts carry into their labels. A
# PredObsPlotData may hold MORE than these (compute_validation's frame holds the
# full 6-column CSV table) but never fewer — see PredObsPlotData.__post_init__.
PLOT_COLUMNS = ("cell", "nfor_obs_ha", _OBS_COL, _PRED_COL)

# Axis domain used when the data carries no finite value at all (empty result,
# all-NaN or all-infinite series). A unit range keeps both renderers valid.
_FALLBACK_AXIS = (0.0, 1.0)


def interval_from_target(name):
    """'forest_loss_2015_2020' -> 5; None if fewer than two 4-digit years."""
    yrs = [int(y) for y in re.findall(r"\d{4}", name or "")]
    return (yrs[1] - yrs[0]) if len(yrs) >= 2 else None


def label_for(pred):
    """Short display label for a prediction (e.g. 'GLM', 'MW_w11')."""
    fam = FAMILY.get(pred.model_key.split("_")[0], pred.model_key)
    return f"{fam}_w{pred.window}" if pred.window is not None else fam


def run_label_for(pred):
    """Display label that stays unique across models and named runs.

    ``label_for`` alone collides once two models of one family predict the
    same dataset (two MW models both render 'MW_w5'), so pickers append the
    run that produced the map: the user-chosen prediction name when the run
    was named, else the model key. Filenames use ``artifact_label_for`` —
    this label is display-only.
    """
    run = getattr(pred, "name", None) or pred.model_key
    return f"{label_for(pred)} · {run}"


def artifact_label_for(pred):
    """Run-qualified, filename-safe label for evaluation rows AND artifacts.

    The charts layer derives plot/CSV paths from a row's ``model`` value, so
    the display label and the artifact stem must be the same string. It also
    keys the shared defrate cache — qualifying it by run keeps two models of
    one family (e.g. two MW models at window 5) from sharing cache entries
    and overwriting each other's plots.
    """
    run = getattr(pred, "name", None) or pred.model_key
    safe_run = re.sub(r"[^A-Za-z0-9_-]+", "_", str(run)).strip("_")
    if not safe_run:
        # A fully non-ASCII run name sanitizes to nothing; fall back to the
        # model key so two such runs cannot share a stem like "GLM_".
        safe_run = re.sub(r"[^A-Za-z0-9_-]+", "_", str(pred.model_key)).strip("_")
    return f"{label_for(pred)}_{safe_run}"


def make_square(raster_file, square_size):
    """Coarse-grid partition (replicates forestatrisk.make_square, no far dep)."""
    ds = gdal.Open(str(raster_file))
    ncol, nrow = ds.RasterXSize, ds.RasterYSize
    del ds
    nsquare_x = int(np.ceil(ncol / square_size))
    nsquare_y = int(np.ceil(nrow / square_size))
    nsquare = nsquare_x * nsquare_y
    x = list(range(0, ncol, square_size))
    y = list(range(0, nrow, square_size))
    nx = [square_size] * nsquare_x
    ny = [square_size] * nsquare_y
    if ncol % square_size > 0:
        nx[-1] = ncol % square_size
    if nrow % square_size > 0:
        ny[-1] = nrow % square_size
    return nsquare, nsquare_x, nsquare_y, x, y, nx, ny


def _finite_mask(points):
    """Boolean mask of rows whose observed AND predicted values are finite."""
    obs = pd.to_numeric(points[_OBS_COL], errors="coerce").to_numpy(dtype="float64")
    pred = pd.to_numeric(points[_PRED_COL], errors="coerce").to_numpy(dtype="float64")
    return np.isfinite(obs) & np.isfinite(pred)


def pred_obs_axis_bounds(points):
    """One common finite axis domain spanning BOTH observed and predicted values.

    Matches the legacy ``p = [min over both columns, max over both columns]``
    whenever the data is well-behaved, but never returns NaN/inf or a zero-width
    domain (which collapses an ECharts axis and breaks a 1:1 reference line):

    * no finite value at all -> ``(0.0, 1.0)``
    * constant series at 0   -> ``(0.0, 1.0)``
    * constant series at v   -> ``(v - |v|, v + |v|)``, i.e. ``(0, 2v)`` for v > 0
    """
    values = np.concatenate(
        [
            pd.to_numeric(points[c], errors="coerce").to_numpy(dtype="float64")
            for c in (_OBS_COL, _PRED_COL)
        ]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return _FALLBACK_AXIS

    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        pad = abs(lo)
        if pad == 0.0:
            return _FALLBACK_AXIS
        return (lo - pad, hi + pad)
    return (lo, hi)


@dataclass(frozen=True, eq=False)
class PredObsPlotData:
    """Everything a predicted-vs-observed chart needs, with no raster access.

    Shared by the archived matplotlib PNG and the interactive ECharts scatter so
    the two can never disagree. ``eq=False`` on purpose: the default dataclass
    ``__eq__`` would compare ``points`` element-wise and raise "truth value of a
    DataFrame is ambiguous" during reacton's prop diffing. Identity equality also
    makes every freshly computed result re-render, which is the safe default.

    ``points`` is a per-cell frame with non-finite rows still in it; renderers
    must use ``finite_points`` instead. It arrives by one of two paths, with
    different widths:

    * from ``compute_validation`` — the full 6-column table, and exactly what
      ``write_pred_obs_csv`` persists;
    * from a GUI loader reading a saved point CSV back
      (``gui.scripts.evaluation_echarts``) — only the 4 columns a chart draws.

    So ``points`` is NOT guaranteed to be CSV-width. Only the 4 columns in
    ``PLOT_COLUMNS`` are guaranteed, and ``__post_init__`` enforces them; pass a
    plot_data to ``write_pred_obs_csv`` only when it came from
    ``compute_validation``.
    """

    model: str
    period: str
    csize_px: int
    csize_ha: float
    points: pd.DataFrame
    axis_min: float
    axis_max: float
    medae: float
    r2: float
    ncell: int

    def __post_init__(self):
        """Structural guard for the "no NaN/inf reaches a renderer" invariant.

        Pure validation only — the dataclass is frozen and this never mutates
        ``self`` (no ``object.__setattr__`` needed). Raises ``ValueError`` if
        ``axis_min``/``axis_max`` are non-finite or form a zero-width/inverted
        domain, if ``ncell`` disagrees with the persisted point count, or if
        ``points`` is missing any of ``PLOT_COLUMNS``.

        The column check is what makes the two construction paths (see the class
        docstring) safe to treat alike: it turns "a renderer will KeyError on
        this frame" and "this CSV was truncated or renamed upstream" into one
        error at construction, where the frame's origin is still on the stack.
        """
        missing = [c for c in PLOT_COLUMNS if c not in self.points.columns]
        if missing:
            raise ValueError(
                f"points is missing required plot column(s) {missing}; "
                f"got {list(self.points.columns)!r}"
            )
        if not (np.isfinite(self.axis_min) and np.isfinite(self.axis_max)):
            raise ValueError(
                f"axis_min/axis_max must be finite, got "
                f"({self.axis_min!r}, {self.axis_max!r})"
            )
        if self.axis_min >= self.axis_max:
            raise ValueError(
                f"axis_min must be strictly less than axis_max (non-degenerate "
                f"domain), got ({self.axis_min!r}, {self.axis_max!r})"
            )
        if self.ncell != len(self.points):
            raise ValueError(
                f"ncell ({self.ncell!r}) does not match len(points) "
                f"({len(self.points)!r})"
            )

    @property
    def title(self):
        """The chart title, formatted exactly as on the archived PNG.

        English, PNG-formatted text: GUI consumers must build their own
        translated string via ``t(...)`` rather than reuse this.
        """
        return (
            f"{self.model} model, {self.period} period\n"
            f"Predicted vs. observed deforestation in {self.csize_ha} ha grid cells."
        )

    @property
    def annotation(self):
        """The 'MedAE / R2 / n' summary block, formatted as on the PNG.

        English, PNG-formatted text: GUI consumers must build their own
        translated string via ``t(...)`` rather than reuse this.
        """
        return (
            f"MedAE = {self.medae:.2f} ha\n"
            f"R2 = {self.r2:.2f}\n"
            f"n = {self.ncell:d}"
        )

    @property
    def x_label(self):
        """Axis title for the observed-deforestation axis."""
        return PRED_OBS_X_LABEL

    @property
    def y_label(self):
        """Axis title for the predicted-deforestation axis."""
        return PRED_OBS_Y_LABEL

    @property
    def finite_points(self):
        """Plottable subset: rows where both values are finite (never NaN/inf)."""
        return self.points[_finite_mask(self.points)]


@dataclass(frozen=True, eq=False)
class ValidationResult:
    """Source of truth for one validation run: metrics + shared chart input."""

    indices: dict[str, Any]
    plot_data: PredObsPlotData


SCAN_CHUNK_ROWS = 256
"""Rows decoded at a time inside one band of :func:`scan_row_bands`.

Bounds a worker's footprint to ``chunk x width x (layer bytes + masks)``
regardless of the band height the caller asked for: a 1000 px coarse cell on
a 49k-wide raster would otherwise hold ~1 GB of temporaries per thread.
"""


def scan_row_bands(
    paths, band_rows, fn, *, combine=None, finalize=None, num_threads=None
):
    """Apply ``fn(band_index, arrays)`` to every full-width row band of ``paths``.

    ``arrays`` is one 2-D array per path, all for the same ``Window(0, row0,
    width, rows)``. Bands are dealt to a thread pool where each worker opens
    its own handles (rasterio handles are not thread-safe), and the results
    come back as a list in band order, so any accumulation the caller does is
    deterministic whatever the thread count. Runs under
    :func:`spatialrisk.parallel.scan_env` so GDAL's block cache does not
    balloon on a single-pass scan.

    Full-width bands rather than squares because the app writes predictions as
    row strips: a square read decodes every strip it touches, and the next
    square in the row decodes them again. One band read decodes each strip
    exactly once, on tiled and stripped files alike.

    When ``band_rows`` exceeds :data:`SCAN_CHUNK_ROWS` and ``combine`` is
    given, each band is decoded in sub-windows of at most that many rows and
    ``combine(acc, part)`` folds the per-chunk results, which keeps worker
    memory flat for large bands. Callers whose tallies are additive (counts,
    sums of counts x weights) pass an elementwise add. ``finalize(acc)``, if
    given, reduces a band's folded result before it is returned, so bulky
    accumulators (a cells x categories count matrix) die inside the worker
    instead of being retained for every band until the scan ends.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = [str(p) for p in paths]
    with rasterio.open(paths[0]) as src:
        height, width = src.height, src.width
    starts = list(range(0, height, band_rows))
    chunk = band_rows if combine is None else min(band_rows, SCAN_CHUNK_ROWS)

    def _read(row0, rows):
        win = Window(0, row0, width, rows)
        arrays = []
        for path in paths:
            with rasterio.open(path) as src:
                arrays.append(src.read(1, window=win))
        return arrays

    def _one(i):
        row0 = starts[i]
        end = min(row0 + band_rows, height)
        acc = None
        for r0 in range(row0, end, chunk):
            part = fn(i, _read(r0, min(chunk, end - r0)))
            acc = part if acc is None else combine(acc, part)
        return acc if finalize is None else finalize(acc)

    if num_threads is None:
        num_threads = worker_threads()
    with scan_env():
        if num_threads <= 1 or len(starts) < 2:
            return [_one(i) for i in range(len(starts))]
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            return list(pool.map(_one, range(len(starts))))


def _add_parts(acc, part):
    """Elementwise in-place sum of two tuples of arrays (additive-tally combiner).

    In place because ``acc`` is always the worker's own first chunk result;
    avoiding a copy matters when a part is a cells x 65535 count matrix.
    """
    for a, b in zip(acc, part):
        a += b
    return acc


def count_categories(values, cat):
    """Count how often each entry of ``cat`` occurs in ``values``.

    Equivalent to ``pd.Categorical(values, categories=cat).value_counts()``
    (values outside ``cat`` are ignored, result in ``cat`` order) but ~50x
    faster: the Categorical rebuilt its 65535-entry index on every call, which
    was 60% of a validation run. For non-negative integer data a single
    ``np.bincount`` is taken and indexed by ``cat``; anything else (floats,
    negatives) goes through ``searchsorted`` so exact-equality semantics hold
    for any dtype.
    """
    cat = np.asarray(cat)
    values = np.asarray(values).ravel()
    if values.size == 0:
        return np.zeros(len(cat), dtype=np.int64)
    if (
        values.dtype.kind in "iu"
        and cat.dtype.kind in "iu"
        and cat.min() >= 0
        and cat.max() < 1 << 24
    ):
        nonneg = values[values >= 0] if values.dtype.kind == "i" else values
        counts = np.bincount(nonneg, minlength=int(cat.max()) + 1)
        return counts[cat]
    order = np.argsort(cat, kind="stable")
    sorted_cat = cat[order]
    pos = np.searchsorted(sorted_cat, values)
    pos_c = np.minimum(pos, len(cat) - 1)
    hit = sorted_cat[pos_c] == values
    counts_sorted = np.bincount(pos_c[hit], minlength=len(cat))
    out = np.empty(len(cat), dtype=np.int64)
    out[order] = counts_sorted
    return out


def compute_validation(
    defor_file,
    forest_file,
    riskmap_file,
    tab_file_defor,
    time_interval,
    csize_coarse_grid=300,
    model_name="model",
    period="calibration",
):
    """Per-cell tally + accuracy indices. Pure computation — writes no files.

    Numerics are frozen: same formulas, same ``round(..., 2)``, same dropped
    cells (``nfor_obs == 0``) and same column order as before the split.

    defor_file   : binary deforestation in the period (1 = deforested)
    forest_file  : binary forest at the START of the period (1 = forest)
    riskmap_file : UInt16 categorical risk (categories 1..65535, nodata 0)
    tab_file_defor: per-category defrate CSV (cols 'cat', 'defor_dens')
    """
    defor_dens_per_cat = pd.read_csv(tab_file_defor)
    cat = defor_dens_per_cat["cat"].values
    defor_dens_period = defor_dens_per_cat["defor_dens"].values * time_interval

    with rasterio.open(str(defor_file)) as src:
        gt = src.transform.to_gdal()
    pix_area = gt[1] * (-gt[5])
    csize_ha = round(csize_coarse_grid * csize_coarse_grid * pix_area / 10000, 2)

    nsquare, nsquare_x, nsquare_y, x, _y, nx, _ny = make_square(
        defor_file, csize_coarse_grid
    )
    # Column edges of the coarse cells; the last cell may be a remainder.
    col_edges = np.asarray(x + [x[-1] + nx[-1]])

    def _tally(_i, arrays):
        defor_data, forest_data, risk_data = arrays
        defor_mask = defor_data == 1
        forest_start = (forest_data == 1) | defor_mask
        # Per-cell pixel counts: column sums, then one reduceat per cell row.
        nfor = np.add.reduceat(forest_start.sum(axis=0), col_edges[:-1])
        ndefor = np.add.reduceat(defor_mask.sum(axis=0), col_edges[:-1])
        # Integer category counts per cell, NOT the weighted sum: chunks are
        # folded with exact integer adds and the float sum runs once per cell
        # below, so the result is bit-identical to the single-pass formula.
        counts = np.empty((nsquare_x, len(cat)), dtype=np.int64)
        for px in range(nsquare_x):
            block = risk_data[:, col_edges[px] : col_edges[px + 1]]
            counts[px] = count_categories(block, cat)
        return nfor, ndefor, counts

    def _weigh(acc):
        nfor, ndefor, counts = acc
        pred = np.array(
            [np.nansum(counts[px] * defor_dens_period) for px in range(nsquare_x)]
        )
        return nfor, ndefor, pred

    parts = scan_row_bands(
        [defor_file, forest_file, riskmap_file],
        csize_coarse_grid,
        _tally,
        combine=_add_parts,
        finalize=_weigh,
    )
    assert len(parts) == nsquare_y
    df = pd.DataFrame(
        {
            "cell": np.arange(nsquare),
            "nfor_obs": np.concatenate([p[0] for p in parts]).astype(np.int64),
            "ndefor_obs": np.concatenate([p[1] for p in parts]).astype(np.int64),
            "nfor_obs_ha": 0.0,
            "ndefor_obs_ha": 0.0,
            "ndefor_pred_ha": np.concatenate([p[2] for p in parts]),
        }
    )

    df = df[df["nfor_obs"] > 0]
    ncell = df.shape[0]
    df["nfor_obs_ha"] = df["nfor_obs"] * pix_area / 10000
    df["ndefor_obs_ha"] = df["ndefor_obs"] * pix_area / 10000

    error_pred = df["ndefor_pred_ha"] - df["ndefor_obs_ha"]
    squared_error = error_pred**2
    RMSE = round(float(np.sqrt(np.mean(squared_error))), 2)
    w = df["nfor_obs_ha"] / df["nfor_obs_ha"].sum()
    wRMSE = round(float(np.sqrt(np.sum(squared_error * w))), 2)
    MedAE = round(float(np.median(np.absolute(error_pred))), 2)
    r = np.corrcoef(df["ndefor_pred_ha"], df["ndefor_obs_ha"])[0, 1]
    r_square = round(float(r**2), 2)

    indices = {
        "RMSE": RMSE,
        "wRMSE": wRMSE,
        "MedAE": MedAE,
        "R2": r_square,
        "ncell": ncell,
        "csize_coarse_grid": csize_coarse_grid,
        "csize_coarse_grid_ha": csize_ha,
    }
    axis_min, axis_max = pred_obs_axis_bounds(df)
    plot_data = PredObsPlotData(
        model=model_name,
        period=period,
        csize_px=csize_coarse_grid,
        csize_ha=csize_ha,
        points=df,
        axis_min=axis_min,
        axis_max=axis_max,
        medae=MedAE,
        r2=r_square,
        ncell=ncell,
    )
    return ValidationResult(indices=indices, plot_data=plot_data)


def write_pred_obs_csv(plot_data, output_path):
    """Persist the per-cell point table (the frozen 6-column CSV).

    Writes ``plot_data.points`` verbatim, so it produces that 6-column file only
    for a plot_data built by ``compute_validation``. A GUI-loaded plot_data
    carries the 4 plotted columns instead (see ``PredObsPlotData``) and would
    write a narrower file under the same name — don't round-trip one through
    here.
    """
    plot_data.points.to_csv(output_path, index=False)
    return output_path


def write_indices_csv(indices, output_path):
    """Persist the one-row accuracy-indices table."""
    pd.DataFrame([indices]).to_csv(output_path, index=False)
    return output_path


def save_pred_obs_png(plot_data, output_path, *, figsize=(6.4, 6.4), dpi=100):
    """Render the archived predicted-vs-observed scatter to a PNG.

    Byte-identical to the pre-split inline matplotlib block for well-behaved
    data. Non-finite rows are dropped and the reference line uses the guaranteed
    finite ``axis_min``/``axis_max``, so degenerate input renders instead of
    producing a blank or NaN-scaled figure.
    """
    import matplotlib

    matplotlib.use("Agg")  # worker-safe: must precede the pyplot import
    import matplotlib.pyplot as plt

    points = plot_data.finite_points
    p = [plot_data.axis_min, plot_data.axis_max]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot(111)
    ax.set_box_aspect(1)
    plt.scatter(
        points[_OBS_COL], points[_PRED_COL], color=None, marker="o", edgecolor="k"
    )
    plt.plot(p, p, "r--")
    plt.title(plot_data.title)
    plt.xlabel(plot_data.x_label)
    plt.ylabel(plot_data.y_label)
    plt.text(0, plot_data.axis_max, plot_data.annotation, ha="left", va="top")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def validate_two_layer(
    defor_file,
    forest_file,
    riskmap_file,
    tab_file_defor,
    time_interval,
    csize_coarse_grid=300,
    indices_file_pred=None,
    tab_file_pred=None,
    fig_file_pred=None,
    model_name="model",
    period="calibration",
    figsize=(6.4, 6.4),
    dpi=100,
):
    """Compute + persist one validation run (compatibility wrapper).

    Two-explicit-layer port of forestatrisk.validation_udef_arp. Kept with its
    original signature and return value (the indices dict) so existing callers
    and the notebook keep working; the computation now lives in
    ``compute_validation`` and the artifacts in ``write_pred_obs_csv`` /
    ``write_indices_csv`` / ``save_pred_obs_png``. Prefer those directly when
    you also need the chart input.

    Each of ``indices_file_pred`` / ``tab_file_pred`` / ``fig_file_pred``
    defaults to ``None``, meaning that artifact is not written; the indices dict
    is returned either way. The former bare-filename defaults ("indices.csv",
    "pred_obs.csv", "pred_obs.png") resolved against the process CWD, which is
    the read-only shared module mount when the app runs on SEPAL, so a caller
    that omitted one crashed with ``Read-only file system``. Callers that want
    the artifacts pass explicit paths.
    """
    result = compute_validation(
        defor_file=defor_file,
        forest_file=forest_file,
        riskmap_file=riskmap_file,
        tab_file_defor=tab_file_defor,
        time_interval=time_interval,
        csize_coarse_grid=csize_coarse_grid,
        model_name=model_name,
        period=period,
    )
    if tab_file_pred is not None:
        write_pred_obs_csv(result.plot_data, tab_file_pred)
    if fig_file_pred is not None:
        save_pred_obs_png(result.plot_data, fig_file_pred, figsize=figsize, dpi=dpi)
    if indices_file_pred is not None:
        write_indices_csv(result.indices, indices_file_pred)
    return result.indices


def _defrate_per_cat(**kwargs):
    """Indirection over rmj.deforrate.defrate_per_cat (monkeypatchable seam)."""
    from spatialrisk import rmj

    return rmj.deforrate.defrate_per_cat(**kwargs)


def resolve_layers(project, pred):
    """Recover the two binary layers + time interval from the prediction's dataset."""
    ds = project.get_dataset(pred.dataset_name)
    if ds is None:
        raise ValueError(f"Dataset '{pred.dataset_name}' not found in project.")
    # Hansen layers created in the GUI carry their parameters in the variable
    # name ("forest_gfc_tc30" for a 30% tree-cover threshold), so an exact match
    # misses every layer added since that feature shipped. The resolver that
    # parses those names lives in gui/, and this package must never import gui/
    # (see the note on PRED_OBS_X_LABEL above), so match the prefix instead.
    # First match wins, as before: picking among several forest layers is a GUI
    # concern (the Predict dialog asks the user); this path stays automatic.
    forest = next(
        (
            f
            for f in ds.features
            if f.name == FOREST_VAR or f.name.startswith(f"{FOREST_VAR}_")
        ),
        None,
    )
    if forest is None:
        raise ValueError(
            f"Feature '{FOREST_VAR}' (optionally parameter-suffixed, e.g. "
            f"'{FOREST_VAR}_tc30') not in dataset '{ds.name}'. "
            f"Available: {[f.name for f in ds.features]}"
        )
    return {
        "defor_file": ds.target.path,
        "forest_file": forest.path,
        "riskmap_file": pred.path,
        "time_interval": interval_from_target(ds.target.name),
        "period": pred.dataset_name,
    }


def evaluate_prediction(project, pred, csizes=(300,), recompute_defrate=True):
    """Defrate + validate one prediction across coarse-grid sizes.

    Returns a list of index dicts, each annotated with prediction/model/period/fig_path.
    Also writes results into ``pred.metrics``.
    """
    lay = resolve_layers(project, pred)
    label, period, ti = label_for(pred), lay["period"], lay["time_interval"]
    evaluation_folder = Path(project.folders.project_folder) / "evaluation"
    period_dir = evaluation_folder / period
    period_dir.mkdir(parents=True, exist_ok=True)

    defrate_csv = period_dir / f"defrate_cat_{label}_{period}.csv"
    if recompute_defrate or not defrate_csv.exists():
        _defrate_per_cat(
            defor_file=lay["defor_file"],
            forest_file=lay["forest_file"],
            riskmap_file=lay["riskmap_file"],
            time_interval=ti,
            tab_file_defrate=defrate_csv,
            verbose=False,
        )

    rows = []
    for csize in csizes:
        fig_path = period_dir / f"pred_obs_{label}_{period}_{csize}.png"
        idx = validate_two_layer(
            defor_file=lay["defor_file"],
            forest_file=lay["forest_file"],
            riskmap_file=lay["riskmap_file"],
            tab_file_defor=defrate_csv,
            time_interval=ti,
            csize_coarse_grid=csize,
            indices_file_pred=period_dir / f"indices_{label}_{period}_{csize}.csv",
            tab_file_pred=period_dir / f"pred_obs_{label}_{period}_{csize}.csv",
            fig_file_pred=fig_path,
            model_name=label,
            period=period,
        )
        idx.update(
            {
                "prediction": pred.storage_key()
                if hasattr(pred, "storage_key")
                else f"{pred.model_key}__{period}",
                "model": label,
                "period": period,
                "fig_path": str(fig_path),
            }
        )
        pred.metrics[f"{period}_{csize}"] = {
            k: idx[k] for k in ("RMSE", "wRMSE", "MedAE", "R2", "ncell")
        }
        rows.append(idx)
    return rows


def _validate_run_component(value, name):
    """One path COMPONENT of the evaluation layout — an identifier, not a path.

    ``truth_tag``/``run_id`` are caller-derived (``evaluate_against_truth`` is
    public), so an absolute value or an embedded ``..`` would resolve outside
    the project's ``evaluation/`` folder before anything checks it. Rejecting
    here — before any ``mkdir`` — is the write-side twin of the deletion
    containment guard in ``gui/tile/evaluation_helpers.delete_run_artifacts``.
    """
    text = str(value)
    if (
        not text
        or text in (".", "..")
        or "/" in text
        or "\\" in text
        or Path(text).is_absolute()
    ):
        raise ValueError(
            f"{name} must be a plain identifier, got the path-like {value!r}"
        )
    return text


def run_output_dir(project, truth_tag, run_id=None):
    """Directory a run's artifacts are written to.

    ``run_id=None`` reproduces the historical shared layout
    ``evaluation/<truth_tag>/``; a run id namespaces the output one level
    deeper, ``evaluation/<truth_tag>/<run_id>/``, so a later run against the
    same truth cannot overwrite an older saved run's files.

    Both components are validated as identifiers (see
    ``_validate_run_component``) and the resolved candidate is re-checked to
    sit below the resolved ``evaluation/`` root — defense in depth; raises
    ``ValueError`` before any directory or file exists.
    """
    eval_root = Path(project.folders.project_folder) / "evaluation"
    out_dir = eval_root / _validate_run_component(truth_tag, "truth_tag")
    if run_id is not None:
        out_dir = out_dir / _validate_run_component(run_id, "run_id")
    resolved_root = eval_root.resolve(strict=False)
    resolved = out_dir.resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise ValueError(
            f"evaluation output {resolved} escapes the project's "
            f"evaluation folder {resolved_root}"
        )
    return out_dir


# --------------------------------------------------------------------------
# TEMPORARY COMPATIBILITY SHIM — REMOVE AFTER ONE RELEASE
#
# Run-scoped output moved the canonical artifacts to
# evaluation/<truth_tag>/<run_id>/, so the newest run also publishes a copy to
# the historical un-scoped evaluation/<truth_tag>/<name> paths for anything
# still reading them by hand. No IN-REPO consumer needs that copy: searched
# twice, no notebook calls ``evaluate_against_truth`` (6.models_evaluation.ipynb
# uses ``evaluate_prediction``/``evaluate_predictions``, whose output layout is
# untouched) and the only non-test caller is the GUI, which reads
# ``EvaluationRecord.artifacts``. The window is kept deliberately, for a user's
# own scripts and for saved runs written before run-scoping — but retiring it
# costs no migration of anything in this repo.
#
# Every call site below is tagged ``# shim`` so
# ``grep -n "# shim" spatialrisk/evaluation.py`` lists every site to touch —
# but grep is not a removal plan by itself. Most tagged calls ARE pure: they
# publish an already-finished artifact to its legacy path and can simply be
# deleted along with the function. The defrate-cache pair inside
# ``_evaluate_one_against_truth`` (the "publish out" call on a cache miss and
# the "mirror in" call on a cache hit, both tagged below) is NOT pure: the
# cache backing ``recompute_defrate=False`` is keyed on the SHARED path, so
# those two calls are load-bearing, not legacy compatibility. Verified:
# stubbing ``_publish_legacy_copy`` to a no-op and running two evaluations
# with ``recompute_defrate=False`` on the second makes ``_defrate_per_cat``
# run TWICE instead of once — the cache silently stops working. Deleting
# those two calls requires first deciding where the defrate cache lives
# post-shim (e.g. keyed on ``truth_tag`` at a fixed non-legacy path, or moved
# onto the record) — that is a design decision, not a delete.
# --------------------------------------------------------------------------
def _publish_legacy_copy(src, dst):
    """Copy a run artifact to its legacy shared path. Temporary — see above."""
    src, dst = Path(src), Path(dst)
    if not src.exists() or src.resolve() == dst.resolve():
        return None
    shutil.copyfile(src, dst)
    return dst


def _evaluate_one_against_truth(
    project,
    pred,
    *,
    defor_file,
    forest_file,
    time_interval,
    truth_tag,
    csizes=(300,),
    recompute_defrate=True,
    run_id=None,
):
    """Defrate + validate ONE prediction against an explicit shared truth.

    Mirrors evaluate_prediction but takes the truth (defor + forest + interval)
    explicitly instead of deriving it from the prediction's own dataset, and
    namespaces all output under evaluation/<truth_tag>/.

    With a ``run_id`` the canonical artifacts go to
    evaluation/<truth_tag>/<run_id>/ and each returned row carries an
    ``EvaluationPlotArtifact`` under the private ``artifact`` key (dropped from
    the aggregate DataFrame by its explicit column list, exactly like
    ``fig_path``). Without one, behaviour is byte-for-byte what it always was.
    """
    from spatialrisk.evaluations import EvaluationPlotArtifact

    label, period = artifact_label_for(pred), pred.dataset_name
    riskmap_file = pred.path
    truth_dir = run_output_dir(project, truth_tag)
    truth_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_output_dir(project, truth_tag, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    defrate_name = f"defrate_cat_{label}_{period}.csv"
    defrate_csv = out_dir / defrate_name
    shared_defrate = truth_dir / defrate_name
    # The defrate cache is deliberately looked up at the SHARED path: making it
    # run-scoped would defeat recompute_defrate=False, which exists to skip this
    # expensive step when a previous run already produced it for this truth.
    if recompute_defrate or not shared_defrate.exists():
        _defrate_per_cat(
            defor_file=defor_file,
            forest_file=forest_file,
            riskmap_file=riskmap_file,
            time_interval=time_interval,
            tab_file_defrate=defrate_csv,
            verbose=False,
        )
        _publish_legacy_copy(
            defrate_csv, shared_defrate
        )  # shim — cache write, NOT pure, see banner
    else:
        # Cache hit: mirror it into the run folder so the run is self-contained.
        _publish_legacy_copy(
            shared_defrate, defrate_csv
        )  # shim — cache mirror, NOT pure, see banner

    rows = []
    for csize in csizes:
        fig_path = out_dir / f"pred_obs_{label}_{period}_{csize}.png"
        points_csv = out_dir / f"pred_obs_{label}_{period}_{csize}.csv"
        indices_csv = out_dir / f"indices_{label}_{period}_{csize}.csv"
        idx = validate_two_layer(
            defor_file=defor_file,
            forest_file=forest_file,
            riskmap_file=riskmap_file,
            tab_file_defor=defrate_csv,
            time_interval=time_interval,
            csize_coarse_grid=csize,
            indices_file_pred=indices_csv,
            tab_file_pred=points_csv,
            fig_file_pred=fig_path,
            model_name=label,
            period=period,
        )
        for src in (points_csv, indices_csv, fig_path):  # shim
            _publish_legacy_copy(src, truth_dir / src.name)

        prediction_key = (
            pred.storage_key()
            if hasattr(pred, "storage_key")
            else f"{pred.model_key}__{period}"
        )
        idx.update(
            {
                "prediction": prediction_key,
                "model": label,
                "period": period,
                "truth": truth_tag,
                "fig_path": str(fig_path),
            }
        )
        if run_id is not None:
            # Only run-scoped paths are recorded: a shared path is not stable
            # enough to promise a saved record its own data.
            idx["artifact"] = EvaluationPlotArtifact(
                prediction_key=prediction_key,
                model=label,
                period=period,
                csize_px=int(csize),
                points_csv=str(points_csv),
                png_path=str(fig_path),
                defrate_csv=str(defrate_csv),
            )
        pred.metrics[f"{truth_tag}__{period}_{csize}"] = {
            k: idx[k] for k in ("RMSE", "wRMSE", "MedAE", "R2", "ncell")
        }
        rows.append(idx)
    return rows


def evaluate_predictions(
    project,
    dataset_filter=None,
    model_filter=None,
    windows=None,
    csizes=(300,),
    recompute_defrate=True,
    auto_save=True,
):
    """Select predictions from the project, evaluate each, return aggregated indices.

    Skips (with a printed warning) any prediction whose layers cannot be resolved,
    rather than aborting the whole batch. Writes <project>/evaluation/indices_all.csv.
    """
    selected = {}
    for key, pred in project.predictions.items():
        if dataset_filter and pred.dataset_name not in dataset_filter:
            continue
        if model_filter and pred.model_key not in model_filter:
            continue
        if (
            windows is not None
            and pred.window is not None
            and pred.window not in windows
        ):
            continue
        selected[key] = pred

    rows = []
    for key, pred in selected.items():
        try:
            rows.extend(
                evaluate_prediction(
                    project, pred, csizes=csizes, recompute_defrate=recompute_defrate
                )
            )
        except Exception as exc:  # broad: skip-and-warn is intentional
            print(f"⚠ skipped {key}: {exc}")

    cols = [
        "prediction",
        "model",
        "period",
        "csize_coarse_grid",
        "csize_coarse_grid_ha",
        "ncell",
        "MedAE",
        "R2",
        "RMSE",
        "wRMSE",
    ]
    df = (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["csize_coarse_grid", "period", "model"])
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=cols)
    )

    evaluation_folder = Path(project.folders.project_folder) / "evaluation"
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(evaluation_folder / "indices_all.csv", index=False)

    if auto_save and rows:
        try:
            project.save()
        except Exception as exc:  # broad: a failed save must not lose results
            print(f"⚠ project.save() after evaluation failed: {exc}")
    return df


def evaluate_against_truth(
    project,
    prediction_keys=None,
    *,
    defor_file,
    forest_file,
    time_interval,
    truth_tag,
    csizes=(300,),
    recompute_defrate=True,
    auto_save=True,
    run_id=None,
):
    """Score selected maps against ONE common truth.

    Unlike evaluate_predictions (which derives each map's truth from its own
    dataset), this applies a single user-chosen truth (defor + forest + interval)
    to every selected map, enabling comparison of maps from different datasets.

    prediction_keys : list[str] or None
        Registry keys of the maps to score. None = all registered predictions.
        Unknown keys are skipped with a printed warning.
    run_id : str or None
        Identifier of THIS run. When given, artifacts are written to
        evaluation/<truth_tag>/<run_id>/ instead of the shared truth folder, and
        the returned DataFrame carries the resulting ``EvaluationPlotArtifact``
        list in ``df.attrs["artifacts"]`` (``attrs`` keeps the public return type
        a plain DataFrame for notebook callers). None = historical layout.
    """
    if prediction_keys is None:
        selected = dict(project.predictions)
    else:
        selected = {}
        for key in prediction_keys:
            pred = project.predictions.get(key)
            if pred is None:
                print(f"⚠ skipped {key}: not registered")
                continue
            selected[key] = pred

    rows = []
    for key, pred in selected.items():
        try:
            rows.extend(
                _evaluate_one_against_truth(
                    project,
                    pred,
                    defor_file=defor_file,
                    forest_file=forest_file,
                    time_interval=time_interval,
                    truth_tag=truth_tag,
                    csizes=csizes,
                    recompute_defrate=recompute_defrate,
                    run_id=run_id,
                )
            )
        except Exception as exc:  # broad: skip-and-warn is intentional
            print(f"⚠ skipped {key}: {exc}")

    # Harvest the per-row artifact objects before they are dropped by the
    # explicit column list below; they travel on df.attrs, not as a column.
    artifacts = [r.pop("artifact") for r in rows if r.get("artifact") is not None]

    cols = [
        "prediction",
        "model",
        "period",
        "truth",
        "csize_coarse_grid",
        "csize_coarse_grid_ha",
        "ncell",
        "MedAE",
        "R2",
        "RMSE",
        "wRMSE",
    ]
    df = (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["csize_coarse_grid", "period", "model"])
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=cols)
    )
    df.attrs["artifacts"] = artifacts
    df.attrs["run_id"] = run_id

    truth_dir = run_output_dir(project, truth_tag)
    truth_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_output_dir(project, truth_tag, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = out_dir / "indices_all.csv"
    df.to_csv(aggregate_csv, index=False)
    _publish_legacy_copy(aggregate_csv, truth_dir / "indices_all.csv")  # shim

    if auto_save and rows:
        try:
            project.save()
        except Exception as exc:  # broad: a failed save must not lose results
            print(f"⚠ project.save() after evaluation failed: {exc}")
    return df
