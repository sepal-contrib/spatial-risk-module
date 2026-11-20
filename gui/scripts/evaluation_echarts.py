"""Interactive predicted-vs-observed scatter: point loading + option (solara-free).

The archived PNG (``spatialrisk.evaluation.save_pred_obs_png``) shows one map's
per-cell predicted vs observed deforestation as a static image. This module
builds the *interactive* twin of that figure for the Evaluation dialog: it finds
the run's saved point CSV, loads it, and returns a plain ECharts option dict.
The widget layer renders it; nothing here imports solara, ipyvuetify or
ipecharts, so the whole chart is unit-testable without a render harness.

**One frozen table, two renderings.** The chart plots the exact float values
sitting in ``pred_obs_{model}_{period}_{csize}.csv`` — no rounding and no
resampling. Past ``PRED_OBS_THIN_POINT_COUNT`` points the chart draws a SUBSET
(one real row per screen-sized bin, see ``_one_point_per_bin``) so the browser
stays responsive; every drawn value is still a CSV value. The PNG and this chart must
never disagree about what a model predicted for a cell, so the numbers travel
from the CSV to the option untouched. That exactness depends on HOW the file is
read: pandas' default CSV float parser is fast but not correctly rounded (it
returns ``0.92`` for the text ``0.9199999999999999``), so the loader reads with
``float_precision="round_trip"`` — see ``_load_cached``. ``PredObsPlotData``
(shared with the PNG writer) supplies the axis bounds and the finite-row filter,
which is what keeps the two figures' domains identical too.

**Non-finite values.** ``PredObsPlotData.points`` here holds the persisted rows
for the four plotted columns (the loader reads no more than the chart draws) and
may include NaN/inf; those are not valid JSON and would break the option on the
wire. Only ``finite_points`` is ever plotted — matching ``save_pred_obs_png``,
which drops the same rows. The annotation's ``n`` stays the full ``ncell``, also
matching the PNG.

**Performance.** Loading is memoized on the artifact's *modification identity*
(path + size + mtime), not just its path: two runs against the same truth
produce identically named files, and keying on the name alone would serve one
run's points under another run's record. ``pred_obs_chart_identity`` wraps that
same key — plus the theme and the caller's label/title text — into one short
string, so the widget can memoize its option build and skip the adapter's
per-render option hash (see ``EChartsChart(option_digest=)``). Because it stands
in for that hash, it has to cover every option input, not just the file's.
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from gui.scripts.echarts_options import (
    DEFAULT_ACCENT,
    RENDERER_CANVAS,
    RENDERER_SVG,
    accent_color,
    accent_fill,
    theme_colors,
)
from gui.scripts.evaluation_charts import pred_obs_artifact_name

logger = logging.getLogger("spatial_risk")

# Point count at which the scatter switches from SVG to canvas + ECharts'
# `large`/`progressive` pipeline. 2000 is ECharts' OWN default for
# `largeThreshold` and (with `progressiveThreshold` at 3000) the point where its
# maintainers judged per-item rendering to stop paying off; it is adopted here
# rather than invented so the number has a source.
#
# MEASURED (2026-07-21, CPython 3.11.10 / pandas 2.x, synthetic point CSVs at
# 500..200k points; see test_large_mode_engages_at_exactly_2000_plotted_points).
# These are order-of-magnitude figures from one dev machine that was also
# running a dev server, not constants: treat the SHAPE (flat until ~10k, then
# superlinear) as the finding, not the millisecond values.
#
#     points     read_csv   _scatter_rows   option (rows warm)   option JSON
#        500        2.6 ms         0.6 ms             0.03 ms       44 KB
#       1000        2.8 ms         0.7 ms             0.04 ms       87 KB
#   ->  2000        4.8 ms         1.0 ms             0.03 ms      173 KB
#       5000       11.6 ms         1.4 ms             0.03 ms      431 KB
#      50000       40.1 ms        12.8 ms             0.04 ms      4.3 MB
#     200000      161.5 ms       282   ms             0.03 ms     17.5 MB
#
# The conclusion is that NOTHING on the Python side argues for a different
# number: at 2000 points the whole option costs ~1 ms to build and 173 KB to
# send, so the cost that decides SVG-vs-canvas is entirely browser-side (one
# <circle> DOM node per point vs one batched canvas draw), which no headless
# measurement can settle. 2000 is therefore kept on its provenance alone, and
# the remaining browser-side confirmation is on the deployed-SEPAL walkthrough
# list. (Writing the same constant into `largeThreshold` below is tidiness, not
# an argument for the value: both sides read this constant, so they agree at
# ANY value, and `large`/`largeThreshold` are only written when large mode is
# already on — ECharts' own default can never contradict the renderer choice.)
PRED_OBS_LARGE_POINT_COUNT = 2000

# Points per progressive-render chunk once large mode is on. ECharts' default
# `progressive` is 400, which is too small here: the whole scatter arrives in one
# option, so larger chunks mean fewer frames to first complete paint.
PRED_OBS_PROGRESSIVE_CHUNK = 5000

# Past this many finite points the scatter is thinned to one point per cell of a
# PRED_OBS_THIN_BINS x PRED_OBS_THIN_BINS grid over the shared axis domain (~1.5
# px on the 380 px card, under the 7 px marker): the default view looks the same,
# every outlier survives, and a 200k-point run stops shipping 17.5 MB of option
# per card. Kept points are real CSV rows, so their values and tooltips stay exact.
PRED_OBS_THIN_POINT_COUNT = 10_000
PRED_OBS_THIN_BINS = 200

# Two caches, two very different per-entry costs — so two constants. Re-measured
# 2026-07-21 on a 200k-point run (the realistic upper end), CPython 3.11.10 /
# pandas 2.x. These are PRE-THINNING figures: since PRED_OBS_THIN_POINT_COUNT a
# row entry holds at most PRED_OBS_THIN_BINS**2 = 40k rows (~9 MB), so the
# 47 MB / 94 MB / 140 MB numbers below are upper bounds that no longer occur.
# The frame figures are unchanged (the full table is still loaded).
#
#   _load_cached entry   (4-column DataFrame)      6.4 MB   -> 8 x  6.4 =  51 MB
#   _scatter_rows entry  (boxed [[float x5], ...]) 47.2 MB  -> 2 x 47.2 =  94 MB
#
# The row cache costs 7.4x the frame it is built from: every value becomes a
# boxed Python float/int inside a per-point list object, which is the price of
# a JSON-serializable option. Sizing both at 8 would have reserved ~430 MB on a
# SEPAL deployment for a dialog that shows one chart at a time.
#
# The loader stays at 8: cheap, and it is what makes flipping between maps and
# cell sizes feel instant.
#
# The row cache is kept at 2, but it is NOT what protects a multi-map dialog.
# Measured by calling `_scatter_rows` directly for three maps (200k/50k/25k
# points) in a loop: 0 hits in 9 calls, ~234 ms per pass — each card's entry is
# evicted by the next, so an LRU of 2 is useless the moment a run has three
# maps. Raising it to one-entry-per-map is the wrong fix (8 x 47 MB), so the
# rebuild is instead removed one level up: _PredObsCard memoizes the whole
# option on `pred_obs_chart_identity` (see gui/widget/evaluation_results.py).
#
# What that memo does and does NOT buy, stated precisely:
#   * WITHIN one continuous mount at one tab state it makes the build
#     once-per-card instead of once-per-re-entering-render, whatever the map
#     count. (Note reacton bails out on `==`-equal props, so a parent re-render
#     with unchanged props never re-enters a card in the first place — the
#     0-hits figure above comes from a direct loop, not from rendering the tab.)
#   * It does NOT survive leaving the Pred-vs-obs tab and coming back: the memo
#     keys on [digest, tab_active], so a round trip re-materializes every card's
#     rows (measured on a 3-card tab: 3 -> 3 -> 6 misses, 0 hits). That is
#     deliberate, not an oversight. The tab round trip already forces a full
#     widget rebuild and a fresh multi-MB option over the websocket (the
#     ipecharts attach-time sizing mitigation, `identity=...|tab{active_tab}`),
#     and dropping the rows while the tab is hidden is what keeps a 3-map 200k
#     dialog from holding ~140 MB of boxed floats for its whole lifetime —
#     which is the same memory argument that keeps this cache at 2.
# What is left for this cache is the narrow case it can actually serve — the
# same plot_data rebuilt under a new digest (a light/dark toggle, a language
# switch) — where 2 is enough.
POINTS_CACHE_SIZE = 8
SCATTER_ROWS_CACHE_SIZE = 2

# Reference line: red dashed, exactly as the PNG's `plt.plot(p, p, "r--")`.
REFERENCE_LINE_COLOR = "#ff0000"

# Opacity of the point cloud, which is painted in the app's ``primary`` accent
# (``accent_fill``) rather than a fixed blue. Translucent so overlapping markers
# read as density; the dark theme gets a little more of it, because its surface
# takes more away. Matplotlib drew unfilled markers with a black edge; on a dark
# surface a black edge disappears, so the edge follows the ink.
_POINT_ALPHA = {True: 0.65, False: 0.55}

# Horizontal breathing room on both sides of the plot box, applied equally so
# the option itself adds no left/right bias. It does NOT centre the box: with
# `containLabel: True` the grid also reserves room for the axis labels and the
# y-axis `name` (nameGap 48) on the left, against only the last x-tick label's
# overflow on the right, so the drawn box sits right of centre by roughly the
# difference. ECharts has no aspect
# lock for a cartesian grid — squareness is a container property (see
# PRED_OBS_SQUARE_HEIGHT), and the option's job is only to not skew it.
_GRID_INSET = 16
# Vertical room under the plot box for the x-axis `name`. `containLabel` grows
# the grid to fit the TICK LABELS only — an axis `name` is never counted — so
# the title is drawn `X_NAME_GAP` below the axis line, i.e. below the band
# containLabel reserved, and with the old `bottom: 8` it fell off the canvas.
# The y-axis title escapes this because its own gap (48) happens to fit inside
# the much wider band the numeric y-labels reserve.
# Centre of the title sits X_NAME_GAP from the axis line, so its lowest ink is
# ~X_NAME_GAP + half a 12px line; containLabel already gives back about
# _AXIS_LABEL_BAND of that.
_X_NAME_GAP = 28
_AXIS_LABEL_BAND = 20
_GRID_BOTTOM = _X_NAME_GAP + 6 - _AXIS_LABEL_BAND + 8  # = 22, incl. breathing
# Side of the square card. Sized so at least three cards fit across the dialog
# (90vw, capped at 1400px) without the tab scrolling vertically: 3 x 380 + two
# 16px gaps = 1172, inside the card's usable width.
PRED_OBS_SQUARE_HEIGHT = "380px"

# English defaults. Every one is overridable via `labels=`: the widget layer
# owns translation (`t(...)`), and this module must not import gui.i18n's
# machinery into a pure builder. The numeric FORMATS are not overridable — they
# are frozen to the PNG's ("MedAE = {:.2f} ha", "R2 = {:.2f}", "n = {:d}") so
# the two figures always quote the same rounded values.
#
# The two axis titles restate `spatialrisk.evaluation.PRED_OBS_X_LABEL` /
# `PRED_OBS_Y_LABEL` (the strings baked into the archived PNG) rather than
# importing them, so this module stays cheap to import; they are pinned equal by
# test_default_axis_labels_match_the_archived_pngs.
DEFAULT_LABELS = {
    "x_axis": "Observed deforestation (ha)",
    "y_axis": "Predicted deforestation (ha)",
    "series": "Grid cells",
    "cell": "Cell",
    "observed": "Observed",
    "predicted": "Predicted",
    "forest": "Forest",
    "residual": "Residual",
    "medae": "MedAE",
    "r2": "R2",
    "n": "n",
    "ha": "ha",
}

# The four columns the chart draws — a local copy of
# `spatialrisk.evaluation.PLOT_COLUMNS`, which is what `PredObsPlotData` requires
# and enforces. Copied rather than imported to keep this module's import cheap;
# pinned equal (contents AND order) by
# test_loader_reads_exactly_the_columns_spatialrisk_requires, because a drift
# would raise inside the loader, be swallowed by its `except Exception`, and
# degrade every interactive scatter to the PNG with only a log line.
_POINT_CSV_COLUMNS = ["cell", "nfor_obs_ha", "ndefor_obs_ha", "ndefor_pred_ha"]


# ---------------------------------------------------------------------------
# Artifact resolution — typed record first, legacy derivation second
# ---------------------------------------------------------------------------


def _record_artifacts(record):
    return list(getattr(record, "artifacts", None) or [])


_ARTIFACT_KIND_EXT = {"points_csv": "csv", "png_path": "png"}


def _typed_artifact_path(record, model, period, csize, kind, prediction_key=None):
    """Recorded path for one map + cell size, or None. Typed tiers only.

    Tier 1 — the artifact whose ``prediction_key`` matches exactly. Its
    presence is decided FIRST, before any blank-field check, and answers the
    whole call on its own: a non-blank ``kind`` returns that path, a blank one
    returns None (-> ``resolve_plot_artifact``'s tier-3 derivation). It never
    falls through to tier 2 in either case — an exact match with a blank
    field is this run's own recorded gap, not license to borrow a SIBLING
    prediction's file for the same slot. Tier 2 — any artifact matching
    ``(model, period, csize)``, first non-blank ``kind``: older manifests
    recorded keys under a different scheme, and the run's own file for that
    map beats a derived path a later run may have overwritten. Two live
    predictions sharing a model+period are told apart by tier 1; tier 2 is
    only reached when NO artifact carries the caller's key at all.
    """
    loose = []
    for art in _record_artifacts(record):
        if (
            getattr(art, "model", None) != model
            or getattr(art, "period", None) != period
            or int(getattr(art, "csize_px", -1)) != int(csize)
        ):
            continue
        if (
            prediction_key is not None
            and getattr(art, "prediction_key", None) == prediction_key
        ):
            path = getattr(art, kind, None)
            return Path(path) if path else None
        path = getattr(art, kind, None)
        if path:
            loose.append(Path(path))
    return loose[0] if loose else None


def resolve_plot_artifact(
    record, *, prediction_key, model, period, csize, kind, fallback_dir=None
):
    """Path of ONE map + cell size's artifact, or None.

    ``kind`` selects which of the pair: ``"points_csv"`` (the interactive
    scatter's table) or ``"png_path"`` (the archived figure). Resolution
    order: exact typed match by prediction key (tier 1), typed match by
    (model, period, csize) for older/partial manifests (tier 2), then the
    deterministic legacy derivation beside the record's own indices CSV
    (tier 3) — see ``_typed_artifact_path`` and ``pred_obs_artifact_name``.
    The derived path stays run-scoped for run-scoped records because their
    ``csv_path`` already points inside the run folder.
    """
    if kind not in _ARTIFACT_KIND_EXT:
        raise ValueError(f"unknown artifact kind: {kind!r}")
    if csize is None or not model or not period:
        return None
    typed = _typed_artifact_path(
        record, model, period, int(csize), kind, prediction_key
    )
    if typed is not None:
        return typed
    if fallback_dir is None:
        csv_path = getattr(record, "csv_path", None)
        if not csv_path:
            return None
        fallback_dir = Path(csv_path).parent
    return Path(fallback_dir) / pred_obs_artifact_name(
        model, period, int(csize), _ARTIFACT_KIND_EXT[kind]
    )


def points_csv_is_expected(record, model, period, csize, *, prediction_key=None):
    """True when THIS map + cell size has a recorded point-CSV artifact.

    A *per-artifact* question, deliberately not a per-record one. A run-scoped
    record can carry a partially populated artifact list — one map recorded, its
    sibling not — and ``resolve_points_csv`` falls back to the derived path for
    the omitted ones on purpose. Asking ``bool(record.artifacts)`` instead would
    treat "this run never recorded a table for this map" as "this run's table
    went missing" and warn about maps that were never promised. Typed tiers
    only — this never reports the tier-3 derivation as "expected".
    """
    if csize is None or not model or not period:
        return False
    return (
        _typed_artifact_path(
            record, model, period, int(csize), "points_csv", prediction_key
        )
        is not None
    )


def resolve_points_csv(
    record, model, period, csize, *, prediction_key=None, fig_dir=None
):
    """Path of the point CSV for one map + cell size, or None.

    A thin wrapper over ``resolve_plot_artifact`` (``kind="points_csv"``) —
    see there for the three-tier resolution order. ``fig_dir`` is this
    entry point's name for that call's ``fallback_dir``.
    """
    return resolve_plot_artifact(
        record,
        prediction_key=prediction_key,
        model=model,
        period=period,
        csize=csize,
        kind="points_csv",
        fallback_dir=fig_dir,
    )


def _index_row_for(record, model, period, csize, prediction_key=None):
    """The record's stored index row for one map + cell size ({} if absent).

    Two live predictions can share (model, period, csize) while scoring
    different maps — the same model rerun against another dataset revision,
    say — and then only the row's own ``"prediction"`` column tells them
    apart. When ``prediction_key`` is given, this PREFERS the row whose
    ``"prediction"`` matches it exactly; only when no row carries that column
    or none matches does it fall back to the first (model, period, csize)
    match, so a legacy record without a ``"prediction"`` column behaves
    exactly as before.
    """
    matches = [
        row
        for row in getattr(record, "indices", None) or []
        if row.get("model") == model
        and row.get("period") == period
        and row.get("csize_coarse_grid") == int(csize)
    ]
    if prediction_key is not None:
        for row in matches:
            if row.get("prediction") == prediction_key:
                return row
    return matches[0] if matches else {}


def _modification_identity(path):
    """``(size, mtime_ns)`` of ``path``, or None when it is not readable.

    The cache key's whole reason for existing: two runs against one truth write
    identically named files, so a path alone does not identify a table. Size and
    mtime together change whenever the bytes do, at no read cost.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def _record_key(record):
    """Short stable id for the record itself (run-scoped when possible)."""
    storage_key = getattr(record, "storage_key", None)
    if callable(storage_key):
        try:
            return str(storage_key())
        except Exception:
            pass
    return str(getattr(record, "run_id", None) or getattr(record, "csv_path", ""))


# ---------------------------------------------------------------------------
# Loading — memoized on the artifact's modification identity
# ---------------------------------------------------------------------------


@lru_cache(maxsize=POINTS_CACHE_SIZE)
def _load_cached(path_str, size, mtime_ns, model, period, csize, csize_ha, medae, r2):
    """Read one point CSV into a PredObsPlotData. Keyed by file identity.

    ``size``/``mtime_ns`` are in the key but unused in the body — that is the
    point: they make a rewritten (or different-run) file a cache miss. All args
    are hashable scalars so ``lru_cache`` can key on them directly.

    Only the four columns the chart draws are read; the other two exist for the
    archived table, not for the plot.

    ``float_precision="round_trip"`` is load-bearing, not a tuning knob.
    pandas' default C float parser is not correctly rounded and can land on a
    neighbouring double: the text ``0.9199999999999999`` parses to ``0.92``, and
    ``123456789.12345679`` to ``123456789.1234568``. ``to_csv`` writes the full
    round-trippable repr, so every bit of loss would be on this side of the
    trip — and since ``save_pred_obs_png`` draws from the in-memory frame, a
    default parse makes the PNG and this chart differ by up to 1 ulp. The
    round-trip parser is exactly ``float()`` on the text and costs a few ms per
    100k rows, once per file (this is memoized).

    Every column is normalized with ``pd.to_numeric(errors="coerce")`` and the
    DRAWABLE rows (finite obs AND pred — the same mask ``finite_points``
    applies) are validated: ``cell`` finite and integral, ``nfor_obs_ha``
    finite. A violation raises ``ValueError`` here, where the caller already
    degrades to None; the guide's "validate data before creating charts" rule.
    Coercion (not ``errors="raise"``) is deliberate: a corrupt value on a row
    the chart never draws must not cost the run its interactive chart.
    """
    import numpy as np
    import pandas as pd

    from spatialrisk.evaluation import PredObsPlotData, pred_obs_axis_bounds

    points = pd.read_csv(
        path_str, usecols=_POINT_CSV_COLUMNS, float_precision="round_trip"
    )[_POINT_CSV_COLUMNS]
    # Every plotted/tooltip column is validated HERE, inside the boundary that
    # already degrades a bad file to the PNG (load_pred_obs_plot_data catches
    # any raise and returns None). Before this, a malformed value slipped
    # through the load and raised out of _scatter_rows while the OPTION was
    # being built — killing the whole Figures render instead of falling back.
    # Malformed text coerces to NaN so a bad value on a row the chart never
    # draws (non-finite obs/pred) stays harmless; the drawable rows are then
    # held to exactly what _scatter_rows will do to them: cell -> int64,
    # nfor_obs_ha -> finite float64.
    for col in _POINT_CSV_COLUMNS:
        points[col] = pd.to_numeric(points[col], errors="coerce")
    drawable = points[
        np.isfinite(points["ndefor_obs_ha"]) & np.isfinite(points["ndefor_pred_ha"])
    ]
    cell = drawable["cell"].to_numpy(dtype="float64")
    if not np.isfinite(cell).all() or (cell != np.rint(cell)).any():
        raise ValueError("cell must hold finite integral ids on drawable rows")
    if not np.isfinite(drawable["nfor_obs_ha"].to_numpy(dtype="float64")).all():
        raise ValueError("nfor_obs_ha must be finite on drawable rows")
    axis_min, axis_max = pred_obs_axis_bounds(points)
    return PredObsPlotData(
        model=model,
        period=period,
        csize_px=int(csize),
        csize_ha=csize_ha,
        points=points,
        axis_min=axis_min,
        axis_max=axis_max,
        medae=medae,
        r2=r2,
        ncell=len(points),
    )


def load_pred_obs_plot_data(
    record, model, period, csize, *, prediction_key=None, fig_dir=None
):
    """Load one map's saved point table as a ``PredObsPlotData``, or None.

    Returns None when the artifact cannot be resolved, does not exist, or cannot
    be parsed — a saved run whose files were moved or deleted must degrade to a
    message in the dialog, never to an exception out of a render.

    ``ncell`` and the axis bounds come from the loaded frame, never from the
    record, so the ``ncell == len(points)`` invariant can never fail on a stale
    index row. ``MedAE``/``R2`` do come from the record's index row: they are the
    already-rounded values the PNG quoted, and recomputing them here would risk
    a different answer than the archived figure.

    Repeated calls with an unchanged file return the SAME object (see
    ``_load_cached``), which is what lets the option builder memoize on it.
    """
    path = resolve_points_csv(
        record, model, period, csize, prediction_key=prediction_key, fig_dir=fig_dir
    )
    if path is None:
        return None
    identity = _modification_identity(path)
    if identity is None:
        # Only an absent table this run PROMISED is a fault. A legacy or
        # PNG-only record resolves to a derived path that was never written, and
        # `_PredObsCard` deliberately shows no warning for it — but this logger
        # feeds the on-map log console at INFO+, so warning here would put in
        # front of the user exactly the message the UI decided not to show.
        # Same rung split as the card's (see `points_csv_is_expected`).
        expected = points_csv_is_expected(
            record, model, period, csize, prediction_key=prediction_key
        )
        log = logger.warning if expected else logger.debug
        log("Evaluation point table is missing: %s", path)
        return None
    row = _index_row_for(record, model, period, csize, prediction_key)
    try:
        return _load_cached(
            str(path),
            identity[0],
            identity[1],
            model,
            period,
            int(csize),
            row.get("csize_coarse_grid_ha"),
            row.get("MedAE"),
            row.get("R2"),
        )
    except Exception as exc:
        logger.warning("Evaluation point table is unreadable (%s): %s", path, exc)
        return None


def _text_identity(labels, title):
    """Short digest of the option's TEXT inputs (``labels`` + ``title``).

    Hashed rather than concatenated only to keep the identity short; these are a
    dozen tiny strings, so the cost is microseconds.
    """
    if not labels and title is None:
        return "-"
    payload = repr((sorted((labels or {}).items()), title))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def pred_obs_chart_identity(
    record,
    model,
    period,
    csize,
    *,
    dark=False,
    labels=None,
    title=None,
    prediction_key=None,
    fig_dir=None,
    accent=DEFAULT_ACCENT,
):
    """Cheap, stable identity of the chart this call would draw.

    Folds in EVERY input of ``pred_obs_scatter_option``: the record, the
    resolved artifact path, that file's modification identity, the cell size,
    the theme, the accent, and the ``labels``/``title`` text. Everything but the
    text comes from a single ``stat()`` — no read, no parse, no option hash —
    and the file half is the SAME key the loader memoizes on, so the two can
    never disagree about what is current.

    The accent is carried in its own right rather than left to ``dark``: the two
    move together in normal use, but a palette edit changes the accent alone,
    and this digest replaces the adapter's content hash — an input it misses is
    an input nothing checks.

    Two uses, both in the widget layer: memoize the option build, and hand the
    result to ``EChartsChart(option_digest=...)`` so the adapter skips hashing an
    option that can carry hundreds of thousands of points.

    **Pass the same ``labels`` and ``title`` you pass to
    ``pred_obs_scatter_option``.** Used as ``option_digest`` this REPLACES the
    adapter's content hash, so an input the identity misses is an input nothing
    checks: switching the app's language would leave the previous language's
    axis titles on screen with no error anywhere. The two signatures share their
    names and defaults precisely so one call site can pass the same values to
    both.
    """
    path = resolve_points_csv(
        record, model, period, csize, prediction_key=prediction_key, fig_dir=fig_dir
    )
    if path is None:
        return None
    size, mtime_ns = _modification_identity(path) or (-1, -1)
    return "|".join(
        str(part)
        for part in (
            _record_key(record),
            path,
            size,
            mtime_ns,
            int(csize),
            "dark" if dark else "light",
            accent_color(accent, dark=dark),
            _text_identity(labels, title),
        )
    )


# ---------------------------------------------------------------------------
# Option building
# ---------------------------------------------------------------------------


def pred_obs_renderer(plot_data):
    """SVG for a small scatter, canvas once it gets dense.

    SVG keeps text and markers crisp and scales with the browser zoom, but costs
    one DOM node per point; past ``PRED_OBS_LARGE_POINT_COUNT`` that is the
    bottleneck and canvas (with ``large``) wins outright.
    """
    return (
        RENDERER_CANVAS
        if _plotted_count(plot_data) >= PRED_OBS_LARGE_POINT_COUNT
        else RENDERER_SVG
    )


def _plotted_count(plot_data):
    return 0 if plot_data is None else len(plot_data.finite_points)


@lru_cache(maxsize=SCATTER_ROWS_CACHE_SIZE)
def _scatter_rows(plot_data):
    """``[[obs, pred, cell, forest_ha, residual], ...]`` for one plot data.

    The only per-point Python work in this module, and it is not cheap:
    measured 1.0 ms at 2k points, 12.8 ms at 50k and 282 ms at 200k.
    ``PredObsPlotData`` is a frozen dataclass with ``eq=False``, so it hashes by
    identity — and the loader hands back the same object for an unchanged file,
    which makes this a hit.

    Kept to ``SCATTER_ROWS_CACHE_SIZE`` (not the loader's size): an entry here
    costs 7.4x the DataFrame rows it holds — 47 MB for 200k rows before
    thinning, at most ~9 MB (40k rows) since — because every value is a boxed
    Python scalar in its own list. That is also why this cache is not the
    multi-map dialog's protection; the widget memoizes the finished option
    instead. See the constants block.

    Values are converted through ``.tolist()`` so they are native Python floats
    and ints, never numpy scalars: the option is serialized to the frontend as
    JSON, and a numpy scalar is not JSON-serializable.

    The residual (predicted - observed) is precomputed into the value array
    because these tooltips are template STRINGS: nothing is computed at hover
    time. (An encoded JS function — ipecharts.tools.encode_js_fn — could
    compute it browser-side, but the template is sufficient and safer.)
    """
    points = plot_data.finite_points
    obs = points["ndefor_obs_ha"].to_numpy(dtype="float64")
    pred = points["ndefor_pred_ha"].to_numpy(dtype="float64")
    if len(points) > PRED_OBS_THIN_POINT_COUNT:
        keep = _one_point_per_bin(obs, pred, plot_data.axis_min, plot_data.axis_max)
        points, obs, pred = points.iloc[keep], obs[keep], pred[keep]
    cells = points["cell"].to_numpy(dtype="int64").tolist()
    forest = points["nfor_obs_ha"].to_numpy(dtype="float64").tolist()
    residual = (pred - obs).tolist()
    return [
        list(row) for row in zip(obs.tolist(), pred.tolist(), cells, forest, residual)
    ]


def _one_point_per_bin(obs, pred, lo, hi):
    """Sorted row positions keeping the first point in each grid bin."""
    import numpy as np

    bins = PRED_OBS_THIN_BINS
    scale = bins / ((float(hi) - float(lo)) or 1.0)
    ix = np.clip(((obs - float(lo)) * scale).astype(np.int64), 0, bins - 1)
    iy = np.clip(((pred - float(lo)) * scale).astype(np.int64), 0, bins - 1)
    _, first = np.unique(ix * bins + iy, return_index=True)
    return np.sort(first)


def pred_obs_annotation(plot_data, labels=None):
    """``MedAE = x ha / R2 = y / n = k``, with the PNG's exact number formats.

    A metric the record never stored is left out rather than printed as ``nan``.
    Only the words are translatable; the formats are frozen to the PNG's.
    """
    text = {**DEFAULT_LABELS, **(labels or {})}
    parts = []
    if plot_data.medae is not None:
        parts.append(f"{text['medae']} = {plot_data.medae:.2f} {text['ha']}")
    if plot_data.r2 is not None:
        parts.append(f"{text['r2']} = {plot_data.r2:.2f}")
    parts.append(f"{text['n']} = {plot_data.ncell:d}")
    return "\n".join(parts)


def pred_obs_scatter_option(
    plot_data, *, dark=False, labels=None, title=None, accent=DEFAULT_ACCENT
):
    """ECharts option for one map's predicted-vs-observed scatter.

    Args:
        plot_data: a ``PredObsPlotData`` (from ``load_pred_obs_plot_data``).
        dark: style for the app's dark theme.
        labels: overrides for ``DEFAULT_LABELS`` — the widget layer passes
            translated strings here.
        accent: the app's ``primary`` colour, which the point cloud is drawn
            in. The 1:1 reference line stays red: it is an annotation, and
            painting it in the accent too would let it read as data.
        title: chart title. ``PredObsPlotData.title`` is deliberately NOT used:
            it is the PNG's English text, and the dialog shows a translated one.

    Returns a plain, JSON-serializable option dict, or None when there is
    nothing to draw.

    **Whatever you pass as ``labels``/``title`` here must also be passed to
    ``pred_obs_chart_identity``** if you use its result as the adapter's
    ``option_digest``: that digest replaces the content hash, so text it does
    not cover cannot trigger a re-render. Both functions take these two by the
    same names and defaults so a single call site can forward them unchanged.

    Structure: one ``scatter`` series carrying every point, plus one ``line``
    series drawing the 1:1 reference. Both axes are pinned to the SAME
    ``axis_min``/``axis_max`` — an independently scaled y-axis would put points
    above or below the 1:1 line for reasons that have nothing to do with the
    model, which is the single most misleading thing this chart could do.

    ``themed_option`` (applied later by the widget) sets only the background and
    the top-level ink, so grid and axis colours are wired in here.
    """
    if plot_data is None:
        return None
    data = _scatter_rows(plot_data)
    if not data:
        return None

    text = {**DEFAULT_LABELS, **(labels or {})}
    colors = theme_colors(dark)
    ink, grid = colors["ink"], colors["grid"]
    axis_min, axis_max = float(plot_data.axis_min), float(plot_data.axis_max)
    large = len(data) >= PRED_OBS_LARGE_POINT_COUNT

    scatter = {
        "type": "scatter",
        "name": text["series"],
        "data": data,
        # cell/forest/residual ride along in the value array purely to reach the
        # tooltip template; only the first two dimensions are coordinates.
        "dimensions": [
            text["observed"],
            text["predicted"],
            text["cell"],
            text["forest"],
            text["residual"],
        ],
        "symbolSize": 7,
        "itemStyle": {
            "color": accent_fill(accent, _POINT_ALPHA[bool(dark)], dark=dark),
            "borderColor": ink,
            "borderWidth": 0.5,
        },
        "emphasis": {"focus": "series"},
        "tooltip": {
            # Template string, not a callback: arbitrary Python callables
            # cannot cross the widget wire. An explicitly encoded JS function
            # COULD (ipecharts.tools.encode_js_fn), but the template is
            # sufficient and safer here. {c0}..{c4} index the value array
            # laid out by _scatter_rows.
            "formatter": (
                f"{text['cell']} {{c2}}<br/>"
                f"{text['observed']} = {{c0}} {text['ha']}<br/>"
                f"{text['predicted']} = {{c1}} {text['ha']}<br/>"
                f"{text['forest']} = {{c3}} {text['ha']}<br/>"
                f"{text['residual']} = {{c4}} {text['ha']}"
            ),
        },
    }
    if large:
        # Canvas-only fast path: one batched draw call instead of one shape per
        # point, rendered in chunks so the first frame does not block.
        scatter.update(
            {
                "large": True,
                "largeThreshold": PRED_OBS_LARGE_POINT_COUNT,
                "progressive": PRED_OBS_PROGRESSIVE_CHUNK,
                "progressiveThreshold": PRED_OBS_LARGE_POINT_COUNT,
            }
        )

    reference = {
        # The PNG's `plt.plot(p, p, "r--")`. Deliberately nameless, silent and
        # tooltip-free: it is an annotation, not data. A name would put it in
        # the legend and let a user toggle away the chart's frame of reference.
        "type": "line",
        "data": [[axis_min, axis_min], [axis_max, axis_max]],
        "showSymbol": False,
        "silent": True,
        "legendHoverLink": False,
        "tooltip": {"show": False},
        "lineStyle": {"color": REFERENCE_LINE_COLOR, "type": "dashed", "width": 1.2},
        "z": 1,
    }

    axis = {
        "type": "value",
        "min": axis_min,
        "max": axis_max,
        "axisLine": {"lineStyle": {"color": grid}},
        "axisTick": {"show": False},
        "splitLine": {"lineStyle": {"color": grid}},
        "axisLabel": {"color": ink},
        "nameTextStyle": {"color": ink},
        "nameLocation": "middle",
    }

    option = {
        "textStyle": {"fontSize": 12},
        "animation": not large,  # animating 200k points helps nobody
        # `top` leaves room for the title plus the MedAE/R2/n block below it;
        # with the toolbox gone, that is all it has to clear.
        "grid": {
            "left": _GRID_INSET,
            "right": _GRID_INSET,
            "top": 44,
            "bottom": _GRID_BOTTOM,
            "containLabel": True,
        },
        "tooltip": {"trigger": "item", "confine": True},
        # Nothing toggleable: one data series, and the reference line must stay.
        "legend": {"show": False},
        # No toolbox: its icon row sat immediately under the card's download
        # button and read as that button's tooltip. Saving is what the card's
        # PNG download already offers, and the `dataZoom: inside` below keeps
        # zooming available without any chrome.
        "toolbox": {"show": False},
        # Wheel/pinch zoom over the plot itself, both axes together so the 1:1
        # line stays at 45 degrees through any zoom.
        "dataZoom": [{"type": "inside", "xAxisIndex": 0, "yAxisIndex": 0}],
        "xAxis": {**axis, "name": text["x_axis"], "nameGap": _X_NAME_GAP},
        "yAxis": {**axis, "name": text["y_axis"], "nameGap": 48},
        "series": [scatter, reference],
        "graphic": [
            {
                # The PNG's top-left MedAE/R2/n block, in the same corner.
                "type": "text",
                "left": 56,
                "top": 48,
                "silent": True,
                "style": {
                    "text": pred_obs_annotation(plot_data, labels),
                    "fill": ink,
                    "fontSize": 12,
                    "lineHeight": 16,
                },
            }
        ],
    }
    if title is not None:
        option["title"] = {
            "text": title,
            "left": "center",
            "top": 0,
            "textStyle": {"color": ink, "fontSize": 13, "fontWeight": "normal"},
        }
    return option
