"""ECharts option builders for the evaluation results dialog (solara-free).

Turns an EvaluationRecord's stored index rows (one per map x cell size) into
the comparison charts shown in the dialog's Charts tab, plus small helpers for
the predicted-vs-observed figures tab. Kept free of solara/ipyvuetify so the
chart contents are unit-testable without a render harness.

The Charts tab used to be one Plotly figure with a subplot per metric. It is
now one self-contained ECharts option per metric (``metric_bar_option``), which
the widget lays out two per row: ECharts has no subplot concept, and a grid of
independent charts is what gives each metric its own legend, tooltip and
y-axis scale. The information design is unchanged — same metrics in the same
order, same titles, one bar series per cell size, one category per map label.

Palette and theme colours come from ``gui.scripts.echarts_options`` (the pure
half of the ECharts adapter), never from plotly. The bars are shades of the
app's ``primary`` accent, handed in by the widget layer.
"""

from pathlib import Path

from gui.scripts.echarts_options import DEFAULT_ACCENT, accent_ramp, theme_colors

# Metric -> (axis title, direction hint). R2 is unitless; the errors are in ha.
_METRIC_TITLES = {
    "MedAE": "MedAE (ha) ↓",
    "R2": "R² ↑",
    "RMSE": "RMSE (ha) ↓",
    "wRMSE": "wRMSE (ha) ↓",
}
_METRIC_ORDER = ["MedAE", "R2", "RMSE", "wRMSE"]


def map_label(row):
    """Display label for a row's map: '<model stem> — <period>'."""
    model = row.get("model") or row.get("prediction") or "?"
    period = row.get("period")
    return f"{model} — {period}" if period else str(model)


def map_title_parts(row, project):
    """``(family, run, "dataset · target")`` for a card header, or None.

    None when the row's prediction is no longer in the project (deleted, or a
    legacy row with no ``prediction`` key): the caller falls back to
    ``map_label``.

    Rows store the prediction's ``storage_key()``, not its registry key, and
    two runs of one model can share it — so the row's ``model`` (the artifact
    label) has to match too.
    """
    from spatialrisk.evaluation import artifact_label_for, label_for

    key = row.get("prediction")
    pred = next(
        (
            p
            for p in (getattr(project, "predictions", None) or {}).values()
            if key
            and p.storage_key() == key
            and artifact_label_for(p) == row.get("model")
        ),
        None,
    )
    if pred is None:
        return None
    ds = project.get_dataset(pred.dataset_name)
    target = getattr(getattr(ds, "target", None), "name", None)
    meta = " · ".join(x for x in (pred.dataset_name, target) if x)
    return label_for(pred), pred.name or pred.model_key, meta


def record_csizes(rows):
    """Sorted unique coarse-grid cell sizes present in the rows."""
    return sorted(
        {r["csize_coarse_grid"] for r in rows if r.get("csize_coarse_grid") is not None}
    )


def record_metrics(rows, selected):
    """Metric keys to chart: the run's selection, else every known metric."""
    keys = [m for m in (selected or _METRIC_ORDER) if m in _METRIC_ORDER]
    return [m for m in keys if any(r.get(m) is not None for r in rows)]


def pred_obs_artifact_name(model, period, csize, ext):
    """File name ``_evaluate_one_against_truth`` writes for one map + cell size.

    The PNG and the point CSV are the same stem with a different extension, and
    both are derived — not stored — for records saved before run-scoping. One
    definition so the two derivations can never drift apart: ``figure_entries``
    uses it for ``png``, the scatter loader for ``csv``.
    """
    return f"pred_obs_{model}_{period}_{csize}.{ext}"


def figure_entries(rows, csize, fig_dir=None):
    """[(label, Path)] of predicted-vs-observed PNGs for one cell size.

    Stored index rows carry NO fig_path column (evaluate_against_truth's
    explicit column list drops it), so the path is derived from ``fig_dir``
    (the run's evaluation folder) + the deterministic file name written by
    _evaluate_one_against_truth: ``pred_obs_{model}_{period}_{csize}.png``.
    A row's own fig_path (in-memory rows, future schema) takes precedence.
    """
    entries = []
    for r in rows:
        if r.get("csize_coarse_grid") != csize:
            continue
        if r.get("fig_path"):
            path = Path(r["fig_path"])
        elif fig_dir is not None and r.get("model") and r.get("period"):
            path = Path(fig_dir) / pred_obs_artifact_name(
                r["model"], r["period"], csize, "png"
            )
        else:
            continue
        entries.append((map_label(r), path))
    return sorted(entries, key=lambda e: e[0])


def csize_series_name(csize):
    """Legend/series name for one cell size — also the tooltip's ``{a}``."""
    return f"csize {csize} px"


def metric_bar_option(rows, metric, dark=False, accent=DEFAULT_ACCENT):
    """Grouped-bar ECharts option for ONE metric: x = map, bars = cell size.

    Args:
        rows: EvaluationRecord.indices (list of dicts).
        metric: a single metric key (``MedAE``, ``R2``, ``RMSE``, ``wRMSE``).
        dark: style for the app's dark theme.
        accent: the app's ``primary`` colour; one shade of it per cell size,
            light -> dark, so the shading encodes their order.

    Returns a plain, JSON-serializable ECharts option dict, or None when this
    metric has nothing chartable in these rows (no rows, no cell sizes, no map
    labels, an unknown metric key, or a metric no row carries a value for).
    The caller drops the Nones rather than drawing an empty frame.

    A missing (label, cell size) pair becomes a ``None`` datum — ECharts draws
    no bar for it and keeps the remaining bars on their own categories, which
    is what Plotly's per-trace ``y`` list of ``None``s did.

    ``themed_option`` (applied later by the widget) sets only the background
    and the text ink, so the grid colour is wired into the axes here.
    """
    if metric not in _METRIC_TITLES:
        return None
    csizes = record_csizes(rows)
    labels = sorted({map_label(r) for r in rows})
    if not csizes or not labels:
        return None
    if not any(r.get(metric) is not None for r in rows):
        return None

    colors = accent_ramp(len(csizes), accent, dark=dark)
    ink, grid = theme_colors(dark)["ink"], theme_colors(dark)["grid"]
    by_key = {(map_label(r), r.get("csize_coarse_grid")): r for r in rows}
    # One cell size means the legend would restate the title, so it is hidden —
    # and then the plot must not go on reserving the legend row's height.
    show_legend = len(csizes) > 1

    series = []
    for ci, csize in enumerate(csizes):
        series.append(
            {
                "type": "bar",
                "name": csize_series_name(csize),
                "data": [
                    (by_key.get((lab, csize)) or {}).get(metric) for lab in labels
                ],
                "itemStyle": {"color": colors[ci]},
                # Plotly's bargroupgap=0.08: the gap between bars inside one group.
                # ECharts reads barGap from the first bar series of the group.
                "barGap": "8%",
            }
        )

    return {
        "title": {
            "text": _METRIC_TITLES[metric],
            "left": "center",
            "top": 0,
            # Plotly drew the subplot titles as 13px annotations in the ink.
            "textStyle": {"color": ink, "fontSize": 13, "fontWeight": "normal"},
        },
        "textStyle": {"fontSize": 12},
        # top clears the title (at y=0) plus, when it is shown, the legend row
        # placed just under it (top=24). Without a legend the plot starts where
        # the legend would have begun instead of leaving a blank band.
        "grid": {
            "left": 8,
            "right": 12,
            "top": 52 if show_legend else 24,
            "bottom": 4,
            "containLabel": True,
        },
        "tooltip": {
            # {b} = category (the map label), {c} = value, {a} = series name
            # (which is "csize N px") — the three fields the Plotly
            # hovertemplate showed. A template string, not a JS function:
            # arbitrary Python callables cannot cross the widget wire (an
            # explicitly encoded one — ipecharts.tools.encode_js_fn — could,
            # but the template is sufficient and safer).
            "trigger": "item",
            "formatter": "{b}<br/>" + metric + " = {c}<br/>{a}",
        },
        "legend": {
            "show": show_legend,
            "data": [s["name"] for s in series],
            "top": 24,
            "right": 0,
            "textStyle": {"color": ink},
        },
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLine": {"lineStyle": {"color": grid}},
            "axisTick": {"show": False},
            "splitLine": {"show": False},  # Plotly: update_xaxes(showgrid=False)
            "axisLabel": {"color": ink, "hideOverlap": True},
        },
        "yAxis": {
            # No explicit min: ECharts already extends a value axis to zero for
            # bar series (Plotly's rangemode="tozero"), and pinning min=0 would
            # clip a negative R2 — which is a real outcome for a bad map.
            "type": "value",
            "axisLine": {"show": False},
            "splitLine": {"lineStyle": {"color": grid}},
            "axisLabel": {"color": ink},
        },
        "series": series,
    }
