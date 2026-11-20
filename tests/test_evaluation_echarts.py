"""Unit tests for the interactive predicted-vs-observed scatter (Task 6).

Module under test: ``gui.scripts.evaluation_echarts`` — the solara-free half of
the scatter. It resolves a saved run's point CSV, loads it, and turns it into a
plain ECharts option dict. The widget half (Task 7) only renders what this
builds, so everything worth asserting about the chart is asserted here.

The headline assertion is ``test_scatter_coordinates_are_exactly_the_saved_csv
_values``: the coordinates the chart plots must be the *exact* floats sitting in
``pred_obs_*.csv``. That is the whole point of the migration — the interactive
chart and the archived PNG are two renderings of one frozen table, and any
rounding or resampling between them would make the two disagree. (Dense runs draw
a subset of real rows — see the thinning tests — never altered values.)

That assertion reads the expected values with ``float()`` on the raw file TEXT,
never with a second ``pd.read_csv``. pandas' default C float parser is fast but
not correctly rounded: it returns ``0.92`` for the text ``0.9199999999999999``.
Comparing one pandas parse against another is self-consistent with that error
and so cannot see it — the file's bytes are the only neutral referee.
"""

import json
from pathlib import Path

import pytest

_POINT_COLUMNS = [
    "cell",
    "nfor_obs",
    "ndefor_obs",
    "nfor_obs_ha",
    "ndefor_obs_ha",
    "ndefor_pred_ha",
]

# Deliberately awkward floats. Each does NOT survive a round(x, 2) or a float32
# trip, and the first, second and fourth are also mis-parsed by pandas' DEFAULT
# CSV float parser (-> 0.92, 0.3, 123456789.1234568), so the headline equality
# assertion fails unless the loader reads with float_precision="round_trip".
_OBS = [0.9199999999999999, 0.30000000000000004, 3.3333333333333335, 123456789.12345679]
_PRED = [1.0700000000000003, 0.7000000000000001, 4.000000000000001, 123456789.12345678]


def _write_points(path, obs=None, pred=None, forest=None):
    """Write the frozen 6-column point CSV and return the frame it holds."""
    import pandas as pd

    obs = list(_OBS if obs is None else obs)
    pred = list(_PRED if pred is None else pred)
    forest = list(range(10, 10 + len(obs))) if forest is None else list(forest)
    frame = pd.DataFrame(
        {
            "cell": list(range(len(obs))),
            "nfor_obs": [100] * len(obs),
            "ndefor_obs": [7] * len(obs),
            "nfor_obs_ha": [float(f) for f in forest],
            "ndefor_obs_ha": obs,
            "ndefor_pred_ha": pred,
        }
    )[_POINT_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _csv_floats(csv_path, *columns):
    """Columns of a point CSV parsed with ``float()`` straight from the TEXT.

    The neutral referee for "the chart plots the saved values". A second
    ``pd.read_csv`` is NOT one: pandas' default float parser is not correctly
    rounded (text ``0.9199999999999999`` parses to ``0.92``), so a pandas-vs-
    pandas comparison is self-consistent with its own error. ``float()`` is
    correctly rounded and is the exact inverse of the round-trippable repr
    ``to_csv`` writes.
    """
    lines = csv_path.read_text().strip().splitlines()
    header = lines[0].split(",")
    idx = [header.index(c) for c in columns]
    return [
        tuple(float(row[i]) for i in idx)
        for row in (line.split(",") for line in lines[1:])
    ]


def _index_row(model="GLM", period="d1", csize=300, **over):
    row = {
        "model": model,
        "period": period,
        "prediction": f"{model}__{period}",
        "csize_coarse_grid": csize,
        "csize_coarse_grid_ha": 90.0,
        "MedAE": 1.5,
        "R2": 0.9,
        "RMSE": 2.0,
        "wRMSE": 1.8,
        "ncell": 4,
    }
    row.update(over)
    return row


def _record(run_dir, *, artifacts=None, indices=None, run_id="run1"):
    """A real EvaluationRecord pointing at ``run_dir``."""
    from spatialrisk.evaluations import EvaluationRecord

    return EvaluationRecord(
        truth_tag="loss_2010",
        truth_defor="/d.tif",
        truth_forest="/f.tif",
        time_interval=5,
        created_at="2026-07-20T10:00:00",
        run_id=run_id,
        csizes=[300],
        indices=list(indices if indices is not None else [_index_row()]),
        csv_path=str(run_dir / "indices_all.csv"),
        artifacts=list(artifacts or []),
    )


def _typed_record(run_dir, *, csize=300, model="GLM", period="d1", **kw):
    from spatialrisk.evaluations import EvaluationPlotArtifact

    artifact = EvaluationPlotArtifact(
        prediction_key=f"{model}__{period}",
        model=model,
        period=period,
        csize_px=csize,
        points_csv=str(run_dir / f"pred_obs_{model}_{period}_{csize}.csv"),
        png_path=str(run_dir / f"pred_obs_{model}_{period}_{csize}.png"),
    )
    return _record(run_dir, artifacts=[artifact], **kw)


def _saved(tmp_path, **kw):
    """Write a run's point CSV and return ``(csv_path, record)``."""
    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    csv_path = run_dir / "pred_obs_GLM_d1_300.csv"
    _write_points(csv_path, **kw)
    return csv_path, _typed_record(run_dir)


def _loaded(tmp_path, **kw):
    """Write a run's CSV, then load it back through the public loader.

    Returns ``(csv_frame, plot_data)`` where ``csv_frame`` is an INDEPENDENT
    read of the file on disk — never the pre-save frame. The CSV is the frozen
    artifact both this chart and the archived PNG render, so it, not whatever
    was in memory before ``to_csv`` wrote it, is what the chart must match.

    ``float_precision="round_trip"`` for the same reason the loader uses it: the
    default parser is not correctly rounded, so a default read here would make
    this helper's frame disagree with the file's own text by up to 1 ulp and
    every test built on it would inherit that error. The headline test skips
    this helper entirely and reads the text with ``float()`` (``_csv_floats``),
    which is the parser-independent check.
    """
    import pandas as pd

    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path, **kw)
    return (
        pd.read_csv(csv_path, float_precision="round_trip"),
        load_pred_obs_plot_data(record, "GLM", "d1", 300),
    )


def _scatter(option):
    return next(s for s in option["series"] if s["type"] == "scatter")


def _refline(option):
    return next(s for s in option["series"] if s["type"] == "line")


# ---------------------------------------------------------------------------
# Cross-layer constants: the GUI's copies must equal spatialrisk's originals
# ---------------------------------------------------------------------------


def test_loader_reads_exactly_the_columns_spatialrisk_requires():
    """``_POINT_CSV_COLUMNS`` is a copy of ``PLOT_COLUMNS`` — pin them equal.

    The loader names its ``usecols`` independently of
    ``spatialrisk.evaluation.PLOT_COLUMNS`` (the GUI module must not grow an
    import-time dependency on a heavy module just to read a tuple), so nothing
    but this test keeps them in step. A column added to ``PLOT_COLUMNS`` and not
    here would make ``PredObsPlotData.__post_init__`` raise inside
    ``_load_cached``; ``load_pred_obs_plot_data``'s blanket ``except Exception``
    would swallow it, and EVERY interactive scatter would silently fall back to
    the PNG with only a log line. Order matters too — the loader re-indexes the
    frame with this list, and the option's dimensions follow it.
    """
    from gui.scripts.evaluation_echarts import _POINT_CSV_COLUMNS
    from spatialrisk.evaluation import PLOT_COLUMNS

    assert _POINT_CSV_COLUMNS == list(PLOT_COLUMNS)


def test_default_axis_labels_match_the_archived_pngs():
    """The English fallbacks are the PNG's axis titles, not a paraphrase.

    The widget layer overrides both with ``t(...)``, so this only bites a caller
    that passes no labels — but that caller's chart sits next to the archived
    PNG in the same dialog, and the two must not word the same axis differently.
    """
    from gui.scripts.evaluation_echarts import DEFAULT_LABELS
    from spatialrisk.evaluation import PRED_OBS_X_LABEL, PRED_OBS_Y_LABEL

    assert DEFAULT_LABELS["x_axis"] == PRED_OBS_X_LABEL
    assert DEFAULT_LABELS["y_axis"] == PRED_OBS_Y_LABEL


# ---------------------------------------------------------------------------
# Headline: the plotted coordinates ARE the saved CSV values
# ---------------------------------------------------------------------------


def test_scatter_coordinates_are_exactly_the_saved_csv_values(tmp_path):
    """No rounding, no resampling: chart floats == CSV floats, bit for bit.

    The interactive scatter and the archived PNG are two renderings of the same
    frozen table. Anything that reshapes the numbers on the way to the option —
    a ``round``, a float32 cast, a downsample, or a lossy CSV *parse* — makes
    the two disagree about what the model predicted, which is the one thing this
    chart exists to show.

    Expected values come from ``float()`` on the file's own text (see
    ``_csv_floats``). Comparing against a second ``pd.read_csv`` would compare
    one lossy parse with an identical lossy parse and pass either way; this
    assertion fails if the loader drops ``float_precision="round_trip"``.
    """
    from gui.scripts.evaluation_echarts import (
        load_pred_obs_plot_data,
        pred_obs_scatter_option,
    )

    csv_path, record = _saved(tmp_path)
    plot_data = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    option = pred_obs_scatter_option(plot_data)

    plotted = [(v[0], v[1]) for v in _scatter(option)["data"]]
    assert plotted == _csv_floats(csv_path, "ndefor_obs_ha", "ndefor_pred_ha")


def test_up_to_the_thin_threshold_every_saved_point_is_plotted(tmp_path):
    """At or below ``PRED_OBS_THIN_POINT_COUNT`` nothing is dropped."""
    from gui.scripts.evaluation_echarts import (
        PRED_OBS_THIN_POINT_COUNT,
        load_pred_obs_plot_data,
        pred_obs_scatter_option,
    )

    n = PRED_OBS_THIN_POINT_COUNT
    obs = [float(i) for i in range(n)]
    pred = [float(i) * 1.1 for i in range(n)]
    _, record = _saved(tmp_path, obs=obs, pred=pred)
    plot_data = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    values = _scatter(pred_obs_scatter_option(plot_data))["data"]

    assert len(values) == n
    assert (values[-1][0], values[-1][1]) == (obs[-1], pred[-1])


def test_a_dense_scatter_keeps_one_real_point_per_bin(tmp_path):
    """Past the threshold: one exact CSV row per occupied bin, outliers kept.

    Thinning must never invent or round a value (each drawn row is a real
    row, tooltip columns included), must keep a lone outlier, and must leave
    the annotation's ``n`` at the full cell count like the PNG.
    """
    from gui.scripts.evaluation_echarts import (
        PRED_OBS_THIN_BINS,
        PRED_OBS_THIN_POINT_COUNT,
        pred_obs_annotation,
        pred_obs_scatter_option,
    )

    n = 3 * PRED_OBS_THIN_POINT_COUNT
    # A dense blob near the origin plus one far outlier.
    obs = [(i % 97) / 1000.0 for i in range(n)] + [500.0]
    pred = [(i % 89) / 1000.0 for i in range(n)] + [3.0]
    frame, plot_data = _loaded(tmp_path, obs=obs, pred=pred)
    values = _scatter(pred_obs_scatter_option(plot_data))["data"]

    assert len(values) < PRED_OBS_THIN_POINT_COUNT
    assert len(values) <= PRED_OBS_THIN_BINS**2
    rows = {
        (c, o, p, f)
        for c, o, p, f in zip(
            frame["cell"],
            frame["ndefor_obs_ha"],
            frame["ndefor_pred_ha"],
            frame["nfor_obs_ha"],
        )
    }
    assert all((v[2], v[0], v[1], v[3]) in rows for v in values)
    assert [500.0, 3.0] in [v[:2] for v in values]
    assert f"n = {n + 1}" in pred_obs_annotation(plot_data)


def test_scatter_values_carry_cell_forest_and_residual(tmp_path):
    """Dimensions 2..4 exist only to reach the tooltip template.

    ECharts tooltips here are template strings (the option is serialized, so no
    callback can cross the wire), which means anything the tooltip shows must
    already be in the value array — including the residual, which is therefore
    precomputed rather than derived at hover time.
    """
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    frame, plot_data = _loaded(tmp_path)
    values = _scatter(pred_obs_scatter_option(plot_data))["data"]

    assert [v[2] for v in values] == frame["cell"].tolist()
    assert [v[3] for v in values] == frame["nfor_obs_ha"].tolist()
    assert [v[4] for v in values] == pytest.approx(
        (frame["ndefor_pred_ha"] - frame["ndefor_obs_ha"]).tolist()
    )


def test_non_finite_rows_are_dropped_but_n_still_counts_them(tmp_path):
    """NaN is not valid JSON — it must never reach the option.

    ``save_pred_obs_png`` drops the same rows from the scatter while its
    annotation keeps quoting the full ``ncell``. Matching that exactly is what
    keeps the interactive chart and the archived PNG telling the same story.
    """
    from gui.scripts.evaluation_echarts import (
        pred_obs_annotation,
        pred_obs_scatter_option,
    )

    _, plot_data = _loaded(
        tmp_path, obs=[1.0, float("nan"), 3.0], pred=[1.5, 2.0, float("inf")]
    )
    option = pred_obs_scatter_option(plot_data)

    assert [(v[0], v[1]) for v in _scatter(option)["data"]] == [(1.0, 1.5)]
    assert "n = 3" in pred_obs_annotation(plot_data)


def test_option_is_json_clean(tmp_path):
    """A numpy scalar would make the adapter's digest fall back to str()."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    option = pred_obs_scatter_option(plot_data, title="GLM — d1")
    assert json.loads(json.dumps(option)) == option


# ---------------------------------------------------------------------------
# Shared axis domain and the 1:1 reference line
# ---------------------------------------------------------------------------


def test_both_axes_share_one_numeric_domain(tmp_path):
    """An independently scaled y-axis would invalidate the 1:1 comparison.

    Points would sit above or below the reference line for reasons that have
    nothing to do with the model — the single most misleading thing this chart
    could do — so the domain is pinned on both axes, not left to autoscale.
    """
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    option = pred_obs_scatter_option(plot_data)

    assert option["xAxis"]["min"] == option["yAxis"]["min"] == plot_data.axis_min
    assert option["xAxis"]["max"] == option["yAxis"]["max"] == plot_data.axis_max


def test_axis_domain_spans_both_columns(tmp_path):
    """The bounds come from observed AND predicted, as the PNG's do."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path, obs=[2.0, 40.0], pred=[1.0, 30.0])
    option = pred_obs_scatter_option(plot_data)

    assert option["xAxis"]["min"] == 1.0  # min lives in the predicted column
    assert option["xAxis"]["max"] == 40.0  # max lives in the observed column


def test_reference_line_spans_the_shared_domain(tmp_path):
    """The 1:1 reference line runs corner to corner of the shared axis domain."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    line = _refline(pred_obs_scatter_option(plot_data))

    assert line["data"] == [
        [plot_data.axis_min, plot_data.axis_min],
        [plot_data.axis_max, plot_data.axis_max],
    ]


def test_reference_line_is_dashed_red_like_the_png(tmp_path):
    """The PNG draws it with ``plt.plot(p, p, "r--")``."""
    from gui.scripts.evaluation_echarts import (
        REFERENCE_LINE_COLOR,
        pred_obs_scatter_option,
    )

    _, plot_data = _loaded(tmp_path)
    line = _refline(pred_obs_scatter_option(plot_data))

    assert line["lineStyle"]["color"] == REFERENCE_LINE_COLOR
    assert line["lineStyle"]["type"] == "dashed"
    assert line["showSymbol"] is False


def test_reference_line_is_silent_and_stays_out_of_legend_and_tooltip(tmp_path):
    """It is an annotation, not data.

    A named series lands in the legend, where a user can toggle away the chart's
    own frame of reference; a hoverable one puts a meaningless two-point series
    in the tooltip. Both are ruled out at the series level, not by hiding the
    legend, so the guarantee survives someone turning the legend back on.
    """
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    option = pred_obs_scatter_option(plot_data)
    line = _refline(option)

    assert "name" not in line
    assert line["silent"] is True
    assert line["tooltip"]["show"] is False
    assert line["legendHoverLink"] is False
    assert option["legend"]["show"] is False


# ---------------------------------------------------------------------------
# Tooltip, annotation, labels, theme
# ---------------------------------------------------------------------------


def test_tooltip_names_every_dimension(tmp_path):
    """Every value slot the tooltip formatter promises is labelled and addressed."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    formatter = _scatter(pred_obs_scatter_option(plot_data))["tooltip"]["formatter"]

    for label in ("Cell", "Observed", "Predicted", "Forest", "Residual"):
        assert label in formatter
    # each value-array slot the tooltip promises is actually addressed
    for slot in ("{c0}", "{c1}", "{c2}", "{c3}", "{c4}"):
        assert slot in formatter


def test_annotation_uses_the_pngs_number_formats(tmp_path):
    """Same values, same rounding, same wording as the archived figure."""
    from gui.scripts.evaluation_echarts import pred_obs_annotation

    _, plot_data = _loaded(tmp_path)

    assert pred_obs_annotation(plot_data) == "MedAE = 1.50 ha\nR2 = 0.90\nn = 4"


def test_annotation_reaches_the_option(tmp_path):
    """The MedAE/R2/n annotation is rendered as graphic text on the option."""
    from gui.scripts.evaluation_echarts import (
        pred_obs_annotation,
        pred_obs_scatter_option,
    )

    _, plot_data = _loaded(tmp_path)
    option = pred_obs_scatter_option(plot_data)
    texts = [g["style"]["text"] for g in option["graphic"]]

    assert pred_obs_annotation(plot_data) in texts


def test_annotation_omits_a_metric_the_record_never_stored(tmp_path):
    """Better a shorter block than 'MedAE = nan ha'."""
    from gui.scripts.evaluation_echarts import (
        load_pred_obs_plot_data,
        pred_obs_annotation,
    )

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    _write_points(run_dir / "pred_obs_GLM_d1_300.csv")
    record = _typed_record(run_dir, indices=[_index_row(MedAE=None, R2=None)])
    plot_data = load_pred_obs_plot_data(record, "GLM", "d1", 300)

    assert pred_obs_annotation(plot_data) == "n = 4"


def test_labels_are_translatable_but_number_formats_are_not(tmp_path):
    """The widget layer owns wording; the formats stay frozen to the PNG's."""
    from gui.scripts.evaluation_echarts import (
        pred_obs_annotation,
        pred_obs_scatter_option,
    )

    _, plot_data = _loaded(tmp_path)
    labels = {
        "x_axis": "Observado (ha)",
        "y_axis": "Predicho (ha)",
        "cell": "Celda",
        "medae": "EMA",
        "n": "n",
    }
    option = pred_obs_scatter_option(plot_data, labels=labels)

    assert option["xAxis"]["name"] == "Observado (ha)"
    assert option["yAxis"]["name"] == "Predicho (ha)"
    assert "Celda {c2}" in _scatter(option)["tooltip"]["formatter"]
    assert pred_obs_annotation(plot_data, labels).startswith("EMA = 1.50 ha")


def test_the_title_is_the_callers_not_the_pngs_english_one(tmp_path):
    """PredObsPlotData.title is PNG text; the dialog shows a translated title."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)

    assert "title" not in pred_obs_scatter_option(plot_data)
    titled = pred_obs_scatter_option(plot_data, title="Mi gráfico")
    assert titled["title"]["text"] == "Mi gráfico"
    # asserted on a substring: json.dumps escapes the newline inside
    # plot_data.title, so comparing the whole string would never match anyway
    assert "Predicted vs. observed deforestation" not in json.dumps(titled)


def test_axes_and_annotation_follow_the_theme(tmp_path):
    """themed_option only sets the top-level ink, so axes are wired here."""
    from gui.scripts.echarts_options import theme_colors
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    for dark in (True, False):
        option = pred_obs_scatter_option(plot_data, dark=dark)
        ink, grid = theme_colors(dark)["ink"], theme_colors(dark)["grid"]
        assert option["xAxis"]["axisLabel"]["color"] == ink
        assert option["yAxis"]["nameTextStyle"]["color"] == ink
        assert option["xAxis"]["splitLine"]["lineStyle"]["color"] == grid
        assert option["yAxis"]["axisLine"]["lineStyle"]["color"] == grid
        assert option["graphic"][0]["style"]["fill"] == ink


# ---------------------------------------------------------------------------
# Interaction, layout, and the large-point-count switch
# ---------------------------------------------------------------------------


def test_toolbox_is_hidden(tmp_path):
    """Its icon row sat under the card's download button and read as a tooltip.

    Zooming survives without it (`dataZoom: inside`), and saving is what the
    card's own PNG download offers.
    """
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)

    assert pred_obs_scatter_option(plot_data)["toolbox"]["show"] is False


def test_inside_zoom_drives_both_axes_together(tmp_path):
    """Zooming one axis alone would tilt the 1:1 line off 45 degrees."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    zooms = pred_obs_scatter_option(plot_data)["dataZoom"]

    assert [z["type"] for z in zooms] == ["inside"]
    assert zooms[0]["xAxisIndex"] == 0 and zooms[0]["yAxisIndex"] == 0


def test_plot_box_is_horizontally_symmetric(tmp_path):
    """Equal insets keep the plot square in a square container.

    ECharts has no aspect lock for a cartesian grid, so squareness comes from
    the container (PRED_OBS_SQUARE_HEIGHT); the option's job is to not skew it.
    """
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    grid = pred_obs_scatter_option(plot_data)["grid"]

    assert grid["left"] == grid["right"]
    assert grid["containLabel"] is True


def test_grid_bottom_leaves_room_for_the_x_axis_title(tmp_path):
    """`containLabel` fits the tick labels, never the axis `name`.

    With too small a `bottom` the "Observed deforestation (ha)" title is drawn
    below the canvas and never appears.
    """
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    _, plot_data = _loaded(tmp_path)
    option = pred_obs_scatter_option(plot_data)

    assert option["xAxis"]["nameLocation"] == "middle"
    # Room for the half of the title that hangs below its own centre, past the
    # band containLabel already reserved for the tick labels.
    assert option["grid"]["bottom"] > option["xAxis"]["nameGap"] - 20


def test_a_small_scatter_stays_on_svg_with_no_large_mode(tmp_path):
    """Small point counts render as SVG without ECharts' large-scatter mode."""
    from gui.scripts.echarts_options import RENDERER_SVG
    from gui.scripts.evaluation_echarts import (
        pred_obs_renderer,
        pred_obs_scatter_option,
    )

    _, plot_data = _loaded(tmp_path)
    scatter = _scatter(pred_obs_scatter_option(plot_data))

    assert pred_obs_renderer(plot_data) == RENDERER_SVG
    assert "large" not in scatter
    assert "progressive" not in scatter


def test_a_dense_scatter_switches_to_canvas_and_large_mode(tmp_path):
    """One batched draw call instead of one DOM node per point."""
    from gui.scripts.echarts_options import RENDERER_CANVAS
    from gui.scripts.evaluation_echarts import (
        PRED_OBS_LARGE_POINT_COUNT,
        PRED_OBS_PROGRESSIVE_CHUNK,
        load_pred_obs_plot_data,
        pred_obs_renderer,
        pred_obs_scatter_option,
    )

    n = PRED_OBS_LARGE_POINT_COUNT
    _, record = _saved(
        tmp_path, obs=[float(i) for i in range(n)], pred=[float(i) for i in range(n)]
    )
    plot_data = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    option = pred_obs_scatter_option(plot_data)
    scatter = _scatter(option)

    assert pred_obs_renderer(plot_data) == RENDERER_CANVAS
    assert scatter["large"] is True
    assert scatter["largeThreshold"] == PRED_OBS_LARGE_POINT_COUNT
    assert scatter["progressive"] == PRED_OBS_PROGRESSIVE_CHUNK
    assert option["animation"] is False  # animating 200k points helps nobody


def test_large_mode_engages_at_exactly_2000_plotted_points(tmp_path):
    """MEASURED THRESHOLD: 1999 plotted points draw as SVG, 2000 as canvas.

    Task 8 measurement, 2026-07-21, CPython 3.11.10 / pandas 2.x, synthetic
    round-trippable point CSVs, best of 5 per cell, on a dev machine that was
    also running a dev server. Order-of-magnitude figures, NOT constants — the
    shape is the finding, and the shape is superlinear past ~10k points:
    ``_scatter_rows`` goes 3.2 ms at 10k to 282 ms at 200k, i.e. 20x the points
    for ~90x the time (allocator pressure from one boxed list per point, and a
    second large frame resident during the 200k trial). Cost of producing ONE
    scatter option at each size:

        points     read_csv   _scatter_rows   option (rows warm)   option JSON
           500        2.6 ms         0.6 ms             0.03 ms       44 KB
          1000        2.8 ms         0.7 ms             0.04 ms       87 KB
      ->  2000        4.8 ms         1.0 ms             0.03 ms      173 KB
          5000       11.6 ms         1.4 ms             0.03 ms      431 KB
         10000       12.7 ms         3.2 ms             0.03 ms      861 KB
         50000       40.1 ms        12.8 ms             0.04 ms      4.3 MB
        200000      161.5 ms       282   ms             0.03 ms     17.5 MB

    Nothing on the Python side argues for moving the number: at 2000 points the
    entire option costs ~1 ms to build and 173 KB to send, three orders of
    magnitude below where per-point work starts to hurt. The switch is a
    browser-side trade — one <circle> DOM node per point against a single
    batched canvas draw — which no headless measurement can settle, so
    ``PRED_OBS_LARGE_POINT_COUNT`` is KEPT at ECharts' own ``largeThreshold``
    default of 2000: provenance, plus no Python-side signal for moving it.

    (The option also writes that same constant into ``largeThreshold``. That is
    tidiness only, not a third reason for the value: both sides read the one
    constant, so they agree at ANY value, and the ``large``/``largeThreshold``
    keys are written only when large mode is already on — so ECharts' own
    default cannot contradict ``pred_obs_renderer`` either way.)

    This test pins the boundary itself, which is the part that is measurable
    here: the last SVG point count and the first canvas one.
    """
    from gui.scripts.echarts_options import RENDERER_CANVAS, RENDERER_SVG
    from gui.scripts.evaluation_echarts import (
        PRED_OBS_LARGE_POINT_COUNT,
        load_pred_obs_plot_data,
        pred_obs_renderer,
        pred_obs_scatter_option,
    )

    assert PRED_OBS_LARGE_POINT_COUNT == 2000, "measured/justified above"

    def option_at(n, where):
        series = [float(i) for i in range(n)]
        _, record = _saved(tmp_path / where, obs=series, pred=list(series))
        plot_data = load_pred_obs_plot_data(record, "GLM", "d1", 300)
        assert len(plot_data.finite_points) == n
        return plot_data, _scatter(pred_obs_scatter_option(plot_data))

    below, below_scatter = option_at(PRED_OBS_LARGE_POINT_COUNT - 1, "below")
    at, at_scatter = option_at(PRED_OBS_LARGE_POINT_COUNT, "at")

    assert pred_obs_renderer(below) == RENDERER_SVG
    assert "large" not in below_scatter and "progressive" not in below_scatter

    assert pred_obs_renderer(at) == RENDERER_CANVAS
    assert at_scatter["large"] is True
    # written from the same constant the renderer choice reads
    assert at_scatter["largeThreshold"] == PRED_OBS_LARGE_POINT_COUNT


# ---------------------------------------------------------------------------
# Artifact resolution — typed record vs legacy derivation
# ---------------------------------------------------------------------------


def test_typed_artifact_wins_over_the_derived_path(tmp_path):
    """The run's own file, even when a same-named one sits in the shared folder.

    This is the whole reason Task 4 introduced typed artifacts: a later run
    against the same truth overwrites ``evaluation/<truth_tag>/``, so a derived
    path can point at another run's numbers.
    """
    from gui.scripts.evaluation_echarts import resolve_points_csv
    from spatialrisk.evaluations import EvaluationPlotArtifact

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    elsewhere = tmp_path / "somewhere_else" / "points.csv"
    record = _record(
        run_dir,
        artifacts=[
            EvaluationPlotArtifact(
                prediction_key="GLM__d1",
                model="GLM",
                period="d1",
                csize_px=300,
                points_csv=str(elsewhere),
                png_path=str(elsewhere.with_suffix(".png")),
            )
        ],
    )

    assert resolve_points_csv(record, "GLM", "d1", 300) == elsewhere


def test_a_legacy_record_derives_the_path_beside_its_indices_csv(tmp_path):
    """No artifacts: same derivation figure_entries uses for the PNG."""
    from gui.scripts.evaluation_charts import pred_obs_artifact_name
    from gui.scripts.evaluation_echarts import resolve_points_csv

    run_dir = tmp_path / "evaluation" / "loss_2010"
    resolved = resolve_points_csv(_record(run_dir), "GLM", "d1", 300)

    assert resolved == run_dir / pred_obs_artifact_name("GLM", "d1", 300, "csv")


def test_the_derived_csv_is_the_pngs_twin(tmp_path):
    """The two artifacts share one stem, so the derivations cannot drift."""
    from gui.scripts.evaluation_charts import figure_entries
    from gui.scripts.evaluation_echarts import resolve_points_csv

    run_dir = tmp_path / "evaluation" / "loss_2010"
    record = _record(run_dir)
    png = figure_entries(record.indices, 300, fig_dir=run_dir)[0][1]
    csv = resolve_points_csv(record, "GLM", "d1", 300)

    assert csv.with_suffix(".png") == png


def test_resolution_matches_on_cell_size(tmp_path):
    """The points CSV is resolved by cell size, not by artifact order."""
    from gui.scripts.evaluation_echarts import resolve_points_csv
    from spatialrisk.evaluations import EvaluationPlotArtifact

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    artifacts = [
        EvaluationPlotArtifact(
            prediction_key="GLM__d1",
            model="GLM",
            period="d1",
            csize_px=c,
            points_csv=str(run_dir / f"p{c}.csv"),
            png_path=str(run_dir / f"p{c}.png"),
        )
        for c in (100, 300)
    ]
    record = _record(run_dir, artifacts=artifacts)

    assert resolve_points_csv(record, "GLM", "d1", 100).name == "p100.csv"
    assert resolve_points_csv(record, "GLM", "d1", 300).name == "p300.csv"


def test_a_typed_png_path_outside_the_derived_directory_wins(tmp_path):
    """An explicit typed PNG path beats the path derived from the run dir."""
    from gui.scripts.evaluation_echarts import resolve_plot_artifact
    from spatialrisk.evaluations import EvaluationPlotArtifact

    elsewhere = tmp_path / "elsewhere" / "pred_obs_GLM_d1_300.png"
    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    art = EvaluationPlotArtifact(
        prediction_key="GLM__d1",
        model="GLM",
        period="d1",
        csize_px=300,
        points_csv=str(run_dir / "pred_obs_GLM_d1_300.csv"),
        png_path=str(elsewhere),
    )
    record = _record(run_dir, artifacts=[art])
    assert (
        resolve_plot_artifact(
            record,
            prediction_key="GLM__d1",
            model="GLM",
            period="d1",
            csize=300,
            kind="png_path",
        )
        == elsewhere
    )


def test_a_partial_manifest_derives_the_png_for_omitted_maps(tmp_path):
    """Tier 3: a map the manifest omits falls back to a deterministic name.

    The derived path sits beside the record's own indices CSV.
    """
    from gui.scripts.evaluation_echarts import resolve_plot_artifact

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    record = _typed_record(run_dir)  # records GLM/d1 only
    assert (
        resolve_plot_artifact(
            record,
            prediction_key="RF__d1",
            model="RF",
            period="d1",
            csize=300,
            kind="png_path",
        )
        == run_dir / "pred_obs_RF_d1_300.png"
    )


def _two_predictions_one_label(run_dir):
    """Two artifacts identical in (model, period, csize), different maps."""
    from spatialrisk.evaluations import EvaluationPlotArtifact

    return [
        EvaluationPlotArtifact(
            prediction_key=key,
            model="GLM",
            period="d1",
            csize_px=300,
            points_csv=str(run_dir / f"{key}.csv"),
            png_path=str(run_dir / f"{key}.png"),
        )
        for key in ("GLM__d1__a", "GLM__d1__b")
    ]


def test_prediction_key_picks_between_artifacts_sharing_a_label(tmp_path):
    """``(model, period, csize)`` is the FILE identity, not the map identity.

    Two predictions can carry the same model+period label — the same model rerun
    against another dataset revision, say — and then only ``prediction_key``
    tells their artifacts apart. Without it the first match wins, which is the
    right default for the single-prediction case but the wrong file here.
    """
    from gui.scripts.evaluation_echarts import resolve_points_csv

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    record = _record(run_dir, artifacts=_two_predictions_one_label(run_dir))

    assert resolve_points_csv(record, "GLM", "d1", 300).name == "GLM__d1__a.csv"
    assert (
        resolve_points_csv(record, "GLM", "d1", 300, prediction_key="GLM__d1__b").name
        == "GLM__d1__b.csv"
    )
    assert (
        resolve_points_csv(record, "GLM", "d1", 300, prediction_key="GLM__d1__a").name
        == "GLM__d1__a.csv"
    )


def test_an_exact_prediction_key_beats_the_loose_typed_match(tmp_path):
    """An exact prediction-key match wins over a looser match on kind alone."""
    from gui.scripts.evaluation_echarts import resolve_plot_artifact

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    art_a, art_b = _two_predictions_one_label(run_dir)
    record = _record(run_dir, artifacts=[art_a, art_b])
    assert resolve_plot_artifact(
        record,
        prediction_key=art_b.prediction_key,
        model=art_b.model,
        period=art_b.period,
        csize=300,
        kind="points_csv",
    ) == Path(art_b.points_csv)


def test_an_unmatched_prediction_key_falls_back_to_the_typed_match(tmp_path):
    """Tier 2: a key the manifest does not know still gets the run's own file.

    (REPLACES the old fall-straight-to-derivation behavior.) The unknown key
    resolves to the run's own typed file for that
    (model, period, csize) — older manifests recorded keys under a different
    scheme, and the run's own file beats a derived path a later run may have
    overwritten.

    Uses ``_two_predictions_one_label`` rather than ``_typed_record``: that
    fixture's artifact is named exactly the tier-3 derivation would produce
    (``pred_obs_GLM_d1_300.csv``), so it cannot tell the tier-2 fix apart from
    the OLD fall-straight-to-derivation behavior it replaces — both land on the
    same path. Here the typed artifacts are named ``GLM__d1__a.csv`` /
    ``GLM__d1__b.csv``, which diverge from the derived name, so only the tier-2
    match (not tier-3 derivation) can produce the asserted path.
    """
    from gui.scripts.evaluation_charts import pred_obs_artifact_name
    from gui.scripts.evaluation_echarts import resolve_points_csv

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    record = _record(run_dir, artifacts=_two_predictions_one_label(run_dir))
    resolved = resolve_points_csv(
        record, "GLM", "d1", 300, prediction_key="not_in_manifest"
    )

    assert resolved == Path(record.artifacts[0].points_csv)
    assert resolved != (
        Path(record.csv_path).parent / pred_obs_artifact_name("GLM", "d1", 300, "csv")
    )


def test_a_blank_field_on_the_exact_match_derives_instead_of_borrowing_a_sibling(
    tmp_path,
):
    """A blank field on the exact-key match must derive, never borrow.

    The exact match's empty slot falls through to tier 3 rather than to a
    SIBLING prediction's file.

    Two artifacts share (model, period, csize) with different prediction
    keys. The exact match (``GLM__d1__a``) recorded no PNG this run
    (``png_path=""``; its ``points_csv`` is populated, so the artifact itself
    is otherwise real). Before the fix, ``_typed_artifact_path``'s
    ``if not path: continue`` ran BEFORE the prediction-key comparison, so
    the exact match was skipped entirely and the loose tier-2 scan served the
    SIBLING's (``GLM__d1__b``) png_path — a different prediction's file
    served under this card. After the fix, the blank exact match returns
    None and ``resolve_plot_artifact`` falls through to the tier-3 derived
    path instead of ever considering ``GLM__d1__b``.
    """
    from gui.scripts.evaluation_charts import pred_obs_artifact_name
    from gui.scripts.evaluation_echarts import resolve_plot_artifact
    from spatialrisk.evaluations import EvaluationPlotArtifact

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    art_a = EvaluationPlotArtifact(
        prediction_key="GLM__d1__a",
        model="GLM",
        period="d1",
        csize_px=300,
        points_csv=str(run_dir / "GLM__d1__a.csv"),
        png_path="",
    )
    art_b = EvaluationPlotArtifact(
        prediction_key="GLM__d1__b",
        model="GLM",
        period="d1",
        csize_px=300,
        points_csv=str(run_dir / "GLM__d1__b.csv"),
        png_path=str(run_dir / "GLM__d1__b.png"),
    )
    record = _record(run_dir, artifacts=[art_a, art_b])

    resolved = resolve_plot_artifact(
        record,
        prediction_key="GLM__d1__a",
        model="GLM",
        period="d1",
        csize=300,
        kind="png_path",
        fallback_dir=run_dir,
    )

    assert resolved != Path(art_b.png_path), "must not borrow the sibling's file"
    assert resolved == run_dir / pred_obs_artifact_name("GLM", "d1", 300, "png")


def test_resolver_rejects_an_unknown_kind(tmp_path):
    """An unknown artifact kind is a programming error, not a silent miss."""
    from gui.scripts.evaluation_echarts import resolve_plot_artifact

    record = _typed_record(tmp_path / "evaluation" / "loss_2010" / "run1")
    with pytest.raises(ValueError):
        resolve_plot_artifact(
            record,
            prediction_key=None,
            model="GLM",
            period="d1",
            csize=300,
            kind="fig_path",
        )


def test_a_record_with_nothing_to_derive_from_resolves_to_nothing():
    """A record with no artifacts, indices or csv_path resolves to None."""
    import types

    from gui.scripts.evaluation_echarts import resolve_points_csv

    record = types.SimpleNamespace(artifacts=[], indices=[], csv_path=None)
    assert resolve_points_csv(record, "GLM", "d1", 300) is None


def test_an_incomplete_selection_resolves_to_nothing_instead_of_raising(tmp_path):
    """The Figures tab has no cell size until a run has one; don't crash on it."""
    from gui.scripts.evaluation_echarts import (
        load_pred_obs_plot_data,
        pred_obs_chart_identity,
        resolve_points_csv,
    )

    record = _record(tmp_path / "evaluation" / "loss_2010")
    for args in (("GLM", "d1", None), (None, "d1", 300), ("GLM", None, 300)):
        assert resolve_points_csv(record, *args) is None
        assert load_pred_obs_plot_data(record, *args) is None
        assert pred_obs_chart_identity(record, *args) is None


def test_a_missing_file_loads_as_none_instead_of_raising(tmp_path):
    """A saved run whose files were deleted must degrade, not crash the dialog."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    assert load_pred_obs_plot_data(_typed_record(run_dir), "GLM", "d1", 300) is None


def test_an_unreadable_file_loads_as_none_instead_of_raising(tmp_path):
    """A CSV without the expected columns loads as None rather than raising."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "pred_obs_GLM_d1_300.csv").write_text("not,a,point,table\n1,2,3,4\n")

    assert load_pred_obs_plot_data(_typed_record(run_dir), "GLM", "d1", 300) is None


def test_only_a_promised_table_warns_when_it_is_missing(tmp_path, caplog):
    """The log must say what the UI says — no louder.

    ``spatial_risk`` feeds the on-map log console at INFO+, so a WARNING is a
    user-visible message. ``_PredObsCard`` deliberately shows nothing for a
    legacy/PNG-only record (rung b): that map was never promised a point table,
    and its derived path is expected not to exist. Warning about it would put in
    front of the user precisely the message the UI decided not to show. A NEW
    record whose RECORDED table has gone missing (rung a) is a real fault and
    still warns, matching the card's own warning.
    """
    import logging

    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"

    with caplog.at_level(logging.DEBUG, logger="spatial_risk"):
        assert (
            load_pred_obs_plot_data(_record(run_dir), "GLM", "d1", 300) is None
        )  # legacy: no artifact
    levels = {r.levelno for r in caplog.records}
    assert levels == {logging.DEBUG}, [r.getMessage() for r in caplog.records]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="spatial_risk"):
        assert (
            load_pred_obs_plot_data(_typed_record(run_dir), "GLM", "d1", 300) is None
        )  # recorded, gone
    assert logging.WARNING in {r.levelno for r in caplog.records}


def test_the_option_of_nothing_is_nothing(tmp_path):
    """No plot data, or a table with no finite row, draws no empty frame."""
    from gui.scripts.evaluation_echarts import pred_obs_scatter_option

    assert pred_obs_scatter_option(None) is None
    _, plot_data = _loaded(tmp_path, obs=[float("nan")], pred=[1.0])
    assert pred_obs_scatter_option(plot_data) is None


# ---------------------------------------------------------------------------
# Malformed point CSVs must degrade to None (-> the PNG), never raise later
# ---------------------------------------------------------------------------


def _rewrite_column(csv_path, column, row_index, text):
    """Corrupt ONE cell of a written point CSV, byte-level (no pandas)."""
    lines = csv_path.read_text().strip().splitlines()
    header = lines[0].split(",")
    cells = lines[1 + row_index].split(",")
    cells[header.index(column)] = text
    lines[1 + row_index] = ",".join(cells)
    csv_path.write_text("\n".join(lines) + "\n")


def test_a_text_cell_id_degrades_to_none_instead_of_raising(tmp_path):
    """A non-numeric cell id degrades to None rather than raising."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path)
    _rewrite_column(csv_path, "cell", 0, "oops")
    assert load_pred_obs_plot_data(record, "GLM", "d1", 300) is None


def test_a_non_integral_cell_id_degrades_to_none(tmp_path):
    """A fractional cell id is not a cell id: the load degrades to None."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path)
    _rewrite_column(csv_path, "cell", 0, "1.5")
    assert load_pred_obs_plot_data(record, "GLM", "d1", 300) is None


def test_nan_forest_area_on_a_drawable_row_degrades_to_none(tmp_path):
    """NaN forest area on a drawable row degrades the whole load to None."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path, forest=[float("nan"), 11.0, 12.0, 13.0])
    assert load_pred_obs_plot_data(record, "GLM", "d1", 300) is None


def test_infinite_forest_area_on_a_drawable_row_degrades_to_none(tmp_path):
    """Infinite forest area on a drawable row degrades the whole load to None."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path, forest=[float("inf"), 11.0, 12.0, 13.0])
    assert load_pred_obs_plot_data(record, "GLM", "d1", 300) is None


def test_a_bad_value_on_a_non_drawable_row_does_not_block_the_chart(tmp_path):
    """Validation is scoped to DRAWABLE rows.

    A corrupt cell id on a row the finite-mask already excludes must not cost
    the run its interactive chart.
    """
    from gui.scripts.evaluation_echarts import (
        load_pred_obs_plot_data,
        pred_obs_scatter_option,
    )

    csv_path, record = _saved(
        tmp_path, obs=[1.0, float("nan")], pred=[1.0, 2.0], forest=[10.0, 11.0]
    )
    _rewrite_column(csv_path, "cell", 1, "oops")  # row 1 is non-drawable
    plot_data = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    assert plot_data is not None
    assert len(plot_data.finite_points) == 1
    assert pred_obs_scatter_option(plot_data) is not None


# ---------------------------------------------------------------------------
# Memoization keyed on file modification identity
# ---------------------------------------------------------------------------


def test_reloading_an_unchanged_file_does_not_re_read_it(tmp_path):
    """Reopening the dialog must not re-parse a 200k-row CSV."""
    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    _, record = _saved(tmp_path)
    first = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    second = load_pred_obs_plot_data(record, "GLM", "d1", 300)

    assert first is second


def test_a_same_size_rewrite_is_never_served_from_the_cache(tmp_path):
    """The exact bug class Task 4 exists to prevent, at the cache layer.

    Two runs against one truth write identically NAMED files. Keying the cache
    on the path alone would hand a record the previous run's points — a wrong
    chart with no error anywhere.

    The rewrite here is byte-for-byte the same LENGTH (same column widths), so
    only the mtime differs: this fails if the key drops mtime and keeps size.
    """
    import os

    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path, obs=[1.0, 2.0], pred=[3.0, 4.0])
    first = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    size_before = csv_path.stat().st_size

    _write_points(csv_path, obs=[5.0, 6.0], pred=[7.0, 8.0])
    os.utime(csv_path, (1, 1))  # a coarse mtime clock must not hide a rewrite
    assert csv_path.stat().st_size == size_before, "rewrite must be same-size"

    second = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    assert second is not first
    assert second.points["ndefor_obs_ha"].tolist() == [5.0, 6.0]


def test_a_same_mtime_rewrite_is_never_served_from_the_cache(tmp_path):
    """The mirror case: mtime pinned identical, only the size moves.

    Fails if the key drops size and keeps mtime. Together with the same-size
    test above, both halves of the modification identity are load-bearing.
    """
    import os

    from gui.scripts.evaluation_echarts import load_pred_obs_plot_data

    csv_path, record = _saved(tmp_path, obs=[1.0, 2.0], pred=[3.0, 4.0])
    os.utime(csv_path, (1, 1))
    first = load_pred_obs_plot_data(record, "GLM", "d1", 300)

    _write_points(csv_path, obs=[1.0, 2.0, 3.0], pred=[3.0, 4.0, 5.0])
    os.utime(csv_path, (1, 1))  # same mtime, more rows
    assert csv_path.stat().st_mtime_ns == 1_000_000_000

    second = load_pred_obs_plot_data(record, "GLM", "d1", 300)
    assert second is not first
    assert second.ncell == 3


def test_the_chart_identity_tracks_everything_the_chart_shows(tmp_path):
    """The widget's memo key and the adapter's digest escape hatch.

    Must move whenever the drawn chart would: a different run, a rewritten file,
    another cell size, the other theme.
    """
    import os

    from gui.scripts.evaluation_echarts import pred_obs_chart_identity

    csv_path, record = _saved(tmp_path)
    base = pred_obs_chart_identity(record, "GLM", "d1", 300)

    assert base == pred_obs_chart_identity(record, "GLM", "d1", 300)
    assert base != pred_obs_chart_identity(record, "GLM", "d1", 300, dark=True)
    assert base != pred_obs_chart_identity(record, "GLM", "d1", 100)

    other = _typed_record(csv_path.parent, run_id="run2")
    assert base != pred_obs_chart_identity(other, "GLM", "d1", 300)

    _write_points(csv_path, obs=[9.0], pred=[9.5])
    os.utime(csv_path, (1, 1))
    assert base != pred_obs_chart_identity(record, "GLM", "d1", 300)


def test_the_chart_identity_moves_when_the_dialog_switches_map(tmp_path):
    """Same record, same cell size, a different model/period.

    The likeliest interaction in the dialog: the map selector changes only
    ``model``/``period`` while the record and the cell size stay put. The
    identity separates the two through the RESOLVED PATH — the file name is
    built from exactly model, period and csize — so a version that dropped the
    path would hand one map's digest to the other map's option, which as an
    ``option_digest`` means the previous map's chart stays on screen.

    The three point CSVs are written with IDENTICAL content (default
    obs/pred/forest values every time), so they are byte-identical and thus
    already the same size. Left alone, ``mtime_ns`` from three sequential
    ``to_csv()`` calls would *usually* differ by OS write-timing jitter alone
    — which would let this assertion pass even with ``path`` dropped from the
    identity, since (size, mtime_ns) would then be the accidental
    discriminator instead. Pinning size (via the identical-content fixture)
    AND mtime_ns (explicitly, via ``os.utime``) to the SAME values on all
    three files removes that accident: ``path`` is left as the only thing
    that can still tell the three identities apart.
    """
    import os

    from gui.scripts.evaluation_echarts import pred_obs_chart_identity
    from spatialrisk.evaluations import EvaluationPlotArtifact

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    maps = [("GLM", "d1"), ("MW_w11", "d1"), ("GLM", "d2")]
    artifacts = []
    csv_paths = []
    for model, period in maps:
        csv_path = run_dir / f"pred_obs_{model}_{period}_300.csv"
        _write_points(csv_path)
        csv_paths.append(csv_path)
        artifacts.append(
            EvaluationPlotArtifact(
                prediction_key=f"{model}__{period}",
                model=model,
                period=period,
                csize_px=300,
                points_csv=str(csv_path),
                png_path=str(csv_path.with_suffix(".png")),
            )
        )
    record = _record(run_dir, artifacts=artifacts)

    sizes = {p.stat().st_size for p in csv_paths}
    assert len(sizes) == 1, "fixture CSVs must be byte-identical in size"
    pinned_ns = 1_700_000_000_000_000_000
    for p in csv_paths:
        os.utime(p, ns=(pinned_ns, pinned_ns))
        assert p.stat().st_mtime_ns == pinned_ns

    identities = [
        pred_obs_chart_identity(record, model, period, 300) for model, period in maps
    ]
    assert len(set(identities)) == len(maps)


def test_the_chart_identity_covers_the_labels_and_title_too(tmp_path):
    """It is passed as ``option_digest``, so it must cover every option input.

    ``labels`` and ``title`` are option inputs like any other — the widget layer
    (Task 7) fills them with ``t(...)`` output. Used as ``option_digest`` this
    string REPLACES the adapter's content hash, so text it did not cover would
    leave the old language's axis titles on screen after a locale switch, with
    no error anywhere.
    """
    from gui.scripts.evaluation_echarts import (
        pred_obs_chart_identity,
        pred_obs_scatter_option,
    )

    _, record = _saved(tmp_path)
    base = pred_obs_chart_identity(record, "GLM", "d1", 300)
    spanish = {"x_axis": "Observado (ha)", "y_axis": "Predicho (ha)"}

    assert base == pred_obs_chart_identity(record, "GLM", "d1", 300, labels={})
    assert base != pred_obs_chart_identity(record, "GLM", "d1", 300, labels=spanish)
    assert base != pred_obs_chart_identity(record, "GLM", "d1", 300, title="Mi gráfico")
    assert pred_obs_chart_identity(
        record, "GLM", "d1", 300, labels=spanish
    ) != pred_obs_chart_identity(record, "GLM", "d1", 300, title="Mi gráfico")
    # and the options those identities stand for really do differ
    _, plot_data = _loaded(tmp_path)
    assert pred_obs_scatter_option(
        plot_data, labels=spanish
    ) != pred_obs_scatter_option(plot_data)


def test_the_chart_identity_never_parses_the_file(tmp_path):
    """One stat() — no read, no parse, no option hash.

    That is the whole point: it exists so a per-render sha1 of an option holding
    hundreds of thousands of points never has to happen. Proven by pointing it
    at a file whose CONTENT is unparsable — a loader would raise or return
    None here, while a stat-only identity is unbothered.
    """
    from gui.scripts.evaluation_echarts import (
        load_pred_obs_plot_data,
        pred_obs_chart_identity,
    )

    run_dir = tmp_path / "evaluation" / "loss_2010" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "pred_obs_GLM_d1_300.csv").write_text("garbage,not,a,table\n")
    record = _typed_record(run_dir)

    identity = pred_obs_chart_identity(record, "GLM", "d1", 300)
    assert isinstance(identity, str) and len(identity) < 500
    assert load_pred_obs_plot_data(record, "GLM", "d1", 300) is None


def test_the_chart_identity_of_an_unresolvable_artifact_is_none():
    """An artifact that resolves to nothing has no chart identity."""
    import types

    from gui.scripts.evaluation_echarts import pred_obs_chart_identity

    record = types.SimpleNamespace(artifacts=[], indices=[], csv_path=None)
    assert pred_obs_chart_identity(record, "GLM", "d1", 300) is None
