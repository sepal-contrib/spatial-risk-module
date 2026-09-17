"""Unit tests for the evaluation step: metrics, artifacts, charts and dialog.

Covers ``spatialrisk.evaluation`` (validation maths, the indices/point CSVs and
the archived predicted-vs-observed PNG) together with the GUI layers built on
it — ``gui.scripts.evaluation_charts`` / ``evaluation_echarts`` for the option
builders and ``gui.widget.evaluation_results`` for the dialog itself.

Every import below the ``importorskip`` is deliberately not at the top of the
file: without rasterio there is nothing here to test, and the skip has to be
decided before the modules that need it are imported. Hence the ``E402``
suppressions.
"""

import contextlib
import types
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

import spatialrisk.evaluation as ev  # noqa: E402
from spatialrisk.evaluation import (  # noqa: E402
    PLOT_COLUMNS,
    PRED_OBS_X_LABEL,
    PRED_OBS_Y_LABEL,
    PredObsPlotData,
    ValidationResult,
    artifact_label_for,
    compute_validation,
    interval_from_target,
    label_for,
    make_square,
    pred_obs_axis_bounds,
    save_pred_obs_png,
    validate_two_layer,
    write_indices_csv,
    write_pred_obs_csv,
)


def test_interval_from_target_parses_two_years():
    """A 'forest_loss_YYYY_YYYY' target yields the span in years."""
    assert interval_from_target("forest_loss_2015_2020") == 5
    assert interval_from_target("forest_loss_2020_2024") == 4


def test_interval_from_target_handles_missing_years():
    """A target with no year pair yields no interval rather than raising."""
    assert interval_from_target("no_years_here") is None


def _pred(model_key, window=None, name=None):
    return types.SimpleNamespace(model_key=model_key, window=window, name=name)


def test_label_for_maps_family_and_window():
    """A prediction's display label names its model family and time window."""
    assert label_for(_pred("glm_glm_v1")) == "GLM"
    assert label_for(_pred("rf_rf_v1")) == "RF"
    assert label_for(_pred("icar_icar_v1")) == "ICAR"
    assert label_for(_pred("jnr_calibration_jnr")) == "JNR"
    assert label_for(_pred("mw_calibration_mw", window=11)) == "MW_w11"


def test_artifact_label_for_is_filename_safe_and_unique():
    """The artifact stem qualifies the family label with a sanitized run."""
    a = _pred("mw_calib_a", window=5)
    b = _pred("mw_calib_b", window=5)
    assert artifact_label_for(a) == "MW_w5_mw_calib_a"
    assert artifact_label_for(a) != artifact_label_for(b)
    named = _pred("mw_calib_a", window=5, name="val 2020!")
    assert artifact_label_for(named) == "MW_w5_val_2020"  # sanitized run name
    fully_sanitized = _pred("mw_calib_a", window=5, name="!!!")
    assert artifact_label_for(fully_sanitized) == "MW_w5_mw_calib_a"


def _write_raster(path, array, pixel=30.0):
    """Write a single-band GeoTIFF (EPSG:3857, square pixels)."""
    array = np.asarray(array)
    transform = from_origin(0, array.shape[0] * pixel, pixel, pixel)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="int32",
        crs="EPSG:3857",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(array.astype("int32"), 1)
    return str(path)


def test_make_square_partitions_600x300_into_two_cells(tmp_path):
    """A 600x300 raster at 300px splits into exactly two coarse cells."""
    r = _write_raster(tmp_path / "r.tif", np.ones((300, 600)))
    nsquare, nsquare_x, nsquare_y, x, y, nx, ny = make_square(r, 300)
    assert nsquare == 2 and nsquare_x == 2 and nsquare_y == 1
    assert x == [0, 300] and y == [0]
    assert nx == [300, 300] and ny == [300]


def test_make_square_handles_remainder(tmp_path):
    """A trailing partial column still becomes a cell, narrower than the rest."""
    r = _write_raster(tmp_path / "r2.tif", np.ones((100, 250)))
    nsquare, nsquare_x, nsquare_y, x, y, nx, ny = make_square(r, 100)
    assert nsquare_x == 3 and nx == [100, 100, 50]  # 250 = 100+100+50
    assert nsquare_y == 1 and ny == [100]


def test_validate_two_layer_perfect_prediction(tmp_path):
    """A prediction equal to the truth scores zero error and R2 = 1."""
    # 700 px wide -> make_square gives 3 cells [300,300,100]; the smaller cell makes
    # predicted/observed vary across cells so corrcoef (R2) is well-defined (=1.0).
    nrow, ncol, pixel = 300, 700, 30.0
    pix_area_ha = (pixel * pixel) / 10000.0  # 0.09 ha
    forest = np.ones((nrow, ncol), dtype="int32")  # all forest

    # 30% deforested per coarse cell (top 90 rows of each 300x300 block).
    defor = np.zeros((nrow, ncol), dtype="int32")
    defor[:90, :] = 1  # top 30% of rows deforested across all 700 cols

    risk = np.ones((nrow, ncol), dtype="int32")  # all category 1

    f = _write_raster(tmp_path / "forest.tif", forest, pixel)
    d = _write_raster(tmp_path / "defor.tif", defor, pixel)
    rk = _write_raster(tmp_path / "risk.tif", risk, pixel)

    # Per-cell: ndefor = 90*300 px; nfor = 300*300 px; cat-1 count = nfor.
    # predicted_ha = count * defor_dens * ti ; want == ndefor*pix_area_ha.
    time_interval = 5
    ndefor_px, nfor_px = 90 * 300, 300 * 300
    defor_dens = (ndefor_px * pix_area_ha) / (nfor_px * time_interval)
    tab = tmp_path / "defrate.csv"
    pd.DataFrame({"cat": [1], "defor_dens": [defor_dens]}).to_csv(tab, index=False)

    idx = validate_two_layer(
        defor_file=d,
        forest_file=f,
        riskmap_file=rk,
        tab_file_defor=str(tab),
        time_interval=time_interval,
        csize_coarse_grid=300,
        indices_file_pred=tmp_path / "indices.csv",
        tab_file_pred=tmp_path / "pred_obs.csv",
        fig_file_pred=tmp_path / "pred_obs.png",
        model_name="TEST",
        period="calibration",
    )
    assert idx["ncell"] == 3
    assert idx["RMSE"] == 0.0
    assert idx["wRMSE"] == 0.0
    assert idx["MedAE"] == 0.0
    assert idx["R2"] == 1.0
    assert (tmp_path / "indices.csv").exists()
    assert (tmp_path / "pred_obs.png").exists()


def _varied_validation_fixture(tmp_path):
    """Three varied coarse cells plus a zero-forest half that must be dropped.

    The three carry DIFFERENT observed/predicted values and two risk categories,
    so metrics, axis bounds and the scatter are all non-degenerate; the bottom
    half (cells 3, 4, 5) has no forest and exists to be cut by the
    ``nfor_obs > 0`` filter.

    Cell 2 is the 100px-wide remainder column, which makes nfor_obs vary too.
    The bottom half (rows 300-599) has no forest and no deforestation at all,
    so cells 3/4/5 get nfor_obs == 0 and are excluded from the result entirely;
    cells 0/1/2 read the exact same pixel region as before, so the previously
    pinned golden values are unaffected by this addition.
    """
    nrow, ncol, pixel = 600, 700, 30.0
    forest = np.ones((nrow, ncol), dtype="int32")
    forest[300:600, :] = 0  # bottom half: no forest recorded at all

    defor = np.zeros((nrow, ncol), dtype="int32")
    defor[:90, 0:300] = 1  # cell 0
    defor[:150, 300:600] = 1  # cell 1
    defor[:40, 600:700] = 1  # cell 2
    # bottom half (cells 3, 4, 5) stays all-zero deforestation too.

    risk = np.ones((nrow, ncol), dtype="int32")
    risk[:, 350:] = 2  # two categories with different densities

    tab = tmp_path / "defrate.csv"
    pd.DataFrame({"cat": [1, 2], "defor_dens": [0.0004, 0.00025]}).to_csv(
        tab, index=False
    )

    return dict(
        defor_file=_write_raster(tmp_path / "defor.tif", defor, pixel),
        forest_file=_write_raster(tmp_path / "forest.tif", forest, pixel),
        riskmap_file=_write_raster(tmp_path / "risk.tif", risk, pixel),
        tab_file_defor=str(tab),
        time_interval=5,
    )


# Golden values captured from validate_two_layer BEFORE the plot-data refactor
# (git show 643a441:spatialrisk/evaluation.py), run against the fixture ABOVE
# (including its zero-forest bottom half). They pin the frozen numerics: metric
# formulas, round(..., 2), CSV columns/order, and the nfor_obs > 0 drop filter.
# Re-verified after the fixture gained the zero-forest cells (3, 4, 5): the
# legacy implementation drops them too, so the surviving-cell values below are
# byte-identical to what was pinned before the fixture change.
_GOLDEN_INDICES = {
    "RMSE": 2619.28,
    "wRMSE": 2964.98,
    "MedAE": 2250.0,
    "R2": 0.43,
    "ncell": 3,
    "csize_coarse_grid": 300,
    "csize_coarse_grid_ha": 8100.0,
}
_GOLDEN_POINT_CSV = (
    "cell,nfor_obs,ndefor_obs,nfor_obs_ha,ndefor_obs_ha,ndefor_pred_ha\n"
    "0,90000,27000,8100.0,2430.0,180.0\n"
    "1,90000,45000,8100.0,4050.0,123.75\n"
    "2,30000,4000,2700.0,360.0,37.5\n"
)
_GOLDEN_INDICES_CSV = (
    "RMSE,wRMSE,MedAE,R2,ncell,csize_coarse_grid,csize_coarse_grid_ha\n"
    "2619.28,2964.98,2250.0,0.43,3,300,8100.0\n"
)


def test_validate_two_layer_matches_golden_metrics_and_csvs(tmp_path):
    """Characterization test: numbers and CSV bytes must never move."""
    lay = _varied_validation_fixture(tmp_path)
    idx = validate_two_layer(
        **lay,
        csize_coarse_grid=300,
        indices_file_pred=tmp_path / "indices.csv",
        tab_file_pred=tmp_path / "pred_obs.csv",
        fig_file_pred=tmp_path / "pred_obs.png",
        model_name="TEST",
        period="calibration",
    )
    assert idx == _GOLDEN_INDICES
    assert (tmp_path / "pred_obs.csv").read_text() == _GOLDEN_POINT_CSV
    assert (tmp_path / "indices.csv").read_text() == _GOLDEN_INDICES_CSV
    assert (tmp_path / "pred_obs.png").stat().st_size > 1000


def test_compute_validation_drops_cells_with_zero_forest(tmp_path):
    """A cell with no forest at the start of the period is dropped entirely.

    The fixture's bottom half (cells 3, 4, 5) has zero forest and zero
    deforestation everywhere; only the top row's cells 0/1/2 may survive the
    ``nfor_obs > 0`` filter.
    """
    lay = _varied_validation_fixture(tmp_path)
    result = compute_validation(
        **lay, csize_coarse_grid=300, model_name="TEST", period="calibration"
    )
    assert set(result.plot_data.points["cell"]) == {0, 1, 2}
    assert result.indices["ncell"] == 3


def test_validate_two_layer_forwards_figsize_and_dpi_to_png(tmp_path):
    """figsize/dpi reach the rendered PNG's pixel dimensions.

    Forwarding them is essentially the wrapper's remaining job, so a non-default
    value must actually land on the image rather than be accepted and silently
    dropped.
    """
    import matplotlib.image as mpimg

    lay = _varied_validation_fixture(tmp_path)
    fig_path = tmp_path / "pred_obs.png"
    validate_two_layer(
        **lay,
        csize_coarse_grid=300,
        indices_file_pred=tmp_path / "indices.csv",
        tab_file_pred=tmp_path / "pred_obs.csv",
        fig_file_pred=fig_path,
        model_name="TEST",
        period="calibration",
        figsize=(3.0, 3.0),
        dpi=50,
    )
    img = mpimg.imread(fig_path)
    height, width = img.shape[0], img.shape[1]
    assert (width, height) == (150, 150)  # figsize(3.0, 3.0) * dpi(50)


def _points(obs, pred):
    """Minimal points frame: the four columns every renderer path requires.

    ``pred_obs_axis_bounds`` only reads the two coordinate columns, but
    ``PredObsPlotData`` requires all of ``PLOT_COLUMNS``, so this builds the
    narrowest frame the dataclass accepts.
    """
    n = len(list(obs))
    return pd.DataFrame(
        {
            "cell": list(range(n)),
            "nfor_obs_ha": [100.0] * n,
            "ndefor_obs_ha": obs,
            "ndefor_pred_ha": pred,
        }
    )


def test_pred_obs_axis_bounds_spans_both_series(tmp_path):
    """The shared axis domain covers observed and predicted alike."""
    lo, hi = pred_obs_axis_bounds(
        _points([2430.0, 4050.0, 360.0], [180.0, 123.75, 37.5])
    )
    assert (lo, hi) == (37.5, 4050.0)


def test_pred_obs_axis_bounds_empty_falls_back_to_unit_range():
    """No points leaves a drawable 0..1 domain rather than an empty one."""
    assert pred_obs_axis_bounds(_points([], [])) == (0.0, 1.0)


def test_pred_obs_axis_bounds_all_nan_falls_back_to_unit_range():
    """All-NaN input is treated as no points at all."""
    assert pred_obs_axis_bounds(_points([np.nan, np.nan], [np.nan, np.nan])) == (
        0.0,
        1.0,
    )


def test_pred_obs_axis_bounds_ignores_infinities():
    """An infinite value must not stretch the domain to infinity."""
    lo, hi = pred_obs_axis_bounds(_points([1.0, np.inf], [-np.inf, 4.0]))
    assert (lo, hi) == (1.0, 4.0)


def test_pred_obs_axis_bounds_all_infinite_falls_back_to_unit_range():
    """All-infinite input is treated as no points at all."""
    assert pred_obs_axis_bounds(_points([np.inf, -np.inf], [np.inf, np.inf])) == (
        0.0,
        1.0,
    )


def test_pred_obs_axis_bounds_constant_series_is_padded():
    """A single repeated value still gets a domain with width."""
    # A zero-width domain would collapse an ECharts axis; pad it symmetrically.
    assert pred_obs_axis_bounds(_points([5.0, 5.0], [5.0, 5.0])) == (0.0, 10.0)


def test_pred_obs_axis_bounds_all_zero_falls_back_to_unit_range():
    """All zeros would give a zero-width domain, so the unit range stands in."""
    assert pred_obs_axis_bounds(_points([0.0, 0.0], [0.0, 0.0])) == (0.0, 1.0)


def test_finite_points_drops_non_finite_rows_without_touching_points():
    """The renderer-safe view is a filter, never a mutation of ``points``."""
    points = _points([1.0, np.nan, 3.0, np.inf], [1.0, 2.0, np.inf, 4.0])
    data = PredObsPlotData(
        model="M",
        period="p",
        csize_px=300,
        csize_ha=8100.0,
        points=points,
        axis_min=1.0,
        axis_max=3.0,
        medae=0.0,
        r2=1.0,
        ncell=4,
    )
    assert len(data.points) == 4  # CSV payload untouched
    assert list(data.finite_points["ndefor_obs_ha"]) == [1.0]
    assert list(data.finite_points["ndefor_pred_ha"]) == [1.0]


def test_compute_validation_returns_indices_and_plot_data(tmp_path):
    """One call yields both the metric rows and the scatter's plot data."""
    lay = _varied_validation_fixture(tmp_path)
    result = compute_validation(
        **lay, csize_coarse_grid=300, model_name="TEST", period="calibration"
    )

    assert isinstance(result, ValidationResult)
    assert result.indices == _GOLDEN_INDICES

    pd_ = result.plot_data
    assert isinstance(pd_, PredObsPlotData)
    assert (pd_.model, pd_.period) == ("TEST", "calibration")
    assert (pd_.csize_px, pd_.csize_ha) == (300, 8100.0)
    assert (pd_.axis_min, pd_.axis_max) == (37.5, 4050.0)
    assert (pd_.medae, pd_.r2, pd_.ncell) == (2250.0, 0.43, 3)
    assert list(pd_.points.columns) == [
        "cell",
        "nfor_obs",
        "ndefor_obs",
        "nfor_obs_ha",
        "ndefor_obs_ha",
        "ndefor_pred_ha",
    ]


def test_compute_validation_writes_nothing(tmp_path):
    """Computing metrics is pure — every artifact is written by its own call."""
    lay = _varied_validation_fixture(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    compute_validation(**lay, csize_coarse_grid=300)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_plot_data_carries_chart_labels():
    """The plot data carries the axis titles the archived PNG uses."""
    data = PredObsPlotData(
        model="GLM",
        period="calibration",
        csize_px=300,
        csize_ha=8100.0,
        points=_points([1.0], [2.0]),
        axis_min=1.0,
        axis_max=2.0,
        medae=1.5,
        r2=0.42,
        ncell=1,
    )
    assert data.title == (
        "GLM model, calibration period\n"
        "Predicted vs. observed deforestation in 8100.0 ha grid cells."
    )
    assert data.annotation == "MedAE = 1.50 ha\nR2 = 0.42\nn = 1"
    assert (data.x_label, data.y_label) == (PRED_OBS_X_LABEL, PRED_OBS_Y_LABEL)


def _base_plot_data_kwargs():
    """Valid PredObsPlotData kwargs: 2 finite points and sane axis bounds.

    Each ``__post_init__`` test overrides only the one field it is about.
    """
    return dict(
        model="M",
        period="p",
        csize_px=300,
        csize_ha=8100.0,
        points=_points([1.0, 3.0], [2.0, 4.0]),
        axis_min=1.0,
        axis_max=4.0,
        medae=0.0,
        r2=1.0,
        ncell=2,
    )


def test_plot_data_rejects_nan_axis_bounds():
    """A NaN bound would break the axis silently, so it raises at construction."""
    kwargs = _base_plot_data_kwargs()
    kwargs["axis_min"] = float("nan")
    with pytest.raises(ValueError):
        PredObsPlotData(**kwargs)


def test_plot_data_rejects_infinite_axis_bounds():
    """An infinite bound raises at construction."""
    kwargs = _base_plot_data_kwargs()
    kwargs["axis_max"] = float("inf")
    with pytest.raises(ValueError):
        PredObsPlotData(**kwargs)


def test_plot_data_rejects_zero_width_axis_domain():
    """A zero-width domain (min == max) leaves nothing to draw, so it raises."""
    kwargs = _base_plot_data_kwargs()
    kwargs["axis_min"] = kwargs["axis_max"] = 5.0
    with pytest.raises(ValueError):
        PredObsPlotData(**kwargs)


def test_plot_data_rejects_inverted_axis_domain():
    """An inverted domain (max < min) raises at construction."""
    kwargs = _base_plot_data_kwargs()
    kwargs["axis_min"], kwargs["axis_max"] = 4.0, 1.0
    with pytest.raises(ValueError):
        PredObsPlotData(**kwargs)


def test_plot_data_rejects_ncell_mismatched_with_points():
    """``ncell`` must equal the row count it claims to describe."""
    kwargs = _base_plot_data_kwargs()
    kwargs["ncell"] = 3  # points only has 2 rows
    with pytest.raises(ValueError):
        PredObsPlotData(**kwargs)


def test_plot_data_accepts_well_formed_bounds_and_ncell():
    """The valid fixture constructs cleanly — the negative tests' control."""
    # Sanity: the valid baseline itself must construct without raising.
    PredObsPlotData(**_base_plot_data_kwargs())


@pytest.mark.parametrize("missing", PLOT_COLUMNS)
def test_plot_data_rejects_points_missing_a_required_plot_column(missing):
    """``points`` has two construction paths of different widths.

    ``compute_validation`` builds the full 6-column CSV table; the GUI loader
    reads back only the plotted columns. Both must carry every column a renderer
    indexes, so the guard is on the columns rather than on the width — which
    also turns a truncated or renamed CSV into an error at construction instead
    of a KeyError inside a render.
    """
    kwargs = _base_plot_data_kwargs()
    kwargs["points"] = kwargs["points"].drop(columns=[missing])
    with pytest.raises(ValueError, match="missing required plot column"):
        PredObsPlotData(**kwargs)


def test_plot_data_accepts_the_wider_compute_validation_frame(tmp_path):
    """The other path: 6 columns is MORE than required, never an error."""
    lay = _varied_validation_fixture(tmp_path)
    points = compute_validation(**lay, csize_coarse_grid=300).plot_data.points

    assert len(points.columns) == 6
    data = PredObsPlotData(
        model="M",
        period="p",
        csize_px=300,
        csize_ha=8100.0,
        points=points,
        axis_min=1.0,
        axis_max=4.0,
        medae=0.0,
        r2=1.0,
        ncell=len(points),
    )
    assert list(data.points.columns) == list(points.columns)


def test_write_pred_obs_csv_matches_golden_bytes(tmp_path):
    """The point CSV is byte-frozen: the chart and the PNG read one file."""
    lay = _varied_validation_fixture(tmp_path)
    result = compute_validation(**lay, csize_coarse_grid=300)
    out = write_pred_obs_csv(result.plot_data, tmp_path / "points.csv")
    assert _Path(out).read_text() == _GOLDEN_POINT_CSV


def test_write_pred_obs_csv_persists_non_finite_rows(tmp_path):
    """The point CSV is the frozen artifact, so non-finite rows reach disk.

    ``points`` is what gets written, not the renderer-safe ``finite_points``
    subset: the table has to hold every row exactly as computed.
    """
    points = _points([1.0, np.nan, 3.0], [1.0, 2.0, np.inf])
    data = PredObsPlotData(
        model="M",
        period="p",
        csize_px=300,
        csize_ha=8100.0,
        points=points,
        axis_min=1.0,
        axis_max=3.0,
        medae=0.0,
        r2=1.0,
        ncell=3,
    )
    assert len(data.finite_points) == 1  # sanity: the renderer subset drops 2 rows

    out = write_pred_obs_csv(data, tmp_path / "points.csv")
    lines = _Path(out).read_text().strip().splitlines()
    assert len(lines) == 1 + 3  # header + all 3 rows, non-finite ones included


def test_write_indices_csv_matches_golden_bytes(tmp_path):
    """The indices CSV is byte-frozen — it is the run's persisted record."""
    lay = _varied_validation_fixture(tmp_path)
    result = compute_validation(**lay, csize_coarse_grid=300)
    out = write_indices_csv(result.indices, tmp_path / "idx.csv")
    assert _Path(out).read_text() == _GOLDEN_INDICES_CSV


def _legacy_pred_obs_png(
    df,
    *,
    model_name,
    period,
    csize_ha,
    MedAE,
    r_square,
    ncell,
    path,
    figsize=(6.4, 6.4),
    dpi=100,
):
    """The pre-refactor matplotlib block, verbatim, for byte-equivalence proof."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = (
        f"{model_name} model, {period} period\n"
        f"Predicted vs. observed deforestation in {csize_ha} ha grid cells."
    )
    p = [
        df[["ndefor_obs_ha", "ndefor_pred_ha"]].min(axis=None),
        df[["ndefor_obs_ha", "ndefor_pred_ha"]].max(axis=None),
    ]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot(111)
    ax.set_box_aspect(1)
    plt.scatter(
        df["ndefor_obs_ha"], df["ndefor_pred_ha"], color=None, marker="o", edgecolor="k"
    )
    plt.plot(p, p, "r--")
    plt.title(title)
    plt.xlabel("Observed deforestation (ha)")
    plt.ylabel("Predicted deforestation (ha)")
    t = f"MedAE = {MedAE:.2f} ha\nR2 = {r_square:.2f}\nn = {ncell:d}"
    y_text = df[["ndefor_obs_ha", "ndefor_pred_ha"]].max(axis=None)
    plt.text(0, y_text, t, ha="left", va="top")
    fig.savefig(path)
    plt.close(fig)
    return path


def test_save_pred_obs_png_is_byte_identical_to_legacy_matplotlib(tmp_path):
    """The archived figure must not drift from the code it replaced."""
    lay = _varied_validation_fixture(tmp_path)
    result = compute_validation(
        **lay, csize_coarse_grid=300, model_name="TEST", period="calibration"
    )

    legacy = _legacy_pred_obs_png(
        result.plot_data.points,
        model_name="TEST",
        period="calibration",
        csize_ha=8100.0,
        MedAE=2250.0,
        r_square=0.43,
        ncell=3,
        path=tmp_path / "legacy.png",
    )
    new = save_pred_obs_png(result.plot_data, tmp_path / "new.png")

    assert _Path(new).read_bytes() == _Path(legacy).read_bytes()


def test_save_pred_obs_png_handles_degenerate_plot_data(tmp_path):
    """Empty / NaN / constant input must still render, with a finite axis."""
    cols = [
        "cell",
        "nfor_obs",
        "ndefor_obs",
        "nfor_obs_ha",
        "ndefor_obs_ha",
        "ndefor_pred_ha",
    ]
    cases = {
        "empty": pd.DataFrame(columns=cols),
        "nan": pd.DataFrame(
            {
                **{c: [0] for c in cols[:4]},
                "ndefor_obs_ha": [np.nan],
                "ndefor_pred_ha": [np.nan],
            }
        ),
        "constant": pd.DataFrame(
            {
                **{c: [0, 0] for c in cols[:4]},
                "ndefor_obs_ha": [7.0, 7.0],
                "ndefor_pred_ha": [7.0, 7.0],
            }
        ),
    }
    for name, points in cases.items():
        lo, hi = pred_obs_axis_bounds(points)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi, name
        data = PredObsPlotData(
            model="M",
            period="p",
            csize_px=300,
            csize_ha=8100.0,
            points=points,
            axis_min=lo,
            axis_max=hi,
            medae=float("nan"),
            r2=float("nan"),
            ncell=len(points),
        )
        out = save_pred_obs_png(data, tmp_path / f"{name}.png")
        assert _Path(out).stat().st_size > 1000, name


def test_the_png_and_the_interactive_chart_share_one_plot_data(tmp_path, monkeypatch):
    """One computed source feeds both renderings — asserted by identity.

    ``validate_two_layer`` computes a single ``PredObsPlotData`` and hands the
    SAME object to the point-CSV writer and to the PNG writer; the interactive
    ECharts scatter is then built from the file that object produced. If either
    renderer ever grew its own computation the archived figure and the
    on-screen chart could disagree about a cell while every test that looks at
    one of them in isolation still passed.

    Verified by behaviour, not by ``inspect.getsource``: the two writers are
    wrapped so the test sees the objects that actually reached them, and the
    chart option is then built by the GUI loader from the CSV they wrote and
    compared back against that very frame.
    """
    from gui.scripts.evaluation_echarts import (
        load_pred_obs_plot_data,
        pred_obs_scatter_option,
    )

    seen = {}
    real_csv, real_png = ev.write_pred_obs_csv, ev.save_pred_obs_png

    def spy_csv(plot_data, output_path):
        seen["csv"] = plot_data
        return real_csv(plot_data, output_path)

    def spy_png(plot_data, output_path, **kw):
        seen["png"] = plot_data
        return real_png(plot_data, output_path, **kw)

    monkeypatch.setattr(ev, "write_pred_obs_csv", spy_csv)
    monkeypatch.setattr(ev, "save_pred_obs_png", spy_png)

    lay = _varied_validation_fixture(tmp_path)
    validate_two_layer(
        **lay,
        csize_coarse_grid=300,
        indices_file_pred=tmp_path / "indices_TEST_calibration_300.csv",
        tab_file_pred=tmp_path / "pred_obs_TEST_calibration_300.csv",
        fig_file_pred=tmp_path / "pred_obs_TEST_calibration_300.png",
        model_name="TEST",
        period="calibration",
    )

    assert seen["csv"] is seen["png"], "two renderings, one PredObsPlotData"

    record = types.SimpleNamespace(
        indices=[
            {
                "model": "TEST",
                "period": "calibration",
                "csize_coarse_grid": 300,
                "csize_coarse_grid_ha": 8100.0,
                "MedAE": 2250.0,
                "R2": 0.43,
            }
        ],
        artifacts=[],
        run_id="run00001",
        csv_path=str(tmp_path / "indices_all.csv"),
    )
    option = pred_obs_scatter_option(
        load_pred_obs_plot_data(record, "TEST", "calibration", 300)
    )

    shared = seen["png"].finite_points
    assert [v[:2] for v in option["series"][0]["data"]] == [
        [o, p] for o, p in zip(shared["ndefor_obs_ha"], shared["ndefor_pred_ha"])
    ]


def _fake_project_with_prediction(tmp_path):
    target = types.SimpleNamespace(
        name="forest_loss_2015_2020", path=tmp_path / "defor.tif"
    )
    forest = types.SimpleNamespace(name="forest_gfc", path=tmp_path / "forest.tif")
    dataset = types.SimpleNamespace(
        name="calibration", target=target, features=[forest]
    )
    pred = types.SimpleNamespace(
        model_key="glm_glm_v1",
        window=None,
        dataset_name="calibration",
        path=tmp_path / "risk.tif",
        metrics={},
    )
    project = types.SimpleNamespace(
        folders=types.SimpleNamespace(project_folder=tmp_path),
        get_dataset=lambda n: dataset if n == "calibration" else None,
        predictions={"glm_glm_v1__calibration_y2015": pred},
        save=lambda: None,
    )
    return project, pred, dataset


def test_resolve_layers_recovers_from_dataset(tmp_path):
    """Missing layer paths are recovered from the dataset rather than failing."""
    project, pred, dataset = _fake_project_with_prediction(tmp_path)
    lay = ev.resolve_layers(project, pred)
    assert lay["defor_file"] == dataset.target.path
    assert lay["forest_file"] == dataset.features[0].path
    assert lay["riskmap_file"] == pred.path
    assert lay["time_interval"] == 5
    assert lay["period"] == "calibration"


def test_evaluate_prediction_runs_defrate_then_validate(tmp_path, monkeypatch):
    """One prediction: the deforestation rate first, then validation on it."""
    project, pred, dataset = _fake_project_with_prediction(tmp_path)
    calls = {}

    def fake_defrate_per_cat(**kw):
        calls["defrate"] = kw
        _Path(kw["tab_file_defrate"]).write_text("cat,defor_dens\n1,0.01\n")

    def fake_validate(**kw):
        calls["validate"] = kw
        return {
            "RMSE": 1.0,
            "wRMSE": 2.0,
            "MedAE": 0.5,
            "R2": 0.9,
            "ncell": 26,
            "csize_coarse_grid": kw["csize_coarse_grid"],
            "csize_coarse_grid_ha": 8100.0,
        }

    monkeypatch.setattr(ev, "_defrate_per_cat", fake_defrate_per_cat)
    monkeypatch.setattr(ev, "validate_two_layer", fake_validate)

    rows = ev.evaluate_prediction(project, pred, csizes=(300,))
    assert len(rows) == 1
    assert rows[0]["model"] == "GLM" and rows[0]["period"] == "calibration"
    assert rows[0]["R2"] == 0.9
    assert calls["defrate"]["time_interval"] == 5
    assert calls["validate"]["csize_coarse_grid"] == 300
    # prediction key: fake pred has no storage_key() -> fallback "{model_key}__{period}"
    assert rows[0]["prediction"] == "glm_glm_v1__calibration"
    # fig_path is added by evaluate_prediction (the fake validate does not return it)
    assert rows[0]["fig_path"].endswith("pred_obs_GLM_calibration_300.png")
    # pred.metrics receives the per-(period,csize) index subset
    assert pred.metrics == {
        "calibration_300": {
            "RMSE": 1.0,
            "wRMSE": 2.0,
            "MedAE": 0.5,
            "R2": 0.9,
            "ncell": 26,
        }
    }


def test_evaluate_predictions_filters_and_aggregates(tmp_path, monkeypatch):
    """Several predictions are filtered to the evaluable ones and pooled."""
    project, pred, dataset = _fake_project_with_prediction(tmp_path)
    # second prediction in a different period that we will filter out
    pred2 = types.SimpleNamespace(
        model_key="rf_rf_v1",
        window=None,
        dataset_name="validation",
        path=tmp_path / "risk2.tif",
        metrics={},
    )
    project.predictions["rf_rf_v1__validation_y2015"] = pred2

    monkeypatch.setattr(
        ev,
        "evaluate_prediction",
        lambda proj, p, csizes=(300,), recompute_defrate=True: [
            {
                "prediction": p.model_key,
                "model": ev.label_for(p),
                "period": p.dataset_name,
                "csize_coarse_grid": 300,
                "csize_coarse_grid_ha": 8100.0,
                "ncell": 26,
                "MedAE": 1.0,
                "R2": 0.5,
                "RMSE": 2.0,
                "wRMSE": 3.0,
                "fig_path": "x.png",
            }
        ],
    )

    df = ev.evaluate_predictions(project, dataset_filter=["calibration"])
    assert list(df["period"].unique()) == ["calibration"]  # validation filtered out
    assert set(["MedAE", "R2", "RMSE", "wRMSE"]).issubset(df.columns)
    assert (_Path(tmp_path) / "evaluation" / "indices_all.csv").exists()


def test_evaluate_one_against_truth_uses_explicit_truth(tmp_path, monkeypatch):
    """An explicit truth layer wins over anything derivable from the project."""
    pred = types.SimpleNamespace(
        model_key="glm_glm_v1",
        window=None,
        dataset_name="validation",
        path=tmp_path / "risk.tif",
        metrics={},
        storage_key=lambda: "glm_glm_v1__validation",
    )
    project = types.SimpleNamespace(
        folders=types.SimpleNamespace(project_folder=tmp_path)
    )
    calls = {}

    def fake_defrate(**kw):
        calls["defrate"] = kw
        _Path(kw["tab_file_defrate"]).write_text("cat,defor_dens\n1,0.01\n")

    def fake_validate(**kw):
        calls["validate"] = kw
        return {
            "RMSE": 1.0,
            "wRMSE": 2.0,
            "MedAE": 0.5,
            "R2": 0.9,
            "ncell": 26,
            "csize_coarse_grid": kw["csize_coarse_grid"],
            "csize_coarse_grid_ha": 8100.0,
        }

    monkeypatch.setattr(ev, "_defrate_per_cat", fake_defrate)
    monkeypatch.setattr(ev, "validate_two_layer", fake_validate)

    truth_defor = tmp_path / "truth_defor.tif"
    truth_forest = tmp_path / "truth_forest.tif"
    rows = ev._evaluate_one_against_truth(
        project,
        pred,
        defor_file=truth_defor,
        forest_file=truth_forest,
        time_interval=7,
        truth_tag="forest_loss_2015_2020",
        csizes=(300,),
    )

    # the SHARED truth is used, not the map's own dataset
    assert calls["defrate"]["defor_file"] == truth_defor
    assert calls["defrate"]["forest_file"] == truth_forest
    assert calls["defrate"]["time_interval"] == 7
    assert calls["validate"]["riskmap_file"] == pred.path
    assert calls["validate"]["time_interval"] == 7
    # row annotations
    assert rows[0]["truth"] == "forest_loss_2015_2020"
    assert rows[0]["period"] == "validation"
    assert rows[0]["model"] == "GLM_glm_glm_v1"
    assert rows[0]["prediction"] == "glm_glm_v1__validation"
    # output namespaced under evaluation/<truth_tag>/
    assert (tmp_path / "evaluation" / "forest_loss_2015_2020").is_dir()
    assert rows[0]["fig_path"].endswith(
        "evaluation/forest_loss_2015_2020/pred_obs_GLM_glm_glm_v1_validation_300.png"
    )
    # metrics keyed by "<tag>__<period>_<csize>"
    assert pred.metrics == {
        "forest_loss_2015_2020__validation_300": {
            "RMSE": 1.0,
            "wRMSE": 2.0,
            "MedAE": 0.5,
            "R2": 0.9,
            "ncell": 26,
        }
    }


def test_evaluate_against_truth_selects_keys_and_namespaces(tmp_path, monkeypatch):
    """Only the selected keys are scored, each in its own namespace."""
    p1 = types.SimpleNamespace(
        model_key="glm_glm_v1",
        window=None,
        dataset_name="calibration",
        path=tmp_path / "r1.tif",
        metrics={},
        storage_key=lambda: "glm_glm_v1__calibration",
    )
    p2 = types.SimpleNamespace(
        model_key="rf_rf_v1",
        window=None,
        dataset_name="validation",
        path=tmp_path / "r2.tif",
        metrics={},
        storage_key=lambda: "rf_rf_v1__validation",
    )
    saved = {"n": 0}
    project = types.SimpleNamespace(
        folders=types.SimpleNamespace(project_folder=tmp_path),
        predictions={"k1": p1, "k2": p2},
        save=lambda: saved.__setitem__("n", saved["n"] + 1),
    )

    monkeypatch.setattr(
        ev,
        "_evaluate_one_against_truth",
        lambda proj, pred, **kw: [
            {
                "prediction": pred.storage_key(),
                "model": ev.artifact_label_for(pred),
                "period": pred.dataset_name,
                "truth": kw["truth_tag"],
                "csize_coarse_grid": 300,
                "csize_coarse_grid_ha": 8100.0,
                "ncell": 26,
                "MedAE": 1.0,
                "R2": 0.5,
                "RMSE": 2.0,
                "wRMSE": 3.0,
                "fig_path": "x.png",
            }
        ],
    )

    df = ev.evaluate_against_truth(
        project,
        prediction_keys=["k1"],
        defor_file=tmp_path / "d.tif",
        forest_file=tmp_path / "f.tif",
        time_interval=5,
        truth_tag="forest_loss_2015_2020",
    )

    assert list(df["period"]) == ["calibration"]  # only k1 was selected
    assert list(df["truth"]) == ["forest_loss_2015_2020"]
    assert "truth" in df.columns
    assert (
        tmp_path / "evaluation" / "forest_loss_2015_2020" / "indices_all.csv"
    ).exists()
    assert saved["n"] == 1  # auto_save ran


def test_evaluate_against_truth_skips_unknown_key(tmp_path, monkeypatch, capsys):
    """An unknown key is skipped with a message, never a crash."""
    project = types.SimpleNamespace(
        folders=types.SimpleNamespace(project_folder=tmp_path),
        predictions={},
        save=lambda: None,
    )

    df = ev.evaluate_against_truth(
        project,
        prediction_keys=["nope"],
        defor_file=tmp_path / "d.tif",
        forest_file=tmp_path / "f.tif",
        time_interval=5,
        truth_tag="t",
    )

    assert len(df) == 0
    assert "skipped nope" in capsys.readouterr().out
    assert (tmp_path / "evaluation" / "t" / "indices_all.csv").exists()


# ---------------------------------------------------------------------------
# Run-scoped, history-safe artifacts (Task 4)
# ---------------------------------------------------------------------------


def _truth_project_and_pred(tmp_path):
    """Fake project holding ONE prediction, scored against an explicit truth."""
    pred = types.SimpleNamespace(
        model_key="glm_glm_v1",
        window=None,
        dataset_name="validation",
        path=tmp_path / "risk.tif",
        metrics={},
        storage_key=lambda: "glm_glm_v1__validation",
    )
    project = types.SimpleNamespace(
        folders=types.SimpleNamespace(project_folder=tmp_path),
        predictions={"k1": pred},
        save=lambda: None,
    )
    return project, pred


def _fake_defrate(**kw):
    _Path(kw["tab_file_defrate"]).write_text("cat,defor_dens\n1,0.01\n")


def _fake_validate_writing(value):
    """validate_two_layer stand-in whose three artifacts encode ``value``.

    Lets a test tell one run's files apart from another's byte-for-byte, with
    the same truth/model/period/cell size — i.e. exactly the collision the
    run-scoped layout must survive.
    """

    def fake_validate(**kw):
        _Path(kw["tab_file_pred"]).write_text(f"cell,ndefor_obs_ha\n0,{value}\n")
        _Path(kw["fig_file_pred"]).write_bytes(f"PNG-{value}".encode())
        _Path(kw["indices_file_pred"]).write_text(f"MedAE\n{value}\n")
        return {
            "RMSE": value,
            "wRMSE": value,
            "MedAE": value,
            "R2": 0.9,
            "ncell": 26,
            "csize_coarse_grid": kw["csize_coarse_grid"],
            "csize_coarse_grid_ha": 8100.0,
        }

    return fake_validate


_TRUTH_TAG = "forest_loss_2015_2020"


def _run_against_truth(project, tmp_path, run_id, value, monkeypatch, csizes=(300,)):
    monkeypatch.setattr(ev, "_defrate_per_cat", _fake_defrate)
    monkeypatch.setattr(ev, "validate_two_layer", _fake_validate_writing(value))
    return ev.evaluate_against_truth(
        project,
        prediction_keys=["k1"],
        defor_file=tmp_path / "d.tif",
        forest_file=tmp_path / "f.tif",
        time_interval=5,
        truth_tag=_TRUTH_TAG,
        csizes=csizes,
        run_id=run_id,
    )


def _spec(tmp_path):
    return {
        "defor_file": str(tmp_path / "d.tif"),
        "forest_file": str(tmp_path / "f.tif"),
        "time_interval": 5,
        "truth_tag": _TRUTH_TAG,
    }


def test_two_runs_same_truth_retain_distinct_artifacts(tmp_path, monkeypatch):
    """HEADLINE REGRESSION: a later run must not overwrite an older saved run.

    Two runs share truth, model, period AND cell size — under the old shared
    ``evaluation/<truth_tag>/`` layout the second silently clobbered the first's
    point CSV and PNG, so reopening the older record showed the newer numbers.
    """
    from gui.scripts.evaluation_charts import figure_entries
    from gui.tile.evaluation_helpers import build_evaluation_record

    project, _pred = _truth_project_and_pred(tmp_path)
    runs = [("run00001", 11.0), ("run00002", 22.0)]
    records = []
    for i, (run_id, value) in enumerate(runs):
        df = _run_against_truth(project, tmp_path, run_id, value, monkeypatch)
        records.append(
            build_evaluation_record(
                project,
                df,
                _spec(tmp_path),
                resolved_keys=["k1"],
                run_id=run_id,
                created_at=f"2026-06-22T14:0{i}:00",
                csizes=(300,),
            )
        )

    for record, (run_id, value) in zip(records, runs):
        assert len(record.artifacts) == 1, "one artifact per map per cell size"
        art = record.artifacts[0]
        assert art.prediction_key == "glm_glm_v1__validation"
        assert art.model == "GLM_glm_glm_v1" and art.period == "validation"
        assert art.csize_px == 300
        # each record's own files survived the other run untouched
        assert _Path(art.points_csv).read_text() == f"cell,ndefor_obs_ha\n0,{value}\n"
        assert _Path(art.png_path).read_bytes() == f"PNG-{value}".encode()
        assert run_id in art.points_csv and run_id in art.png_path
        # ...and the path the Figures tab derives resolves to that same file
        fig_dir = _Path(record.csv_path).parent
        entries = figure_entries(record.indices, 300, fig_dir=fig_dir)
        assert [str(p) for _, p in entries] == [art.png_path]
        assert entries[0][1].read_bytes() == f"PNG-{value}".encode()

    assert records[0].artifacts[0].png_path != records[1].artifacts[0].png_path
    assert records[0].csv_path != records[1].csv_path


def test_run_scoped_artifacts_live_under_run_directory(tmp_path, monkeypatch):
    """A run-scoped evaluation keeps every artifact inside its own folder."""
    project, _pred = _truth_project_and_pred(tmp_path)
    _run_against_truth(project, tmp_path, "run00001", 11.0, monkeypatch)

    run_dir = tmp_path / "evaluation" / _TRUTH_TAG / "run00001"
    assert run_dir.is_dir()
    for name in (
        "defrate_cat_GLM_glm_glm_v1_validation.csv",
        "pred_obs_GLM_glm_glm_v1_validation_300.csv",
        "indices_GLM_glm_glm_v1_validation_300.csv",
        "pred_obs_GLM_glm_glm_v1_validation_300.png",
        "indices_all.csv",
    ):
        assert (run_dir / name).exists(), name


def test_run_scoped_evaluation_also_publishes_legacy_shared_paths(
    tmp_path, monkeypatch
):
    """Dual-publish shim: notebooks reading the old shared paths keep working."""
    project, _pred = _truth_project_and_pred(tmp_path)
    _run_against_truth(project, tmp_path, "run00001", 11.0, monkeypatch)
    _run_against_truth(project, tmp_path, "run00002", 22.0, monkeypatch)

    shared = tmp_path / "evaluation" / _TRUTH_TAG
    for name in (
        "defrate_cat_GLM_glm_glm_v1_validation.csv",
        "pred_obs_GLM_glm_glm_v1_validation_300.csv",
        "indices_GLM_glm_glm_v1_validation_300.csv",
        "pred_obs_GLM_glm_glm_v1_validation_300.png",
        "indices_all.csv",
    ):
        assert (shared / name).exists(), name
    # the shared copy tracks the LATEST run
    latest_png = shared / "pred_obs_GLM_glm_glm_v1_validation_300.png"
    assert latest_png.read_bytes() == b"PNG-22.0"
    # while the older run's own copy is untouched
    assert (
        shared.parent / _TRUTH_TAG / "run00001" / latest_png.name
    ).read_bytes() == b"PNG-11.0"


def test_evaluate_against_truth_without_run_id_keeps_legacy_layout(
    tmp_path, monkeypatch
):
    """The notebook path (no run_id) writes exactly where it always did."""
    project, _pred = _truth_project_and_pred(tmp_path)
    df = _run_against_truth(project, tmp_path, None, 11.0, monkeypatch)

    shared = tmp_path / "evaluation" / _TRUTH_TAG
    png = shared / "pred_obs_GLM_glm_glm_v1_validation_300.png"
    assert png.read_bytes() == b"PNG-11.0"
    assert (shared / "indices_all.csv").exists()
    # no run sub-directory was created, and no artifacts are claimed
    assert [p for p in shared.iterdir() if p.is_dir()] == []
    assert df.attrs.get("artifacts", []) == []


def test_evaluate_against_truth_threads_run_id_into_each_prediction(
    tmp_path, monkeypatch
):
    """The run id reaches every per-prediction call, so artifacts stay scoped."""
    seen = {}

    def fake_one(proj, pred, **kw):
        seen.update(kw)
        return []

    project, _pred = _truth_project_and_pred(tmp_path)
    monkeypatch.setattr(ev, "_evaluate_one_against_truth", fake_one)
    ev.evaluate_against_truth(
        project,
        prediction_keys=["k1"],
        defor_file=tmp_path / "d.tif",
        forest_file=tmp_path / "f.tif",
        time_interval=5,
        truth_tag=_TRUTH_TAG,
        run_id="run00001",
    )
    assert seen["run_id"] == "run00001"


def test_one_artifact_per_prediction_per_cell_size(tmp_path, monkeypatch):
    """Each (prediction, cell size) pair records exactly one artifact."""
    project, _pred = _truth_project_and_pred(tmp_path)
    df = _run_against_truth(
        project, tmp_path, "run00001", 11.0, monkeypatch, csizes=(100, 300)
    )
    arts = df.attrs["artifacts"]
    assert sorted(a.csize_px for a in arts) == [100, 300]
    assert {a.model for a in arts} == {"GLM_glm_glm_v1"}
    assert len({a.points_csv for a in arts}) == 2


def test_legacy_record_without_artifacts_still_shows_its_pngs(tmp_path):
    """Acceptance: pre-Task-4 records keep loading AND keep displaying figures.

    Such a record has ``artifacts == []`` and a ``csv_path`` pointing at the
    SHARED evaluation/<truth_tag>/indices_all.csv, so the figure directory is
    the shared folder and the derived-filename branch must still find the PNG.
    """
    from gui.scripts.evaluation_charts import figure_entries
    from spatialrisk.evaluations import EvaluationRecord

    shared = tmp_path / "evaluation" / _TRUTH_TAG
    shared.mkdir(parents=True)
    png = shared / "pred_obs_GLM_validation_300.png"
    png.write_bytes(b"PNG-legacy")

    record = EvaluationRecord(
        truth_tag=_TRUTH_TAG,
        truth_defor="d",
        truth_forest="f",
        time_interval=5,
        prediction_keys=["k1"],
        csizes=[300],
        created_at="2026-06-01T10:00:00",
        indices=[
            {
                "model": "GLM",
                "period": "validation",
                "csize_coarse_grid": 300,
                "MedAE": 1.0,
            }
        ],
        csv_path=str(shared / "indices_all.csv"),
        run_id="legacy00",
    )

    assert record.artifacts == []
    entries = figure_entries(record.indices, 300, fig_dir=_Path(record.csv_path).parent)
    assert [p for _, p in entries] == [png]
    assert entries[0][1].read_bytes() == b"PNG-legacy"


def test_evaluation_tile_threads_run_id_and_orders_delete():
    """The tile passes the run id through and deletes in dependency order."""
    import inspect

    import gui.tile.evaluation_tile as et

    src = inspect.getsource(et)
    # the run id reaches the computation, not just the record builder
    assert "run_id=job_id" in src
    # deletion goes through the helper that commits BEFORE removing files
    assert "delete_evaluation_run" in src


def test_evaluation_tile_does_not_publish_a_failed_deletion():
    """on_delete must return BEFORE project.set when deletion did not commit."""
    import inspect

    import gui.tile.evaluation_tile as et

    src = inspect.getsource(et)
    body = src[src.index("def on_delete") : src.index("def on_dismiss")]
    assert body.index("if not deleted") < body.index("project.set")
    # the message reports a failed deletion, not a completed removal
    assert "could not be deleted" in body


def test_evaluation_results_widget_exports_list_and_dialog():
    """The widget module exposes both the saved-runs list and the dialog."""
    import inspect

    import gui.widget.evaluation_results as er

    assert hasattr(er, "EvaluationResults")
    assert hasattr(er, "EvaluationTableDialog")
    src = inspect.getsource(er)
    # list reads the persisted registry; dialog renders the saved table
    assert "evaluations" in src
    assert "on_open" in src and "on_delete" in src
    assert "solara.DataFrame" in src
    # Regression: the open action must be an explicit clickable action — on_click
    # on ListItem/ListItemContent does NOT fire in this reacton.ipyvuetify setup,
    # so the row uses ProductTable's "open" ActionSpec (a real Button, icon
    # mdi-table-eye) to open the popup, rather than a clickable row/list item.
    assert '"kind": "open"' in src


def test_evaluation_tile_wires_record_and_dialog():
    """The tile hands the selected record to the dialog."""
    import inspect

    import gui.tile.evaluation_tile as et

    src = inspect.getsource(et)
    assert "build_evaluation_record" in src
    assert "add_evaluation" in src
    assert "EvaluationTableDialog" in src
    assert "delete_evaluation" in src
    # background job does the mutate-then-replace re-render
    assert "model_copy()" in src
    # eval_indices transient table is gone
    assert "eval_indices" not in src


# ---------------------------------------------------------------------------
# Chart builders (gui/scripts/evaluation_charts.py)
# ---------------------------------------------------------------------------


def _chart_rows():
    rows = []
    for model, base in [("glm", 1.0), ("rf", 0.7)]:
        for csize in (100, 300):
            rows.append(
                {
                    "prediction": f"{model}__d1",
                    "model": model,
                    "period": "d1",
                    "csize_coarse_grid": csize,
                    "ncell": 40,
                    "MedAE": base * csize / 100,
                    "R2": 0.5,
                    "RMSE": base,
                    "wRMSE": base,
                    "fig_path": f"/tmp/pred_obs_{model}_d1_{csize}.png",
                }
            )
    return rows


# --- metric_bar_option: one serializable ECharts option per metric ---------
#
# ECharts has no subplot concept, so the Charts tab's single multi-subplot
# figure became one independent option dict per metric, laid out two-per-row by
# the widget. The assertions below pin the information design that carried over
# (metrics, order, titles, one bar series per cell size, one category per map).


def test_metric_bar_option_categories_are_the_map_labels():
    """X axis = one category per model/period label, sorted (as Plotly did)."""
    from gui.scripts.evaluation_charts import metric_bar_option

    option = metric_bar_option(_chart_rows(), "MedAE")
    assert option["xAxis"]["type"] == "category"
    assert option["xAxis"]["data"] == ["glm — d1", "rf — d1"]


def test_metric_bar_option_has_one_bar_series_per_cell_size():
    """One bar series per coarse-grid cell size."""
    from gui.scripts.evaluation_charts import metric_bar_option

    option = metric_bar_option(_chart_rows(), "MedAE")
    assert [s["type"] for s in option["series"]] == ["bar", "bar"]
    assert [s["name"] for s in option["series"]] == ["csize 100 px", "csize 300 px"]


def test_metric_bar_option_series_data_is_the_metric_values_per_label():
    """Values follow the category order, one list per cell size."""
    from gui.scripts.evaluation_charts import metric_bar_option

    option = metric_bar_option(_chart_rows(), "MedAE")
    # _chart_rows: MedAE = base * csize / 100, base 1.0 (glm) / 0.7 (rf)
    assert option["series"][0]["data"] == [1.0, 0.7]
    assert option["series"][1]["data"] == [3.0, pytest.approx(2.1)]


def test_metric_bar_option_leaves_a_missing_label_csize_pair_empty():
    """A map evaluated at only one cell size must not shift the other bars."""
    from gui.scripts.evaluation_charts import metric_bar_option

    rows = [
        r
        for r in _chart_rows()
        if not (r["model"] == "rf" and r["csize_coarse_grid"] == 300)
    ]
    option = metric_bar_option(rows, "MedAE")
    assert option["xAxis"]["data"] == ["glm — d1", "rf — d1"]
    assert option["series"][1]["data"] == [3.0, None]


def test_metric_bar_option_title_carries_the_direction_hint():
    """Each title states the unit and whether lower or higher is better."""
    from gui.scripts.evaluation_charts import metric_bar_option

    titles = {
        m: metric_bar_option(_chart_rows(), m)["title"]["text"]
        for m in ("MedAE", "R2", "RMSE", "wRMSE")
    }
    assert titles == {
        "MedAE": "MedAE (ha) ↓",
        "R2": "R² ↑",
        "RMSE": "RMSE (ha) ↓",
        "wRMSE": "wRMSE (ha) ↓",
    }


def test_metric_bar_option_shows_the_legend_only_for_several_cell_sizes():
    """Matches the old showlegend=len(csizes) > 1."""
    from gui.scripts.evaluation_charts import metric_bar_option

    many = metric_bar_option(_chart_rows(), "MedAE")
    assert many["legend"]["show"] is True
    assert many["legend"]["data"] == ["csize 100 px", "csize 300 px"]

    one = metric_bar_option(
        [r for r in _chart_rows() if r["csize_coarse_grid"] == 100], "MedAE"
    )
    assert one["legend"]["show"] is False


def test_metric_bar_option_reserves_the_legend_row_only_when_it_shows():
    """A hidden legend must not leave a blank band above the plot.

    grid.top has to clear the title always, and the legend row (placed at
    top=24) only when there is more than one cell size to name.
    """
    from gui.scripts.evaluation_charts import metric_bar_option

    with_legend = metric_bar_option(_chart_rows(), "MedAE")
    assert with_legend["legend"]["show"] is True
    assert with_legend["grid"]["top"] == 52

    one_csize = metric_bar_option(
        [r for r in _chart_rows() if r["csize_coarse_grid"] == 100], "MedAE"
    )
    assert one_csize["legend"]["show"] is False
    assert one_csize["grid"]["top"] == with_legend["legend"]["top"] == 24


def test_metric_bar_option_colors_bars_from_the_app_accent():
    """Shades of the app's "primary", never a palette of the chart's own."""
    from gui.scripts.echarts_options import accent_color, accent_ramp
    from gui.scripts.evaluation_charts import metric_bar_option

    accent = "#5BB624"
    option = metric_bar_option(_chart_rows(), "MedAE", accent=accent)
    assert [s["itemStyle"]["color"] for s in option["series"]] == accent_ramp(2, accent)

    # One cell size means shading would encode nothing, so the bar is the
    # accent itself — the same colour as every color="primary" control.
    single = metric_bar_option(
        [r for r in _chart_rows() if r["csize_coarse_grid"] == 100],
        "MedAE",
        accent=accent,
    )
    assert single["series"][0]["itemStyle"]["color"] == accent_color(accent)


def test_metric_bar_option_bars_follow_a_changed_accent():
    """Recolouring the app's primary recolours the charts — no frozen hexes."""
    from gui.scripts.evaluation_charts import metric_bar_option

    green = metric_bar_option(_chart_rows(), "MedAE", accent="#5BB624")
    gold = metric_bar_option(_chart_rows(), "MedAE", accent="#76591e")
    assert [s["itemStyle"]["color"] for s in green["series"]] != [
        s["itemStyle"]["color"] for s in gold["series"]
    ]


def test_metric_bar_option_tooltip_shows_label_value_and_cell_size():
    """ECharts template tokens: {b} category, {c} value, {a} series name."""
    from gui.scripts.evaluation_charts import metric_bar_option

    tooltip = metric_bar_option(_chart_rows(), "MedAE")["tooltip"]
    assert tooltip["trigger"] == "item"
    assert tooltip["formatter"] == "{b}<br/>MedAE = {c}<br/>{a}"
    # the series name is what carries "csize N px" into {a}
    assert metric_bar_option(_chart_rows(), "R2")["series"][1]["name"] == "csize 300 px"


def test_metric_bar_option_uses_the_theme_ink_and_grid_colors():
    """themed_option only sets ink; the grid colour must be wired in here."""
    from gui.scripts.echarts_options import theme_colors
    from gui.scripts.evaluation_charts import metric_bar_option

    for dark in (False, True):
        colors = theme_colors(dark)
        option = metric_bar_option(_chart_rows(), "MedAE", dark=dark)
        assert option["title"]["textStyle"]["color"] == colors["ink"]
        assert option["legend"]["textStyle"]["color"] == colors["ink"]
        assert option["xAxis"]["axisLabel"]["color"] == colors["ink"]
        assert option["yAxis"]["axisLabel"]["color"] == colors["ink"]
        assert option["xAxis"]["axisLine"]["lineStyle"]["color"] == colors["grid"]
        assert option["yAxis"]["splitLine"]["lineStyle"]["color"] == colors["grid"]


def test_metric_bar_option_keeps_the_x_axis_gridlines_off():
    """Plotly parity: update_xaxes(showgrid=False)."""
    from gui.scripts.evaluation_charts import metric_bar_option

    option = metric_bar_option(_chart_rows(), "MedAE")
    assert option["xAxis"]["splitLine"]["show"] is False


def test_metric_bar_option_is_json_serializable():
    """EChartsRawWidget hands the dict straight to the frontend."""
    import json

    from gui.scripts.evaluation_charts import metric_bar_option

    json.dumps(metric_bar_option(_chart_rows(), "MedAE"))


def test_metric_bar_option_returns_none_when_nothing_is_chartable():
    """Nothing to draw yields None, so the caller can drop the chart."""
    from gui.scripts.evaluation_charts import metric_bar_option

    assert metric_bar_option([], "RMSE") is None
    # a metric no row carries has nothing to draw
    rows = [{k: v for k, v in r.items() if k != "wRMSE"} for r in _chart_rows()]
    assert metric_bar_option(rows, "wRMSE") is None
    # an unknown metric key has no title and no data
    assert metric_bar_option(_chart_rows(), "nope") is None
    # rows without a cell size cannot be split into series
    assert (
        metric_bar_option([{"model": "glm", "period": "d1", "MedAE": 1.0}], "MedAE")
        is None
    )


def test_record_metrics_keeps_the_canonical_order_and_drops_empties():
    """Unchanged behaviour — the widget builds one option per returned metric."""
    from gui.scripts.evaluation_charts import record_metrics

    rows = _chart_rows()
    assert record_metrics(rows, []) == ["MedAE", "R2", "RMSE", "wRMSE"]
    assert record_metrics(rows, ["R2", "MedAE"]) == ["R2", "MedAE"]
    assert record_metrics(rows, ["R2", "bogus"]) == ["R2"]
    stripped = [{k: v for k, v in r.items() if k != "R2"} for r in rows]
    assert record_metrics(stripped, ["R2", "MedAE"]) == ["MedAE"]


# --- widget identity --------------------------------------------------------
#
# There is no chart_identity helper any more: EChartsChart digests the option
# it is handed, so nothing the chart DRAWS has to be re-listed by a caller (see
# tests/test_echarts_adapter.py). What the option cannot show — the active tab,
# which run is on screen — is asserted on the rendered tab further down.


def test_figure_entries_and_csizes():
    """Figure entries and cell sizes come back sorted and paired."""
    from gui.scripts.evaluation_charts import figure_entries, record_csizes

    rows = _chart_rows()
    assert record_csizes(rows) == [100, 300]
    entries = figure_entries(rows, 300)
    assert [label for label, _ in entries] == ["glm — d1", "rf — d1"]
    assert str(entries[0][1]).endswith("pred_obs_glm_d1_300.png")


def test_figure_entries_derives_paths_without_fig_path_column():
    """A record with no fig_path column still yields figure entries.

    Real records store indices WITHOUT it (evaluate_against_truth's explicit
    column list drops the field), so the paths must be derived from the record's
    evaluation folder instead of coming up empty.
    """
    from pathlib import Path

    from gui.scripts.evaluation_charts import figure_entries

    rows = [{k: v for k, v in r.items() if k != "fig_path"} for r in _chart_rows()]
    fig_dir = Path("/data/proj/evaluation/loss_2010")
    entries = figure_entries(rows, 300, fig_dir=fig_dir)
    assert [label for label, _ in entries] == ["glm — d1", "rf — d1"]
    assert entries[0][1] == fig_dir / "pred_obs_glm_d1_300.png"
    # without a fig_dir there is nothing to derive from
    assert figure_entries(rows, 300) == []


def test_evaluation_dialog_has_tabs_and_csize_select():
    """The dialog carries its three tabs and the cell-size selector."""
    import inspect

    import gui.widget.evaluation_results as er

    src = inspect.getsource(er)
    assert "metric_bar_option" in src and "EChartsChart" in src
    assert "figure_entries" in src and "csize_select_label" in src
    assert "rv.Tabs" in src and "rv.TabsItems" in src


def test_evaluation_dialog_drops_the_plotly_modebar_workaround():
    """The modebar rule was a FigurePlotly artefact; the table width rule stays.

    Asserted on the CSS the dialog actually RENDERS (``solara.Style`` emits a
    ``VuetifyTemplate`` carrying its stylesheet inline), not on the module's
    source text: a stale ``.modebar { display: none }`` would be dead weight
    shipped to every browser, and only the rendered output can show that it is
    gone while the table-width rule the dialog still needs survives.
    """
    import ipyvuetify as vw
    import reacton
    import solara

    from gui.i18n import t as _t

    _t("common.close")  # warm the translator before the first render
    from gui.widget.evaluation_results import EvaluationTableDialog

    p = types.SimpleNamespace(evaluations={"run-a": _chart_record()})
    project = solara.reactive(p, equals=lambda a, b: a is b)
    _, rc = reacton.render(
        EvaluationTableDialog(
            project=project, eval_key="run-a", on_close=lambda *_: None
        ),
        handle_error=False,
    )
    css = "\n".join(
        w.template
        for w in rc.find(vw.VuetifyTemplate).widgets
        if isinstance(getattr(w, "template", None), str)
    )
    assert "modebar" not in css
    assert ".evaluation-table-dialog" in css and "width: 100%" in css
    rc.close()


def _run_blocked(blocked, body):
    """Run ``body`` in a subprocess where importing ``blocked`` raises."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        f"BLOCKED = {tuple(blocked)!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCKED:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n" + body
    )
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )


_CHART_SMOKE = (
    "rows = [{'model': m, 'period': 'd1', 'csize_coarse_grid': c,\n"
    "         'MedAE': 1.0, 'R2': 0.5, 'RMSE': 1.0, 'wRMSE': 1.0}\n"
    "        for m in ('glm', 'rf') for c in (100, 300)]\n"
    "opt = ec.metric_bar_option(rows, 'MedAE')\n"
    "assert opt['series'][0]['itemStyle']['color'].startswith('#')\n"
    "assert opt['title']['text'] and opt['legend']['show'] is True\n"
)

# The scatter half of the same layering rule. Builds from an in-memory
# PredObsPlotData rather than a file so the smoke needs no fixture on disk —
# the point is that the option BUILDS, which is where a lazy import would bite.
_SCATTER_SMOKE = (
    "import pandas as pd\n"
    "import gui.scripts.evaluation_echarts as ee\n"
    "from spatialrisk.evaluation import PredObsPlotData\n"
    "pts = pd.DataFrame({'cell': [0, 1], 'nfor_obs_ha': [9.0, 8.0],\n"
    "                    'ndefor_obs_ha': [1.0, 2.0],\n"
    "                    'ndefor_pred_ha': [1.5, 2.5]})\n"
    "pd_ = PredObsPlotData(model='glm', period='d1', csize_px=300,\n"
    "                      csize_ha=90.0, points=pts, axis_min=1.0,\n"
    "                      axis_max=2.5, medae=0.5, r2=0.9, ncell=2)\n"
    "sopt = ee.pred_obs_scatter_option(pd_)\n"
    "assert [v[:2] for v in sopt['series'][0]['data']] == [[1.0, 1.5], [2.0, 2.5]]\n"
    "assert sopt['xAxis']['min'] == sopt['yAxis']['min'] == 1.0\n"
    "assert ee.pred_obs_renderer(pd_) == 'svg'\n"
)


def test_evaluation_gui_path_imports_no_plotly():
    """No module the Evaluation tile pulls in may still reach for plotly.

    Blocks 'plotly' at import time in a subprocess and then actually BUILDS a
    chart option: the old code imported plotly lazily inside the figure
    builder, so importing the modules alone would not have caught it. The
    palette call is the one that used to land in plotly.colors.
    """
    proc = _run_blocked(
        ["plotly"],
        "import gui.scripts.evaluation_charts as ec\n"
        "import gui.widget.evaluation_results  # noqa: F401\n"
        "import gui.tile.evaluation_tile  # noqa: F401\n"
        + _CHART_SMOKE
        + _SCATTER_SMOKE
        + "assert 'plotly' not in sys.modules\n"
        "print('OK')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_evaluation_charts_builds_options_without_solara():
    """gui/scripts/* is solara-free by the app's layering rule.

    The chart builders moved from plotly to ECharts but must still import (and
    run) with solara, ipyvuetify and ipecharts all blocked — the widget half of
    the adapter is the only module allowed to know ipecharts exists. Covers
    BOTH builders: the metric bars and the predicted-vs-observed scatter.
    """
    proc = _run_blocked(
        ["solara", "reacton", "ipyvuetify", "ipecharts", "plotly"],
        "import gui.scripts.evaluation_charts as ec\n"
        + _CHART_SMOKE
        + _SCATTER_SMOKE
        + "print('OK')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# Charts tab — headless render (the evaluation dialog is not covered by any
# other render test, and EChartsChart's identity contract can only fail here)
# ---------------------------------------------------------------------------


def _chart_record(metrics=("MedAE", "R2"), rows=None):
    return types.SimpleNamespace(
        indices=list(_chart_rows() if rows is None else rows),
        metrics=list(metrics),
        csv_path="/data/proj/evaluation/loss_2010/indices_all.csv",
        truth_tag="loss_2010",
    )


@contextlib.contextmanager
def _dark_theme():
    """Put the SESSION theme state in dark mode, and restore after.

    The charts follow pysepal's resolved ``ThemeState.dark`` (via
    ``use_theme_dark``), not solara's internal theme traitlet: the app runs
    under ``@with_sepal_sessions`` and MapApp's ThemeToggle drives the session
    state. Headless, ``get_current_theme_state()`` hands back the module-level
    fallback state — the same object the hook observes.
    """
    from pysepal.solara import get_current_theme_state

    state = get_current_theme_state()
    before = (state.mode, state.dark)
    try:
        state.set_mode("dark")
        yield state
    finally:
        state.mode, state.dark = before


def _render_charts_tab(**kwargs):
    import ipecharts
    import reacton

    from gui.i18n import t as _t

    _t("common.close")  # warm the translator before the first render
    from gui.widget.evaluation_results import _ChartsTab

    kwargs.setdefault("record", _chart_record())
    kwargs.setdefault("eval_key", "run-a")
    kwargs.setdefault("active_tab", 1)
    _, rc = reacton.render(_ChartsTab(**kwargs), handle_error=False)
    return rc, ipecharts.EChartsRawWidget


def test_charts_tab_renders_one_chart_per_selected_metric():
    """One chart per metric the run selected."""
    rc, cls = _render_charts_tab()
    assert len(rc.find(cls).widgets) == 2


def test_charts_tab_renders_all_four_metrics_when_none_were_selected():
    """No stored selection means every known metric is charted."""
    rc, cls = _render_charts_tab(record=_chart_record(metrics=()))
    assert len(rc.find(cls).widgets) == 4


def test_charts_tab_charts_carry_the_per_metric_options():
    """Each chart gets its own metric's option, not a shared one."""
    rc, cls = _render_charts_tab()
    titles = [w.option["title"]["text"] for w in rc.find(cls).widgets]
    assert titles == ["MedAE (ha) ↓", "R² ↑"]


def test_charts_tab_builds_its_options_from_the_live_theme():
    """The dark theme must reach metric_bar_option, not just the widget.

    themed_option (applied inside the adapter) only sets the top-level ink, so
    a `dark=` that never reaches the builder leaves the title, the legend, both
    axis labels and both grid colours on their light values over a dark
    surface. Asserted on title.textStyle.color specifically: the option's
    top-level textStyle would still be dark via the adapter and would hide the
    bug.
    """
    from gui.scripts.echarts_options import theme_colors

    with _dark_theme():
        rc, cls = _render_charts_tab()
        option = rc.find(cls).widgets[0].option
        assert option["title"]["textStyle"]["color"] == theme_colors(True)["ink"]
        assert option["xAxis"]["axisLabel"]["color"] == theme_colors(True)["ink"]
        assert (
            option["yAxis"]["splitLine"]["lineStyle"]["color"]
            == theme_colors(True)["grid"]
        )
        rc.close()

    rc, cls = _render_charts_tab()
    assert (
        rc.find(cls).widgets[0].option["title"]["textStyle"]["color"]
        == theme_colors(False)["ink"]
    )
    rc.close()


def test_charts_tab_repaints_when_the_theme_is_toggled_in_place():
    """A LIVE theme toggle must reach charts that are already on screen.

    The test above renders twice from scratch, so it can only prove the theme
    is read at build time — a fresh render reads the current theme whether or
    not anything subscribed to it. This one flips the theme on a mounted render
    context and touches nothing else, which is what a user does.

    That distinction is the whole bug. ``solara.lab.theme`` is an ipyvuetify
    *traitlet* behind a ``Proxy``, not a ``Reactive``: reading ``theme.dark`` in
    a render body sets up no subscription, so a toggle re-renders nothing, and
    reacton's prop-equality bailout blocks the component even when its parent
    does re-render. ``use_theme_dark()`` observes ``ThemeState.dark`` instead —
    pysepal's session-scoped state that MapApp's ThemeToggle actually drives —
    so the flip itself schedules the re-render.
    """
    from gui.scripts.echarts_options import theme_colors

    rc, cls = _render_charts_tab()
    assert (
        rc.find(cls).widgets[0].option["title"]["textStyle"]["color"]
        == theme_colors(False)["ink"]
    )
    with _dark_theme():
        assert (
            rc.find(cls).widgets[0].option["title"]["textStyle"]["color"]
            == theme_colors(True)["ink"]
        )
    rc.close()


def test_charts_tab_follows_an_auto_mode_theme_resolution():
    """A mounted chart follows an auto-mode theme resolution, both directions.

    In auto mode the frontend resolves prefers-color-scheme and pysepal writes
    the RESOLVED value onto ``ThemeState.dark``; the chart must follow that
    write going dark->light as well as light->dark.
    """
    from pysepal.solara import get_current_theme_state

    from gui.scripts.echarts_options import theme_colors

    state = get_current_theme_state()
    before = (state.mode, state.dark)
    try:
        state.set_mode("auto")
        state.set_dark(False)
        rc, cls = _render_charts_tab()
        assert (
            rc.find(cls).widgets[0].option["title"]["textStyle"]["color"]
            == theme_colors(False)["ink"]
        )
        state.set_dark(True)  # auto-mode resolution flips to dark
        assert (
            rc.find(cls).widgets[0].option["title"]["textStyle"]["color"]
            == theme_colors(True)["ink"]
        )
        state.set_dark(False)  # and back — dark -> light
        assert (
            rc.find(cls).widgets[0].option["title"]["textStyle"]["color"]
            == theme_colors(False)["ink"]
        )
        rc.close()
    finally:
        state.mode, state.dark = before


def test_charts_tab_draws_the_bar_charts_with_the_svg_renderer():
    """Renderer is a deliberate per-call-site choice, not a default to drift.

    Small bar charts: SVG (crisp text, tiny DOM). Canvas here would be a silent
    performance/quality change, which is what resolve_renderer exists to stop.
    """
    rc, cls = _render_charts_tab()
    assert {w.renderer for w in rc.find(cls).widgets} == {"svg"}


def test_charts_tab_lays_metrics_out_in_two_columns():
    """Several metrics lay out two per row."""
    import ipyvuetify as vw

    rc, _ = _render_charts_tab()
    grids = [
        w
        for w in rc.find(vw.Html).widgets
        if "grid-template-columns" in (w.style_ or "")
    ]
    assert grids and "repeat(2," in grids[0].style_


def test_charts_tab_uses_a_single_column_for_a_single_metric():
    """A lone metric gets the full width instead of half of it."""
    import ipyvuetify as vw

    rc, _ = _render_charts_tab(record=_chart_record(metrics=("R2",)))
    grids = [
        w
        for w in rc.find(vw.Html).widgets
        if "grid-template-columns" in (w.style_ or "")
    ]
    assert grids and "repeat(1," in grids[0].style_


def test_charts_tab_swaps_the_chart_when_the_metric_selection_changes():
    """The identity must carry the metric: same run, different selection.

    Position 0 held MedAE; after the re-render it must hold R2. If the metric
    were missing from the identity, use_memo would hand back the stale MedAE
    widget with no error at all.
    """
    from gui.widget.evaluation_results import _ChartsTab

    rc, cls = _render_charts_tab()
    assert rc.find(cls).widgets[0].option["title"]["text"] == "MedAE (ha) ↓"
    rc.render(
        _ChartsTab(
            record=_chart_record(metrics=("R2",)), eval_key="run-a", active_tab=1
        )
    )
    assert rc.find(cls).widgets[0].option["title"]["text"] == "R² ↑"


def test_charts_tab_rebuilds_the_chart_when_the_charted_values_change():
    """Same run key, different index rows — the option must reach the widget."""
    from gui.widget.evaluation_results import _ChartsTab

    rc, cls = _render_charts_tab()
    first = rc.find(cls).widgets[0]
    bumped = [{**r, "MedAE": r["MedAE"] + 5} for r in _chart_rows()]
    rc.render(
        _ChartsTab(record=_chart_record(rows=bumped), eval_key="run-a", active_tab=1)
    )
    second = rc.find(cls).widgets[0]
    assert second is not first
    assert second.option["series"][0]["data"] == [6.0, 5.7]


def test_charts_tab_keeps_its_charts_across_a_tab_switch():
    """The tab index must NOT be in the chart identity.

    Rebuilding on every tab switch is what squished the charts: the leaving
    tab's fresh widgets re-attach inside a ``display:none`` v-window-item,
    ipecharts measures width 0 (it sizes on attach only) and draws its 100px
    fallback — permanently, since there is no ResizeObserver. The widget must
    survive the switch; re-measuring on re-entry is the adapter's
    ``visible``-nudge job, not a teardown's.
    """
    from gui.widget.evaluation_results import _ChartsTab

    rc, cls = _render_charts_tab(active_tab=1)
    first = rc.find(cls).widgets[0]
    rc.render(_ChartsTab(record=_chart_record(), eval_key="run-a", active_tab=2))
    assert rc.find(cls).widgets[0] is first
    rc.render(_ChartsTab(record=_chart_record(), eval_key="run-a", active_tab=1))
    assert rc.find(cls).widgets[0] is first
    rc.close()


def test_charts_tab_tells_the_adapter_whether_its_tab_is_shown(monkeypatch):
    """``visible`` must track the active tab — it drives the resize nudge."""
    import gui.widget.evaluation_results as er
    from gui.widget.evaluation_results import _ChartsTab

    seen = []
    real = er.EChartsChart

    def spy(*args, **kwargs):
        seen.append(kwargs.get("visible"))
        return real(*args, **kwargs)

    monkeypatch.setattr(er, "EChartsChart", spy)
    rc, _ = _render_charts_tab(active_tab=1)
    assert seen and set(seen) == {True}

    seen.clear()
    rc.render(_ChartsTab(record=_chart_record(), eval_key="run-a", active_tab=2))
    assert seen and set(seen) == {False}
    rc.close()


def test_charts_tab_says_so_when_there_is_nothing_to_chart():
    """An empty run shows a message rather than a blank grid."""
    import ipecharts

    rc, _ = _render_charts_tab(record=_chart_record(rows=[]))
    assert rc.find(ipecharts.EChartsRawWidget).widgets == []


def test_evaluation_table_dialog_mounts_with_its_charts():
    """Mount smoke for the whole dialog — nothing else renders it.

    Catches a prop mismatch between the dialog and _ChartsTab (the tab index is
    threaded through now), which no source-substring test can see. `eager=True`
    on the dialog means the tab bodies are built even though nothing is
    visible headlessly.
    """
    import ipecharts
    import reacton
    import solara

    from gui.i18n import t as _t

    _t("common.close")  # warm the translator before the first render
    from gui.widget.evaluation_results import EvaluationTableDialog

    p = types.SimpleNamespace(evaluations={"run-a": _chart_record()})
    project = solara.reactive(p, equals=lambda a, b: a is b)
    _, rc = reacton.render(
        EvaluationTableDialog(
            project=project, eval_key="run-a", on_close=lambda *_: None
        ),
        handle_error=False,
    )
    assert len(rc.find(ipecharts.EChartsRawWidget).widgets) == 2
    rc.close()


# ---------------------------------------------------------------------------
# Figures tab — interactive predicted-vs-observed scatter (headless render)
#
# The image-only tab is now an explorable scatter, one card per map, that keeps
# the PNG reachable. These render tests cover the fallback ladder (missing CSV /
# legacy PNG-only / neither artifact), cell-size switching, lazy loading and
# theme, none of which a source-substring test can see.
# ---------------------------------------------------------------------------

_FIG_MODEL = "GLM"
_FIG_PERIOD = "validation"


def _write_points(path, obs, pred):
    """Write a real 6-column point CSV (the scatter loader reads 4 of them)."""
    import pandas as pd

    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(obs)
    pd.DataFrame(
        {
            "cell": list(range(n)),
            "nfor_obs": [10] * n,
            "ndefor_obs": [1] * n,
            "nfor_obs_ha": [9.0] * n,
            "ndefor_obs_ha": list(obs),
            "ndefor_pred_ha": list(pred),
        }
    ).to_csv(path, index=False)


def _fig_index_row(
    csize=300,
    ha=90.0,
    medae=0.5,
    r2=0.9,
    model=_FIG_MODEL,
    period=None,
    prediction=None,
):
    return {
        "model": model,
        "period": period or _FIG_PERIOD,
        "csize_coarse_grid": csize,
        "csize_coarse_grid_ha": ha,
        "MedAE": medae,
        "R2": r2,
        "prediction": prediction or f"{model.lower()}__validation",
    }


def _figures_record(*, indices, csv_path, artifacts=()):
    return types.SimpleNamespace(
        indices=list(indices),
        metrics=[],
        csv_path=str(csv_path),
        truth_tag="loss_2010",
        run_id="run00001",
        artifacts=list(artifacts),
    )


def _fig_artifact(points_csv, png_path, csize=300, model=_FIG_MODEL):
    return types.SimpleNamespace(
        prediction_key=f"{model.lower()}__validation",
        model=model,
        period=_FIG_PERIOD,
        csize_px=csize,
        points_csv=str(points_csv),
        png_path=str(png_path),
    )


def _render_figures_tab(**kwargs):
    import ipecharts
    import reacton

    from gui.i18n import t as _t

    _t("common.close")  # warm the translator before the first render
    from gui.widget.evaluation_results import _FiguresTab

    kwargs.setdefault("eval_key", "run-a")
    kwargs.setdefault("active_tab", 2)  # predicted-vs-observed is the third tab
    _, rc = reacton.render(_FiguresTab(**kwargs), handle_error=False)
    return rc, ipecharts.EChartsRawWidget


def _scatter_data(widget):
    """The [obs, pred] pairs the scatter series carries (drops the ride-alongs)."""
    return [v[:2] for v in widget.option["series"][0]["data"]]


def test_two_predictions_with_identical_labels_render_distinct_cards(tmp_path):
    """Two same-label predictions get one card each, on their own artifacts.

    ``rows_by_label`` used to collapse them into a single card. Every index row
    must now get its own, wired to its OWN artifact — including the on-chart
    MedAE annotation, not just the plotted points.

    The two index rows are given DIFFERENT MedAE values so a card that quotes
    the wrong row's stats (``_index_row_for`` matching on (model, period,
    csize) alone, ignoring which prediction the card is actually drawing) is
    caught even when both predictions' points render correctly.
    """
    csv_a = tmp_path / "a" / "pred_obs_GLM_validation_300.csv"
    csv_b = tmp_path / "b" / "pred_obs_GLM_validation_300.csv"
    _write_points(csv_a, obs=[1.0, 2.0], pred=[1.0, 2.0])
    _write_points(csv_b, obs=[1.0, 2.0, 3.0], pred=[1.0, 2.0, 3.0])
    arts = [
        _fig_artifact(points_csv=csv_a, png_path=csv_a.with_suffix(".png")),
        types.SimpleNamespace(
            prediction_key="glm__validation__2",
            model=_FIG_MODEL,
            period=_FIG_PERIOD,
            csize_px=300,
            points_csv=str(csv_b),
            png_path=str(csv_b.with_suffix(".png")),
        ),
    ]
    rows = [
        _fig_index_row(medae=1.5),
        _fig_index_row(prediction="glm__validation__2", medae=9.5),
    ]
    record = _figures_record(
        indices=rows, csv_path=tmp_path / "indices_all.csv", artifacts=arts
    )
    rc, cls = _render_figures_tab(record=record)
    widgets = rc.find(cls).widgets
    assert len(widgets) == 2
    by_point_count = {
        len(next(s for s in w.option["series"] if s["type"] == "scatter")["data"]): w
        for w in widgets
    }
    assert sorted(by_point_count) == [2, 3]  # each card drew ITS prediction's file

    def annotation_text(widget):
        return widget.option["graphic"][0]["style"]["text"]

    # csv_a's card (2 points, row medae=1.5) and csv_b's card (3 points, row
    # medae=9.5) must each quote THEIR OWN row's MedAE, not the sibling's.
    assert "MedAE = 1.50" in annotation_text(by_point_count[2])
    assert "MedAE = 9.50" in annotation_text(by_point_count[3])
    rc.close()


def test_the_figures_tab_shows_the_typed_png_even_outside_the_derived_dir(tmp_path):
    """png_path resolution goes through the typed artifact, not derivation."""
    import ipywidgets

    png = tmp_path / "elsewhere" / "archived.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    art = _fig_artifact(points_csv=tmp_path / "gone.csv", png_path=png)
    record = _figures_record(
        indices=[_fig_index_row()],
        csv_path=tmp_path / "indices_all.csv",
        artifacts=[art],
    )
    rc, cls = _render_figures_tab(record=record)
    assert rc.find(ipywidgets.Image).widgets  # the typed PNG shows
    rc.close()


def test_figures_tab_renders_the_interactive_scatter(tmp_path):
    """The primary path: a readable point CSV becomes an ECharts scatter."""
    from gui.i18n import t as _t
    from gui.scripts.evaluation_echarts import PRED_OBS_SQUARE_HEIGHT

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv",
        obs=[1.0, 2.0, 3.0],
        pred=[1.5, 2.5, 2.0],
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    rc, cls = _render_figures_tab(record=record)
    widgets = rc.find(cls).widgets
    assert len(widgets) == 1
    assert _scatter_data(widgets[0]) == [[1.0, 1.5], [2.0, 2.5], [3.0, 2.0]]
    # a small scatter draws with SVG, at the card's fixed pixel height
    assert widgets[0].renderer == "svg"
    assert widgets[0].height == PRED_OBS_SQUARE_HEIGHT
    # the translated axis label reached the option, not the PNG's English string
    assert widgets[0].option["xAxis"]["name"] == _t(
        "widgets.evaluation_results.chart_x_axis"
    )
    rc.close()


def test_figures_tab_titles_the_chart_with_the_cell_size_in_hectares(tmp_path):
    """The PNG states the cell size in ha; the interactive twin must not drop it.

    The card's own header names the map and the selector is labelled in PIXELS,
    so without this title the hectare figure the archived image carries appears
    nowhere in the interactive tab. It comes from the record's stored
    ``csize_coarse_grid_ha`` (never recomputed) and is translated — the PNG's
    own English sentence (``PredObsPlotData.title``) is deliberately not reused.
    """
    from gui.i18n import t as _t

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row(ha=90.0)], csv_path=tmp_path / "indices_all.csv"
    )
    rc, cls = _render_figures_tab(record=record)
    title = rc.find(cls).widgets[0].option["title"]["text"]
    assert "90.0" in title
    assert title == _t("widgets.evaluation_results.chart_csize_title", csize_ha=90.0)
    rc.close()


def test_figures_tab_moves_the_chart_title_when_the_language_changes(tmp_path):
    """The title is option TEXT, so the digest must carry it too.

    ``pred_obs_chart_identity`` is handed to the adapter as ``option_digest``,
    i.e. INSTEAD of hashing the option. Pass the title to
    ``pred_obs_scatter_option`` but not to the identity and a language switch
    leaves the previous language's title on screen with no error anywhere —
    the same coupling ``labels=`` has.
    """
    from pysepal.translator import Translator

    from gui import i18n
    from gui.widget.evaluation_results import _FiguresTab

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row(ha=90.0)], csv_path=tmp_path / "indices_all.csv"
    )
    before = i18n._translator.value
    try:
        i18n._translator.value = Translator(i18n.MESSAGES_DIR, target="en")
        rc, cls = _render_figures_tab(record=record)
        english = rc.find(cls).widgets[0].option["title"]["text"]

        i18n._translator.value = Translator(i18n.MESSAGES_DIR, target="es-ES")
        rc.render(_FiguresTab(record=record, eval_key="run-a", active_tab=2))
        spanish = rc.find(cls).widgets[0].option["title"]["text"]
        rc.close()
    finally:
        i18n._translator.value = before

    assert english == "Grid cells of 90.0 ha"
    assert spanish != english
    assert "90.0" in spanish


def test_figures_tab_switches_the_cell_size(tmp_path):
    """The cell-size selector shows for >1 size and reloads that size's points."""
    import ipyvuetify as vw

    _write_points(tmp_path / "pred_obs_GLM_validation_100.csv", obs=[1.0], pred=[1.1])
    _write_points(tmp_path / "pred_obs_GLM_validation_300.csv", obs=[2.0], pred=[2.9])
    record = _figures_record(
        indices=[_fig_index_row(csize=100), _fig_index_row(csize=300)],
        csv_path=tmp_path / "indices_all.csv",
    )
    rc, cls = _render_figures_tab(record=record)
    sel = rc.find(vw.Select).widgets
    assert len(sel) == 1
    assert _scatter_data(rc.find(cls).widgets[0]) == [[1.0, 1.1]]  # first size
    sel[0].v_model = 300  # user picks the coarser grid
    assert _scatter_data(rc.find(cls).widgets[0]) == [[2.0, 2.9]]
    rc.close()


def test_figures_tab_hides_the_selector_for_a_single_cell_size(tmp_path):
    """One cell size means the selector offers no choice, so it is hidden."""
    import ipyvuetify as vw

    _write_points(tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0], pred=[1.0])
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    rc, _ = _render_figures_tab(record=record)
    assert rc.find(vw.Select).widgets == []
    rc.close()


def test_figures_tab_missing_csv_falls_back_to_png_with_a_warning(tmp_path):
    """(a) A NEW record whose point CSV vanished: show the PNG AND warn."""
    import ipyvuetify as vw
    import ipywidgets

    png = tmp_path / "pred_obs_GLM_validation_300.png"
    png.write_bytes(b"PNG-typed")
    missing_csv = tmp_path / "pred_obs_GLM_validation_300.csv"  # never written
    record = _figures_record(
        indices=[_fig_index_row()],
        csv_path=tmp_path / "indices_all.csv",
        artifacts=[_fig_artifact(missing_csv, png)],
    )
    rc, cls = _render_figures_tab(record=record)
    assert rc.find(cls).widgets == []  # no interactive chart
    assert rc.find(ipywidgets.Image).widgets  # the saved PNG shows
    warnings = [a for a in rc.find(vw.Alert).widgets if a.type == "warning"]
    assert warnings  # non-fatal warning
    rc.close()


def test_figures_tab_legacy_png_only_is_not_treated_as_broken(tmp_path):
    """(b) A LEGACY record (no artifacts, no CSV) shows its PNG, no warning."""
    import ipyvuetify as vw
    import ipywidgets

    png = tmp_path / "pred_obs_GLM_validation_300.png"
    png.write_bytes(b"PNG-legacy")
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv", artifacts=[]
    )
    rc, cls = _render_figures_tab(record=record)
    assert rc.find(cls).widgets == []
    assert rc.find(ipywidgets.Image).widgets  # PNG shows
    warnings = [a for a in rc.find(vw.Alert).widgets if a.type == "warning"]
    assert warnings == []  # legacy: no warning
    rc.close()


def test_figures_tab_missing_both_artifacts_shows_the_resolved_path(tmp_path):
    """(c) Neither CSV nor PNG: the missing-figure message names the path."""
    import ipyvuetify as vw
    import ipywidgets

    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv", artifacts=[]
    )
    rc, cls = _render_figures_tab(record=record)
    assert rc.find(cls).widgets == []
    assert rc.find(ipywidgets.Image).widgets == []
    infos = [a for a in rc.find(vw.Alert).widgets if a.type == "info"]
    text = " ".join(str(c) for a in infos for c in (a.children or []))
    # The RESOLVED path, not just the file name: the message exists so a user
    # can go and look for the file, and a bare basename does not say where.
    assert str(tmp_path / "pred_obs_GLM_validation_300.png") in text
    rc.close()


def test_figures_tab_unreadable_csv_falls_back_to_png_with_a_warning(tmp_path):
    """(a) again, for a point CSV that EXISTS but cannot be parsed.

    The loader handles two different failures on the same rung — no file, and a
    file it cannot read (truncated, half-written, columns renamed upstream) —
    and only the first was covered. Both must land on the PNG, not on a
    traceback.
    """
    import ipyvuetify as vw
    import ipywidgets

    png = tmp_path / "pred_obs_GLM_validation_300.png"
    png.write_bytes(b"PNG-typed")
    csv = tmp_path / "pred_obs_GLM_validation_300.csv"
    csv.write_text("cell,nfor_obs_ha\n1,9.0\n")  # the plotted columns are gone
    record = _figures_record(
        indices=[_fig_index_row()],
        csv_path=tmp_path / "indices_all.csv",
        artifacts=[_fig_artifact(csv, png)],
    )
    rc, cls = _render_figures_tab(record=record)
    assert rc.find(cls).widgets == []  # no interactive chart
    assert rc.find(ipywidgets.Image).widgets  # the saved PNG shows
    warnings = [a for a in rc.find(vw.Alert).widgets if a.type == "warning"]
    assert warnings  # non-fatal warning
    rc.close()


def test_figures_tab_unplottable_points_fall_back_without_taking_the_tab_down(tmp_path):
    """A loadable table with NO finite rows must degrade like any missing one.

    ``finite_points`` drops non-finite rows, so an all-NaN point CSV loads fine
    and then ``pred_obs_scatter_option`` returns None — its documented "nothing
    to draw". Handing that None to the chart adapter raises
    ``TypeError: 'NoneType' object is not iterable`` out of this render, which
    takes the SIBLING map's chart and both cards' PNGs with it. The poisoned
    card must fall through to its own PNG and leave the tab standing.
    """
    import ipywidgets

    nan = float("nan")
    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv",
        obs=[nan, nan, nan],
        pred=[nan, nan, nan],
    )
    (tmp_path / "pred_obs_GLM_validation_300.png").write_bytes(b"PNG-poisoned")
    _write_points(
        tmp_path / "pred_obs_RF_validation_300.csv", obs=[1.0, 2.0], pred=[1.5, 2.5]
    )
    (tmp_path / "pred_obs_RF_validation_300.png").write_bytes(b"PNG-good")
    record = _figures_record(
        indices=[_fig_index_row(), _fig_index_row(model="RF")],
        csv_path=tmp_path / "indices_all.csv",
    )

    rc, cls = _render_figures_tab(record=record)
    charts = rc.find(cls).widgets
    assert len(charts) == 1  # the sibling still renders
    assert _scatter_data(charts[0]) == [[1.0, 1.5], [2.0, 2.5]]
    assert len(rc.find(ipywidgets.Image).widgets) == 1  # the poisoned card's PNG
    rc.close()


def test_a_failing_option_builder_spares_the_sibling_cards(tmp_path, monkeypatch):
    """One card's option-builder failure must not raise out of the render.

    That card falls back — here to its missing-figure message — and the healthy
    sibling keeps its interactive chart.
    """
    import gui.widget.evaluation_results as er

    for period in ("validation", "calibration"):
        _write_points(
            tmp_path / f"pred_obs_GLM_{period}_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
        )
    rows = [_fig_index_row(), _fig_index_row(period="calibration")]
    record = _figures_record(indices=rows, csv_path=tmp_path / "indices_all.csv")

    real = er.pred_obs_scatter_option

    def boom(plot_data, **kwargs):
        if plot_data.period == "calibration":
            raise ValueError("synthetic builder failure")
        return real(plot_data, **kwargs)

    monkeypatch.setattr(er, "pred_obs_scatter_option", boom)
    rc, cls = _render_figures_tab(record=record)
    assert len(rc.find(cls).widgets) == 1  # the healthy sibling still renders
    rc.close()


def test_figures_tab_does_not_warn_for_a_map_the_run_never_recorded(tmp_path):
    """The chart-unavailable warning is per-ARTIFACT, not per-record.

    A run-scoped record may name a point table for one map and not for its
    sibling; ``resolve_points_csv`` falls back to the derived path for the
    omitted ones on purpose. Deciding on ``bool(record.artifacts)`` would warn
    about a map the run never promised a table for.
    """
    import ipyvuetify as vw
    import ipywidgets

    glm_csv = tmp_path / "pred_obs_GLM_validation_300.csv"
    _write_points(glm_csv, obs=[1.0, 2.0], pred=[1.0, 2.0])
    glm_png = tmp_path / "pred_obs_GLM_validation_300.png"
    glm_png.write_bytes(b"PNG-glm")
    # RF: no artifact entry, no point CSV — only the PNG on disk.
    (tmp_path / "pred_obs_RF_validation_300.png").write_bytes(b"PNG-rf")
    record = _figures_record(
        indices=[_fig_index_row(), _fig_index_row(model="RF")],
        csv_path=tmp_path / "indices_all.csv",
        artifacts=[_fig_artifact(glm_csv, glm_png)],
    )

    rc, cls = _render_figures_tab(record=record)
    assert len(rc.find(cls).widgets) == 1  # GLM charts
    assert len(rc.find(ipywidgets.Image).widgets) == 1  # RF shows its PNG
    warnings = [a for a in rc.find(vw.Alert).widgets if a.type == "warning"]
    assert warnings == []  # nothing was promised
    rc.close()


def test_figures_tab_moves_the_axis_titles_when_the_language_changes(tmp_path):
    """The digest must carry the SAME labels the option was built from.

    ``_PredObsCard`` hands ``pred_obs_chart_identity`` the labels it hands
    ``pred_obs_scatter_option``, and that identity is passed to the adapter as
    ``option_digest`` — i.e. INSTEAD of hashing the option. Drop ``labels=``
    from the digest call and a language switch no longer moves the digest, so
    use_memo returns the previous language's chart: stale axis titles, no error
    anywhere.
    """
    from pysepal.translator import Translator

    from gui import i18n
    from gui.widget.evaluation_results import _FiguresTab

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    before = i18n._translator.value
    try:
        i18n._translator.value = Translator(i18n.MESSAGES_DIR, target="en")
        rc, cls = _render_figures_tab(record=record)
        english = rc.find(cls).widgets[0].option["xAxis"]["name"]

        i18n._translator.value = Translator(i18n.MESSAGES_DIR, target="es-ES")
        rc.render(_FiguresTab(record=record, eval_key="run-a", active_tab=2))
        spanish = rc.find(cls).widgets[0].option["xAxis"]["name"]
        rc.close()
    finally:
        i18n._translator.value = before

    assert english == "Observed deforestation (ha)"
    assert spanish != english
    assert (
        spanish
        == Translator(i18n.MESSAGES_DIR, target="es-ES")["widgets"][
            "evaluation_results"
        ]["chart_x_axis"]
    )


def test_figures_tab_scatter_survives_tab_changes_without_a_rebuild(tmp_path):
    """``active_tab`` gates the LOAD; it must stay out of the chart identity.

    A ``|tab{n}`` identity rebuilt every scatter on every tab switch, and the
    leaving tab's fresh widget re-attached inside a ``display:none`` container
    where ipecharts measures width 0 and stays squished (it sizes on attach
    only). The transitions now are:

    * ``None -> 2``: nothing about the chart changed — the SAME widget is
      reused, which is exactly what a tab-free identity buys;
    * ``2 -> 0``: the load gate empties the option, the card drops to the PNG
      rung (unchanged);
    * ``0 -> 2``: a fresh chart mounts while the tab is shown; its sizing is
      the adapter's ``visible``-nudge job.
    """
    from gui.widget.evaluation_results import _FiguresTab

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )

    rc, cls = _render_figures_tab(record=record, active_tab=None)
    mounted = rc.find(cls).widgets[0]
    rc.render(_FiguresTab(record=record, eval_key="run-a", active_tab=2))
    assert rc.find(cls).widgets[0] is mounted  # no teardown while shown

    rc.render(_FiguresTab(record=record, eval_key="run-a", active_tab=0))
    assert rc.find(cls).widgets == []  # the card drops the chart
    rc.render(_FiguresTab(record=record, eval_key="run-a", active_tab=2))
    assert len(rc.find(cls).widgets) == 1  # and remounts it on re-entry
    rc.close()


def test_figures_tab_marks_its_scatter_visible_only_when_shown(tmp_path, monkeypatch):
    """The card hands the adapter ``visible=True`` only when its tab is active.

    That flag is what schedules the post-transition resize nudge.
    """
    import gui.widget.evaluation_results as er

    seen = []
    real = er.EChartsChart

    def spy(*args, **kwargs):
        seen.append(kwargs.get("visible"))
        return real(*args, **kwargs)

    monkeypatch.setattr(er, "EChartsChart", spy)
    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    rc, cls = _render_figures_tab(record=record, active_tab=2)
    assert seen and set(seen) == {True}
    rc.close()


def test_figures_tab_hands_the_adapter_its_own_option_digest(tmp_path, monkeypatch):
    """The scatter supplies ``option_digest``, so the adapter never hashes it.

    Hashing a scatter option costs ~118 ms at 50k points and ~470 ms at 200k
    (2026-07-21; order-of-magnitude and machine-dependent, not constants) — per
    render, in a dialog the user is interacting with. Dropping the argument is
    invisible except as latency, so pin it where it is passed.
    """
    import gui.widget.echarts as ec

    hashed = []
    real = ec._option_digest
    monkeypatch.setattr(
        ec, "_option_digest", lambda option: (hashed.append(1), real(option))[1]
    )

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    rc, cls = _render_figures_tab(record=record)
    assert len(rc.find(cls).widgets) == 1  # a chart really was built
    assert hashed == []  # and the adapter did not hash it
    rc.close()


def test_figures_tab_builds_the_option_from_the_live_theme(tmp_path):
    """The dark theme must reach the scatter option's own axis colours."""
    from gui.scripts.echarts_options import theme_colors

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    with _dark_theme():
        rc, cls = _render_figures_tab(record=record)
        opt = rc.find(cls).widgets[0].option
        assert opt["xAxis"]["axisLabel"]["color"] == theme_colors(True)["ink"]
        assert (
            opt["xAxis"]["splitLine"]["lineStyle"]["color"]
            == theme_colors(True)["grid"]
        )
        rc.close()

    rc, cls = _render_figures_tab(record=record)
    assert (
        rc.find(cls).widgets[0].option["xAxis"]["axisLabel"]["color"]
        == theme_colors(False)["ink"]
    )
    rc.close()


def test_figures_tab_repaints_the_scatter_when_the_theme_is_toggled_in_place(tmp_path):
    """A LIVE theme toggle must reach a scatter that is already on screen.

    The sibling test above renders twice from scratch and therefore cannot see
    this: a fresh render reads the current theme whether or not the component
    ever subscribed to it. Here the theme flips on a MOUNTED render context and
    nothing else moves — the user's actual gesture.

    ``solara.lab.theme`` is an ipyvuetify traitlet behind a ``Proxy``, not a
    ``Reactive``, so ``dark = theme.dark`` in the render body creates no
    subscription: the toggle re-renders nothing, and reacton's prop-equality
    bailout stops the card even when the dialog above it does re-render — the
    scatter keeps its light ink (``#52514e``) on a dark surface, and
    ``pred_obs_scatter_option`` is never even called. ``use_theme_dark()``
    observes pysepal's ``ThemeState.dark`` instead, so the flip itself
    schedules the re-render.
    """
    from gui.scripts.echarts_options import theme_colors

    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.0, 2.0]
    )
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    rc, cls = _render_figures_tab(record=record)
    assert (
        rc.find(cls).widgets[0].option["xAxis"]["axisLabel"]["color"]
        == theme_colors(False)["ink"]
    )
    with _dark_theme():
        assert (
            rc.find(cls).widgets[0].option["xAxis"]["axisLabel"]["color"]
            == theme_colors(True)["ink"]
        )
    rc.close()


def test_figures_tab_defers_point_load_until_its_tab_is_active(tmp_path, monkeypatch):
    """Point data is parsed only when the tab is active, not on dialog open."""
    import gui.widget.evaluation_results as er

    _write_points(tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0], pred=[1.0])
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    calls = []
    real = er.load_pred_obs_plot_data

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(er, "load_pred_obs_plot_data", spy)

    rc, cls = _render_figures_tab(record=record, active_tab=0)  # some other tab
    assert calls == []  # nothing parsed
    assert rc.find(cls).widgets == []
    rc.close()

    rc, cls = _render_figures_tab(record=record, active_tab=2)  # figures active
    assert calls and len(rc.find(cls).widgets) == 1
    rc.close()


def test_figures_tab_offers_a_png_download(tmp_path):
    """An explicit action keeps the canonical PNG reachable from the UI."""
    import ipyvuetify as vw

    from gui.i18n import t as _t

    _write_points(tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0], pred=[1.0])
    png = tmp_path / "pred_obs_GLM_validation_300.png"
    png.write_bytes(b"PNG-bytes")
    record = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    rc, _ = _render_figures_tab(record=record)
    labels = [
        c
        for b in rc.find(vw.Btn).widgets
        for c in (b.children or [])
        if isinstance(c, str)
    ]
    assert _t("widgets.evaluation_results.download_png") in labels
    assert png.read_bytes() == b"PNG-bytes"  # PNG still present on disk
    rc.close()


def test_figures_tab_builds_each_option_once_not_once_per_render(tmp_path):
    """A multi-map figures tab must not rebuild every option when it re-renders.

    Measured 2026-07-21 (CPython 3.11.10 / pandas 2.x; order-of-magnitude, not
    constants): materializing an option's point rows costs 1.0 ms at 2k points,
    12.8 ms at 50k and 282 ms at 200k, and ``_scatter_rows``' module-level LRU
    cannot absorb a third map at ``SCATTER_ROWS_CACHE_SIZE = 2`` — calling it
    for three maps (200k/50k/25k points) in a loop scores ZERO hits in 9 calls,
    ~234 ms of pure rebuild. The fix is one level up: ``_PredObsCard`` memoizes
    the finished option on ``pred_obs_chart_identity``, once per card whatever
    the map count.

    Scope, so the claim is not overstated: reacton bails out on ``==``-equal
    props, so a parent re-render that leaves this card's props alone never
    re-enters it and never rebuilt anything. The cost this removes is on the
    passes that DO re-enter — which is why the re-render below has to move an
    extrinsic prop to happen at all. ``eval_key`` only feeds the chart widget's
    rebuild ``identity`` and is deliberately NOT part of the option's digest, so
    the cards genuinely re-render while the option memo must hold. Building the
    option in the render body instead doubles the row builder's miss count,
    which is what this asserts.

    The memo also keys on ``tab_active``, so leaving the Pred-vs-obs tab and
    returning DOES rebuild every card's rows (3 -> 3 -> 6 misses). That is
    deliberate — see the constants block in ``gui/scripts/evaluation_echarts``:
    the tab round trip already forces a fresh widget and a fresh option over the
    wire, and dropping the rows while the tab is hidden is what keeps the memory
    bounded.
    """
    from gui.scripts.evaluation_echarts import _scatter_rows
    from gui.widget.evaluation_results import _FiguresTab

    models = ("GLM", "MW", "RF")
    for m in models:
        _write_points(
            tmp_path / f"pred_obs_{m}_validation_300.csv",
            obs=[1.0, 2.0, 3.0],
            pred=[1.5, 2.5, 2.0],
        )
    record = _figures_record(
        indices=[_fig_index_row(model=m) for m in models],
        csv_path=tmp_path / "indices_all.csv",
    )

    _scatter_rows.cache_clear()
    rc, cls = _render_figures_tab(record=record, eval_key="run-a")
    assert len(rc.find(cls).widgets) == 3
    after_first = _scatter_rows.cache_info().misses
    assert after_first == 3, "one row build per map on the first render"

    rc.render(_FiguresTab(record=record, eval_key="run-b", active_tab=2))
    assert len(rc.find(cls).widgets) == 3
    assert _scatter_rows.cache_info().misses == after_first, (
        "the option must be memoized on the chart identity, not rebuilt in the"
        " render body — an LRU of 2 cannot cover three cards"
    )
    rc.close()


def test_evaluation_table_dialog_shows_the_scatter_on_the_figures_tab(tmp_path):
    """End-to-end: driving the dialog to the third tab loads the scatter.

    Extends the mount smoke: the figures tab is inactive on open (no scatter),
    and switching to it parses the point CSV and draws the interactive chart.
    """
    import ipecharts
    import ipyvuetify as vw
    import reacton
    import solara

    from gui.i18n import t as _t

    _t("common.close")  # warm the translator before the first render
    _write_points(
        tmp_path / "pred_obs_GLM_validation_300.csv", obs=[1.0, 2.0], pred=[1.5, 2.5]
    )
    from gui.widget.evaluation_results import EvaluationTableDialog

    rec = _figures_record(
        indices=[_fig_index_row()], csv_path=tmp_path / "indices_all.csv"
    )
    p = types.SimpleNamespace(evaluations={"run-a": rec})
    project = solara.reactive(p, equals=lambda a, b: a is b)
    _, rc = reacton.render(
        EvaluationTableDialog(
            project=project, eval_key="run-a", on_close=lambda *_: None
        ),
        handle_error=False,
    )

    def scatters():
        return [
            w
            for w in rc.find(ipecharts.EChartsRawWidget).widgets
            if any(s.get("type") == "scatter" for s in w.option.get("series", []))
        ]

    assert scatters() == []  # inactive figures tab: no scatter parsed yet
    rc.find(vw.Tabs).widgets[0].v_model = 2  # user opens Pred. vs obs.
    assert len(scatters()) == 1
    assert _scatter_data(scatters()[0]) == [[1.0, 1.5], [2.0, 2.5]]
    rc.close()


def test_run_scoped_artifact_records_defrate_csv(tmp_path, monkeypatch):
    """The artifact points at the run's own defrate table."""
    from types import SimpleNamespace

    from spatialrisk.predictions.prediction import Prediction

    project = SimpleNamespace(
        folders=SimpleNamespace(project_folder=tmp_path), predictions={}
    )
    pred = Prediction(
        path=tmp_path / "prob.tif", model_key="glm_m", dataset_name="ds_2020"
    )
    written = {}

    def fake_defrate(**kw):
        """Mock defrate function."""
        _Path(kw["tab_file_defrate"]).write_text("cat,defor_dens\n1,0\n")
        written["path"] = _Path(kw["tab_file_defrate"])

    monkeypatch.setattr(ev, "_defrate_per_cat", fake_defrate)
    monkeypatch.setattr(
        ev,
        "validate_two_layer",
        lambda **kw: {
            "RMSE": 0,
            "wRMSE": 0,
            "MedAE": 0,
            "R2": 0,
            "ncell": 1,
            "csize_coarse_grid": 300,
            "csize_coarse_grid_ha": 9.0,
        },
    )

    rows = ev._evaluate_one_against_truth(
        project,
        pred,
        defor_file=tmp_path / "d.tif",
        forest_file=tmp_path / "f.tif",
        time_interval=5,
        truth_tag="truth",
        run_id="abcd1234",
    )

    art = rows[0]["artifact"]
    assert art.defrate_csv == str(written["path"])
    assert (
        _Path(art.defrate_csv).parent == tmp_path / "evaluation" / "truth" / "abcd1234"
    )
