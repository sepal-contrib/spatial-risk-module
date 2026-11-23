"""Read-only Details dialog on the Harmonization rows."""

from types import SimpleNamespace

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_details_dialog import (  # noqa: E402
    VariableDetailsDialog,
    harmonized_details,
)
from spatialrisk.harmonization import HarmonizationStatus  # noqa: E402
from spatialrisk.variables.models import RasterType  # noqa: E402

K = "widgets.variable_details."
CURRENT = HarmonizationStatus(pending=[], current=["slope_2020"])
PENDING = HarmonizationStatus(pending=["slope_2020"], current=[])


def _named(type_name, **attrs):
    return type(type_name, (), attrs)()


def _project(output, raster_type=RasterType.continuous):
    raw = _named(
        "LocalRasterVar",
        name="slope",
        data_type="raster",
        year=2020,
        raster_type=raster_type,
        path="/x/slope.tif",
    )
    return SimpleNamespace(
        raw_variables={"slope_2020": raw},
        processed_variables={"slope_2020": output} if output is not None else {},
    )


def _output(**extra):
    fields = dict(
        name="slope",
        data_type="raster",
        year=2020,
        raster_type=RasterType.continuous,
        path="/x/out/slope.tif",
        default_crs="EPSG:32633",
        default_resolution=30.0,
        processing_history=["reprojected_matched"],
        categorical_levels=None,
        resampling="bilinear",
    )
    fields.update(extra)
    return _named("LocalRasterVar", **fields)


def _rows(project, status=None):
    return dict(harmonized_details(project, "slope_2020", status).rows)


def test_ready_row_shows_its_output_and_the_stored_method():
    """A harmonized row: grid, stored resampling, steps and path."""
    details = harmonized_details(_project(_output()), "slope_2020", CURRENT)
    assert details.name == "slope (2020)"
    rows = dict(details.rows)
    assert rows[t(K + "source_layer")] == "slope_2020"
    assert rows[t(K + "target_crs")] == "EPSG:32633"
    assert rows[t(K + "resolution")] == 30.0
    assert rows[t(K + "resampling")] == "bilinear"
    assert rows[t(K + "matches_grid")] is True
    assert rows[t(K + "processing_steps")] == ["reprojected_matched"]
    assert rows[t(K + "file_path")] == "/x/out/slope.tif"


def test_output_without_a_stored_method_shows_the_rule_default():
    """Older outputs and rows not harmonized yet: the rule's pick, marked."""
    legacy = _rows(_project(_output(resampling=None)), CURRENT)
    assert legacy[t(K + "resampling")] == t(
        K + "resampling_default", method="bilinear", raster_type="continuous"
    )
    never = _rows(_project(None, RasterType.categorical), PENDING)
    assert never[t(K + "resampling")] == t(
        K + "resampling_default", method="nearest", raster_type="categorical"
    )
    assert never[t(K + "matches_grid")] is None
    assert never[t(K + "file_path")] is None
    untyped = _rows(_project(None, None), PENDING)
    assert untyped[t(K + "resampling")] == t(
        K + "resampling_default_untyped", method="nearest"
    )


def test_grid_field_follows_the_lists_status():
    """Moved reference reads No; other reasons are named; no status, no answer."""
    p = _project(_output())
    assert _rows(p, PENDING)[t(K + "matches_grid")] is False
    assert _rows(p, None)[t(K + "matches_grid")] is None
    retyped = _project(_output(raster_type=RasterType.categorical))
    assert _rows(retyped, PENDING)[t(K + "matches_grid")] == t(
        K + "grid_reason_raster_type"
    )
    no_levels = _project(
        _output(raster_type=RasterType.categorical), RasterType.categorical
    )
    assert _rows(no_levels, PENDING)[t(K + "matches_grid")] == t(
        K + "grid_reason_categories"
    )


def test_gone_rows_have_no_details():
    """No project, no key or a deleted variable: nothing to show, nothing raised."""
    assert harmonized_details(None, "slope_2020") is None
    assert harmonized_details(_project(None), None) is None
    assert harmonized_details(_project(None), "nope") is None


def test_dialog_never_opens_a_file(monkeypatch):
    """Rendering reads the project only: the paths here do not exist."""
    import builtins

    import rasterio

    def _boom(*a, **k):
        raise AssertionError("the details dialog opened a file")

    monkeypatch.setattr(rasterio, "open", _boom)
    monkeypatch.setattr(builtins, "open", _boom)
    harmonized_details(_project(_output()), "slope_2020", CURRENT)


def test_dialog_renders_the_rows_and_closes():
    """Open on a key, one field per row; Close calls on_close; no key, closed."""
    project = solara.reactive(_project(_output()))
    closed = []
    _, rc = reacton.render(
        VariableDetailsDialog(
            project=project,
            key="slope_2020",
            on_close=lambda: closed.append(1),
            status=CURRENT,
        ),
        handle_error=False,
    )
    try:
        assert rc.find(vw.Dialog).widget.v_model is True
        fields = {f.label: f.v_model for f in rc.find(vw.TextField).widgets}
        assert fields[t(K + "resampling")] == "bilinear"
        assert fields[t(K + "matches_grid")] == t("common.yes")
        assert len(fields) == 7
        close = [b for b in rc.find(vw.Btn).widgets if t("common.close") in b.children]
        close[0].click()
        assert closed == [1]
    finally:
        rc.close()
    _, rc = reacton.render(
        VariableDetailsDialog(project=project, key=None, on_close=lambda: None),
        handle_error=False,
    )
    try:
        assert rc.find(vw.Dialog).widget.v_model is False
    finally:
        rc.close()
