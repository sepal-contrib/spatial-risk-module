"""Read-only Details dialog on the Harmonization rows."""

from types import SimpleNamespace

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_details_dialog import (  # noqa: E402
    VariableDetailsDialog,
    derived_details,
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


def _rivers_project(**raw_attrs):
    raw = _named(
        "LocalRasterVar",
        name="rivers_wlmc",
        data_type="raster",
        year=None,
        raster_type=RasterType.categorical,
        path="/x/rivers_wlmc.tif",
        **raw_attrs,
    )
    return SimpleNamespace(raw_variables={"rivers_wlmc": raw}, processed_variables={})


def test_predefined_row_lists_its_stored_params():
    """The values saved on the layer show as fields, defaults included."""
    project = _rivers_project(params={"classes": ["large", "major", "canal"]})
    rows = dict(harmonized_details(project, "rivers_wlmc").rows)
    assert rows[t("vars.modal.param_water_classes")] == (
        f"{t('vars.modal.water_large')}; {t('vars.modal.water_major')}; "
        f"{t('vars.modal.water_canal')}"
    )


def test_details_show_default_display():
    """A catalogue layer on its default display says so; a plain layer has no row."""
    rows = dict(
        harmonized_details(_rivers_project(vis_params=None), "rivers_wlmc").rows
    )
    assert rows[t(K + "display")] == t(K + "display_default")
    plain = SimpleNamespace(
        raw_variables={"mine": _named("LocalRasterVar", name="mine", year=None)},
        processed_variables={},
    )
    assert t(K + "display") not in dict(harmonized_details(plain, "mine").rows)


def test_details_show_custom_display():
    """A catalogue layer with its own display says Customised."""
    project = _rivers_project(vis_params={"min": 0, "max": 1})
    rows = dict(harmonized_details(project, "rivers_wlmc").rows)
    assert rows[t(K + "display")] == t(K + "display_custom")


def test_layer_saved_without_params_reads_them_from_its_name():
    """Projects written before params were stored still show them."""
    stored = dict(
        harmonized_details(
            _rivers_project(params={"classes": ["large", "major", "canal"]}),
            "rivers_wlmc",
        ).rows
    )
    legacy = dict(harmonized_details(_rivers_project(), "rivers_wlmc").rows)
    assert legacy == stored


def test_interval_layer_lists_its_start_and_end_year():
    """Part C: the period shows as two plain-year fields, stored or from the name."""
    name = "deforestation_gfc_tc30_2015_2020"
    raw = _named(
        "LocalRasterVar",
        name=name,
        data_type="raster",
        year=None,
        raster_type=RasterType.categorical,
        path=f"/x/{name}.tif",
        params=None,
    )
    project = SimpleNamespace(raw_variables={name: raw}, processed_variables={})
    details = harmonized_details(project, name)
    rows = dict(details.rows)
    assert details.name == name
    assert rows[t("vars.modal.param_tree_cover_threshold")] == "30"
    assert rows[t("vars.modal.param_start_year")] == "2015"
    assert rows[t("vars.modal.param_end_year")] == "2020"


def test_legacy_default_layer_lists_its_default_params():
    """A bare-named layer saved without params shows the defaults, like a new one."""
    raw = _named(
        "LocalRasterVar",
        name="roads",
        data_type="raster",
        year=None,
        raster_type=RasterType.categorical,
        path="/x/roads.tif",
    )
    project = SimpleNamespace(raw_variables={"roads": raw}, processed_variables={})
    rows = dict(harmonized_details(project, "roads").rows)
    assert rows[t("vars.modal.param_roads_source")] == t("vars.modal.roads_source_osm")


def test_gone_rows_have_no_details():
    """No project, no key or a deleted variable: nothing to show, nothing raised."""
    assert harmonized_details(None, "slope_2020") is None
    assert harmonized_details(_project(None), None) is None
    assert harmonized_details(_project(None), "nope") is None


def _derived_project():
    raw = {
        "forest_2020": _named("LocalRasterVar", name="forest", year=2020),
        "forest_2024": _named("LocalRasterVar", name="forest", year=2024),
        "roads": _named("LocalRasterVar", name="roads", year=None),
    }
    processed = {
        "forest_edge_2020": _named(
            "LocalRasterVar",
            name="forest_edge",
            year=2020,
            processing_history=["reprojected_matched", "edge"],
            tags=[],
            path="/x/out/forest_edge_2020.tif",
        ),
        "loss_forest_2020_2024": _named(
            "LocalRasterVar",
            name="loss_forest_2020_2024",
            year=None,
            processing_history=[],
            tags=["loss", "change", "2020_2024"],
            path="/x/out/loss.tif",
        ),
        "loss_forest_2020_2024_dist": _named(
            "LocalRasterVar",
            name="loss_forest_2020_2024_dist",
            year=None,
            processing_history=["dist"],
            tags=["loss", "change", "2020_2024"],
            path=None,
        ),
    }
    return SimpleNamespace(raw_variables=raw, processed_variables=processed)


def test_derived_edge_shows_its_operation_source_and_year():
    """An edge layer: the source the list names, the inherited year, the path."""
    details = derived_details(_derived_project(), "forest_edge_2020")
    assert details.name == "forest_edge (2020)"
    rows = dict(details.rows)
    assert rows[t(K + "operation")] == t("tiles.postprocess.op_label_edge")
    assert rows[t(K + "inputs")] == ["forest (2020)"]
    assert rows[t(K + "years")] == [2020]
    assert rows[t(K + "processing_steps")] == ["reprojected_matched", "edge"]
    assert rows[t(K + "file_path")] == "/x/out/forest_edge_2020.tif"


def test_derived_change_layer_names_both_inputs_and_years():
    """A loss layer: both layers and both years, read off its name."""
    rows = dict(derived_details(_derived_project(), "loss_forest_2020_2024").rows)
    assert rows[t(K + "operation")] == t("tiles.postprocess.op_label_loss")
    assert rows[t(K + "inputs")] == ["forest (2020)", "forest (2024)"]
    assert rows[t(K + "years")] == [2020, 2024]


def test_distance_of_a_loss_layer_is_a_distance():
    """The history wins over the inherited loss name and tags."""
    p = _derived_project()
    rows = dict(derived_details(p, "loss_forest_2020_2024_dist").rows)
    assert rows[t(K + "operation")] == t("tiles.postprocess.op_label_dist")
    assert rows[t(K + "years")] is None
    assert rows[t(K + "file_path")] is None
    assert derived_details(p, "nope") is None
    assert derived_details(None, "forest_edge_2020") is None


def test_dialog_shows_a_derived_row():
    """``kind="derived"`` reads the processed key."""
    _, rc = reacton.render(
        VariableDetailsDialog(
            project=solara.reactive(_derived_project()),
            kind="derived",
            key="loss_forest_2020_2024",
            on_close=lambda: None,
        ),
        handle_error=False,
    )
    try:
        assert rc.find(vw.Dialog).widget.v_model is True
        fields = {f.label: f.v_model for f in rc.find(vw.TextField).widgets}
        assert fields[t(K + "inputs")] == "forest (2020), forest (2024)"
        assert fields[t(K + "years")] == "2020, 2024"
    finally:
        rc.close()


def test_dialog_never_opens_a_file(monkeypatch):
    """Rendering reads the project only: the paths here do not exist."""
    import builtins

    import rasterio

    def _boom(*a, **k):
        raise AssertionError("the details dialog opened a file")

    monkeypatch.setattr(rasterio, "open", _boom)
    monkeypatch.setattr(builtins, "open", _boom)
    harmonized_details(_project(_output()), "slope_2020", CURRENT)
    derived_details(_derived_project(), "forest_edge_2020")


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
        # 7 fields + slope's 2 catalogue defaults (DEM source, slope unit)
        # + the display row
        assert len(fields) == 10
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
