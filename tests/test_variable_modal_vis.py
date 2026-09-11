"""The Variables modal lets a custom layer pick its own palette.

Categorical (the default raster type now) shows presence swatches plus a hex
field; continuous shows the named ramps with an invert switch. The choice is
submitted as ``vis_params`` and prefilled when editing.
"""

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

import gui.widget.variable_modal as mod  # noqa: E402
from gui.scripts import variable_palettes as vp  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_asset_selector(monkeypatch):
    """Stand in for pysepal's AssetSelectComponent on the GEE branch.

    The real one schedules GEE coroutines with ``solara.lab.use_task``, which
    needs a running event loop that a bare ``reacton.render`` does not have.
    The selector's own wiring is covered by test_variable_modal_asset_select.
    """

    @solara.component
    def FakeSelector(**_):
        solara.Text("asset-select stub")

    monkeypatch.setattr(mod, "AssetSelectComponent", FakeSelector)


def _find(widget, cls, out=None):
    """Collect widgets of ``cls`` in the rendered tree."""
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _render(initial_entry=None):
    """Render the modal open on ``initial_entry``."""
    box, _rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            initial_entry=initial_entry,
        )
    )
    return box


def _btns(box, css_class):
    """Buttons carrying ``css_class``."""
    return [b for b in _find(box, vw.Btn) if css_class in (b.class_ or "")]


def _custom_gee(**extra):
    """A custom GEE-asset entry with ``extra`` fields."""
    return {
        "source": "custom",
        "type": "GEEVar",
        "name": "x",
        "asset_id": "projects/p/assets/x",
        **extra,
    }


def test_raster_type_defaults_to_categorical():
    """Most custom layers are 0/1 masks, so the select opens on categorical."""
    box = _render(_custom_gee())
    label = t("vars.modal.custom_raster_type_label")
    select = next(s for s in _find(box, vw.Select) if s.label == label)
    assert select.v_model == "categorical"


def test_categorical_shows_swatches_and_a_hex_field_not_ramps():
    """A categorical layer offers the swatch row and the hex field."""
    box = _render(_custom_gee(raster_type="categorical"))
    assert len(_btns(box, "sr-swatch")) == len(vp.PRESENCE_COLORS)
    assert _btns(box, "sr-ramp") == []
    hex_field = next(
        f
        for f in _find(box, vw.TextField)
        if f.label == t("vars.modal.presence_hex_label")
    )
    assert hex_field.v_model == f"#{vp.DEFAULT_PRESENCE}"


def test_continuous_shows_ramps_and_an_invert_switch_not_swatches():
    """A continuous layer offers the ramp row and the invert switch."""
    box = _render(_custom_gee(raster_type="continuous"))
    assert len(_btns(box, "sr-ramp")) == len(vp.RAMPS)
    assert _btns(box, "sr-swatch") == []
    switch = next(
        s for s in _find(box, vw.Switch) if s.label == t("vars.modal.ramp_invert_label")
    )
    assert switch.v_model is False

    # Vector layers are rasterized later; nothing to pick.
    box = _render(
        {
            "source": "custom",
            "type": "LocalVectorVar",
            "name": "r",
            "path": "/tmp/r.shp",
        }
    )
    assert _btns(box, "sr-swatch") == [] and _btns(box, "sr-ramp") == []


def test_editing_prefills_the_saved_presence_colour():
    """Editing a saved mask shows its colour in the hex field."""
    box = _render(
        _custom_gee(raster_type="categorical", vis_params=vp.categorical_vis("#7b1fa2"))
    )
    hex_field = next(
        f
        for f in _find(box, vw.TextField)
        if f.label == t("vars.modal.presence_hex_label")
    )
    assert hex_field.v_model == "#7b1fa2"


def test_editing_prefills_the_saved_ramp_and_inversion():
    """Editing a saved ramp reselects it and restores the invert switch."""
    box = _render(
        _custom_gee(
            raster_type="continuous",
            vis_params=vp.continuous_vis("viridis", invert=True),
        )
    )
    switch = next(
        s for s in _find(box, vw.Switch) if s.label == t("vars.modal.ramp_invert_label")
    )
    assert switch.v_model is True
    pressed = [
        b for b in _btns(box, "sr-ramp") if "sr-ramp--selected" in (b.class_ or "")
    ]
    labels = [
        c for h in _find(pressed[0], vw.Html) for c in h.children if isinstance(c, str)
    ]
    assert len(pressed) == 1 and labels == [t("vars.modal.ramp_viridis")]


# ---- entry <-> variable round-trips (variables_tile) --------------------------


def test_variable_to_entry_carries_vis_params_for_custom_layers():
    """Both custom raster classes round-trip vis_params into the edit entry."""
    from gui.tile.variables_tile import _variable_to_entry
    from spatialrisk import Project
    from spatialrisk.variables.gee_var import GEEVar
    from spatialrisk.variables.local_raster_var import LocalRasterVar
    from spatialrisk.variables.models import DataType, RasterType

    Project._ensure_model_schemas()
    vis = vp.categorical_vis("#795548")
    project = type("P", (), {"base_raster": None})()

    gee = GEEVar(
        name="custom",
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        path="projects/p/assets/x",
        vis_params=vis,
    )
    assert _variable_to_entry("custom", gee, project)["vis_params"] == vis

    local = LocalRasterVar(
        name="custom",
        path="/tmp/x.tif",
        raster_type=RasterType.categorical,
        vis_params=vis,
    )
    assert _variable_to_entry("custom", local, project)["vis_params"] == vis


def test_build_variable_passes_vis_params_through(monkeypatch):
    """The factory hands vis_params to both custom raster classes."""
    import ee

    import gui.scripts.predefined_variables as predefined
    from gui.store.state_manager import app_state
    from gui.tile.variables_tile import _build_variable
    from spatialrisk import Project
    from spatialrisk.variables.models import DataType, RasterType

    monkeypatch.setattr(app_state.aoi_result, "value", object())
    monkeypatch.setattr(
        predefined, "resolve_aoi_ee", lambda _r: ee.Geometry.__new__(ee.Geometry)
    )
    project = Project(project_name="p")
    vis = vp.continuous_vis("oranges")

    gee = _build_variable(
        {
            "source": "custom",
            "type": "GEEVar",
            "name": "c",
            "year": None,
            "path": "projects/p/assets/x",
            "default_scale": None,
            "data_type": DataType.raster,
            "raster_type": RasterType.continuous,
            "vis_params": vis,
        },
        project,
    )
    assert gee.vis_params == vis

    local = _build_variable(
        {
            "source": "custom",
            "type": "LocalRasterVar",
            "name": "c",
            "year": None,
            "path": "/tmp/c.tif",
            "data_type": DataType.raster,
            "raster_type": RasterType.continuous,
            "vis_params": vis,
        },
        project,
    )
    assert local.vis_params == vis


def test_selected_swatch_is_marked_and_the_picker_is_outlined():
    """The chosen swatch shows a check mark; the picker sits in a legend box."""
    box = _render(
        _custom_gee(raster_type="categorical", vis_params=vp.categorical_vis("#2196f3"))
    )
    selected = [b for b in _btns(box, "sr-swatch") if "sr-swatch--selected" in b.class_]
    assert len(selected) == 1
    icons = _find(selected[0], vw.Icon)
    assert icons and icons[0].children == ["mdi-check"]
    unselected = [b for b in _btns(box, "sr-swatch") if b is not selected[0]]
    assert all(not _find(b, vw.Icon) for b in unselected)
    fieldsets = [h for h in _find(box, vw.Html) if h.tag == "fieldset"]
    assert len(fieldsets) == 1
    legend = next(h for h in _find(fieldsets[0], vw.Html) if h.tag == "legend")
    assert legend.children == [t("vars.modal.presence_color_label")]
