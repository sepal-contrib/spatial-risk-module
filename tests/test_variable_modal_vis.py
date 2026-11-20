"""The Variables modal lets a custom layer pick its own palette.

Categorical (the default raster type now) shows a class list (a colour, label
and visibility per value); continuous shows the named ramps with an invert
switch and an optional Min/Max. The choice is submitted as ``vis_params`` and
prefilled when editing.
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
    # A solara.lab.Menu holds its button in ``activator``, not in ``children``.
    children = list(getattr(widget, "children", []) or [])
    children += list(getattr(widget, "activator", []) or [])
    for child in children:
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


def _mask(color):
    """The ``vis_params`` of a 0/1 mask saved before class lists existed."""
    return {"palette": ["ffffff", color], "min": 0, "max": 1}


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


def _fields(box, label_key):
    """Text fields labelled ``label_key``."""
    return [f for f in _find(box, vw.TextField) if f.label == t(label_key)]


def _ramp_select(box):
    """The colour ramp select, or None."""
    label = t("vars.modal.ramp_label")
    return next((s for s in _find(box, vw.Select) if s.label == label), None)


def _class_rows(box):
    """The class rows as ``(value, label)``, in list order."""
    rows = [h for h in _find(box, vw.Html) if h.class_ == "sr-class-row"]
    return [
        (
            next(h.children[0] for h in _find(r, vw.Html) if h.tag == "span"),
            _find(r, vw.TextField)[0].v_model,
        )
        for r in rows
    ]


def test_categorical_shows_the_class_list_not_a_ramp():
    """A categorical layer opens on the 0/1 mask's two classes."""
    box = _render(_custom_gee(raster_type="categorical"))
    assert _class_rows(box) == [
        ("0", t("legend.class.absent")),
        ("1", t("legend.class.present")),
    ]
    assert len(_btns(box, "sr-palette-menu")) == 1
    assert _ramp_select(box) is None
    assert _fields(box, "vars.modal.range_min") == []


def test_continuous_shows_the_ramp_range_and_invert_not_classes():
    """A continuous layer offers the ramp select, Invert, Min and Max."""
    box = _render(_custom_gee(raster_type="continuous"))
    assert [i["value"] for i in _ramp_select(box).items] == list(vp.RAMPS)
    assert _class_rows(box) == []
    switch = next(
        s for s in _find(box, vw.Switch) if s.label == t("vars.modal.ramp_invert_label")
    )
    assert switch.v_model is False
    # Empty Min and Max: the renderer stretches the ramp.
    assert _fields(box, "vars.modal.range_min")[0].v_model == ""
    assert _fields(box, "vars.modal.range_max")[0].v_model == ""


def _vector(**extra):
    """A custom vector entry."""
    return {
        "source": "custom",
        "type": "LocalVectorVar",
        "name": "r",
        "path": "/tmp/r.shp",
        **extra,
    }


def test_a_vector_mask_shows_its_two_classes_with_the_background_hidden():
    """Rasterized as binary a vector is 0 and 1: nothing to find, no ramp."""
    box = _render(_vector())
    assert _class_rows(box) == [
        ("0", t("legend.class.absent")),
        ("1", t("legend.class.present")),
    ]
    eyes = [_find(b, vw.Icon)[0].children for b in _btns(box, "sr-class-hide")]
    assert eyes == [["mdi-eye-off"], ["mdi-eye"]]
    assert _ramp_select(box) is None
    find = t("vars.modal.find_classes")
    assert not [b for b in _find(box, vw.Btn) if find in b.children]


def test_a_vector_burnt_per_feature_starts_empty_with_a_find_button():
    """Rasterized as unique the classes are the file's features: found, not seeded."""
    box = _render(_vector(rasterization_method="unique"))
    assert _class_rows(box) == []
    find = t("vars.modal.find_classes")
    assert not next(b for b in _find(box, vw.Btn) if find in b.children).disabled


def test_editing_a_mask_saved_before_classes_reads_as_two_classes():
    """A saved presence colour becomes the colour of class 1."""
    box = _render(_custom_gee(raster_type="categorical", vis_params=_mask("7b1fa2")))
    swatches = _btns(box, "sr-class-swatch")
    assert [("ffffff" in b.style_, "7b1fa2" in b.style_) for b in swatches] == [
        (True, False),
        (False, True),
    ]


def test_editing_prefills_the_saved_classes():
    """Editing a class list shows its values, labels and hidden classes."""
    classes = [
        {"value": 10, "color": "006400", "label": "Trees", "visible": True},
        {"value": 80, "color": "0064c8", "label": "Water", "visible": False},
    ]
    box = _render(
        _custom_gee(raster_type="categorical", vis_params={"classes": classes})
    )
    assert _class_rows(box) == [("10", "Trees"), ("80", "Water")]
    eyes = [_find(b, vw.Icon)[0].children for b in _btns(box, "sr-class-hide")]
    assert eyes == [["mdi-eye"], ["mdi-eye-off"]]
    # The legend preview draws the visible class only.
    preview = next(h for h in _find(box, vw.Html) if h.class_ == "sr-legend-preview")
    texts = [
        c for h in _find(preview, vw.Html) for c in h.children if isinstance(c, str)
    ]
    assert texts == ["Trees"]


def test_editing_prefills_the_saved_ramp_inversion_and_range():
    """Editing a saved ramp reselects it, with its Invert switch and range."""
    box = _render(
        _custom_gee(
            raster_type="continuous",
            vis_params=vp.continuous_vis("viridis", invert=True, span=(0, 2500)),
        )
    )
    switch = next(
        s for s in _find(box, vw.Switch) if s.label == t("vars.modal.ramp_invert_label")
    )
    assert switch.v_model is True
    assert _ramp_select(box).v_model == "viridis"
    assert _fields(box, "vars.modal.range_min")[0].v_model == "0"
    assert _fields(box, "vars.modal.range_max")[0].v_model == "2500"


def test_editing_a_greyscale_ramp_pinned_to_zero_one_keeps_its_range():
    """It has an old mask's shape (two colours, 0..1) but is a ramp."""
    box = _render(
        _custom_gee(
            raster_type="continuous",
            vis_params=vp.continuous_vis("grayscale", span=(0.0, 1.0)),
        )
    )
    assert _ramp_select(box).v_model == "grayscale"
    assert _fields(box, "vars.modal.range_min")[0].v_model == "0"
    assert _fields(box, "vars.modal.range_max")[0].v_model == "1"


def test_a_class_row_can_be_removed():
    """Its close button drops the class from the list."""
    box = _render(_custom_gee(raster_type="categorical"))
    _btns(box, "sr-class-remove")[0].click()
    assert [value for value, _ in _class_rows(box)] == ["1"]


def test_probes_need_a_raster_and_for_an_asset_the_gee_interface():
    """Without a way to read the layer the probe button is disabled."""
    label = t("vars.modal.find_classes")
    box = _render(_custom_gee(raster_type="categorical"))  # no gee_interface
    assert next(b for b in _find(box, vw.Btn) if label in b.children).disabled
    box = _render(
        {
            "source": "custom",
            "type": "LocalRasterVar",
            "name": "r",
            "path": "/tmp/r.tif",
            "raster_type": "categorical",
        }
    )
    assert not next(b for b in _find(box, vw.Btn) if label in b.children).disabled


def test_custom_vis_is_the_class_list_or_the_ramp():
    """What the modal submits for each raster type."""
    classes = [{"value": 1, "color": "2e7d32", "label": "x", "visible": True}]
    assert mod.custom_vis("categorical", classes, "blues", False) == {
        "classes": classes
    }
    assert mod.custom_vis("continuous", classes, "blues", True, (1, 9)) == {
        "palette": list(reversed(vp.RAMPS["blues"])),
        "min": 1,
        "max": 9,
    }


# ---- entry <-> variable round-trips (variables_tile) --------------------------


def test_variable_to_entry_carries_vis_params_for_custom_layers():
    """Both custom raster classes round-trip vis_params into the edit entry."""
    from gui.tile.variables_tile import _variable_to_entry
    from spatialrisk import Project
    from spatialrisk.variables.gee_var import GEEVar
    from spatialrisk.variables.local_raster_var import LocalRasterVar
    from spatialrisk.variables.models import DataType, RasterType

    Project._ensure_model_schemas()
    vis = _mask("795548")
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
