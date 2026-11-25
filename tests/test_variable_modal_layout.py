"""Add/Edit Variable modal: one field order for predefined and user variables.

Rows mirror each other across the two forms; visualization is a collapsed
panel near the end; a predefined layer's description sits behind an info icon.
"""

import ipyvuetify as vw
import reacton
import reacton.ipyvuetify as rv
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.scripts import variable_classes as vc  # noqa: E402
from gui.scripts.predefined import PREDEFINED_CATALOGUE, param_specs  # noqa: E402
from gui.widget import variable_modal  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402

TOGGLE = "vars.modal.display_toggle"


def _render(initial_entry=None):
    _box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            initial_entry=initial_entry,
        ),
        handle_error=False,
    )
    return rc


def _predefined(key, params=None):
    return {"source": "predefined", "predefined_key": key, "params": params}


def _labels(widget):
    """Field labels and panel headers in on-screen (tree) order."""
    if isinstance(widget, (vw.Select, vw.TextField)):
        out = [widget.label]
    elif isinstance(widget, vw.ExpansionPanelHeader):
        out = [" ".join(str(c) for c in widget.children)]
    else:
        out = []
    for child in getattr(widget, "children", None) or ():
        if not isinstance(child, str):
            out += _labels(child)
    return out


def _labels_in_page_order(rc):
    # the form dialog is the one holding the Source select (not an info popup)
    source = t("vars.modal.source_label")
    for dialog in rc.find(vw.Dialog).widgets:
        labels = _labels(dialog)
        if source in labels:
            return labels
    raise AssertionError("no form dialog")


def _texts(widget):
    """Every string child under ``widget``, depth first."""
    out = []
    for child in getattr(widget, "children", None) or ():
        out += [child] if isinstance(child, str) else _texts(child)
    return out


def _subsequence(seq, wanted):
    """True when every item of ``wanted`` appears in ``seq``, in that order.

    A label matches an item that starts with it, so a panel header carrying
    "(changed)" still counts.
    """
    it = iter(seq)
    return all(any(str(s).startswith(w) for s in it) for w in wanted)


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _text_field(rc, label_key):
    label = t(label_key)
    matches = [f for f in rc.find(vw.TextField).widgets if f.label == label]
    return matches[0] if matches else None


def _sections(rc, text):
    """The collapsible panels whose header contains ``text``."""
    return [
        p
        for p in rc.find(vw.ExpansionPanels).widgets
        if text in str(p.children[0].children[0].children)
    ]


def _submit(rc):
    text = t("common.add")
    [b for b in rc.find(vw.Btn).widgets if text in str(b.children)][0].click()


def _alert_text(rc):
    return " ".join(str(a.children) for a in rc.find(vw.Alert).widgets)


def _year_labels(key):
    return [t(s["label_key"]) for s in param_specs(key) if s.get("type") == "year"]


# ---- Task 1: predefined form -------------------------------------------------


def test_predefined_fields_follow_the_mirror_order():
    """Read-only fields sit on the rows of their user-form twins."""
    rc = _render(_predefined("protected_area"))
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            t("vars.modal.source_label"),
            t("vars.modal.predefined_variable_label"),
            t("vars.modal.predefined_var_type_label"),
            t("vars.modal.param_pa_timing"),
            t("vars.modal.predefined_raster_type_label"),
            t("vars.modal.advanced_toggle"),
            t(TOGGLE),
        ],
    )


def test_layer_description_sits_behind_the_info_icon():
    """The layer blurb opens from the select's icon, no longer inline."""
    rc = _render(_predefined("protected_area"))
    sel = _select(rc, "vars.modal.predefined_variable_label")
    assert sel.prepend_inner_icon == "mdi-information-outline"
    assert "field-info-icon" in (sel.class_ or "")
    assert "click:prepend-inner" in sel._event_handlers_map
    entry = PREDEFINED_CATALOGUE["protected_area"]
    title = t("vars.modal.layer_info_title", label=t(entry["label_key"]))
    desc = t(entry["description_key"])[:40]
    dialogs = rc.find(vw.Dialog).widgets
    popups = [d for d in dialogs if title in str(d.children)]
    assert popups and desc in str(popups[0].children)
    # no longer inline in the form: the form dialog holds it only via the popup
    form = [d for d in dialogs if t("vars.modal.source_label") in _labels(d)][0]
    assert str(form.children).count(desc) == 1


def test_no_info_icon_without_a_layer():
    """Nothing picked yet: no icon, so no empty popup."""
    rc = _render()  # predefined source, nothing picked yet
    assert not _select(rc, "vars.modal.predefined_variable_label").prepend_inner_icon


def test_interval_years_come_before_the_read_only_raster_type():
    """Start/end years shown above the fields stay above Raster type."""
    rc = _render(_predefined("deforestation_gfc"))
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            t("vars.modal.predefined_var_type_label"),
            *_year_labels("deforestation_gfc"),
            t("vars.modal.predefined_raster_type_label"),
            t(TOGGLE),
        ],
    )


def test_logging_pair_follows_its_timing_switch():
    """The switch, then the start/end pair it reveals, then Raster type."""
    rc = _render(_predefined("logging_mapbiomas", {"timing": "period"}))
    years = _year_labels("logging_mapbiomas")
    assert years
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            t("vars.modal.param_logging_timing"),
            *years,
            t("vars.modal.predefined_raster_type_label"),
            t(TOGGLE),
        ],
    )


# ---- Task 2: user form -------------------------------------------------------


def _gee_layer():
    return {
        "source": "custom",
        "type": "GEEVar",
        "name": "roads",
        "asset_id": "projects/x/assets/roads",
        "raster_type": "continuous",
    }


@solara.component
def _NoAssetPicker(asset_id, on_asset_id):
    """Stand-in for the asset picker, which drives async GEE calls."""
    rv.Html(tag="div")


def test_user_fields_follow_the_mirror_order(monkeypatch):
    """Input before year, then raster type, then the Visualization panel."""
    monkeypatch.setattr(variable_modal, "_GeeAssetField", _NoAssetPicker)
    rc = _render(_gee_layer())
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            t("vars.modal.source_label"),
            t("vars.modal.custom_name_label"),
            t("vars.modal.custom_type_label"),
            t("vars.modal.custom_scale_label"),  # input block before the year
            t("vars.modal.custom_year_label"),
            t("vars.modal.custom_raster_type_label"),
            t(TOGGLE),
        ],
    )


def test_user_visualization_is_a_collapsed_panel():
    """User-layer style fields live in a collapsed panel, probes included."""
    rc = _render(
        {
            "source": "custom",
            "type": "LocalVectorVar",
            "name": "r",
            "path": "/r.gpkg",
            "rasterization_method": "unique",  # classes are found, not seeded
        }
    )
    panels = _sections(rc, t(TOGGLE))
    assert len(panels) == 1 and panels[0].v_model is None
    # built while closed: its probe button lives inside the panel
    assert t("vars.modal.find_classes") in _texts(panels[0])


def test_switching_source_keeps_the_panel_collapsed():
    """An open panel on one source does not open the other's."""
    rc = _render(_predefined("protected_area"))
    _sections(rc, t(TOGGLE))[0].v_model = 0
    _select(rc, "vars.modal.source_label").v_model = "custom"
    _select(rc, "vars.modal.custom_type_label").v_model = "LocalVectorVar"
    panels = _sections(rc, t(TOGGLE))
    assert panels and all(p.v_model is None for p in panels)


# ---- Task 3: a visualization error opens the panel ---------------------------


def _remove_all_classes(rc):
    while True:
        buttons = [
            b for b in rc.find(vw.Btn).widgets if "sr-class-remove" in (b.class_ or "")
        ]
        if not buttons:
            return
        buttons[0].click()


def test_a_range_error_opens_the_user_visualization_panel(monkeypatch):
    """Min above Max on submit shows the range fields."""
    monkeypatch.setattr(variable_modal, "_GeeAssetField", _NoAssetPicker)
    rc = _render(_gee_layer())
    _text_field(rc, "vars.modal.range_min").v_model = "5"
    _text_field(rc, "vars.modal.range_max").v_model = "1"
    _submit(rc)
    assert t("vars.modal.err_range") in _alert_text(rc)
    assert _sections(rc, t(TOGGLE))[0].v_model == 0


def test_a_class_error_opens_the_predefined_visualization_panel():
    """An empty class list on submit shows the class list."""
    rc = _render(_predefined("protected_area"))
    _remove_all_classes(rc)
    _submit(rc)
    assert t(vc.ERR_NO_CLASSES) in _alert_text(rc)
    assert _sections(rc, t(TOGGLE))[0].v_model == 0


# ---- Year placement: what, then when; the month reads after its year ---------

YEAR = t("vars.modal.predefined_year_label")
RASTER = t("vars.modal.predefined_raster_type_label")


def _param_label(key, param_key):
    spec = next(s for s in param_specs(key) if s["key"] == param_key)
    return t(spec["label_key"])


def test_only_the_month_period_goes_after_the_year():
    """The flag lives on the shared month param only."""
    for key in PREDEFINED_CATALOGUE:
        for spec in param_specs(key):
            assert bool(spec.get("after_year")) == (
                spec["label_key"] == "vars.modal.param_period"
            ), (key, spec["key"])


def test_month_period_follows_the_year():
    """Precipitation reads Year, then the month."""
    rc = _render(_predefined("precipitation"))
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            t("vars.modal.predefined_var_type_label"),
            YEAR,
            _param_label("precipitation", "period"),
            RASTER,
        ],
    )


def test_controllers_come_before_the_year():
    """Region and product set the year list, so they sit above it."""
    rc = _render(_predefined("mapbiomas_fire"))
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            _param_label("mapbiomas_fire", "region"),
            _param_label("mapbiomas_fire", "product"),
            YEAR,
            _param_label("mapbiomas_fire", "period"),
            RASTER,
        ],
    )


def test_timing_switch_comes_before_the_year_it_reveals():
    """Protected areas By year: the switch, then Year."""
    rc = _render(_predefined("protected_area", {"timing": "yearly"}))
    assert _subsequence(
        _labels_in_page_order(rc), [t("vars.modal.param_pa_timing"), YEAR, RASTER]
    )


def test_year_pair_follows_the_basic_params():
    """MapBiomas deforestation: Region, then start/end years."""
    rc = _render(_predefined("deforestation_mapbiomas"))
    assert _subsequence(
        _labels_in_page_order(rc),
        [
            _param_label("deforestation_mapbiomas", "region"),
            *_year_labels("deforestation_mapbiomas"),
            RASTER,
        ],
    )


def test_inactive_period_leaves_the_year_alone():
    """Fire history product: no Period, Year still before Raster type."""
    rc = _render(_predefined("mapbiomas_fire", {"product": "since"}))
    labels = _labels_in_page_order(rc)
    assert _param_label("mapbiomas_fire", "period") not in labels
    assert _subsequence(
        labels, [_param_label("mapbiomas_fire", "product"), YEAR, RASTER]
    )


def test_class_buttons_still_work_below_the_year():
    """Year now sits mid-form; the class list below it still responds."""
    rc = _render(_predefined("protected_area", {"timing": "yearly"}))
    year = _select(rc, "vars.modal.predefined_year_label")
    year.v_model = year.items[0]  # past the "Year is required" check
    _remove_all_classes(rc)
    _submit(rc)
    assert t(vc.ERR_NO_CLASSES) in _alert_text(rc)
