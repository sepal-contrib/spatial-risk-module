"""A year left over from another layer or method never reaches a saved layer.

Hansen GFC starts in 2001 and GHSL towns stop at 2020, so a 1985 kept from
MapBiomas or a 2025 kept from SMOD would save a layer that does not exist.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal  # noqa: E402

LAYER = "vars.modal.predefined_variable_label"
YEAR = "vars.modal.predefined_year_label"


def _render(added):
    _box, rc = reacton.render(
        VariableModal(open_=solara.reactive(True), on_add=added.append),
        handle_error=False,
    )
    return rc


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _submit(rc):
    text = t("common.add")
    [b for b in rc.find(vw.Btn).widgets if text in str(b.children)][0].click()


def _sections(rc, text):
    """The collapsible panels whose header contains ``text``."""
    return [
        p
        for p in rc.find(vw.ExpansionPanels).widgets
        if text in str(p.children[0].children[0].children)
    ]


def test_switching_layer_replaces_a_year_the_new_layer_lacks():
    """1985 picked on another layer becomes forest_gfc's newest year (#35)."""
    rc = _render([])
    try:
        _select(rc, LAYER).v_model = "precipitation"
        _select(rc, YEAR).v_model = 1985
        _select(rc, LAYER).v_model = "forest_gfc"
        assert _select(rc, YEAR).v_model == max(_select(rc, YEAR).items)
    finally:
        rc.close()


def test_year_selects_list_newest_first_and_default_to_the_newest():
    """#35: the most recent data is at the top and picked by default."""
    rc = _render([])
    try:
        _select(rc, LAYER).v_model = "forest_gfc"
        year = _select(rc, YEAR)
        assert year.items == sorted(year.items, reverse=True)
        assert year.v_model == year.items[0] == 2026
        # 2026 is a 1 January state, not a year of data the source lacks.
        assert year.hint == t("vars.modal.year_hint_forest", year=2026, previous=2025)
        _select(rc, LAYER).v_model = "deforestation_gfc"
        start = _select(rc, "vars.modal.param_start_year")
        assert start.items == sorted(start.items, reverse=True)
        assert start.v_model is None  # a period is always an explicit choice
    finally:
        rc.close()


def _custom_year(rc):
    label = t("vars.modal.custom_year_label")
    return [f for f in rc.find(vw.TextField).widgets if f.label == label][0]


def test_the_predefined_year_never_lands_in_the_custom_year_field():
    """Each source keeps its own year."""
    rc = _render([])
    try:
        _select(rc, LAYER).v_model = "forest_gfc"
        _select(rc, "vars.modal.source_label").v_model = "custom"
        assert _custom_year(rc).v_model == ""
        _custom_year(rc).v_model = "2015"
        _select(rc, "vars.modal.source_label").v_model = "predefined"
        assert _select(rc, YEAR).v_model == 2026
        _select(rc, "vars.modal.source_label").v_model = "custom"
        assert _custom_year(rc).v_model == "2015"
    finally:
        rc.close()


def test_switching_layer_keeps_a_year_the_new_layer_offers():
    """2020 is offered by both layers, so it stays."""
    rc = _render([])
    try:
        _select(rc, LAYER).v_model = "precipitation"
        _select(rc, YEAR).v_model = 2020
        _select(rc, LAYER).v_model = "forest_gfc"
        assert _select(rc, YEAR).v_model == 2020
    finally:
        rc.close()


def test_a_year_the_method_does_not_offer_is_refused():
    """towns: 2025 from SMOD, method back to GHSL (to 2020) cannot be saved."""
    added = []
    rc = _render(added)
    try:
        _select(rc, LAYER).v_model = "towns"
        _sections(rc, t("vars.modal.advanced_toggle"))[0].v_model = 0
        _select(rc, "vars.modal.param_towns_method").v_model = "smod"
        _select(rc, YEAR).v_model = 2025
        _select(rc, "vars.modal.param_towns_method").v_model = "ghsl"
        _submit(rc)
        assert added == []
        alerts = [str(a.children) for a in rc.find(vw.Alert).widgets]
        assert any(t(YEAR) in a for a in alerts), alerts
    finally:
        rc.close()
