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


def test_switching_layer_clears_a_year_the_new_layer_lacks():
    """1985 picked on another layer is cleared on forest_gfc (from 2001)."""
    added = []
    rc = _render(added)
    try:
        _select(rc, LAYER).v_model = "precipitation"
        _select(rc, YEAR).v_model = 1985
        _select(rc, LAYER).v_model = "forest_gfc"
        assert _select(rc, YEAR).v_model is None
        _submit(rc)
        assert added == []
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
