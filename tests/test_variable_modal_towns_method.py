"""The Add Variable modal shows only the towns knobs of the chosen method.

Editing a non-default towns layer opens Advanced automatically (Part A);
depends_on (Part D) hides the other method's fields.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal  # noqa: E402

SMOD_DEFAULT = ["centre", "dense", "semidense", "suburban"]


def _fields(params, editing_key):
    """``{label: v_model}`` of every text field and select the modal renders."""
    box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            editing_key=editing_key,
            initial_entry={
                "source": "predefined",
                "predefined_key": "towns",
                "params": params,
                "year": 2020,
            },
        ),
        handle_error=False,
    )
    try:
        widgets = rc.find(vw.TextField).widgets + rc.find(vw.Select).widgets
        return {w.label: w.v_model for w in widgets}
    finally:
        rc.close()


def test_editing_a_smod_layer_shows_classes_not_thresholds():
    """towns_smod_ucd reopens with the class picker and no threshold boxes."""
    fields = _fields(
        {
            "method": "smod",
            "population_threshold": 15,
            "built_threshold": 90,
            "smod_classes": ["centre", "dense"],
        },
        "towns_smod_ucd_2020",
    )
    assert fields[t("vars.modal.param_towns_method")] == "smod"
    assert fields[t("vars.modal.param_towns_smod_classes")] == ["centre", "dense"]
    assert t("vars.modal.param_towns_population") not in fields
    assert t("vars.modal.param_towns_built") not in fields


def test_editing_a_ghsl_layer_shows_thresholds_not_classes():
    """towns_p50 reopens with the threshold boxes and no class picker."""
    fields = _fields(
        {
            "method": "ghsl",
            "population_threshold": 50,
            "built_threshold": 90,
            "smod_classes": SMOD_DEFAULT,
        },
        "towns_p50_2020",
    )
    assert fields[t("vars.modal.param_towns_population")] == "50"
    assert fields[t("vars.modal.param_towns_built")] == "90"
    assert t("vars.modal.param_towns_smod_classes") not in fields
