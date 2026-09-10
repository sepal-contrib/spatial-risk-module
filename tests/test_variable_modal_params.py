"""A catalogue param field shows exactly what the modal would submit.

The field used to fall back to the catalogue default for *display* while
``coerce_param_values`` read the raw form state with no such fallback. Whenever
the two disagreed the user saw ``30`` in the box and got "must be a whole number
between 1 and 100" on submit — an error with nothing to act on. What is shown is
what is submitted; blank stays blank and fails validation visibly.
"""

import inspect

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

import gui.widget.variable_modal as mod  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _threshold_field(initial_params):
    box, _rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            initial_entry={
                "source": "predefined",
                "predefined_key": "forest_gfc",
                "params": initial_params,
                "year": 2020,
            },
        )
    )
    label = t("vars.modal.param_tree_cover_threshold")
    return next(f for f in _find(box, vw.TextField) if f.label == label)


def test_param_field_renders_the_seeded_value():
    """The normal path is unaffected: seeded params still show their value."""
    assert _threshold_field({"tree_cover_threshold": 45}).v_model == "45"


def test_param_field_does_not_invent_a_value_the_state_lacks():
    """A params dict without this key leaves the field blank, not defaulted.

    Displaying the catalogue default here would promise a value that
    ``coerce_param_values`` never sees, producing an unactionable range error
    on submit.
    """
    assert _threshold_field({"some_other_param": 1}).v_model == ""


def test_param_edits_use_a_functional_state_update():
    """Two params edited in quick succession must not clobber each other.

    Closing over the render's ``params_raw`` writes a snapshot that may already
    be stale by the time the handler runs. (Source inspection: reacton gives no
    hook here to interleave two handler calls within one render.)
    """
    src = " ".join(inspect.getsource(mod._render_predefined_fields).split())
    assert "set_params_raw(" in src
    assert "lambda prev: {**prev," in src  # the update reads the LATEST state
    assert "{**params_raw" not in src  # never the render's snapshot


def _temperature_select(initial_params):
    box, _rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            initial_entry={
                "source": "predefined",
                "predefined_key": "temperature_2m",
                "params": initial_params,
                "year": 2020,
            },
        )
    )
    label = t("vars.modal.param_aggregation")
    return next(s for s in _find(box, vw.Select) if s.label == label)


def test_choice_param_renders_a_select_seeded_with_default():
    """No params in the entry -> prefill seeds the catalogue default (median)."""
    sel = _temperature_select(None)

    assert sel.v_model == "median"
    assert [item["value"] for item in sel.items] == ["mean", "max", "min", "median"]


def test_choice_param_prefills_the_saved_metric():
    """Editing temperature_2m_max reopens the dropdown on max."""
    assert _temperature_select({"aggregation": "max"}).v_model == "max"


def test_choice_param_does_not_render_a_text_field():
    """The metric must never be free-typed — only the dropdown."""
    box, _rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            initial_entry={
                "source": "predefined",
                "predefined_key": "temperature_2m",
                "params": None,
                "year": 2020,
            },
        )
    )
    label = t("vars.modal.param_aggregation")
    assert not [f for f in _find(box, vw.TextField) if f.label == label]


def test_choice_error_message_lists_the_options():
    """A choice spec has no min/max, so the range message cannot be used.

    Source inspection (the codebase's pattern for handler-closure behaviour):
    the submit path must branch on the spec type and format the choice key.
    """
    src = " ".join(inspect.getsource(mod.VariableModal).split())
    assert "error_param_choice" in src
    assert "error_param_range" in src  # int path must remain


def test_custom_gee_entry_submits_raster_type():
    """The raster-type select shown for GEE assets must land in the entry.

    It was rendered for ``GEEVar`` but only written for ``LocalRasterVar``, so
    ``to_local_raster`` later failed with "raster_type must be provided".
    """
    src = inspect.getsource(mod.VariableModal.f)
    gee_branch = src.split('elif var_type == "GEEVar":', 1)[1].split("elif", 1)[0]
    assert 'entry["raster_type"] = RasterType(raster_type)' in gee_branch
