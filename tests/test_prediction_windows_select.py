"""The Predict dialog offers a per-window choice for MW models only.

An MW run fans out one risk map per trained window size; the field lets the
user narrow that fan-out (e.g. re-run only the window that won evaluation).
Every other family resolves no windows at all and must never see the field.
"""

import types

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.widget.prediction_form_dialog import PredictionFormDialog  # noqa: E402


def _find(widget, cls, out=None):
    """Collect every widget of type *cls* in the rendered tree."""
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _texts(widget, out=None):
    """Collect every string leaf in the rendered tree."""
    out = [] if out is None else out
    for child in getattr(widget, "children", []) or []:
        if isinstance(child, str):
            out.append(child)
        else:
            _texts(child, out)
    return out


def _project(tmp_path, predictions=None):
    """Fake project with one MW model (two trained windows) and one ML model."""
    mw = types.SimpleNamespace(
        ldefrate_files={"5": "a.tif", "11": "b.tif"}, win_size_list=[5, 11]
    )
    dataset = types.SimpleNamespace(name="validation", features=[])
    return types.SimpleNamespace(
        models={"mw_calibration_mw": mw, "glm_glm_v1": object()},
        datasets={"validation": dataset},
        processed_variables={},
        predictions=predictions if predictions is not None else {},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def _render(tmp_path, model_key, predictions=None):
    """Render the dialog open and seeded, through the prefill path."""
    project = solara.reactive(_project(tmp_path, predictions))
    open_ = solara.reactive(True)
    prefill = solara.reactive(
        {
            "kind": "model",
            "model_key": model_key,
            "dataset_key": "validation",
            "name": "val_2020",
        }
    )
    box, rc = reacton.render(
        PredictionFormDialog(project, open_, on_submit=lambda e: None, prefill=prefill)
    )
    return box, rc


def _window_selects(box):
    """Every Select carrying the window-sizes label."""
    label = t("tiles.inference.windows_label")
    return [s for s in _find(box, vw.Select) if s.label == label]


def _create(box):
    """Click the dialog's Create button."""
    label = t("common.add")
    btn = next(b for b in _find(box, vw.Btn) if label in str(b.children))
    btn.fire_event("click", {})


def test_mw_model_shows_the_window_select_with_all_selected(tmp_path):
    """An MW model reveals the field, seeded with every trained window."""
    box, _ = _render(tmp_path, "mw_calibration_mw")
    (select,) = _window_selects(box)
    assert select.multiple
    assert sorted(select.v_model) == [5, 11]


def test_ml_model_shows_no_window_select(tmp_path):
    """Non-MW families resolve no windows — the field would be meaningless."""
    box, _ = _render(tmp_path, "glm_glm_v1")
    assert _window_selects(box) == []


def test_replace_confirmation_names_every_colliding_window_key(tmp_path):
    """One run can overwrite several ``_w<size>`` maps — list them all."""
    box, _ = _render(
        tmp_path,
        "mw_calibration_mw",
        predictions={"val_2020_w5": object(), "val_2020_w11": object()},
    )
    _create(box)
    assert any("val_2020_w5, val_2020_w11" in s for s in _texts(box))
