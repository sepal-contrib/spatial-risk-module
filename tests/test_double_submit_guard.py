"""A slow UI let the Create button be clicked twice, launching two jobs.

Closing the dialog is a browser round-trip, so a second click queued behind
the first is processed after it — and nothing in the shared CreationDialog
stopped it. Training was the reported case: two workers trained the same
model, the second overwrote the first's files, and the two completed job
rows collapsed behind the single registered model. Every dialog built on
the frame had the same hole.

Two layers guard it now: the frame refuses a second launch per opening,
and the dialogs whose jobs register asynchronously reject the name of a
job that is still running (a re-opened dialog with the same name would
otherwise skip the overwrite confirm, because the first result is not
registered until its worker finishes).
"""

import types

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_model_form_dialog_render.py: warm the translator before the first
# render — the first t() *during* a render breaks reacton's widget map.
t("common.cancel")

from gui.widget.allocation_form import AllocationFormDialog  # noqa: E402
from gui.widget.creation_dialog import CreationDialog  # noqa: E402
from gui.widget.model_form_dialog import ModelFormDialog  # noqa: E402
from gui.widget.prediction_form_dialog import PredictionFormDialog  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402
from spatialrisk.project import Project  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _button(box, label):
    # Buttons carry a leading icon, so children are [Icon, label].
    return next(b for b in _find(box, vw.Btn) if label in str(b.children))


def _click(box, label, times=1):
    btn = _button(box, label)
    for _ in range(times):
        btn.fire_event("click", None)


def _alert_text(box):
    alerts = [a for a in _find(box, vw.Alert) if a.type == "error"]
    return alerts[0].children[0] if alerts else None


def _text_field(box, label):
    return next(f for f in _find(box, vw.TextField) if f.label == label)


def _select(box, label):
    return next(s for s in _find(box, vw.Select) if s.label == label)


# --- The shared frame -----------------------------------------------------


def _render_frame(open_, launches, will_replace=lambda: None):
    box, rc = reacton.render(
        CreationDialog(
            open_=open_,
            title="t",
            create_label="Create",
            validate=lambda: None,
            will_replace=will_replace,
            launch=lambda: launches.append(1),
        ),
        handle_error=False,
    )
    return box, rc


def test_double_click_on_create_launches_once():
    """The reported bug: two queued Create clicks must not launch twice."""
    launches = []
    open_ = solara.reactive(True)
    box, rc = _render_frame(open_, launches)
    try:
        _click(box, "Create", times=2)
        assert len(launches) == 1
        assert open_.value is False
    finally:
        rc.close()


def test_guard_resets_when_the_dialog_reopens():
    """One launch per opening — not one per dialog lifetime."""
    launches = []
    open_ = solara.reactive(True)
    box, rc = _render_frame(open_, launches)
    try:
        _click(box, "Create")
        open_.set(True)
        _click(box, "Create")
        assert len(launches) == 2
    finally:
        rc.close()


def test_double_click_on_replace_confirm_launches_once():
    """The confirm-replace button routes through the same guarded launch."""
    launches = []
    open_ = solara.reactive(True)
    box, rc = _render_frame(open_, launches, will_replace=lambda: "taken")
    try:
        _click(box, "Create")
        assert not launches  # the confirm step is in the way
        _click(box, t("common.replace"), times=2)
        assert len(launches) == 1
    finally:
        rc.close()


# --- Running-name validation ------------------------------------------------


def test_model_dialog_rejects_a_name_that_is_still_training():
    """A storage key held by a running training job is a hard refusal."""
    submitted = []
    project = solara.reactive(Project(project_name="p"))
    box, rc = reacton.render(
        ModelFormDialog(
            project=project,
            open_=solara.reactive(True),
            on_submit=submitted.append,
            running_keys=frozenset({"rf_twice"}),
        ),
        handle_error=False,
    )
    try:
        _select(box, t("tiles.train.model_select_label")).v_model = "rf"
        _text_field(box, t("tiles.train.model_name_label")).v_model = "twice"
        _click(box, t("tiles.train.train_button"))
        assert not submitted
        assert _alert_text(box) == t("tiles.train.error_name_running", name="twice")
    finally:
        rc.close()


def _prediction_project(tmp_path):
    dataset = types.SimpleNamespace(name="calibration", features=[])
    return types.SimpleNamespace(
        models={"glm_glm_v1": object()},
        datasets={"calibration": dataset},
        processed_variables={},
        predictions={},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def test_prediction_dialog_rejects_a_name_that_is_still_running(tmp_path):
    """A prediction name held by a running job is refused before model checks."""
    submitted = []
    box, rc = reacton.render(
        PredictionFormDialog(
            project=solara.reactive(_prediction_project(tmp_path)),
            open_=solara.reactive(True),
            on_submit=submitted.append,
            running_names=frozenset({"twice"}),
        ),
        handle_error=False,
    )
    try:
        _text_field(box, t("tiles.inference.pred_name_label")).v_model = "twice"
        _click(box, t("tiles.inference.run_button"))
        assert not submitted
        assert _alert_text(box) == t("tiles.inference.error_name_running", name="twice")
    finally:
        rc.close()


def _allocation_project(tmp_path):
    return types.SimpleNamespace(
        allocations={},
        predictions={},
        processed_variables={},
        models={},
        datasets={},
        filter_predictions=lambda **kw: [],
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def test_allocation_dialog_rejects_a_name_that_is_still_running(tmp_path):
    """Run keys never collide, so a running name must be refused outright."""
    launched = []
    box, rc = reacton.render(
        AllocationFormDialog(
            open_=solara.reactive(True),
            project=solara.reactive(_allocation_project(tmp_path)),
            on_launch=launched.append,
            on_close=lambda: None,
            running_names=frozenset({"twice"}),
            active_names=frozenset({"twice"}),
        ),
        handle_error=False,
    )
    try:
        _text_field(box, t("toolbox.allocation.field_name")).v_model = "twice"
        _click(box, t("toolbox.allocation.run"))
        assert not launched
        assert _alert_text(box) == t(
            "toolbox.allocation.error_name_running", name="twice"
        )
    finally:
        rc.close()


# --- The variable modal builds its own dialog ----------------------------


def test_variable_modal_double_submit_adds_once():
    """Pins the modal's own protection: submit resets the form before closing.

    The modal is not built on CreationDialog. It was already safe because
    reset() clears the selection before the second click's validation runs;
    this test keeps that from regressing if the submit order ever changes.
    """
    added = []
    box, rc = reacton.render(
        VariableModal(open_=solara.reactive(True), on_add=added.append),
        handle_error=False,
    )
    try:
        _select(box, t("vars.modal.predefined_variable_label")).v_model = "altitude"
        _click(box, t("vars.modal.submit_add"), times=2)
        assert len(added) == 1
    finally:
        rc.close()
