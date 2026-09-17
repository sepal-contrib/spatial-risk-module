"""Destructive deletes route through a confirm dialog instead of deleting on click.

Variables and datasets are removed from the project (data loss), so their list
"remove" buttons must open a ConfirmDialog rather than delete immediately. These
guard the wiring so it cannot be silently dropped.
"""

import inspect


def test_variables_tile_confirms_remove():
    """Removing a source variable goes through the confirm dialog."""
    from gui.tile.variables_tile import VariablesTile

    src = inspect.getsource(VariablesTile)
    assert "ConfirmDialog" in src
    # The list's remove button opens the dialog rather than deleting directly.
    assert "on_remove=set_pending_remove" in src


def test_dataset_tile_confirms_remove():
    """Removing a dataset goes through the confirm dialog."""
    from gui.tile.dataset_tile import DatasetTile

    src = inspect.getsource(DatasetTile)
    assert "ConfirmDialog" in src
    assert "on_remove=set_pending_remove" in src


def test_train_tile_confirms_and_really_deletes_model():
    """A confirmed model delete unregisters the model."""
    from gui.tile.train_tile import TrainTile

    src = inspect.getsource(TrainTile)
    assert "ConfirmDialog" in src
    assert "delete_model" in src  # removal actually unregisters the model
    # Model deletion (not job dismissal) drives the confirm dialog — see
    # docs/spec: completed runs render as model rows and "delete" on a model
    # row deletes the model.
    assert "on_delete=set_pending_delete" in src


def test_inference_tile_confirms_and_really_deletes_predictions():
    """A confirmed prediction delete unregisters the predictions."""
    from gui.tile.inference_tile import InferenceTile

    src = inspect.getsource(InferenceTile)
    assert "ConfirmDialog" in src
    assert "delete_prediction" in src  # removal actually unregisters the predictions
    assert "on_delete=set_pending_delete" in src


def test_manage_projects_widget_confirms_delete():
    """The Manage dialog hands the target up; it never deletes."""
    from gui.widget.manage_projects import (
        ConfirmDeleteProjectDialog,
        ManageProjectsDialog,
    )

    assert callable(ManageProjectsDialog) and callable(ConfirmDeleteProjectDialog)

    src = inspect.getsource(ManageProjectsDialog)
    assert "on_delete(" in src  # the row hands the target up; it never deletes
    assert "delete_project" not in src  # the widget must not touch the disk
    assert "rv.Btn(" not in src  # rv.Btn silently drops clicks in this codebase
    # A disabled rv.ListItem suppresses its children, which would make the trash
    # button dead on exactly the corrupt projects we most need to remove.
    assert "disabled=not info.readable" not in src

    confirm = inspect.getsource(ConfirmDeleteProjectDialog)
    assert "delete_confirm_valid" in confirm  # type-the-name gating
    assert "writer_active" in confirm  # refused while a task is writing


def test_project_panel_confirms_and_really_deletes_projects():
    """The panel confirms, then really removes the folder."""
    import gui.solara_app as app

    src = inspect.getsource(app.ProjectPanel)
    assert "ConfirmDeleteProjectDialog" in src
    assert "delete_project" in src  # removal actually hits the disk
    assert "on_delete=open_delete" in src  # the trash button opens the confirm dialog
    assert "close_project_state" in src  # deleting the open project closes it
    assert "is_writing" in src  # refused while a task is writing to it


def test_project_panel_delete_is_not_re_entrant_and_owns_its_error():
    """The two ways the confirm button can go wrong on an uncancellable rmtree.

    The button's `disabled` only reaches the browser a round-trip later, so a real
    double-click invokes the task twice — and TaskAsyncio.__call__ cancels the
    in-flight one, which does not stop the rmtree but does skip its continuation.
    The confirm button must therefore go through the `pending` guard, never straight
    to the task. And a Task keeps `.error`/`.exception` until the next invoke, so the
    failure text must be owned by the panel or it leaks into the next target's dialog.
    """
    import gui.solara_app as app

    src = inspect.getsource(app.ProjectPanel)
    assert "on_confirm=confirm_delete" in src  # not on_confirm=delete_task
    assert "if delete_task.pending:" in src  # the synchronous re-entrancy guard
    assert "error=delete_error.value" in src  # the panel owns the message …
    assert "delete_task.exception" not in src  # … and never the task's sticky one


def test_project_panel_guards_stage_load_and_save_against_a_delete_in_flight():
    """A delete in flight must also close Load, Save and staging a new target.

    The rmtree behind a confirmed delete cannot be called back either — and it
    is not only the confirm button that can re-enter it. Staging a new target
    (open_delete), loading a different project (do_load), or saving the one
    already open (do_save) are each gated ONLY by a render-time `disabled=`/
    `busy=` prop elsewhere in this file, which reaches the browser a round-trip
    after delete_task starts — the same window that makes the confirm button's
    own `disabled` insufficient. Missing this on do_load/do_save is the more
    dangerous half: Load succeeds while a delete for that same project is still
    running (its manifest dies last, so it is still listable), and a subsequent
    Save does mkdir(parents=True, exist_ok=True) and writes the manifest straight
    back into the folder the in-flight rmtree is erasing — a manifest-only
    zombie project, the same harm confirm_delete's guard exists to prevent,
    arriving through a different door.
    """
    import gui.solara_app as app

    src = inspect.getsource(app.ProjectPanel)
    lines = src.splitlines()

    def guard_follows(signature: str, within: int = 4) -> bool:
        """Is `signature`'s def line followed by the synchronous pending guard?

        Within a few lines, to allow for a leading docstring.
        """
        for i, line in enumerate(lines):
            if signature in line:
                window = "\n".join(lines[i + 1 : i + 1 + within])
                return "if delete_task.pending:" in window
        return False

    for signature in (
        "def open_delete(info):",
        "def do_load():",
        "def do_save():",
        "def _really_save():",
    ):
        assert guard_follows(signature), (
            f"{signature} must bail out on delete_task.pending "
            "before doing anything else"
        )

    # Cosmetic, but the UI should reflect it too — the handler guards above are
    # what actually hold this closed.
    # Manage dialog's Load button, then the panel's own Save button.
    assert "busy=load_busy or delete_task.pending" in src
    assert "disabled=delete_task.pending" in src
