"""Project load/create/save/delete messages are toasts, not the status banner.

ProjectPanel needs DATA_DIR scanning and dialogs to render, so — like
test_delete_confirm — its wiring is asserted at the source level.
"""

import inspect


def test_project_panel_toasts_status_and_errors():
    """ProjectPanel reports load/create/save/delete outcomes via toasts."""
    import gui.solara_app as app

    src = inspect.getsource(app.ProjectPanel)
    assert "use_notifications()" in src
    assert "status_message" not in src
    assert "error_message" not in src
    for key in (
        "project.status_loaded",
        "project.status_deleted",
        "project.status_created",
        "project.status_saved",
        "project.error_no_project_to_save",
    ):
        assert key in src, f"{key} lost its call site"


def test_workflow_panel_has_no_status_clearing_effect():
    """With no status reactive there is nothing to clear on tab switch."""
    import gui.solara_app as app

    src = inspect.getsource(app)
    assert "_clear_status_on_tab_switch" not in src
