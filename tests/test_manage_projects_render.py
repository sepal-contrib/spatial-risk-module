"""The manage-projects widgets actually mount — and every row deletes ITS OWN row.

The other manage-projects tests are source-substring checks. ``assert "on_delete("
in src`` passes just as happily against a late-binding closure (``lambda:
on_delete(info)``, no ``info=info`` default), where *every* row's trash button
hands up the LAST project in the list — i.e. it offers to delete GUY and erases
someone else's folder instead. On an irreversible, multi-GB delete that is the
worst bug in the feature, and nothing but a render can catch it: these components
are only exercised here.

Row order is the widget-tree order, so the trash buttons come back in list order —
the broken closure yields ``["broken", "broken"]`` where the fixed one yields
``["GUY", "broken"]``.
"""

from datetime import datetime, timedelta

import ipyvuetify as vw
import reacton

from gui.i18n import t

# Warm the translator BEFORE the first render. A first t() *during* the first
# render is a known way to corrupt reacton's widget map (it blows up use_event, and
# only in isolated runs). These components happen not to trip it today; the warm-up
# is a free guarantee that a future one cannot either.
t("common.cancel")

from gui.scripts.project_io import ProjectInfo  # noqa: E402
from gui.widget.manage_projects import (  # noqa: E402
    ConfirmDeleteProjectDialog,
    ManageProjectsDialog,
)


def _infos():
    """One healthy project and one corrupt one.

    The corrupt row must render (and stay deletable) — cleaning it up is the whole
    point of the feature. ``modified`` is a real datetime whenever ``readable`` is
    True: that is ``list_project_infos``'s invariant, and ``format_relative`` needs it.
    """
    return [
        ProjectInfo(
            name="GUY",
            raw_count=6,
            processed_count=6,
            model_count=3,
            modified=datetime.now() - timedelta(hours=5),
            readable=True,
            trained_model_count=1,
            prediction_count=2,
        ),
        ProjectInfo(
            name="broken",
            raw_count=0,
            processed_count=0,
            model_count=0,
            modified=None,
            readable=False,
            error="unreadable project file",
        ),
    ]


def _render_manage(on_delete=lambda info: None, **kwargs):
    el = ManageProjectsDialog(
        open=True,
        infos=kwargs.pop("infos", _infos()),
        selected=kwargs.pop("selected", "GUY"),
        on_select=lambda name: None,
        on_load=lambda: None,
        on_delete=on_delete,
        on_cancel=lambda: None,
        **kwargs,
    )
    return reacton.render(el, handle_error=False)


def _render_confirm(**kwargs):
    el = ConfirmDeleteProjectDialog(
        open=True,
        name=kwargs.pop("name", "GUY"),
        size_bytes=kwargs.pop("size_bytes", 3_400_000_000),
        on_cancel=lambda: None,
        on_confirm=lambda: None,
        **kwargs,
    )
    return reacton.render(el, handle_error=False)


def _trash_buttons(rc):
    """The rendered row trash buttons, in row order."""
    return [
        btn
        for btn in rc.find(vw.Btn).widgets
        if any(
            isinstance(child, vw.Icon) and child.children == ["mdi-delete-outline"]
            for child in btn.children
        )
    ]


def _button(rc, label):
    """The rendered button carrying this exact label."""
    return next(btn for btn in rc.find(vw.Btn).widgets if btn.children == [label])


def _open_chips_by_row(rc):
    """{row name: [open-chip labels in that row]} — which row wears the badge.

    Asserting only that *a* chip exists would pass with the badge on the wrong
    project, which is exactly the failure that matters: it would tell the user
    the project they are about to load is already the open one.
    """
    label = t("project.chip_open")
    rows = {}
    for item in rc.find(vw.ListItem).widgets:
        title = next(
            w for w in item.children[0].children if isinstance(w, vw.ListItemTitle)
        )
        name = next(c for c in title.children if isinstance(c, str))
        rows[name] = [
            c.children[0]
            for c in title.children
            if isinstance(c, vw.Chip) and c.children == [label]
        ]
    return rows


# --- the row -> target binding (the bug the substring tests cannot see) --------


def test_each_row_trash_button_deletes_its_own_project():
    """Each row trash button deletes its own project."""
    deleted = []
    box, rc = _render_manage(on_delete=lambda info: deleted.append(info.name))

    buttons = _trash_buttons(rc)
    assert len(buttons) == 2  # one per row, corrupt project included
    for btn in buttons:
        btn.fire_event("click", None)

    # Each button captured its own row. A late-binding closure gives ["broken",
    # "broken"] here — every row deleting the last project in the list.
    assert deleted == ["GUY", "broken"]
    rc.close()


def test_row_hands_up_the_whole_info_not_just_a_name():
    """The row must pass the record itself, not the name it happens to display.

    open_delete() prices the folder off the ProjectInfo.
    """
    captured = []
    box, rc = _render_manage(on_delete=captured.append)

    _trash_buttons(rc)[0].fire_event("click", None)

    assert isinstance(captured[0], ProjectInfo) and captured[0].name == "GUY"
    rc.close()


# --- "this one is already open" -----------------------------------------------


def test_the_open_project_row_wears_the_open_chip():
    """The open project row wears the open chip."""
    box, rc = _render_manage(current="GUY")

    assert _open_chips_by_row(rc) == {"GUY": [t("project.chip_open")], "broken": []}
    rc.close()


def test_no_row_is_badged_when_no_project_is_open():
    """The panel's empty state can still reach Manage, so `current` is None there."""
    box, rc = _render_manage(current=None)

    assert _open_chips_by_row(rc) == {"GUY": [], "broken": []}
    rc.close()


def test_an_open_project_that_was_never_saved_badges_nothing():
    """An unsaved project has no row to badge, nor a same-named stranger's.

    `current` is the in-memory project name; until its first save it has no row
    on disk.
    """
    box, rc = _render_manage(current="unsaved-draft")

    assert _open_chips_by_row(rc) == {"GUY": [], "broken": []}
    rc.close()


# --- mount smoke: hook order + the states nothing else exercises ---------------


def test_manage_dialog_mounts_with_a_corrupt_row():
    """The unreadable row renders its error instead of counts, and keeps its trash.

    A disabled ListItem would kill the button on exactly the projects to remove.
    """
    box, rc = _render_manage()

    subtitles = [str(w.children) for w in rc.find(vw.ListItemSubtitle).widgets]
    assert any("unreadable project file" in s for s in subtitles)
    assert len(_trash_buttons(rc)) == 2
    rc.close()


def test_manage_dialog_mounts_with_no_projects():
    """Manage dialog mounts with no projects."""
    box, rc = _render_manage(infos=[], selected=None)
    assert _trash_buttons(rc) == []
    assert _button(rc, t("common.load")).disabled  # nothing to load
    rc.close()


def test_load_is_disabled_for_an_unreadable_selection():
    """Load is disabled for an unreadable selection."""
    box, rc = _render_manage(selected="broken")
    assert _button(rc, t("common.load")).disabled
    rc.close()


def test_confirm_dialog_mounts_in_every_state():
    """Confirm dialog mounts in every state."""
    for kwargs in (
        {},
        {"is_open_project": True},
        {"writer_active": True},
        {"busy": True},
        {"error": "Permission denied"},
    ):
        box, rc = _render_confirm(**kwargs)
        assert _button(rc, t("project.dialog_delete_confirm")) is not None
        rc.close()


def test_confirm_dialog_refuses_while_a_writer_holds_the_project():
    """Confirm dialog refuses while a writer holds the project."""
    box, rc = _render_confirm(writer_active=True)

    assert _button(rc, t("project.dialog_delete_confirm")).disabled
    # No type-the-name field either: there is nothing the user can type to proceed.
    assert rc.find(vw.TextField).widgets == []
    assert any(a.type == "error" for a in rc.find(vw.Alert).widgets)
    rc.close()


# --- an rmtree in flight cannot be called back --------------------------------


def test_a_delete_in_flight_cannot_be_dismissed():
    """While the rmtree runs, the dialog must not hand the app back.

    Cancel and ESC/scrim-click do not stop the delete — it lands seconds later
    regardless — so offering them just reveals a live Manage dialog (Load enabled)
    over a folder that is being erased, and the delete's continuation then closes
    whatever project the user opened in the meantime.
    """
    box, rc = _render_confirm(busy=True)

    assert rc.find(vw.Dialog).widget.persistent is True  # ESC / scrim click blocked
    assert _button(rc, t("common.cancel")).disabled  # and Cancel is not a way out
    assert _button(rc, t("project.dialog_delete_confirm")).disabled  # no re-entry
    rc.close()


def test_an_idle_confirm_dialog_stays_dismissable():
    """An idle confirm dialog stays dismissable."""
    box, rc = _render_confirm()

    assert not rc.find(vw.Dialog).widget.persistent
    assert not _button(rc, t("common.cancel")).disabled
    rc.close()


# --- rename pencil + folder sizes ----------------------------------------------


def _icon_buttons(rc, icon):
    return [
        btn
        for btn in rc.find(vw.Btn).widgets
        if any(isinstance(c, vw.Icon) and c.children == [icon] for c in btn.children)
    ]


def test_each_row_pencil_renames_its_own_project():
    """Each pencil hands up its own row's name, not the last row's."""
    renamed = []
    box, rc = _render_manage(on_rename=renamed.append)
    for btn in _icon_buttons(rc, "mdi-pencil-outline"):
        btn.fire_event("click", None)
    assert renamed == ["GUY", "broken"]  # not the last row twice
    rc.close()


def test_no_pencil_without_a_rename_callback():
    """Rename is opt-in: no callback, no pencil."""
    box, rc = _render_manage()
    assert _icon_buttons(rc, "mdi-pencil-outline") == []
    rc.close()


def test_rows_show_their_size_or_a_placeholder_while_counting():
    """A counted folder shows its size; one still being counted shows an ellipsis."""
    box, rc = _render_manage(sizes={"GUY": 3 * 1024**3})
    texts = [
        w.children[0]
        for w in rc.find(vw.Html).widgets
        if (w.attributes or {}).get("title") == t("project.size_on_disk")
    ]
    assert texts == ["3.0 GB", "…"]
    rc.close()
