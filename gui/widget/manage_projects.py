"""Manage-projects dialog: search, load and delete saved projects.

Both components are presentational — every action is a callback supplied by
``ProjectPanel``, which owns the app state. The confirm dialog is rendered once
at the panel's top level (never inside a list row): nested button -> Dialog
toggles have proved unreliable here, the same constraint
``gui/widget/confirm_dialog.py`` documents.

The shared ``ConfirmDialog`` is not reused: six tiles depend on it and it has no
text field, and deleting a project — irreversible, up to several GB — is gated on
typing the project's name.
"""

from datetime import datetime

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.project_ui_helpers import (
    delete_confirm_valid,
    filter_project_infos,
    format_relative,
    format_size,
    project_count_chips,
)
from gui.widget.enter_key import ENTER_DEFAULT


@solara.component
def ManageProjectsDialog(
    open,
    infos,
    selected,
    on_select,
    on_load,
    on_delete,
    on_cancel,
    current=None,
    busy=False,
    error=None,
):
    """List / search / load / delete saved projects.

    Args:
        open: bool — whether the dialog is shown.
        infos: list[ProjectInfo] — every saved project (already scanned).
        selected: str | None — name of the selected project.
        current: str | None — name of the project already open in the app, badged
            in the list. Distinct from `selected`, which is the row the user has
            clicked *in this dialog*: the two are highlighted by different things
            (chip vs ListItemGroup) because they answer different questions.
            An unsaved new project has no row here, so nothing is badged.
        on_select: callback(name) — a row was clicked.
        on_load: the footer's Load action.
        on_cancel: the footer's Cancel action, also run on ESC / an outside click.
        on_delete: callback(ProjectInfo) — the row's trash button; the caller
            opens the confirm dialog. This widget never touches the disk.
        busy: a load is in progress.
        error: the load's error text, if any.
    """
    query, set_query = solara.use_state("")

    # A stale filter would hide rows the next time the dialog opens.
    def _reset_query():
        if open:
            set_query("")

    solara.use_effect(_reset_query, [open])

    shown = filter_project_infos(infos, query)
    now = datetime.now()

    # One expression, three ways Load must not fire: nothing selected; the
    # selection scrolled out of the current search filter (Load would open a
    # project the user cannot see); the selection is corrupt.
    sel = next((i for i in shown if i.name == selected), None)
    load_disabled = busy or sel is None or not sel.readable

    with rv.Dialog(
        v_model=open,
        on_v_model=lambda v: None if v else on_cancel(),
        max_width="480px",
        eager=True,
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("project.dialog_load_title"))
            with rv.CardText():
                if infos:
                    rv.TextField(
                        label=t("project.dialog_manage_search"),
                        v_model=query,
                        on_v_model=set_query,
                        prepend_inner_icon="mdi-magnify",
                        dense=True,
                        outlined=True,
                        clearable=True,
                        hide_details=True,
                    )
                if not infos:
                    solara.Info(t("project.dialog_load_empty"))
                elif not shown:
                    solara.Info(t("project.dialog_manage_no_match", query=query))
                else:
                    with rv.List(three_line=True):
                        with rv.ListItemGroup(v_model=selected, on_v_model=on_select):
                            for info in shown:
                                # Deliberately NOT disabled for unreadable rows: a
                                # disabled Vuetify list item suppresses interaction
                                # on its children, which would kill the trash button
                                # on exactly the corrupt projects this feature is
                                # for. Load is gated instead (load_disabled above).
                                with rv.ListItem(value=info.name):
                                    with rv.ListItemContent():
                                        # Outlined, so the badge cannot be mistaken
                                        # for one of the filled count chips below it.
                                        title = [info.name]
                                        if current and info.name == current:
                                            title.append(
                                                rv.Chip(
                                                    children=[t("project.chip_open")],
                                                    x_small=True,
                                                    outlined=True,
                                                    color="primary",
                                                    class_="ml-2",
                                                )
                                            )
                                        rv.ListItemTitle(children=title)
                                        if info.readable:
                                            with rv.Row(
                                                style_="flex-wrap: wrap; gap: 4px; "
                                                "margin: 2px 0;"
                                            ):
                                                for chip in project_count_chips(info):
                                                    rv.Chip(
                                                        children=[chip.label],
                                                        x_small=True,
                                                        color="primary"
                                                        if chip.accent
                                                        else None,
                                                        text_color="white"
                                                        if chip.accent
                                                        else None,
                                                    )
                                            rv.ListItemSubtitle(
                                                children=[
                                                    t(
                                                        "project.dialog_load_modified",
                                                        time_ago=format_relative(
                                                            info.modified, now
                                                        ),
                                                    )
                                                ]
                                            )
                                        else:
                                            rv.ListItemSubtitle(
                                                children=[
                                                    info.error
                                                    or t(
                                                        "project.dialog_load_unreadable"
                                                    )
                                                ]
                                            )
                                    with rv.ListItemAction():
                                        solara.Button(
                                            icon_name="mdi-delete-outline",
                                            icon=True,
                                            small=True,
                                            color="error",
                                            on_click=lambda info=info: on_delete(info),
                                        )
                if busy:
                    rv.ProgressLinear(indeterminate=True)
                if error:
                    solara.Error(error)
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"), on_click=on_cancel, text=True, small=True
                )
                solara.Button(
                    t("common.load"),
                    on_click=on_load,
                    color="primary",
                    small=True,
                    disabled=load_disabled,
                    # Enter presses it — a disabled button ignores the press.
                    classes=[ENTER_DEFAULT],
                )


@solara.component
def ConfirmDeleteProjectDialog(
    open,
    name,
    size_bytes,
    on_cancel,
    on_confirm,
    is_open_project=False,
    writer_active=False,
    busy=False,
    error=None,
):
    """Type-the-name confirmation for deleting a project's whole folder.

    Args:
        open: bool — whether the dialog is shown.
        name: str — the project being deleted (also what must be typed).
        size_bytes: int — folder size, so the user sees what they are losing.
        on_cancel: callback — dismissed. The caller closes the dialog.
        on_confirm: callback — delete. The caller closes the dialog.
        is_open_project: the target is the project currently open — it will close.
        writer_active: a background task is still writing into this project, so
            delete is refused: its auto-save would re-create the folder.
        busy: a delete is in progress.
        error: the delete's failure text, if any.
    """
    typed, set_typed = solara.use_state("")

    # Never carry a previous target's typed name into the next confirmation.
    def _reset_typed():
        if open:
            set_typed("")

    solara.use_effect(_reset_typed, [open, name])

    can_delete = (
        delete_confirm_valid(typed, name or "") and not writer_active and not busy
    )

    def confirm_if_typed():
        # The disabled button is the browser's copy of this gate; a press sent
        # before the last keystroke reached the kernel still lands here.
        if can_delete:
            on_confirm()

    with rv.Dialog(
        v_model=open,
        on_v_model=lambda v: None if v else on_cancel(),
        max_width="420px",
        eager=True,
        # An rmtree cannot be called back once it starts, so while it runs the
        # dialog must not pretend otherwise. ESC / a scrim click would dismiss it
        # and hand back a live Manage dialog (Load enabled) over a folder that is
        # being erased; the delete would land anyway, seconds later, on whatever
        # the user had moved on to. `persistent` blocks both while busy.
        persistent=busy,
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("project.dialog_delete_title"))
            with rv.CardText():
                solara.Text(
                    t(
                        "project.dialog_delete_message",
                        name=name or "",
                        size=format_size(size_bytes or 0),
                    )
                )
                if writer_active:
                    solara.Error(t("project.dialog_delete_busy"))
                elif is_open_project:
                    solara.Warning(t("project.dialog_delete_open_warning"))
                if not writer_active:
                    rv.TextField(
                        label=t("project.dialog_delete_type_label", name=name or ""),
                        v_model=typed,
                        on_v_model=set_typed,
                        dense=True,
                        outlined=True,
                        autofocus=True,
                        hide_details=True,
                        # Vuetify spacing helper (24px, !important) — an inline
                        # margin would lose to the card text's own utilities.
                        # The field is outlined, so its floating label sits on
                        # the top border and needs clear air under the warning.
                        class_="mt-6",
                    )
                if busy:
                    rv.ProgressLinear(indeterminate=True)
                if error:
                    solara.Error(error)
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"),
                    on_click=on_cancel,
                    text=True,
                    small=True,
                    # Cancel cannot stop the rmtree, so it must not offer to: while
                    # busy it would only close the dialog over a live delete.
                    disabled=busy,
                )
                solara.Button(
                    t("project.dialog_delete_confirm"),
                    on_click=confirm_if_typed,
                    color="error",
                    small=True,
                    disabled=not can_delete,
                    # Enter in the name field deletes — but only once the name
                    # is typed: until then the button is disabled, and a
                    # disabled button ignores the press.
                    classes=[ENTER_DEFAULT],
                )
