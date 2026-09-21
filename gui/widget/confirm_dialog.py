"""Reusable confirmation dialog for destructive actions.

Follows the proven ProjectPanel discard/overwrite pattern: a single ``rv.Dialog``
rendered at the component's top level and toggled by ``use_state`` (a row button
merely sets the pending target). Nested button -> Dialog toggles have proved
unreliable, so callers should render this once per tile, not inside a list row.

Four optional slots extend the plain confirm/cancel shape:

``checkbox_label`` / ``checkbox_value`` / ``on_checkbox``
    An opt-in that rides along with the action — "also delete the files from
    disk". The caller owns the state, so it can reset it every time the dialog
    opens; a sticky "yes, delete" is exactly what a destructive tick must not be.
``details``
    Monospace lines under the message (the files a choice would touch).
``note``
    A dimmed explanation, for when the checkbox cannot be offered at all.
``secondary_label`` / ``on_secondary``
    A third button between Cancel and the confirm, for a choice with two
    non-destructive outcomes (Keep existing / Re-download).
"""

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.widget.text_style import MUTED

# Paths and file names: monospace keeps them scannable and stops a long one
# from re-flowing the sentence above it.
_DETAIL_STYLE = (
    "font-family: ui-monospace, SFMono-Regular, Menlo, monospace;"
    "font-size: 0.78rem; line-height: 1.5; overflow-wrap: anywhere;" + MUTED
)
_NOTE_STYLE = "font-size: 0.8rem; line-height: 1.4;" + MUTED


@solara.component
def ConfirmDialog(
    open,
    on_cancel,
    on_confirm,
    title=None,
    message=None,
    confirm_label=None,
    confirm_color="error",
    checkbox_label=None,
    checkbox_value=False,
    on_checkbox=None,
    details=None,
    note=None,
    secondary_label=None,
    on_secondary=None,
):
    """Modal confirm dialog for a destructive action.

    Args:
        open: bool — whether the dialog is shown.
        on_cancel: callback() — dismiss without acting (also on ESC / outside click).
        on_confirm: callback() — act; the caller is responsible for closing.
        title: dialog title.
        message: dialog body copy.
        confirm_label: label of the confirm button.
        confirm_color: colour of the confirm button.
        checkbox_label: label of an optional opt-in checkbox (None = no checkbox).
        checkbox_value: its current value — owned by the caller.
        on_checkbox: callback(bool) — fired when the user ticks or unticks it.
        details: iterable of lines listed in monospace under the message.
        note: dimmed explanation shown under the message.
        secondary_label: label of an optional third button, left of the confirm.
        on_secondary: callback() — fired by that third button.
    """
    title = title if title is not None else t("dialog.confirm_default_title")
    message = message if message is not None else t("dialog.confirm_default_message")
    if confirm_label is None:
        confirm_label = t("dialog.confirm_default_label")
    # A plain confirm keeps its narrow box; file paths and a third button need
    # the extra room, or every path wraps over three lines.
    wide = bool(details or checkbox_label or secondary_label)
    with rv.Dialog(
        v_model=open,
        on_v_model=lambda v: None if v else on_cancel(),
        max_width="440px" if wide else "380px",
        eager=True,
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(title)
            with rv.CardText():
                solara.Text(message)
                if note:
                    solara.Text(note, style=_NOTE_STYLE)
                for line in details or []:
                    solara.Text(str(line), style=_DETAIL_STYLE)
                if checkbox_label:
                    solara.Checkbox(
                        label=checkbox_label,
                        value=bool(checkbox_value),
                        on_value=on_checkbox,
                        style="margin-top: 4px;",
                    )
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"), on_click=on_cancel, text=True, small=True
                )
                if secondary_label:
                    solara.Button(
                        secondary_label,
                        on_click=on_secondary,
                        outlined=True,
                        small=True,
                    )
                solara.Button(
                    confirm_label, on_click=on_confirm, color=confirm_color, small=True
                )
