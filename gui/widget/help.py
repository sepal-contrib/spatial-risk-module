"""Shared help widgets for tiles.

``InfoButton`` is the standard way to give a step longer explanatory text
(what runs, why, data sources) without crowding the form: a small info icon
button that opens a popup dialog rendering Markdown (links included). Short
per-field help belongs in the field's own ``hint``/``persistent_hint`` props,
not here.
"""

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t


@solara.component
def InfoButton(title: str, markdown: str, icon: str = "mdi-information-outline"):
    """Info icon button opening an 'About …' popup dialog with Markdown content.

    Uses the proven top-level ``use_state`` + ``rv.Dialog(eager=True)`` toggle
    (same as ``ConfirmDialog``). Render it once per tile near the component's
    top level — nested button→dialog toggles (e.g. inside list rows) have
    proved unreliable.

    ``eager=True`` stays on the dialogs that set it, under the native MapApp
    shell too: a lazy VDialog does not take focus the first time it opens, so
    Esc does nothing until the user clicks inside it (checked in a browser on
    this popup, issue #57).
    """
    open_, set_open = solara.use_state(False)

    # solara.Button wires the click through ipyvue.use_event — a raw
    # rv.Btn(on_click=...) observes a nonexistent 'click' trait and silently
    # drops clicks.
    solara.Button(
        icon_name=icon,
        icon=True,
        small=True,
        on_click=lambda: set_open(True),
    )

    InfoPopup(title, markdown, open_, set_open, icon=icon)


@solara.component
def InfoPopup(
    title: str,
    markdown: str,
    open_: bool,
    set_open,
    icon: str = "mdi-information-outline",
):
    """The 'About …' popup on its own, driven by caller-owned state.

    Use it when the trigger is not a button of ours — e.g. a field's own
    ``prepend-inner`` icon, whose click arrives as a Vuetify ``click:*`` event
    (see ``ModelFormDialog``). ``InfoButton`` is this plus the icon button.
    """
    with rv.Dialog(
        v_model=open_,
        on_v_model=set_open,
        max_width="640px",
        eager=True,
        scrollable=True,
    ):
        with rv.Card():
            with rv.CardTitle(style_="gap:8px;"):
                rv.Icon(children=[icon])
                solara.Text(title)
            with rv.CardText():
                solara.Markdown(markdown)
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=lambda: set_open(False),
                    text=True,
                    small=True,
                )
