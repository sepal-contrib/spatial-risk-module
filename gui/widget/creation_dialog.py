"""Shared creation dialog: form frame + Create action + replace confirmation.

Every list-first tab opens its "New <thing>" form through this frame so the
layout, button order, validation flow and duplicate policy stay identical:
Create validates on click (button never disabled), an existing key asks for
confirmation (never a silent overwrite, never a hard refusal), and the
caller's `launch` performs the actual mutation / job spawn.
"""

from typing import Callable, List, Optional

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.widget.confirm_dialog import ConfirmDialog

# Restyle the Advanced-parameters panel to sit in the form's flow: same
# border/height/label colour as the outlined dense fields, no 24px inset.
# Injected here so every creation form's `class_="advanced-params"` panel
# looks identical.
_ADVANCED_PANEL_CSS = """
.advanced-params .v-expansion-panel {
  border: 1px solid rgba(0, 0, 0, .38); border-radius: 4px;
}
.theme--dark .advanced-params .v-expansion-panel {
  border-color: rgba(255, 255, 255, .24);
}
.advanced-params .v-expansion-panel::before { box-shadow: none; }
.advanced-params .v-expansion-panel-header {
  min-height: 40px; padding: 0 12px; font-size: 14px; color: rgba(0, 0, 0, .6);
}
.theme--dark .advanced-params .v-expansion-panel-header {
  color: rgba(255, 255, 255, .7);
}
.advanced-params .v-expansion-panel-content__wrap { padding: 16px 12px 4px; }

/* A field's own help icon (`class_="field-info-icon"`). It has to be the
   prepend-inner icon: that is the only icon slot whose click Vuetify
   preventDefaults and stops propagating (VInput.genIcon, and only while a
   click:prepend-inner listener exists), so on a v-select it opens the popup
   without also dropping the menu open. Vuetify only renders that slot on the
   left, so park it just clear of the caret — 12px field padding + a 24px
   caret means 38px of right offset — and reserve room so a long value never
   runs under it. */
.field-info-icon .v-input__slot { position: relative; }
.field-info-icon .v-input__prepend-inner {
  position: absolute; right: 38px; top: 50%; transform: translateY(-50%);
  margin: 0 !important; padding: 0 !important; z-index: 1;
}
.field-info-icon .v-select__selections { padding-right: 30px; }
/* VTextField.labelPosition slides the floating label left by the measured
   prepend-inner width, which here would drag "Model" out of the outline's
   notch. In LTR it writes that offset to `left` (`{left: offset, right:
   'auto'}`), inline, so undoing it takes !important on both. */
.field-info-icon .v-label { left: 0 !important; right: auto !important; }

/* Multi-select with chips (`class_="multi-chips"`). A dense field pins its
   slot to a one-line height, so a second row of chips spills out of the
   outlined border. Let the slot grow with its content instead and keep the
   single-row height as the floor. */
.multi-chips .v-input__slot { height: auto !important; min-height: 40px; }
.multi-chips .v-select__slot { min-height: 40px; }
.multi-chips .v-select__selections { flex-wrap: wrap; padding: 3px 0; }
.multi-chips .v-select__selections .v-chip { margin: 2px 4px 2px 0; }

/* Formula textarea: patsy code reads better monospaced. */
.formula-field textarea { font-family: monospace; font-size: 13px; }

/* A field that owns a FieldHint (`gui/widget/text_style.py`). Vuetify reserves
   a details row under every input for validation messages; the hinted fields
   have no `rules`, so that row is always empty and pushes the hint ~22px down.
   Collapse it when EMPTY only — an actual error message still renders at its
   natural height, which `hide-details` would have suppressed. Needed as CSS
   rather than a prop because FileInputComponent, AdminLevelSelector and
   AssetSelectComponent expose no hide-details argument at all. */
.sr-tight-field .v-text-field__details { min-height: 0; margin-bottom: 0; }
.sr-tight-field .v-messages { min-height: 0; }
"""


@solara.component
def CreationDialog(
    open_,
    title: str,
    create_label: str,
    validate: Callable[[], Optional[str]],
    will_replace: Callable[[], Optional[str]],
    launch: Callable[[], None],
    create_icon: str = "mdi-plus",
    on_close: Optional[Callable[[], None]] = None,
    replace_title: Optional[str] = None,
    replace_message: Optional[Callable[[str], str]] = None,
    max_width: str = "560px",
    children: List = [],
):
    """Creation-form frame with the unified Create flow.

    Args:
        open_: solara.Reactive[bool] — dialog visibility (owned by the tile).
        title: dialog heading.
        create_label: label for the submit button (e.g. "Register", "Save").
        create_icon: icon on the submit button; the default "+" suits a
            creation form, a form that *sets* something names its own.
        validate: () -> error message | None; runs on Create click.
        will_replace: () -> existing storage key | None; runs after validate.
            A returned key opens the confirm-replace dialog instead of
            launching directly.
        launch: () -> None; performs the creation. The dialog closes and the
            caller's form resets (on_close) after it returns.
        on_close: reset callback fired on Cancel/ESC and after a launch.
        replace_title: heading of the confirm-replace dialog; defaults to the
            generic widgets.creation_dialog.replace_title copy.
        replace_message: key -> confirmation body; defaults to the generic
            widgets.creation_dialog.replace_message copy.
        max_width: CSS max-width of the dialog card.
        children: the caller's form fields, rendered in the card body.
    """
    error, set_error = solara.use_state(None)
    pending_replace, set_pending_replace = solara.use_state(None)
    # One launch per opening. Closing the dialog is a browser round-trip, so
    # a second Create (or Replace) click queued behind the first is handled
    # after it, with the form still valid — the reset re-suggests a valid
    # name, and a job only registers its product when its worker finishes,
    # so neither validate() nor will_replace() sees the first launch. A ref,
    # not state: the second handler runs before any re-render could carry a
    # new closure, so it has to read what the first handler wrote.
    launched = solara.use_ref(False)

    def _rearm():
        if open_.value:
            launched.current = False

    solara.use_effect(_rearm, [open_.value])

    def close():
        set_error(None)
        set_pending_replace(None)
        open_.set(False)
        if on_close is not None:
            on_close()

    def do_launch():
        if launched.current:
            return
        launched.current = True
        launch()
        close()

    def on_create():
        set_error(None)
        err = validate()
        if err:
            set_error(err)
            return
        key = will_replace()
        if key:
            set_pending_replace(key)
            return
        do_launch()

    # `persistent` takes the form out of the click-outside path. Without it a
    # click meant to dismiss an open v-select menu closed the whole form: the
    # menu renders detached, so Vuetify's overlay stack does not put it above
    # the dialog here and both read the same click as their own. Vuetify's
    # persistent "shake" would fire on exactly that click, so it is off too.
    dialog = rv.Dialog(
        v_model=open_.value,
        on_v_model=lambda v: None if v else close(),
        max_width=max_width,
        eager=True,
        persistent=True,
        no_click_animation=True,
    )
    # ESC back, which `persistent` otherwise disables: VDialog emits `keydown`
    # regardless, and VSelect stops ESC propagating while its menu is open — so
    # ESC closes an open dropdown first and the form only once none is open.
    # rv.use_event is a hook — call it unconditionally.
    rv.use_event(dialog, "keydown.esc", lambda *_: close())
    with dialog:
        with rv.Card():
            with rv.CardTitle():
                solara.Text(title)
            with rv.CardText():
                solara.Style(_ADVANCED_PANEL_CSS)
                solara.Column(style="gap:4px;", children=children)
                if error:
                    # ``type=``, not ``type_=``: reacton's Alert has no ``type_``
                    # parameter, so it is swallowed and the alert renders grey
                    # and iconless — under the warning-coloured advisory a few
                    # pixels above it, with the severity hierarchy inverted.
                    rv.Alert(type="error", dense=True, children=[error])
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(t("common.cancel"), on_click=close, text=True, small=True)
                solara.Button(
                    create_label,
                    icon_name=create_icon,
                    color="primary",
                    small=True,
                    on_click=on_create,
                )

    _msg = (
        replace_message(pending_replace)
        if replace_message is not None and pending_replace
        else t("widgets.creation_dialog.replace_message", key=pending_replace or "")
    )
    ConfirmDialog(
        open=pending_replace is not None,
        on_cancel=lambda: set_pending_replace(None),
        on_confirm=do_launch,
        title=replace_title or t("widgets.creation_dialog.replace_title"),
        message=_msg,
        confirm_label=t("common.replace"),
        confirm_color="warning",
    )
