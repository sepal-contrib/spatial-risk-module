"""Visible answer to a click outside a form dialog.

Form dialogs — the shared ``CreationDialog`` frame and the Add/Edit Variable
modal — are ``persistent``, so a click outside them never closes them. That is
deliberate: in the Vuetify build solara ships (2.2.34) an open v-select menu is
invisible to the dialog's "am I the topmost overlay?" check (``stackable``
looks for ``v-menu__content--active``; menus are marked
``menuable__content__active``), so a backdrop click that only meant to close a
dropdown also counts as a click outside the form. A non-persistent form would
close on it and lose everything typed so far.

Persistent alone left that click with no reaction at all (issue #36). Vuetify's
own answer, the "shake" (``no_click_animation=False``), is no use here: it also
fires on ESC, which these forms handle from Python to *close* — the form would
wobble and then close. So the frames keep ``no_click_animation`` and nudge the
dialog from Python instead: VDialog still emits ``click:outside`` while
persistent, and every one replays the same scale bump Vuetify uses. A backdrop
click that closes a dropdown nudges too, exactly as Vuetify's shake would; the
dropdown closes and the form stays, which is the point.

The class goes on the ``.v-dialog`` element itself (``rv.Dialog``'s
``content_class``), where Vuetify's own shake runs — never on the card inside
it. ``.v-dialog`` is the dialog's scroll box (``overflow-y: auto``): a card
scaled inside it overflows, so with classic scrollbars every nudge flashed a
vertical and a horizontal scrollbar, re-wrapped the fields and clipped the
card's edges.
"""

from typing import Callable, Tuple

import solara

# Replaying a CSS animation needs a *different* animation name: re-applying the
# same one is a no-op. Two classes with identical keyframes, alternated on each
# click, restart it every time. Values copied from Vuetify's `animate-dialog`.
_NUDGE_NAMES = ("sr-dialog-nudge-a", "sr-dialog-nudge-b")

NUDGE_CSS = "".join(
    f"""
@keyframes {name} {{
  0% {{ transform: scale(1); }}
  50% {{ transform: scale(1.03); }}
  100% {{ transform: scale(1); }}
}}
.{name} {{ animation: {name} .15s cubic-bezier(.25, .8, .25, 1); }}
"""
    for name in _NUDGE_NAMES
)


def nudge_class(count: int) -> str:
    """Dialog class after ``count`` outside clicks ("" before the first one)."""
    if count <= 0:
        return ""
    return _NUDGE_NAMES[(count - 1) % 2]


def use_outside_click_nudge(is_open: bool) -> Tuple[str, Callable[..., None]]:
    """Hook: the nudge state of a persistent dialog.

    Returns ``(nudge, on_click_outside)``. The caller passes ``nudge`` as the
    ``rv.Dialog``'s ``content_class``, wires ``on_click_outside`` with
    ``rv.use_event(dialog, "click:outside", on_click_outside)`` and renders
    ``solara.Style(NUDGE_CSS)`` inside the dialog. The class has to be known
    before the ``rv.Dialog`` element is built, which is why the event wiring is
    left to the caller rather than done here.

    The count resets whenever the dialog opens or closes: a dialog that goes
    from ``display: none`` back to shown replays its animation, so a stale class
    would nudge the form the moment it reopens. Holds hooks: call it
    unconditionally.
    """
    count, set_count = solara.use_state(0)

    def _reset():
        set_count(0)

    solara.use_effect(_reset, [is_open])

    def on_click_outside(*_):
        set_count(lambda n: n + 1)

    return nudge_class(count), on_click_outside
