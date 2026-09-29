"""A click outside a modal either closes it or visibly answers (issue #36).

Simple dialogs (confirmations, help popups) hold nothing the user typed, so an
outside click closes them — Vuetify's default for a non-persistent dialog.

Form dialogs cannot: in the Vuetify build solara ships (2.2.34) an open
v-select menu is invisible to the dialog's "am I the topmost overlay?" check
(``stackable`` looks for ``v-menu__content--active``, menus are marked
``menuable__content__active``), so a backdrop click that only meant to close a
dropdown also counts as a click outside the form. The forms therefore stay
``persistent`` — a stray click must never throw away a half-filled form — and
answer the click with a short nudge of the dialog instead of ignoring it.

The nudge lands on the ``.v-dialog`` element (``content_class``), not on the
card inside it: ``.v-dialog`` is the dialog's scroll box (``overflow-y: auto``),
so a card scaled inside it overflows and, with classic scrollbars, flashes two
scrollbars and re-wraps the form on every click.

Rendered rather than grepped: the nudge is a class that has to reach the dialog
widget, and a source check cannot see whether it does. (The translator warm-up
these renders need is done once for the suite in ``tests/conftest.py``.)
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.creation_dialog import CreationDialog
from gui.widget.dialog_nudge import NUDGE_CSS, nudge_class
from gui.widget.help import InfoPopup
from gui.widget.variable_modal import VariableModal

NUDGE_CLASSES = {nudge_class(1), nudge_class(2)}


def _find(widget, cls, out=None):
    """Collect every widget of ``cls`` under ``widget``, depth first."""
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _dialog(box):
    """The one v-dialog the component renders first (the form itself)."""
    return _find(box, vw.Dialog)[0]


def _dialog_classes(box):
    """The ``.v-dialog`` element's extra classes — where the nudge lands."""
    return (_dialog(box).content_class or "").split()


def _card_classes(box):
    """The class list of the form's card, which must never scale on its own."""
    card = _find(_dialog(box), vw.Card)[0]
    return (card.class_ or "").split()


def _nudges(box):
    """The nudge classes currently on the form's dialog element."""
    return [c for c in _dialog_classes(box) if c in NUDGE_CLASSES]


def _click_outside(box):
    """Send the event VDialog emits for a click on its backdrop."""
    _dialog(box).fire_event("click:outside", {})


def _button(box, label):
    """The button whose children carry ``label``."""
    return next(b for b in _find(box, vw.Btn) if label in str(b.children))


def _styles(box):
    """Every CSS template solara.Style rendered under ``box``."""
    return [w.template for w in _find(box, vw.VuetifyTemplate)]


# --- The nudge itself -------------------------------------------------------


def test_nudge_class_is_empty_until_the_first_outside_click():
    """A form that nobody clicked around must not animate on open."""
    assert nudge_class(0) == ""


def test_nudge_class_alternates_so_every_click_restarts_the_animation():
    """Re-applying one CSS animation name does not replay it; alternating does."""
    classes = [nudge_class(n) for n in range(1, 6)]
    assert all(classes)
    assert all(a != b for a, b in zip(classes, classes[1:]))
    assert len(set(classes)) == 2


def test_each_nudge_class_owns_its_own_keyframes():
    """Two class names sharing one @keyframes would still not restart."""
    for cls in NUDGE_CLASSES:
        assert f".{cls}" in NUDGE_CSS
        assert f"@keyframes {cls}" in NUDGE_CSS


# --- Forms: stay open, answer the click --------------------------------------


def _render_frame(open_, closes=None):
    """Render the shared creation frame with a do-nothing form."""
    return reacton.render(
        CreationDialog(
            open_=open_,
            title="t",
            create_label="Create",
            validate=lambda: None,
            will_replace=lambda: None,
            launch=lambda: None,
            on_close=None if closes is None else (lambda: closes.append(1)),
        ),
        handle_error=False,
    )


def test_creation_form_stays_open_on_an_outside_click():
    """The half-filled form survives: persistent, and no reset fires."""
    open_ = solara.reactive(True)
    closes = []
    box, rc = _render_frame(open_, closes)
    try:
        assert _dialog(box).persistent is True
        _click_outside(box)
        assert open_.value is True
        assert closes == []
    finally:
        rc.close()


def test_creation_form_nudges_on_every_outside_click():
    """The reported problem: an outside click used to get no reaction at all."""
    box, rc = _render_frame(solara.reactive(True))
    try:
        assert _nudges(box) == []
        _click_outside(box)
        first = _nudges(box)
        assert len(first) == 1
        _click_outside(box)
        second = _nudges(box)
        assert len(second) == 1 and second != first
    finally:
        rc.close()


def test_creation_form_nudges_the_dialog_not_the_card_inside_it():
    """Scaling the card inside the ``.v-dialog`` scroll box overflows it.

    With classic scrollbars that flashed a vertical and a horizontal scrollbar,
    re-wrapped the fields and clipped the card's edges on every outside click.
    Scaling the ``.v-dialog`` itself moves the frame and the card together.
    """
    box, rc = _render_frame(solara.reactive(True))
    try:
        _click_outside(box)
        assert len(_nudges(box)) == 1
        assert not NUDGE_CLASSES & set(_card_classes(box))
    finally:
        rc.close()


def test_creation_form_ships_the_nudge_css():
    """The class is inert without its keyframes on the page."""
    box, rc = _render_frame(solara.reactive(True))
    try:
        assert any(NUDGE_CSS in s for s in _styles(box))
    finally:
        rc.close()


def test_creation_form_does_not_replay_the_nudge_when_reopened():
    """A card that goes from display:none to shown replays its animation."""
    open_ = solara.reactive(True)
    box, rc = _render_frame(open_)
    try:
        _click_outside(box)
        assert _nudges(box)
        _button(box, t("common.cancel")).fire_event("click", None)
        assert open_.value is False
        open_.set(True)
        assert _nudges(box) == []
    finally:
        rc.close()


def test_variable_modal_stays_open_and_nudges_on_an_outside_click():
    """The Add/Edit Variable form predates the frame and needs the same answer."""
    open_ = solara.reactive(True)
    box, rc = reacton.render(
        VariableModal(open_=open_, on_add=lambda entry: None),
        handle_error=False,
    )
    try:
        assert _dialog(box).persistent is True
        assert any(NUDGE_CSS in s for s in _styles(box))
        _click_outside(box)
        assert open_.value is True
        assert len(_nudges(box)) == 1
        assert not NUDGE_CLASSES & set(_card_classes(box))
        open_.set(False)
        open_.set(True)
        assert _nudges(box) == []
    finally:
        rc.close()


# --- Simple dialogs: close ---------------------------------------------------


def test_confirm_dialog_closes_on_an_outside_click():
    """Nothing typed, nothing to lose: the backdrop click cancels.

    Vuetify closes a non-persistent dialog on its own and reports it through
    v-model, which the dialog routes to ``on_cancel``.
    """
    cancels = []
    box, rc = reacton.render(
        ConfirmDialog(
            open=True,
            on_cancel=lambda: cancels.append(1),
            on_confirm=lambda: None,
        ),
        handle_error=False,
    )
    try:
        dialog = _dialog(box)
        assert not dialog.persistent
        dialog.v_model = False  # what the browser sends after an outside click
        assert cancels == [1]
    finally:
        rc.close()


def test_help_popup_closes_on_an_outside_click():
    """An 'About …' popup is read-only and closes like any plain dialog."""
    state = []
    box, rc = reacton.render(
        InfoPopup("About", "Some *help*.", True, state.append),
        handle_error=False,
    )
    try:
        dialog = _dialog(box)
        assert not dialog.persistent
        dialog.v_model = False
        assert state == [False]
    finally:
        rc.close()
