"""Enter confirms a modal: the key presses the open dialog's default button.

A dialog opts in by tagging its primary button with ``ENTER_DEFAULT``
(``solara.Button(..., classes=[ENTER_DEFAULT])``); ``EnterKeyListener``,
mounted once by the app's ``Page``, does the rest in the browser. One default
button per dialog: with none, or with two, Enter does nothing there.

Why the browser and not ``rv.use_event(dialog, "keydown.enter", ...)`` — the way
the creation frame wires ESC back in: ipyvue flattens every DOM node in an event
payload to ``{"id": node.id}``, and a Vuetify input's id is an opaque
``input-<uid>``. Python therefore cannot tell a text field from a select, a
button or a list item, and each of those owns Enter itself. VSelect (and
VAutocomplete) does not stop Enter propagating either: it opens its menu or
picks the highlighted item, and only calls ``preventDefault`` a tick later — so
a Python handler would have submitted the form with the value from *before* the
pick. That decision needs the DOM.

Why *press the button* instead of calling the dialog's submit handler: Enter
then takes exactly the path a click takes — the same validation, the same
replace prompt, the same one-launch-per-opening guard in ``CreationDialog`` and
the same handler-side ``pending`` guards — and ``HTMLElement.click()`` on a
disabled button is a no-op, so a button that is disabled (Load with nothing
selected, Delete before the project name is typed) cannot fire via Enter.

When Enter presses the button — HTML's implicit form submission, applied to
the dialog on top:

* the focus is on a single-line text input (text, number, search, ...), a
  checkbox or a radio — none of which does anything with Enter — or on a dialog
  itself (VDialog focuses its content when it opens; a click on a dialog's
  blank space lands there too), or on nothing at all;
* never while the focus is on a control that owns Enter: a textarea (new line;
  Vuetify stops it anyway), a select / autocomplete / combobox (opens its menu,
  or picks the highlighted item — the form must not also submit), a button or
  link (the browser already activates it), a list item, a tab, an expansion
  header — nor on anything outside a dialog;
* the button pressed is the one of the *topmost* open dialog — the one on
  screen — and only a button that belongs to that dialog itself;
* when the focus was left behind the topmost dialog, in a dialog under it,
  Enter is for the topmost one whatever the focused control is. That is the
  state a replace prompt opens in: the form's own retain-focus handler pulls
  the focus straight back to the form's first focusable (seen in the browser)
  — a select in every real form, the Train form's info icon (a button). So the
  listener also takes Enter in the capture phase, before that control's own
  handler could open its menu or click it behind the prompt, and presses the
  prompt's Replace;
* not a held key's auto-repeat (it would run on into the next dialog — Create,
  then that prompt's Replace), not an IME composition, not a modifier chord,
  and not an event something else already handled (``defaultPrevented``).

The press waits for the kernel. ipywidgets throttles each model's updates: while
one is in flight, the field's newest value is held back in the browser until the
kernel reports idle. A button click is a custom message and is never held back,
so a click right after the last keystroke — the normal way to use Enter — could
overtake the typed name and create an artifact or a project under a prefix of
it. So Enter first sends a barrier message through this widget, and clicks only
when the kernel echoes it back (``vue_enter_barrier``): the kernel reported idle
for the in-flight update before it handled the barrier, so the held-back value
went out before the echo came back, and the click queues behind it.

That value is only *sent* by then, not handled: the kernel reads it after the
barrier, so a button the last keystrokes enable (Delete, once the project's
name is complete) is still disabled when the first echo comes back. So a press
that finds its button disabled waits for one more echo — the held-back value
is ahead of that second barrier, and so is the re-render that enables the
button — and is dropped only if the button is still disabled then. ESC, or the
dialog closing, while the press waits drops it.

A click on the backdrop hands the keyboard back to the dialog on top. The
scrim is not focusable, so that click leaves the focus on ``<body>`` — and a
persistent form, which stays open and only nudges (``gui/widget/dialog_nudge``),
then no longer heard ESC: VDialog listens for it on its own content element.
So after any click on a ``.v-overlay`` the listener focuses the topmost open
dialog, once Vuetify has handled the click (a dropdown closed, a
non-persistent prompt dismissed — then the dialog under it is the one on top).

Destructive confirmations are deliberately included: Enter mirrors ESC, the
dialog is itself the confirmation step behind a deliberate click, and the one
irreversible multi-GB delete (a whole project) keeps its button disabled until
the project's name is typed.
"""

import ipyvuetify as v
import traitlets

#: CSS class that marks a dialog's default button — the one Enter presses.
ENTER_DEFAULT = "sr-enter-default"

# Everything lives inside module.exports: ipyvue's legacy loader (its fallback
# today) evaluates only the object literal, so nothing may sit before it.
_TEMPLATE = """
<template>
  <span style="display: none"></span>
</template>
<script>
module.exports = {
  mounted() {
    // Presses waiting for the kernel's echo: barrier token -> button.
    this.srWaiting = new Map();
    this.srNextToken = 0;
    // Capture phase: runs before the focused control's own handler, so Enter
    // left behind the topmost dialog reaches that dialog instead.
    document.addEventListener("keydown", this.onKeydownCapture, true);
    document.addEventListener("keydown", this.onEnterKeydown);
    document.addEventListener("click", this.onBackdropClick, true);
  },
  beforeDestroy() {
    document.removeEventListener("keydown", this.onKeydownCapture, true);
    document.removeEventListener("keydown", this.onEnterKeydown);
    document.removeEventListener("click", this.onBackdropClick, true);
    this.srWaiting.clear();
  },
  methods: {
    isPlainEnter(e) {
      return (
        e.key === "Enter" &&
        !e.repeat &&
        !e.isComposing &&
        !e.defaultPrevented &&
        !(e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) &&
        e.target instanceof Element
      );
    },
    onKeydownCapture(e) {
      // ESC dismisses what a waiting press was meant for.
      if (e.key === "Escape") {
        this.srWaiting.clear();
        return;
      }
      if (!this.isPlainEnter(e)) return;
      const top = this.topmostDialog();
      const around = e.target.closest(".v-dialog__content");
      // Only the focus left in a dialog under the topmost one; everything
      // else goes the normal way, below.
      if (!top || !around || around === top) return;
      const button = this.defaultButton(top);
      if (!button) return;
      e.preventDefault();
      e.stopPropagation();
      this.press(button);
    },
    onEnterKeydown(e) {
      if (!this.isPlainEnter(e) || !this.enterIsFree(e.target)) return;
      const top = this.topmostDialog();
      const button = top && this.defaultButton(top);
      if (!button) return;
      e.preventDefault();
      this.press(button);
    },
    onBackdropClick(e) {
      if (!(e.target instanceof Element) || !e.target.closest(".v-overlay")) {
        return;
      }
      // A macrotask, so Vue has patched in what the click closed first.
      setTimeout(() => {
        const top = this.topmostDialog();
        if (top && !top.contains(document.activeElement)) {
          top.focus({ preventScroll: true });
        }
      }, 0);
    },
    enterIsFree(target) {
      // Nothing focused, or a dialog itself: v-dialog__content is the
      // focusable wrapper VDialog focuses when it opens.
      if (target === document.body || target === document.documentElement) {
        return true;
      }
      if (target.matches(".v-dialog__content")) return true;
      if (!target.closest(".v-dialog__content")) return false;
      if (target.tagName !== "INPUT") return false;
      // v-select also marks v-autocomplete, v-combobox and v-overflow-btn.
      if (target.closest(".v-select")) return false;
      // Inputs where Enter has no job of its own (HTML implicit submission).
      return [
        "text", "search", "email", "url", "tel", "password", "number",
        "checkbox", "radio",
      ].includes(target.type);
    },
    topmostDialog() {
      // Highest z-index wins (VDialog stacks each newly opened dialog 2
      // above the others); on a tie the later one paints on top.
      let top = null;
      let topZ = -Infinity;
      for (const el of document.querySelectorAll(".v-dialog__content--active")) {
        const z = parseInt(window.getComputedStyle(el).zIndex, 10) || 0;
        if (z >= topZ) {
          top = el;
          topZ = z;
        }
      }
      return top;
    },
    defaultButton(dialog) {
      // The dialog's own tagged button; none or two means Enter stays out.
      const buttons = Array.from(
        dialog.querySelectorAll("button.%(cls)s")
      ).filter((b) => b.closest(".v-dialog__content") === dialog);
      return buttons.length === 1 ? buttons[0] : null;
    },
    press(button) {
      // Once per button per wait: a second Enter before the echo is the same
      // press.
      for (const waiting of this.srWaiting.values()) {
        if (waiting.button === button) return;
      }
      this.barrier(button, 2);
    },
    barrier(button, rounds) {
      const token = ++this.srNextToken;
      this.srWaiting.set(token, { button, rounds });
      this.enter_barrier(token);
    },
    jupyter_enter_barrier_passed(token) {
      const waiting = this.srWaiting.get(token);
      if (!waiting) return;
      this.srWaiting.delete(token);
      const { button, rounds } = waiting;
      // A macrotask, so Vue has patched in the state that arrived ahead of
      // the echo. Nothing if the dialog closed while the press waited.
      setTimeout(() => {
        if (!button.isConnected || !button.closest(".v-dialog__content--active")) {
          return;
        }
        if (!button.disabled) {
          // A click, so Enter runs the button's own handler.
          button.click();
        } else if (rounds > 1) {
          // The value that enables it may be queued behind the first
          // barrier: wait for one more echo before giving up.
          this.barrier(button, rounds - 1);
        }
      }, 0);
    },
  },
};
</script>
""" % {
    "cls": ENTER_DEFAULT
}


class EnterKeyListenerWidget(v.VuetifyTemplate):
    """Invisible widget whose script turns Enter into a default-button click."""

    template = traitlets.Unicode(_TEMPLATE).tag(sync=True)

    def vue_enter_barrier(self, token):
        """Echo ``token``: the kernel has handled everything sent before it.

        The script clicks the default button only when this echo arrives, so
        the click reaches Python after the field updates it would otherwise
        overtake. Returns at once — it runs inside the session's websocket
        receive loop.
        """
        self.send({"method": "enter_barrier_passed", "args": [token]})


def EnterKeyListener():
    """Mount the page-wide Enter listener (once, in the app's ``Page``).

    It listens on ``document``, so where it sits in the tree does not matter.
    Mounting it twice would still press a button once: the first listener
    marks the event handled (``preventDefault``) and the second skips it.
    """
    return EnterKeyListenerWidget.element()
