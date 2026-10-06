"""Enter confirms a modal: every dialog tags exactly one default button (#33).

The key itself is handled in the browser — ``gui/widget/enter_key.py`` explains
why Python cannot tell a text field from a select in a keydown payload — and it
*presses* the tagged button. So the Python half of the contract, pinned here on
rendered widgets, is which button that is: the dialog's primary action, never
Cancel or a secondary choice; a nested prompt tags its own; the button Enter
presses is the one whose click runs the full create flow; and a primary that is
disabled stays disabled (the browser's ``click()`` then does nothing).

The browser half is pinned at the bottom: the kernel's echo that the press
waits for, and — where node is installed — the real script run against a small
stand-in DOM.
"""

import inspect
import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta

import ipyvuetify as vw
import ipywidgets
import pytest
import reacton
import solara

from gui.i18n import t

# Warm the translator before the first render (see test_manage_projects_render).
t("common.cancel")

from gui.scripts.project_io import ProjectInfo  # noqa: E402
from gui.widget.confirm_dialog import ConfirmDialog  # noqa: E402
from gui.widget.creation_dialog import CreationDialog  # noqa: E402
from gui.widget.enter_key import (  # noqa: E402
    ENTER_DEFAULT,
    EnterKeyListener,
    EnterKeyListenerWidget,
)
from gui.widget.manage_projects import (  # noqa: E402
    ConfirmDeleteProjectDialog,
    ManageProjectsDialog,
)
from gui.widget.variable_modal import VariableModal  # noqa: E402


def _buttons(widget, dialog=None, out=None):
    """(enclosing dialog, button) for every button under ``widget``."""
    out = [] if out is None else out
    if isinstance(widget, vw.Dialog):
        dialog = widget
    if isinstance(widget, vw.Btn):
        out.append((dialog, widget))
    for child in getattr(widget, "children", None) or []:
        if isinstance(child, ipywidgets.Widget):
            _buttons(child, dialog, out)
    return out


def _is_default(button):
    """True when the button carries the Enter-default class."""
    return ENTER_DEFAULT in (button.class_ or "").split()


def _label(button):
    """The button's text label (its icon, if any, is a separate child)."""
    return next((c for c in button.children if isinstance(c, str)), None)


def _defaults(box):
    """{dialog: [default buttons]} for every dialog rendered under ``box``."""
    found = {}
    for dialog, button in _buttons(box):
        found.setdefault(dialog, [])
        if _is_default(button):
            found[dialog].append(button)
    return found


def _only_default(box):
    """The single dialog's single default button."""
    defaults = _defaults(box)
    assert len(defaults) == 1, f"expected one dialog, found {len(defaults)}"
    (buttons,) = defaults.values()
    assert len(buttons) == 1, f"expected one default button, found {len(buttons)}"
    return buttons[0]


def _infos():
    """One healthy saved project, as ``list_project_infos`` returns it."""
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
    ]


# --- the shared frames -------------------------------------------------------


def _render_creation(launches, will_replace=lambda: None):
    """A CreationDialog that records each launch."""
    return reacton.render(
        CreationDialog(
            open_=solara.reactive(True),
            title="t",
            create_label="Create",
            validate=lambda: None,
            will_replace=will_replace,
            launch=lambda: launches.append(1),
        ),
        handle_error=False,
    )


def test_creation_form_and_its_replace_prompt_each_default_to_their_own_confirm():
    """Two dialogs, one default each: Create in the form, Replace in the prompt.

    The replace prompt is its own dialog, stacked above the form, and the
    browser half presses the default of the *topmost* open dialog — so Enter
    while the prompt is up must find Replace there, not the form's Create.
    """
    box, rc = _render_creation([])
    try:
        defaults = _defaults(box)
        assert len(defaults) == 2
        labels = sorted([_label(b) for b in buttons] for buttons in defaults.values())
        assert labels == sorted([["Create"], [t("common.replace")]])
    finally:
        rc.close()


def test_enter_in_a_creation_form_goes_through_the_create_click_path():
    """The button Enter presses runs validate, the replace prompt and the guard.

    A key handler that called ``launch`` directly would skip the overwrite
    confirmation and the one-launch-per-opening guard; pressing the default
    button cannot, because it is the same button a click lands on.
    """
    launches = []
    box, rc = _render_creation(launches, will_replace=lambda: "existing_key")
    try:
        form, prompt = sorted(
            _defaults(box).items(), key=lambda kv: _label(kv[1][0]) != "Create"
        )
        form_dialog, (create,) = form
        prompt_dialog, (replace,) = prompt

        create.fire_event("click", None)
        assert launches == [], "an existing key must ask before replacing it"
        assert prompt_dialog.v_model is True, "the replace prompt did not open"

        replace.fire_event("click", None)
        replace.fire_event("click", None)  # a second Enter queued behind it
        assert launches == [1]
    finally:
        rc.close()


def test_confirm_dialog_defaults_to_confirm_not_cancel_or_the_secondary_choice():
    """Remove / Replace / Discard prompts: Enter confirms, as the colour says."""
    box, rc = reacton.render(
        ConfirmDialog(
            open=True,
            on_cancel=lambda: None,
            on_confirm=lambda: None,
            confirm_label="Remove",
            secondary_label="Keep existing",
            on_secondary=lambda: None,
        ),
        handle_error=False,
    )
    try:
        assert _label(_only_default(box)) == "Remove"
    finally:
        rc.close()


def test_variable_modal_defaults_to_its_submit_button():
    """The Add Variable modal predates the frame; it tags its own submit."""
    box, rc = reacton.render(
        VariableModal(open_=solara.reactive(True), on_add=lambda entry: None),
        handle_error=False,
    )
    try:
        assert _label(_only_default(box)) == t("vars.modal.submit_add")
    finally:
        rc.close()


# --- project dialogs ---------------------------------------------------------


def _render_manage(selected):
    """ManageProjectsDialog with one saved project and the given selection."""
    return reacton.render(
        ManageProjectsDialog(
            open=True,
            infos=_infos(),
            selected=selected,
            on_select=lambda name: None,
            on_load=lambda: None,
            on_delete=lambda info: None,
            on_cancel=lambda: None,
        ),
        handle_error=False,
    )


def test_load_is_the_default_and_enter_cannot_load_nothing():
    """With no project selected Load is disabled — so Enter cannot press it."""
    box, rc = _render_manage(selected=None)
    try:
        load = _only_default(box)
        assert _label(load) == t("common.load")
        assert load.disabled is True
    finally:
        rc.close()

    box, rc = _render_manage(selected="GUY")
    try:
        assert _only_default(box).disabled is False
    finally:
        rc.close()


def test_project_delete_default_waits_for_the_typed_name():
    """Enter may confirm the irreversible delete only once the name is typed.

    The button stays disabled until then, and the browser's ``click()`` on a
    disabled button is a no-op — this gate is what makes a destructive default
    safe here.
    """
    box, rc = reacton.render(
        ConfirmDeleteProjectDialog(
            open=True,
            name="GUY",
            size_bytes=3_400_000_000,
            on_cancel=lambda: None,
            on_confirm=lambda: None,
        ),
        handle_error=False,
    )
    try:
        delete = _only_default(box)
        assert _label(delete) == t("project.dialog_delete_confirm")
        assert delete.disabled is True
        (field,) = rc.find(vw.TextField).widgets
        field.v_model = "GUY"
        assert _only_default(box).disabled is False
    finally:
        rc.close()


def test_project_delete_rechecks_the_typed_name_before_deleting():
    """A press the browser sent before the name reached the kernel deletes nothing.

    The disabled state is the browser's copy of the gate; a press that
    overtakes the last keystroke's update still reaches Python, so the confirm
    handler re-checks the name the kernel has.
    """
    confirms = []
    box, rc = reacton.render(
        ConfirmDeleteProjectDialog(
            open=True,
            name="GUY",
            size_bytes=3_400_000_000,
            on_cancel=lambda: None,
            on_confirm=lambda: confirms.append(1),
        ),
        handle_error=False,
    )
    try:
        (field,) = rc.find(vw.TextField).widgets
        field.v_model = "GU"
        _only_default(box).fire_event("click", None)
        assert confirms == [], "deleted a project whose name was not typed"
        field.v_model = "GUY"
        _only_default(box).fire_event("click", None)
        assert confirms == [1]
    finally:
        rc.close()


def test_project_panel_dialogs_each_default_to_their_confirm(tmp_path, monkeypatch):
    """New / Discard / Overwrite / Load / Delete: one default per dialog.

    The first three are inline ``rv.Dialog``s in ProjectPanel rather than the
    shared frames, so they are the easiest to miss.
    """
    import gui.solara_app as app

    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    box, rc = reacton.render(app.ProjectPanel(), handle_error=False)
    try:
        defaults = _defaults(box)
        # The panel's own New / Manage / Save buttons sit outside any dialog.
        assert defaults.pop(None) == [], "a panel button is tagged as a default"
        assert all(len(buttons) == 1 for buttons in defaults.values()), [
            [_label(b) for b in buttons] for buttons in defaults.values()
        ]
        assert sorted(_label(bs[0]) for bs in defaults.values()) == sorted(
            [
                t("common.create"),
                t("project.dialog_discard_confirm"),
                t("project.dialog_overwrite_confirm"),
                t("common.load"),
                t("project.dialog_delete_confirm"),
            ]
        )
    finally:
        rc.close()


# --- the listener ------------------------------------------------------------


def test_page_mounts_the_enter_listener():
    """The tags do nothing without the one page-wide listener."""
    import gui.solara_app as app

    assert "EnterKeyListener()" in inspect.getsource(app.Page)


def test_listener_renders_as_one_hidden_template():
    """The listener joins the widget tree and takes no space on the page."""

    @solara.component
    def Host():
        EnterKeyListener()

    box, rc = reacton.render(Host(), handle_error=False)
    try:
        (listener,) = rc.find(EnterKeyListenerWidget).widgets
        assert 'style="display: none"' in listener.template
    finally:
        rc.close()


def _listener_script():
    """The ``<script>`` body of the listener's template."""
    template = EnterKeyListenerWidget.class_traits()["template"].default_value
    return re.search(r"<script>(.*)</script>", template, re.S).group(1)


def test_listener_script_is_all_inside_module_exports():
    """Nothing sits outside ``module.exports``.

    ipyvue 1.10+ runs the script as a CommonJS module, but its fallback (and
    older ipyvue) evaluates only the object from the first ``{`` onwards — a
    top-level constant would then be undefined on every key press.
    """
    assert _listener_script().strip().startswith("module.exports = {")


def test_the_kernel_echoes_each_enter_barrier_back(monkeypatch):
    """Python half of the barrier: hand the token straight back to the script.

    The browser holds its press until this echo arrives. The echo can only
    come after the kernel handled every message sent before the barrier — and
    by then the browser has released the field updates ipywidgets was holding
    back (a model's next update waits until its previous one went idle).
    """
    widget = EnterKeyListenerWidget()
    try:
        sent = []
        monkeypatch.setattr(
            widget, "send", lambda content, buffers=None: sent.append(content)
        )
        widget._handle_event(None, {"event": "enter_barrier", "data": 7}, [])
        assert sent == [{"method": "enter_barrier_passed", "args": [7]}]
        # ...and both names match the script's side of the round trip.
        assert "enter_barrier" in widget.events
        script = _listener_script()
        assert "this.enter_barrier(" in script
        assert "jupyter_enter_barrier_passed(" in script
    finally:
        widget.close()


def test_listener_script_keeps_its_guards():
    """Pin the browser-side guards where node is missing (see below).

    Act only from inside a dialog, keep Enter for selects and every other
    control that owns it, press only the topmost dialog's own button — also
    when the focus was left behind it, taking the key in the capture phase —
    wait for the kernel before pressing, ignore auto-repeat, IME composition
    and handled events, and press by clicking so a disabled button stays put.
    """
    script = _listener_script()
    for guard in (
        'target.closest(".v-select")',
        'querySelectorAll(".v-dialog__content--active")',
        '.closest(".v-dialog__content") === dialog',
        'addEventListener("keydown", this.onKeydownCapture, true)',
        "e.stopPropagation()",
        'addEventListener("click", this.onBackdropClick, true)',
        "top.focus(",
        "this.enter_barrier(",
        "e.repeat",
        "e.isComposing",
        "e.defaultPrevented",
        f"button.{ENTER_DEFAULT}",
        ".click()",
    ):
        assert guard in script, f"the listener lost its guard: {guard}"


# --- the listener's script, run ------------------------------------------------
#
# The script decides in the browser, where pytest cannot follow — so run the
# real script under node against a small stand-in DOM: dialogs stacked by
# z-index, inputs, a VSelect that opens its menu on Enter, and the browser's
# own Enter-activates-a-focused-button. The kernel's barrier echo is released
# by hand, so each scenario can check what happened before and after it.

_NODE_HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const CLS = process.argv[3];

let log = [];
const listeners = { capture: [], bubble: [] };
const clickListeners = [];

class Element {
  constructor(tag, classes, attrs) {
    attrs = attrs || {};
    this.tagName = tag.toUpperCase();
    this.classes = new Set(classes || []);
    this.children = [];
    this.parentNode = null;
    this.style = { zIndex: attrs.z === undefined ? "auto" : String(attrs.z) };
    this.type = attrs.type || "";
    this.disabled = !!attrs.disabled;
    this.label = attrs.label || tag;
  }
  append(...kids) {
    for (const kid of kids) {
      kid.parentNode = this;
      this.children.push(kid);
    }
    return this;
  }
  get isConnected() {
    let node = this;
    while (node.parentNode) node = node.parentNode;
    return node === document.documentElement;
  }
  matches(selector) {
    const m = /^([a-z]*)((?:\.[\w-]+)*)$/.exec(selector);
    if (!m) throw new Error("selector not supported here: " + selector);
    if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
    return m[2].split(".").filter(Boolean).every((c) => this.classes.has(c));
  }
  closest(selector) {
    for (let node = this; node instanceof Element; node = node.parentNode) {
      if (node.matches(selector)) return node;
    }
    return null;
  }
  querySelectorAll(selector) {
    const found = [];
    const walk = (node) => {
      for (const kid of node.children) {
        if (kid.matches(selector)) found.push(kid);
        walk(kid);
      }
    };
    walk(this);
    return found;
  }
  contains(other) {
    for (let node = other; node; node = node.parentNode) {
      if (node === this) return true;
    }
    return false;
  }
  click() {
    if (!this.disabled) log.push("click:" + this.label);
  }
  focus() {
    document.activeElement = this;
    log.push("focus:" + this.label);
  }
}

const html = new Element("html");
const body = new Element("body");
html.append(body);
const phase = (capture) =>
  capture === true || (capture && capture.capture) ? "capture" : "bubble";
global.Element = Element;
global.document = {
  documentElement: html,
  body,
  activeElement: body,
  addEventListener(type, fn, capture) {
    if (type === "keydown") listeners[phase(capture)].push(fn);
    if (type === "click" && phase(capture) === "capture") clickListeners.push(fn);
  },
  removeEventListener(type, fn, capture) {
    const list =
      type === "click" && phase(capture) === "capture"
        ? clickListeners
        : listeners[phase(capture)];
    if (list.includes(fn)) list.splice(list.indexOf(fn), 1);
  },
  querySelectorAll: (selector) => html.querySelectorAll(selector),
};
global.window = { getComputedStyle: (node) => ({ zIndex: node.style.zIndex }) };

const mod = { exports: {} };
new Function("module", "exports", src)(mod, mod.exports);
const spec = mod.exports;

function keydown(target, key, extra) {
  const e = Object.assign(
    {
      key: key || "Enter",
      repeat: false,
      isComposing: false,
      altKey: false,
      ctrlKey: false,
      metaKey: false,
      shiftKey: false,
    },
    extra || {},
    {
      target,
      defaultPrevented: false,
      propagationStopped: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      stopPropagation() {
        this.propagationStopped = true;
      },
    }
  );
  for (const fn of listeners.capture.slice()) fn(e);
  if (!e.propagationStopped) {
    // The focused control's own handler: VSelect opens its menu on Enter.
    if (e.key === "Enter" && target.closest(".v-select")) log.push("menu-opened");
    for (const fn of listeners.bubble.slice()) fn(e);
  }
  // The browser's default action: Enter activates a focused button.
  if (e.key === "Enter" && !e.defaultPrevented && target.tagName === "BUTTON") {
    target.click();
  }
  return e;
}

// A click that lands on `target`; `during` runs before the listener's
// macrotask, the way Vuetify closes what the click dismissed in a microtask.
function click(target, during) {
  for (const fn of clickListeners.slice()) fn({ target });
  if (during) during();
}

const el = (tag, classes, attrs, ...kids) =>
  new Element(tag, classes, attrs).append(...kids);
const dialog = (z, ...kids) =>
  el("div", ["v-dialog__content", "v-dialog__content--active"],
     { z, label: "dialog@" + z },
     el("div", ["v-dialog"], {}, ...kids));
// A dialog's backdrop: the scrim inside its v-overlay, a sibling of the dialog.
const scrim = () => {
  const inner = el("div", ["v-overlay__scrim"]);
  body.append(el("div", ["v-overlay", "v-overlay--active"], {}, inner));
  return inner;
};
const button = (label, attrs) =>
  el("button", attrs && attrs.primary ? ["v-btn", CLS] : ["v-btn"],
     Object.assign({ label }, attrs));
const text = () => el("input", [], { type: "text" });
const selectField = () => {
  const input = el("input", [], { type: "text" });
  return { input, select: el("div", ["v-input", "v-select"], {}, input) };
};

const scenarios = {
  async text_field(h) {
    const field = text();
    body.append(
      dialog(202, field, button("Cancel"), button("Create", { primary: true }))
    );
    const e = keydown(field);
    const before = h.log();
    await h.echo();
    return {
      before,
      barriers: h.barriers(),
      prevented: e.defaultPrevented,
      after: h.log(),
    };
  },
  async enabled_by_the_typed_name(h) {
    const field = text();
    const del = button("Delete", { primary: true, disabled: true });
    body.append(dialog(202, field, del));
    keydown(field);
    del.disabled = false; // the name's update lands ahead of the echo
    await h.echo();
    return { after: h.log() };
  },
  async still_disabled(h) {
    const field = text();
    const del = button("Delete", { primary: true, disabled: true });
    body.append(dialog(202, field, del));
    keydown(field);
    await h.echo();
    await h.echo();
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
  async enabled_after_the_first_echo(h) {
    // The typed name was held back behind the first barrier: the kernel
    // enables Delete only between the first echo and the second.
    const field = text();
    const del = button("Delete", { primary: true, disabled: true });
    body.append(dialog(202, field, del));
    keydown(field);
    await h.echo();
    const afterFirst = h.log();
    del.disabled = false;
    await h.echo();
    return { afterFirst, barriers: h.barriers(), after: h.log() };
  },
  async one_press_per_wait(h) {
    const field = text();
    body.append(dialog(202, field, button("Create", { primary: true })));
    keydown(field);
    keydown(field);
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
  async escape_cancels(h) {
    const field = text();
    body.append(dialog(202, field, button("Create", { primary: true })));
    keydown(field);
    keydown(field, "Escape");
    await h.echo();
    return { after: h.log() };
  },
  async dialog_closed_meanwhile(h) {
    const field = text();
    const form = dialog(202, field, button("Create", { primary: true }));
    body.append(form);
    keydown(field);
    form.classes.delete("v-dialog__content--active");
    await h.echo();
    return { after: h.log() };
  },
  async stranded_on_a_select(h) {
    const { input, select } = selectField();
    body.append(
      dialog(202, select, text(), button("Create", { primary: true })),
      dialog(204, button("Cancel"), button("Replace", { primary: true }))
    );
    const e = keydown(input);
    await h.echo();
    return { after: h.log(), stopped: e.propagationStopped };
  },
  async stranded_on_an_icon_button(h) {
    const { select } = selectField();
    const info = button("info");
    body.append(
      dialog(202, info, select, button("Train", { primary: true })),
      dialog(204, button("Cancel"), button("Replace", { primary: true }))
    );
    keydown(info);
    await h.echo();
    return { after: h.log() };
  },
  async select_keeps_enter(h) {
    const { input, select } = selectField();
    body.append(dialog(202, select, button("Create", { primary: true })));
    keydown(input);
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
  async button_keeps_enter(h) {
    const cancel = button("Cancel");
    body.append(dialog(202, text(), cancel, button("Create", { primary: true })));
    keydown(cancel);
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
  async textarea_keeps_enter(h) {
    const area = el("textarea");
    body.append(dialog(202, area, button("Create", { primary: true })));
    keydown(area);
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
  async auto_repeat_ignored(h) {
    const field = text();
    body.append(dialog(202, field, button("Create", { primary: true })));
    keydown(field, "Enter", { repeat: true });
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
  async backdrop_refocuses_the_dialog(h) {
    body.append(dialog(202, text(), button("Create", { primary: true })));
    click(scrim());
    const before = h.log();
    await h.echo();
    return { before, after: h.log() };
  },
  async backdrop_closes_the_prompt_on_top(h) {
    const prompt = dialog(204, button("Replace", { primary: true }));
    body.append(dialog(202, text()), prompt);
    click(scrim(), () => prompt.classes.delete("v-dialog__content--active"));
    await h.echo();
    return { after: h.log() };
  },
  async backdrop_keeps_focus_already_inside(h) {
    const field = text();
    body.append(dialog(202, field));
    document.activeElement = field;
    click(scrim());
    await h.echo();
    return { after: h.log() };
  },
  async non_backdrop_click_ignored(h) {
    const item = el("div", ["v-list-item"]);
    const menu = el("div", ["menuable__content__active"], {}, item);
    body.append(dialog(202, text()), menu);
    click(item);
    await h.echo();
    return { after: h.log() };
  },
  async page_field_ignored(h) {
    const field = text();
    body.append(field, dialog(202, button("Create", { primary: true })));
    keydown(field);
    await h.echo();
    return { barriers: h.barriers(), after: h.log() };
  },
};

async function run(scenario) {
  body.children = [];
  document.activeElement = body;
  log = [];
  const sent = [];
  let barriers = 0;
  const vm = {};
  for (const [name, fn] of Object.entries(spec.methods)) vm[name] = fn.bind(vm);
  vm.enter_barrier = (token) => {
    barriers += 1;
    sent.push(token);
  };
  spec.mounted.call(vm);
  const echo = async () => {
    while (sent.length) vm.jupyter_enter_barrier_passed(sent.shift());
    await new Promise((resolve) => setTimeout(resolve, 20));
  };
  try {
    return await scenario({ echo, barriers: () => barriers, log: () => log.slice() });
  } finally {
    spec.beforeDestroy.call(vm);
  }
}

(async () => {
  const results = {};
  for (const [name, scenario] of Object.entries(scenarios)) {
    results[name] = await run(scenario);
  }
  process.stdout.write(JSON.stringify(results));
})().catch((err) => {
  console.error((err && err.stack) || err);
  process.exit(1);
});
"""


@pytest.fixture(scope="module")
def script_run(tmp_path_factory):
    """What the listener's real script did in each ``_NODE_HARNESS`` scenario."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; test_listener_script_keeps_its_guards")
    workdir = tmp_path_factory.mktemp("enter_key")
    (workdir / "listener.js").write_text(_listener_script())
    (workdir / "harness.js").write_text(_NODE_HARNESS)
    proc = subprocess.run(
        [node, str(workdir / "harness.js"), str(workdir / "listener.js")]
        + [ENTER_DEFAULT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_enter_presses_the_default_only_once_the_kernel_has_caught_up(script_run):
    """Type a name, press Enter at once: the press must not overtake the name.

    ipywidgets holds a field's newest value back while its previous update is
    in flight; a button's click is never held back. So the press waits for
    the kernel to echo a barrier, and it is dropped if ESC or a close came in
    between. The held-back name is handled only after that first barrier, so
    a button still disabled at the first echo gets a second one before the
    press gives up (seen in the browser: Delete-project stayed disabled).
    """
    run = script_run["text_field"]
    assert run["before"] == [], "pressed before the kernel caught up"
    assert run["barriers"] == 1
    assert run["prevented"] is True
    assert run["after"] == ["click:Create"]
    assert script_run["enabled_by_the_typed_name"]["after"] == ["click:Delete"]
    assert script_run["still_disabled"] == {"barriers": 2, "after": []}
    assert script_run["enabled_after_the_first_echo"] == {
        "afterFirst": [],
        "barriers": 2,
        "after": ["click:Delete"],
    }
    assert script_run["one_press_per_wait"] == {
        "barriers": 1,
        "after": ["click:Create"],
    }
    assert script_run["escape_cancels"]["after"] == []
    assert script_run["dialog_closed_meanwhile"]["after"] == []


def test_enter_behind_the_topmost_dialog_confirms_the_topmost(script_run):
    """The replace prompt is up, but the focus sits in the form behind it.

    The form's retain-focus handler hands the focus back to the form's first
    focusable — a select in every real form, the info icon (a button) in
    Train. Enter must press Replace, and neither open the select's menu nor
    click the icon behind the prompt.
    """
    run = script_run["stranded_on_a_select"]
    assert run["after"] == ["click:Replace"]
    assert run["stopped"] is True
    assert script_run["stranded_on_an_icon_button"]["after"] == ["click:Replace"]


def test_controls_that_own_enter_keep_it(script_run):
    """Selects, buttons and textareas keep Enter; repeats and page fields too."""
    assert script_run["select_keeps_enter"] == {
        "barriers": 0,
        "after": ["menu-opened"],
    }
    assert script_run["button_keeps_enter"] == {
        "barriers": 0,
        "after": ["click:Cancel"],
    }
    for name in ("textarea_keeps_enter", "auto_repeat_ignored", "page_field_ignored"):
        assert script_run[name] == {"barriers": 0, "after": []}, name


def test_a_backdrop_click_hands_the_keyboard_back_to_the_dialog(script_run):
    """A click on the scrim leaves the focus on ``<body>``: ESC went nowhere.

    A persistent form stays open on that click (and nudges), but VDialog hears
    ESC only on its own content element — so the form could no longer be
    dismissed from the keyboard. The listener focuses the topmost dialog once
    the click is handled; if the click dismissed a prompt, the form under it.
    """
    run = script_run["backdrop_refocuses_the_dialog"]
    assert run["before"] == [], "focused before Vuetify handled the click"
    assert run["after"] == ["focus:dialog@202"]
    assert script_run["backdrop_closes_the_prompt_on_top"]["after"] == [
        "focus:dialog@202"
    ]
    assert script_run["backdrop_keeps_focus_already_inside"]["after"] == []
    assert script_run["non_backdrop_click_ignored"]["after"] == []
