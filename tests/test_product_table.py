"""Unit tests for the shared ProductTable widget's pure helpers."""


def test_action_icons_standardized():
    """Every action kind resolves to its one canonical icon."""
    from gui.widget.product_table import action_icon

    assert action_icon("map_toggle", is_on=False) == "mdi-map-plus"
    assert action_icon("map_toggle", is_on=True) == "mdi-map-minus"
    assert action_icon("edit") == "mdi-pencil-outline"
    assert action_icon("delete") == "mdi-delete-outline"
    assert action_icon("run") == "mdi-play"
    assert action_icon("retry") == "mdi-refresh"
    assert action_icon("rerun") == "mdi-refresh"
    assert action_icon("cancel") == "mdi-stop-circle"
    # The details action reads as "what is this / how was it made", not
    # "peek at a table" — several of the dialogs it opens show no table.
    assert action_icon("open") == "mdi-information-outline"
    assert action_icon("dismiss") == "mdi-close"


def test_action_colors_are_theme_tokens_never_literal_grey():
    """The off-state map toggle must not be a literal grey.

    "grey darken-1" renders nearly black on the dark theme, which reads as a
    *disabled* button — the icon is enabled and clickable. Returning None lets
    Vuetify use the theme's default text colour (white on dark, dark on light),
    matching the edit/delete icons beside it.
    """
    from gui.widget.product_table import action_color

    assert action_color("map_toggle", is_on=False) is None
    assert action_color("map_toggle", is_on=True) == "primary"
    assert action_color("edit") is None
    assert action_color("delete") is None
    assert action_color("run") == "primary"
    assert action_color("retry") == "primary"
    assert action_color("rerun") is None
    assert action_color("open") == "primary"
    # An explicit per-action override still wins.
    assert action_color("delete", override="error") == "error"


def test_grid_style_builds_template_columns():
    """The column list becomes a CSS grid template."""
    from gui.widget.product_table import grid_style

    style = grid_style(["minmax(0,1fr)", "90px", "112px"])
    assert "display:grid" in style
    assert "grid-template-columns:minmax(0,1fr) 90px 112px;" in style


def test_status_maps_cover_all_states():
    """No job status falls through to the unknown-status placeholder."""
    from gui.widget.product_table import STATUS_COLORS, STATUS_ICONS

    for status in ("running", "ready", "completed", "failed", "cancelled"):
        assert status in STATUS_ICONS
        assert status in STATUS_COLORS
    assert STATUS_ICONS["running"] == "mdi-timer-sand sr-hourglass"
    assert STATUS_ICONS["completed"] == "mdi-check-circle"
    # Status tones are Vuetify theme tokens, not literal colours, so they track
    # the app palette in light and dark ("cancelled" stays a neutral grey).
    assert STATUS_COLORS["failed"] == "error"
    assert STATUS_COLORS["ready"] == "success"


def test_product_table_renders_duplicate_row_keys():
    """Rendering must not require unique row keys.

    Rows may share a name (a temporal variable per year, summary rows keyed
    by display name). Regression: reacton raised KeyError "Duplicate key
    'forest_gfc'" on project load once rows were mounted with
    .key(row["key"]).
    """
    import solara

    from gui.i18n import t
    from gui.widget.product_table import ProductTable

    t("common.close")  # prime the catalog: first t() inside a first render
    # corrupts reacton's widget map (known harness artifact)

    rows = [
        {
            "key": "forest_gfc",
            "cells": [{"type": "text", "value": "forest_gfc"}],
            "actions": [],
            "error": None,
        }
        for _ in range(2)
    ]
    box, rc = solara.render(
        ProductTable(
            title="Vars", columns=[{"label": "Name"}], rows=rows, empty_text="none"
        ),
        handle_error=False,
    )
    rc.close()


def test_harmonization_statuses_and_action_are_in_the_vocabulary():
    """Step 3's per-variable list reuses the status cell and action column."""
    from gui.i18n import t
    from gui.widget.product_table import (
        STATUS_COLORS,
        STATUS_ICONS,
        action_color,
        action_icon,
    )

    for status in ("harmonized", "pending", "not_downloaded", "checking"):
        assert status in STATUS_ICONS
        assert status in STATUS_COLORS
        assert t(f"widgets.product_table.status_{status}") != (
            f"widgets.product_table.status_{status}"
        )
    assert STATUS_COLORS["harmonized"] == "success"
    assert STATUS_COLORS["pending"] == "grey"
    assert STATUS_COLORS["not_downloaded"] == "warning"
    assert action_icon("run") == "mdi-play"
    assert action_color("run") == "primary"


def test_busy_actions_show_the_hourglass():
    """Any loading action swaps its icon for the hourglass (#38)."""
    from gui.widget.product_table import BUSY_ICON, action_icon

    assert BUSY_ICON == "mdi-timer-sand"
    assert action_icon("map_toggle", is_on=False, loading=True) == BUSY_ICON
    assert action_icon("map_toggle", is_on=True, loading=True) == BUSY_ICON
    assert action_icon("run", loading=True) == BUSY_ICON
    assert action_icon("retry", loading=True) == BUSY_ICON
    assert action_icon("run") == "mdi-play"


def test_the_hourglass_is_one_grey_everywhere():
    """Same grey in every section, whatever tone the button or status had (#38).

    The CSS pins it per theme on the icon itself, and a busy row action drops
    its own colour so the button's ripple/hover is neutral too.
    """
    import reacton

    from gui.widget.product_table import HOURGLASS_CSS, ProductTable

    assert "color: rgba(0, 0, 0, 0.6) !important" in HOURGLASS_CSS
    assert "color: rgba(255, 255, 255, 0.7) !important" in HOURGLASS_CSS
    for theme in ("light", "dark"):
        assert f".v-icon.theme--{theme}.sr-hourglass" in HOURGLASS_CSS
        assert f".sr-hourglass .v-icon.theme--{theme}" in HOURGLASS_CSS

    rows = [
        {
            "key": "a",
            "cells": [{"value": "a"}],
            "actions": [
                {"kind": "run", "on_click": lambda: None, "loading": True},
                {"kind": "open", "on_click": lambda: None},
            ],
        }
    ]
    _box, rc = reacton.render(
        ProductTable(title="t", columns=[{"label": "n"}], rows=rows, empty_text=""),
        handle_error=False,
    )
    try:
        import ipyvuetify as vw

        btns = {
            str(b.children[0].children[0]): b
            for b in rc.find(vw.Btn).widgets
            if b.icon and b.children and getattr(b.children[0], "children", None)
        }
        assert not btns["mdi-timer-sand"].color  # busy: the CSS grey
        assert btns["mdi-information-outline"].color == "primary"  # idle keeps its tone
    finally:
        rc.close()


# --- Stage 1 Task 1: clipping, width tokens, subtitle, status cell -----------

CLIP_DECLS = (
    "min-width:0;",
    "overflow:hidden;",
    "white-space:nowrap;",
    "text-overflow:ellipsis;",
)


def _render_table(cells, actions=None):
    import reacton

    from gui.i18n import t
    from gui.widget.product_table import ProductTable

    t("common.close")  # prime the catalog (first t() in a first render)
    rows = [{"key": "r", "cells": cells, "actions": actions or [], "error": None}]
    cols = [{"label": f"c{i}"} for i in range(len(cells))]
    _box, rc = reacton.render(
        ProductTable(title="t", columns=cols, rows=rows, empty_text=""),
        handle_error=False,
    )
    return rc


def _leaf_holders(rc):
    """Html widgets whose only child is a string: the text holders.

    ``rc.find`` only sees top-level widgets, so walk the widget tree by hand.
    """
    import ipyvuetify as vw

    out = []

    def walk(w):
        if isinstance(w, vw.Html) and len(w.children) == 1:
            if isinstance(w.children[0], str):
                out.append(w)
        for c in getattr(w, "children", None) or []:
            if not isinstance(c, str):
                walk(c)

    for w in rc.find(vw.Html).widgets:
        walk(w)
    return out


def test_width_tokens_match_the_spec():
    """Width tokens carry the spec values."""
    from gui.widget.product_table import (
        ACTIONS_COL_WIDTH,
        W_CHIP,
        W_COUNT,
        W_NAME,
        W_STATUS,
        W_YEAR,
    )

    assert (W_CHIP, W_YEAR, W_COUNT, W_STATUS, ACTIONS_COL_WIDTH) == (
        "64px",
        "48px",
        "56px",
        "96px",
        "120px",
    )
    assert W_NAME == "minmax(0,1fr)"


def test_every_cell_clips_and_carries_a_title():
    """Every text holder clips with an ellipsis and carries its full text as title."""
    from gui.widget.product_table import CLIP

    for d in CLIP_DECLS:
        assert d in CLIP
    cells = [
        {"type": "text", "value": "m1", "subtitle": "ds_a · sample_b"},
        {"type": "text", "value": "dataset_2015_loss_amazonia_v2"},
        {"type": "chip", "value": "icar_long_chip"},
        {"type": "status", "status": "ready"},
    ]
    rc = _render_table(cells)
    try:
        holders = {w.children[0]: w for w in _leaf_holders(rc)}
        for text in (
            "m1",
            "ds_a · sample_b",
            "dataset_2015_loss_amazonia_v2",
            "icar_long_chip",
            "ready",
        ):
            w = holders[text]
            for d in CLIP_DECLS:
                assert d in w.style_, (text, w.style_)
            assert w.attributes.get("title") == text
    finally:
        rc.close()


def test_subtitle_renders_as_a_muted_second_line():
    """A subtitle is a muted 0.75rem second line."""
    from gui.widget.text_style import MUTED

    rc = _render_table([{"type": "text", "value": "m1", "subtitle": "ds_a · sample_b"}])
    try:
        holders = {w.children[0]: w for w in _leaf_holders(rc)}
        assert "m1" in holders
        sub = holders["ds_a · sample_b"]
        assert MUTED in sub.style_ and "font-size:0.75rem" in sub.style_
    finally:
        rc.close()


def test_running_status_shows_progress_percent():
    """A running status with progress shows the percentage."""
    rc = _render_table([{"type": "status", "status": "running", "progress": 0.42}])
    try:
        assert "42 %" in {w.children[0] for w in _leaf_holders(rc)}
    finally:
        rc.close()


def test_status_label_override_wins():
    """An explicit status label replaces the i18n word."""
    label = "Needs model icar_2021"
    rc = _render_table([{"type": "status", "status": "blocked", "label": label}])
    try:
        holders = {w.children[0]: w for w in _leaf_holders(rc)}
        assert holders[label].attributes.get("title") == label
    finally:
        rc.close()


def test_cell_title_by_cell_type():
    """cell_title yields the full text for text/chip/status cells only."""
    from gui.widget.product_table import cell_title

    assert cell_title({"type": "text", "value": "abc"}) == {"title": "abc"}
    assert cell_title({"value": "abc"}) == {"title": "abc"}
    assert cell_title({"type": "chip", "value": "ref"}) == {"title": "ref"}
    assert cell_title({"type": "status", "status": "ready", "label": "Go"}) == {
        "title": "Go"
    }
    assert cell_title({"type": "render", "fn": lambda: None}) == {}
    assert cell_title({"type": "text", "value": None}) == {}


def test_new_states_are_in_the_vocabulary():
    """blocked/queued/outdated join the status vocabulary; pending is neutral."""
    from gui.widget.product_table import STATUS_COLORS, STATUS_ICONS

    for s in ("blocked", "queued", "outdated"):
        assert s in STATUS_ICONS and s in STATUS_COLORS
    assert STATUS_COLORS["pending"] == "grey"
    assert STATUS_COLORS["outdated"] == "warning"
    assert STATUS_ICONS["queued"] == "mdi-timer-sand"
    assert STATUS_ICONS["blocked"] == "mdi-link-variant-off"
    assert STATUS_ICONS["outdated"] == "mdi-update"


def test_status_labels_fit_in_every_locale():
    """Status words fit the 96px status column in every locale."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "gui" / "messages"
    dirs = [d for d in root.iterdir() if (d / "widgets.json").exists()]
    assert len(dirs) >= 4
    for d in dirs:
        table = json.loads((d / "widgets.json").read_text())["widgets"]["product_table"]
        for k, v in table.items():
            if k.startswith("status_") and k != "status_blocked":
                assert len(v) <= 12, (d.name, k, v)
        for k in ("status_blocked", "status_queued", "status_outdated"):
            assert table[k], (d.name, k)
        assert "{input}" in table["status_blocked"], d.name


def test_status_blocked_without_a_label_has_no_raw_placeholder():
    """A blocked cell with no label never shows the literal "{input}"."""
    from gui.widget.product_table import status_label

    text = status_label({"type": "status", "status": "blocked"})
    assert "{" not in text and text.startswith("needs")


def test_status_words_are_lowercase_initial_in_every_locale():
    """Status column casing: every word starts lowercase."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "gui" / "messages"
    for d in root.iterdir():
        f = d / "widgets.json"
        if not f.exists():
            continue
        table = json.loads(f.read_text())["widgets"]["product_table"]
        for k, v in table.items():
            if k.startswith("status_"):
                assert v[0] == v[0].lower(), (d.name, k, v)


def test_subtitle_name_lets_the_caller_size_win():
    """The default 0.875rem comes before the caller's size, so theirs wins."""
    rc = _render_table(
        [{"type": "text", "value": "m1", "subtitle": "s", "size": "1.2rem"}]
    )
    try:
        holders = {w.children[0]: w for w in _leaf_holders(rc)}
        style = holders["m1"].style_
        assert style.index("font-size:0.875rem;") < style.index("font-size:1.2rem;")
    finally:
        rc.close()


# --- Stage 1 Task 2: fixed action slots, more menu, tooltips ------------------


def _kids(w):
    """Children plus a solara Menu's activator (held in its own trait)."""
    kids = list(getattr(w, "activator", None) or [])
    kids += list(getattr(w, "children", None) or [])
    return [c for c in kids if not isinstance(c, str)]


def _noop():
    return None


def _all_widgets(rc):
    import ipyvuetify as vw

    out = []

    def walk(w):
        out.append(w)
        for c in _kids(w):
            walk(c)

    for w in rc.find(vw.Html).widgets:
        walk(w)
    return out


def _slot_boxes(rc):
    """The four 28px action boxes of the (single) row."""
    return [
        w
        for w in _all_widgets(rc)
        if "width:28px" in (getattr(w, "style_", None) or "")
        and "justify-content:center" in w.style_
    ]


def _btn_icon(b):
    return str(b.children[0].children[0])


def _acts(*kinds):
    return [{"kind": k, "on_click": _noop} for k in kinds]


def test_actions_render_in_fixed_slots():
    """Scrambled actions land in their fixed slots; delete hides in the menu."""
    import ipyvuetify as vw

    rc = _render_table(
        [{"type": "text", "value": "a"}],
        actions=_acts("delete", "map_toggle", "open", "run"),
    )
    try:
        boxes = _slot_boxes(rc)
        assert len(boxes) == 4
        icons = [[_btn_icon(b) for b in _descendants(box, vw.Btn)] for box in boxes]
        assert icons == [
            ["mdi-play"],
            ["mdi-map-plus"],
            ["mdi-information-outline"],
            ["mdi-dots-vertical", "mdi-delete-outline"],  # button, then menu item
        ]
        # delete lives only inside the menu, never in a slot 1-3 box
        for box in boxes[:3]:
            assert "mdi-delete-outline" not in [
                _btn_icon(b) for b in _descendants(box, vw.Btn)
            ]
        assert "mdi-delete-outline" in [
            _btn_icon(b) for b in _descendants(boxes[3], vw.Btn)
        ]
    finally:
        rc.close()


def _descendants(w, cls):
    out = []

    def walk(x):
        if isinstance(x, cls):
            out.append(x)
        for c in _kids(x):
            walk(c)

    walk(w)
    return out


def _all_btns(rc):
    import ipyvuetify as vw

    return [w for w in _all_widgets(rc) if isinstance(w, vw.Btn)]


def test_empty_slots_keep_icons_aligned():
    """Every row has four slot boxes; the map button is always in box 2."""
    import ipyvuetify as vw

    for kinds in (("map_toggle",), ("run", "map_toggle", "open", "delete")):
        rc = _render_table([{"type": "text", "value": "a"}], actions=_acts(*kinds))
        try:
            boxes = _slot_boxes(rc)
            assert len(boxes) == 4
            maps = [_btn_icon(b) for b in _descendants(boxes[1], vw.Btn)]
            assert maps == ["mdi-map-plus"]
        finally:
            rc.close()


def test_every_action_has_a_tooltip_and_aria_label():
    """Every button carries a title equal to its aria-label."""
    rc = _render_table(
        [{"type": "text", "value": "a"}],
        actions=_acts("run", "map_toggle", "open", "delete"),
    )
    try:
        btns = _all_btns(rc)
        assert len(btns) >= 4
        for b in btns:
            title = b.attributes.get("title")
            assert title and title == b.attributes.get("aria-label")
    finally:
        rc.close()


def test_disabled_action_tooltip_gives_the_reason():
    """A disabled action's tooltip is its reason."""
    acts = [
        {
            "kind": "run",
            "on_click": _noop,
            "disabled": True,
            "disabled_reason": "Needs download",
        }
    ]
    rc = _render_table([{"type": "text", "value": "a"}], actions=acts)
    try:
        run = [b for b in _all_btns(rc) if _btn_icon(b) == "mdi-play"][0]
        assert run.disabled
        assert run.attributes["title"] == "Needs download"
    finally:
        rc.close()


def test_explicit_label_is_the_tooltip():
    """An explicit action label is the tooltip and aria-label."""
    acts = [{"kind": "run", "on_click": _noop, "label": "Download"}]
    rc = _render_table([{"type": "text", "value": "a"}], actions=acts)
    try:
        run = [b for b in _all_btns(rc) if _btn_icon(b) == "mdi-play"][0]
        assert run.attributes["title"] == run.attributes["aria-label"] == "Download"
    finally:
        rc.close()


def test_two_actions_in_one_single_slot_is_an_error():
    """Two actions for one single-action slot is a programming error."""
    import pytest

    from gui.widget.product_table import _slot_actions

    with pytest.raises(AssertionError):
        _slot_actions(_acts("run", "cancel"))


def test_more_menu_lists_edit_duplicate_dismiss_delete_in_order():
    """The more menu orders edit, duplicate, dismiss, delete."""
    from gui.i18n import t
    from gui.widget.product_table import _slot_actions, action_label

    slots = _slot_actions(_acts("delete", "dismiss", "duplicate", "edit", "run"))
    assert [a["kind"] for a in slots[4]] == [
        "edit",
        "duplicate",
        "dismiss",
        "delete",
    ]
    assert [action_label(a) for a in slots[4]] == [
        t(f"widgets.product_table.action_{k}")
        for k in ("edit", "duplicate", "dismiss", "delete")
    ]


def test_more_menu_renders_its_items_in_order():
    """The rendered menu shows the button then its items in order."""
    import reacton

    from gui.widget.product_table import _MoreMenu, _slot_actions

    slots = _slot_actions(_acts("delete", "edit", "dismiss", "duplicate"))
    _box, rc = reacton.render(_MoreMenu(slots[4]), handle_error=False)
    try:
        icons = [_btn_icon(b) for b in _all_btns_of(rc) if b.children]
        assert icons == [
            "mdi-dots-vertical",
            "mdi-pencil-outline",
            "mdi-content-copy",
            "mdi-close",
            "mdi-delete-outline",
        ]
    finally:
        rc.close()


def _all_btns_of(rc):
    import ipyvuetify as vw

    out = []

    def walk(x):
        if isinstance(x, vw.Btn):
            out.append(x)
        for c in _kids(x):
            walk(c)

    for w in rc.find().widgets:
        walk(w)
    return out


def test_i18n_action_keys_exist_in_every_locale():
    """Action and verb keys exist in every locale."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "gui" / "messages"
    keys = [
        f"action_{k}"
        for k in (
            "run cancel retry rerun map_on map_off open edit duplicate delete "
            "dismiss more"
        ).split()
    ] + ["verb_download", "verb_harmonize"]
    for d in root.iterdir():
        f = d / "widgets.json"
        if f.exists():
            table = json.loads(f.read_text())["widgets"]["product_table"]
            for k in keys:
                assert table.get(k), (d.name, k)


# --- Stage 1 Task 3: clamped row errors with a details dialog ----------------


def _texts(w):
    """Every string under a widget, in document order."""
    out = []

    def walk(x):
        for c in getattr(x, "children", None) or []:
            if isinstance(c, str):
                out.append(c)
            else:
                walk(c)

    walk(w)
    return out


def test_error_line_is_clamped_with_a_details_link():
    """A long row error clamps to two lines and opens its full text in a dialog."""
    import ipyvuetify as vw
    import reacton

    from gui.i18n import t
    from gui.widget.product_table import ProductTable

    t("common.close")  # prime the catalog (first t() in a first render)
    msg = "E" * 600
    rows = [
        {
            "key": "r",
            "cells": [{"type": "text", "value": "a"}],
            "actions": [],
            "error": msg,
        }
    ]
    _box, rc = reacton.render(
        ProductTable(title="t", columns=[{"label": "c"}], rows=rows, empty_text=""),
        handle_error=False,
    )
    try:
        widgets = _all_widgets(rc)
        errs = [
            w
            for w in widgets
            if isinstance(w, vw.Html)
            and getattr(w, "children", None) == [msg]
            and "-webkit-line-clamp:2" in (w.style_ or "")
        ]
        assert len(errs) == 1
        assert "-webkit-box-orient:vertical" in errs[0].style_
        assert "overflow:hidden" in errs[0].style_

        details = t("widgets.product_table.error_details")
        btns = [b for b in _all_btns(rc) if details in _texts(b)]
        assert len(btns) == 1

        dialogs = [w for w in widgets if isinstance(w, vw.Dialog)]
        assert len(dialogs) == 1
        assert dialogs[0].v_model is False

        btns[0].fire_event("click", {})

        dialogs = [w for w in _all_widgets(rc) if isinstance(w, vw.Dialog)]
        assert dialogs[0].v_model is True
        pre_nodes = [
            w
            for w in _all_widgets(rc)
            if isinstance(w, vw.Html)
            and w.tag == "pre"
            and getattr(w, "children", None) == [msg]
        ]
        assert pre_nodes and "white-space:pre-wrap" in pre_nodes[0].style_
    finally:
        rc.close()


def test_more_menu_delete_item_runs_its_callback():
    """Clicking the delete item in the more menu calls that action's callback."""
    import reacton

    from gui.i18n import t
    from gui.widget.product_table import _MoreMenu, _slot_actions

    calls = []
    slots = _slot_actions(
        [{"kind": "delete", "on_click": lambda: calls.append("delete")}]
    )
    _box, rc = reacton.render(_MoreMenu(slots[4]), handle_error=False)
    try:
        delete = t("widgets.product_table.action_delete")
        items = [b for b in _all_btns_of(rc) if delete in _texts(b)]
        assert len(items) == 1
        items[0].fire_event("click", {})
        assert calls == ["delete"]
    finally:
        rc.close()


def test_blocked_status_fallback_uses_the_translated_input_word():
    """The blocked fallback fills {input} from the i18n word, not an English literal."""
    from gui.i18n import t
    from gui.widget.product_table import status_label

    expected = t(
        "widgets.product_table.status_blocked",
        input=t("widgets.product_table.input_generic"),
    )
    assert status_label({"type": "status", "status": "blocked"}) == expected
    assert t("widgets.product_table.input_generic") != (
        "widgets.product_table.input_generic"
    )
