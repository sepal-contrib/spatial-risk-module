"""Shared product-list table used by every workflow tab and the Summary popup.

One CSS-grid table with a collapsible count header, declarative cells, and a
standard trailing Actions column. All product lists (variables, datasets,
samples, models, predictions, evaluations, summary tables) render through this
component so styling, icons, truncation, and empty states stay consistent.

This is the single home of the grid styling formerly duplicated across
variable_list.py / dataset_list.py / sample_set_list.py / summary_lists.py and
of the job-status icon maps formerly duplicated across train_model_list.py /
inference_output_list.py.
"""

from typing import Optional

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.widget.text_style import MUTED

GRID_BASE = "display:grid;align-items:center;width:100%;column-gap:8px;"
HEADER_EXTRA = (
    "padding:4px 8px 6px;border-bottom:2px solid rgba(0,0,0,0.15);"
    "font-size:0.72rem;font-weight:600;"
    "text-transform:uppercase;letter-spacing:0.05em;"
)
ROW_EXTRA = "padding:5px 8px;border-bottom:1px solid rgba(0,0,0,0.08);"
CELL = "display:flex;align-items:center;gap:4px;min-width:0;"
CELL_RIGHT = "display:flex;align-items:center;justify-content:flex-end;gap:0;"
NAME_CELL = "display:flex;align-items:center;gap:6px;min-width:0;overflow:hidden;"
# Header labels clip like data cells so a narrow column never overlaps its
# neighbour (grid items need min-width:0 + block display for ellipsis to apply).
HEADER_CELL = (
    "display:block;min-width:0;overflow:hidden;"
    "text-overflow:ellipsis;white-space:nowrap;"
)
# Full-width line under a failed row (grid rows have no subtitle slot).
ERROR_LINE = "grid-column:1/-1;padding:0 8px 5px;font-size:0.75rem;"

# Chips live in flexible columns, so a long label (a "Base" tag next to a long
# variable name, a dataset's target name) squeezes the chip below its content
# width. Vuetify's `.v-chip__content` then overflows one edge and the label
# reads as pushed off-centre. Pin the content to the chip's box and truncate
# the label itself so it stays centred with an ellipsis instead.
CHIP_CSS = """
.sr-chip .v-chip__content {
  width: 100%; min-width: 0; max-width: 100%; justify-content: center;
}
"""
CHIP_LABEL = "min-width:0;overflow:hidden;text-overflow:ellipsis;"
# Tag chips trailing a name ("ref") are short and carry meaning the row can't
# convey otherwise: the long, ellipsised name yields the space, not the tag.
CHIP_NO_SHRINK = "flex:0 0 auto;max-width:none;"

# One x-small icon button is 20px wide; 28px leaves breathing room between
# them. The 8px tail keeps the right-aligned buttons off the panel edge and
# gives the "ACTIONS" header label its own space.
ACTIONS_COL_WIDTH = "120px"

# Column width tokens shared by every list (spec: one row anatomy). The name
# column takes the remainder; everything else is fixed so header and data grids
# (separate containers, hence no ``max-content``) always line up.
W_NAME = "minmax(0,1fr)"
W_CHIP = "64px"
W_YEAR = "48px"
W_COUNT = "56px"
W_STATUS = "96px"

# Every cell's text element clips to its column with an ellipsis; the full text
# rides along as a native ``title`` (see ``cell_title``).
CLIP = "min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"
SUBTITLE = MUTED + "font-size:0.75rem;"

# Vuetify theme tokens, so status chips follow the app accent in light and dark.
# "cancelled" stays a literal grey: it is deliberately neutral, not a theme tone.
STATUS_COLORS = {
    "running": "info",
    "tiling": "info",
    "ready": "success",
    "completed": "success",
    "failed": "error",
    "cancelled": "grey",
    # Step 3 per-variable harmonization states (gui/widget/variable_list.py).
    "harmonized": "success",
    "pending": "grey",
    "not_downloaded": "warning",
    "checking": "grey",
    # List lifecycle states (create-then-run lists).
    "blocked": "grey",
    "queued": "grey",
    "outdated": "warning",
}
STATUS_ICONS = {
    # Flip and grey ship with the table (HOURGLASS_CSS), overriding the tone.
    "running": "mdi-timer-sand sr-hourglass",
    "tiling": "mdi-timer-sand sr-hourglass",
    "ready": "mdi-check-circle",
    "completed": "mdi-check-circle",
    "failed": "mdi-alert-circle",
    "cancelled": "mdi-cancel",
    "harmonized": "mdi-check-circle",
    "pending": "mdi-clock-outline",
    "not_downloaded": "mdi-cloud-outline",
    "checking": "mdi-help-circle-outline",
    "blocked": "mdi-link-variant-off",
    "queued": "mdi-timer-sand",  # static: queued is waiting, not working
    "outdated": "mdi-update",
}

_ACTION_ICONS = {
    "edit": "mdi-pencil-outline",
    "duplicate": "mdi-content-copy",
    "delete": "mdi-delete-outline",
    "download": "mdi-cloud-download-outline",
    "harmonize": "mdi-hammer",
    "cancel": "mdi-stop-circle",
    "open": "mdi-information-outline",
    "dismiss": "mdi-close",
}


# Busy state, app-wide: an hourglass that flips instead of Vuetify's spinner
# (#38). ``sr-hourglass`` animates the icon (on an ancestor or on the icon
# itself). ``sr-busy`` makes a button ignore clicks without ``disabled``, whose
# faded grey reads as "unavailable" rather than "working". Both are render-time
# props, so the handlers' in-flight claims stay the real double-click guard.
# The hourglass is one neutral grey everywhere, whatever colour its button or
# status would otherwise give it: the theme's secondary-text grey, keyed on the
# icon's own theme class (set by Vuetify's theme provider, so it is right in
# voila's teleported dialogs too, unlike the ``text--*`` helpers).
BUSY_ICON = "mdi-timer-sand"
BUSY_CLASSES = ["sr-hourglass", "sr-busy"]
HOURGLASS_CSS = """
.sr-hourglass .v-icon, .v-icon.sr-hourglass {
  animation: sr-hourglass-flip 2.4s ease-in-out infinite;
}
/* No fade in from the icon it replaces: Vuetify eases .v-icon colour over
   0.3 s, longer than many adds take, so a map-minus hourglass showed primary. */
.sr-hourglass .v-icon, .v-icon.sr-hourglass { transition: none !important; }
.v-icon.theme--light.sr-hourglass, .sr-hourglass .v-icon.theme--light {
  color: rgba(0, 0, 0, 0.6) !important;
}
.v-icon.theme--dark.sr-hourglass, .sr-hourglass .v-icon.theme--dark {
  color: rgba(255, 255, 255, 0.7) !important;
}
.sr-busy { pointer-events: none; }
@keyframes sr-hourglass-flip {
  0%, 40% { transform: rotate(0deg); }
  50%, 90% { transform: rotate(180deg); }
  100% { transform: rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
  .sr-hourglass .v-icon, .v-icon.sr-hourglass { animation: none; }
}
"""


def action_icon(kind: str, is_on: bool = False, loading: bool = False) -> str:
    """Standard icon for an action kind (map_toggle switches on is_on).

    A loading action of any kind shows the hourglass instead.
    """
    if loading:
        return BUSY_ICON
    if kind == "map_toggle":
        return "mdi-map-minus" if is_on else "mdi-map-plus"
    return _ACTION_ICONS[kind]


def action_color(kind: str, is_on: bool = False, override=None):
    """Vuetify colour token for an action icon (None = theme default text).

    Only "on"/primary states name a colour. Everything else returns None so the
    icon inherits the theme's text colour — a literal grey looked *disabled* on
    the dark theme, which is exactly what an enabled toggle must not look like.
    """
    if override is not None:
        return override
    if kind == "map_toggle":
        return "primary" if is_on else None
    if kind in ("download", "harmonize", "open"):
        return "primary"
    return None


def status_label(cell: dict) -> str:
    """Text of a status cell: explicit label, else progress %, else the word."""
    status = cell.get("status", "")
    if cell.get("label"):
        return str(cell["label"])
    progress = cell.get("progress")
    if status == "running" and progress is not None:
        return f"{round(min(max(progress, 0.0), 1.0) * 100)} %"
    if status in STATUS_ICONS:
        return t(f"widgets.product_table.status_{status}")
    return str(status)


def cell_title(cell: dict) -> dict:
    """Native ``title`` attributes so a clipped cell is readable on hover."""
    ctype = cell.get("type", "text")
    if ctype == "status":
        return {"title": status_label(cell)}
    if ctype in ("text", "chip") and cell.get("value") is not None:
        return {"title": str(cell["value"])}
    return {}


def grid_style(widths) -> str:
    """Grid container style for the given column-width list."""
    return GRID_BASE + "grid-template-columns:" + " ".join(widths) + ";"


def _render_chip(item: dict, shrink: bool = True):
    """One list chip.

    ``shrink=False`` keeps it at its natural width so a tight column truncates
    the neighbouring text instead of the chip's own label.
    """
    children = []
    if item.get("icon"):
        children.append(rv.Icon(children=[item["icon"]], x_small=True, left=True))
    value = str(item.get("value", ""))
    children.append(
        rv.Html(
            tag="span",
            style_=CHIP_LABEL + "white-space:nowrap;",
            attributes={"title": value},
            children=[value],
        )
    )
    rv.Chip(
        children=children,
        class_="sr-chip",
        style_="" if shrink else CHIP_NO_SHRINK,
        x_small=True,
        outlined=item.get("outlined", True),
        color=item.get("color"),
    )


def _clipped(text: str, title: str, style: str = ""):
    """One ellipsised text holder carrying its full text as ``title``."""
    rv.Html(
        tag="span",
        style_=CLIP + style,
        attributes={"title": title},
        children=[text],
    )


def _render_cell(cell: dict, first: bool):
    ctype = cell.get("type", "text")
    wrapper = NAME_CELL if first else CELL + "overflow:hidden;"

    if ctype == "render":
        with rv.Html(tag="div", style_=wrapper):
            cell["fn"]()
        return
    if ctype == "status":
        status = cell.get("status", "")
        label = status_label(cell)
        with rv.Html(tag="div", style_=CELL + "overflow:hidden;"):
            rv.Icon(
                children=[STATUS_ICONS.get(status, "mdi-help-circle")],
                color=STATUS_COLORS.get(status, "grey"),
                small=True,
                style_="flex:0 0 auto;",
            )
            _clipped(label, label, MUTED + "font-size:0.8rem;")
        return
    if ctype == "chip":
        with rv.Html(tag="div", style_=CELL + "overflow:hidden;"):
            _render_chip(cell)
        return
    if ctype == "chips":
        with rv.Html(tag="div", style_=CELL + "overflow:hidden;"):
            for item in cell.get("items", []):
                _render_chip(item)
        return

    # "text" (default)
    style = ""
    if cell.get("muted"):
        style += MUTED
    if cell.get("size"):
        style += f"font-size:{cell['size']};"
    text = str(cell.get("value", ""))
    subtitle = cell.get("subtitle")
    with rv.Html(tag="div", style_=wrapper):
        if subtitle:
            # Two-line name: the column clips each line on its own.
            with rv.Html(
                tag="div",
                style_="display:flex;flex-direction:column;min-width:0;flex:1 1 auto;",
            ):
                _clipped(text, text, style + "font-size:0.875rem;")
                _clipped(str(subtitle), str(subtitle), SUBTITLE)
        else:
            _clipped(text, text, style)
        for item in cell.get("chips", []):
            _render_chip(item, shrink=False)


def _render_action(act: dict):
    kind = act["kind"]
    is_on = act.get("is_on", False)
    color = action_color(kind, is_on, override=act.get("color"))
    loading = act.get("loading", False)
    # A busy action is the grey hourglass that ignores clicks (BUSY_CLASSES),
    # not a disabled button, whose faded look reads as "unavailable".
    solara.Button(
        "",
        icon_name=action_icon(kind, is_on, loading=loading),
        on_click=act["on_click"],
        icon=True,
        text=True,
        x_small=True,
        color=None if loading else color,
        disabled=act.get("disabled", False) and not loading,
        classes=BUSY_CLASSES if loading else [],
    )


@solara.component
def _Row(row: dict, grid: str, show_actions: bool):
    """One grid row.

    Its own component so the ``rv.use_event`` click hook is called exactly once
    per row instance regardless of how many rows the table renders (hooks must
    not live in the caller's loop).

    An optional ``row["on_click"]`` makes the content cells clickable. The
    listener sits on a ``display:contents`` wrapper around the content cells —
    not the row div — so clicks on the trailing action buttons never bubble
    into it.
    """
    on_click = row.get("on_click")
    with rv.Html(tag="div", style_=grid + ROW_EXTRA):
        content = rv.Html(
            tag="div",
            style_="display:contents;" + ("cursor:pointer;" if on_click else ""),
        )
        with content:
            for i, cell in enumerate(row.get("cells", [])):
                _render_cell(cell, first=i == 0)
        if show_actions:
            with rv.Html(tag="div", style_=CELL_RIGHT):
                for act in row.get("actions", []):
                    _render_action(act)
        if row.get("error"):
            rv.Html(
                tag="span",
                class_="error--text",
                style_=ERROR_LINE,
                children=[str(row["error"])],
            )
    # rv.use_event is a hook — call it unconditionally, gate inside the handler.
    rv.use_event(content, "click", lambda *_: on_click() if on_click else None)


@solara.component
def ProductTable(
    title: str,
    columns: list,
    rows: list,
    empty_text: str,
    collapsible: bool = True,
    show_actions: bool = True,
    banner: Optional[str] = None,
    actions_width: Optional[str] = None,
):
    """Uniform product table: collapsible count header + CSS-grid rows.

    Args:
        title: header label; rendered as ``TITLE (n)``.
        columns: content columns ``[{"label", "width"?}]``; an Actions column
            is appended automatically when ``show_actions``.
        rows: ``[{"key", "cells": [CellSpec], "actions": [ActionSpec],
            "error": str|None, "on_click": callable|None}]``. ``on_click``
            makes the row's content cells clickable (actions excluded). See
            module docstring / spec for CellSpec and ActionSpec shapes.
        empty_text: grey placeholder when there are no rows.
        collapsible: show the collapse chevron (expanded by default).
        show_actions: False = read-only mode (Summary popup).
        banner: optional stats line under the header (Summary popup).
        actions_width: Actions column width; defaults to ``ACTIONS_COL_WIDTH``
            so every list's action column is the same size.
    """
    collapsed, set_collapsed = solara.use_state(False)

    widths = [c.get("width", W_NAME) for c in columns]
    labels = [c["label"] for c in columns]
    if show_actions:
        widths.append(actions_width or ACTIONS_COL_WIDTH)
        labels.append(t("common.actions"))
    grid = grid_style(widths)

    with solara.Column(style="gap:0;width:100%;"):
        solara.Style(CHIP_CSS + HOURGLASS_CSS)
        with solara.Row(style="align-items:center;gap:8px;padding:4px 0;"):
            solara.Text(
                f"{title} ({len(rows)})",
                style=(
                    "font-weight:600;font-size:0.8rem;"
                    "text-transform:uppercase;letter-spacing:0.05em;"
                ),
            )
            if collapsible:
                solara.Button(
                    "",
                    icon_name="mdi-chevron-down" if collapsed else "mdi-chevron-up",
                    on_click=lambda: set_collapsed(not collapsed),
                    icon=True,
                    text=True,
                    x_small=True,
                )
        if not collapsed:
            if banner:
                solara.Text(
                    banner,
                    style=MUTED + "font-size:0.78rem;padding:2px 8px 8px;",
                )
            if not rows:
                solara.Text(
                    empty_text,
                    style=MUTED + "padding:4px 8px;",
                )
            else:
                with rv.Html(tag="div", style_=MUTED + grid + HEADER_EXTRA):
                    for i, lbl in enumerate(labels):
                        # Right-align the Actions header to sit over its
                        # right-aligned (flex-end) action buttons.
                        is_actions = show_actions and i == len(labels) - 1
                        rv.Html(
                            tag="span",
                            style_=HEADER_CELL
                            + ("text-align:right;" if is_actions else ""),
                            children=[lbl],
                        )
                # No .key(row["key"]) here: row keys are display identity, not
                # guaranteed unique in one table (a temporal variable repeats
                # per year; summary rows key on name) and reacton raises on
                # duplicate sibling keys. _Row is stateless, so positional
                # reconciliation is fine.
                for row in rows:
                    _Row(row, grid, show_actions)
