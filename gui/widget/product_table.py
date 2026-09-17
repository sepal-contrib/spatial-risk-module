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
NAME_TEXT = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
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
ACTION_BTN_WIDTH = 28
ACTIONS_MIN_WIDTH = 56
ACTIONS_COL_WIDTH = "112px"

# Vuetify theme tokens, so status chips follow the app accent in light and dark.
# "cancelled" stays a literal grey: it is deliberately neutral, not a theme tone.
STATUS_COLORS = {
    "running": "info",
    "ready": "success",
    "completed": "success",
    "failed": "error",
    "cancelled": "grey",
    # Step 3 per-variable harmonization states (gui/widget/variable_list.py).
    "harmonized": "success",
    "pending": "warning",
    "not_downloaded": "warning",
    "checking": "grey",
}
STATUS_ICONS = {
    "running": "mdi-loading mdi-spin",
    "ready": "mdi-check-circle",
    "completed": "mdi-check-circle",
    "failed": "mdi-alert-circle",
    "cancelled": "mdi-cancel",
    "harmonized": "mdi-check-circle",
    "pending": "mdi-clock-outline",
    "not_downloaded": "mdi-cloud-outline",
    "checking": "mdi-help-circle-outline",
}

_ACTION_ICONS = {
    "edit": "mdi-pencil-outline",
    "delete": "mdi-delete-outline",
    "download": "mdi-cloud-download-outline",
    "harmonize": "mdi-hammer",
    "cancel": "mdi-stop-circle",
    "open": "mdi-information-outline",
    "dismiss": "mdi-close",
}


def action_icon(kind: str, is_on: bool = False) -> str:
    """Standard icon for an action kind (map_toggle switches on is_on)."""
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


def actions_width_for(rows) -> str:
    """Actions-column width that fits the busiest row and no more.

    A fixed width had to assume the maximum any table might use (four buttons),
    so two-button tables paid ~50px of dead space right where the flexible name
    column was being ellipsised. Both the header and the data rows read this one
    computed value, so the columns still line up (they are separate grid
    containers, hence no ``max-content``).
    """
    most = max((len(r.get("actions") or []) for r in rows), default=0)
    return f"{max(most * ACTION_BTN_WIDTH + 8, ACTIONS_MIN_WIDTH)}px"


def name_tooltip(cell: dict) -> dict:
    """Native ``title`` attributes so an ellipsised name is readable on hover."""
    if cell.get("type", "text") == "text" and cell.get("value") is not None:
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
    children.append(
        rv.Html(tag="span", style_=CHIP_LABEL, children=[str(item.get("value", ""))])
    )
    rv.Chip(
        children=children,
        class_="sr-chip",
        style_="" if shrink else CHIP_NO_SHRINK,
        x_small=True,
        outlined=item.get("outlined", True),
        color=item.get("color"),
    )


def _render_cell(cell: dict, first: bool):
    ctype = cell.get("type", "text")
    wrapper = NAME_CELL if first else CELL

    if ctype == "render":
        with rv.Html(tag="div", style_=wrapper):
            cell["fn"]()
        return
    if ctype == "status":
        status = cell.get("status", "")
        with rv.Html(tag="div", style_=CELL):
            rv.Icon(
                children=[STATUS_ICONS.get(status, "mdi-help-circle")],
                color=STATUS_COLORS.get(status, "grey"),
                small=True,
            )
            label = (
                t(f"widgets.product_table.status_{status}")
                if status in STATUS_ICONS
                else str(status)
            )
            solara.Text(label, style=MUTED + "font-size:0.8rem;")
        return
    if ctype == "chip":
        with rv.Html(tag="div", style_=CELL):
            _render_chip(cell)
        return
    if ctype == "chips":
        with rv.Html(tag="div", style_=CELL):
            for item in cell.get("items", []):
                _render_chip(item)
        return

    # "text" (default)
    style = NAME_TEXT if first else ""
    if cell.get("muted"):
        style += MUTED
    if cell.get("size"):
        style += f"font-size:{cell['size']};"
    with rv.Html(
        tag="div", style_=wrapper, attributes=name_tooltip(cell) if first else {}
    ):
        solara.Text(str(cell.get("value", "")), style=style)
        for item in cell.get("chips", []):
            _render_chip(item, shrink=False)


def _render_action(act: dict):
    kind = act["kind"]
    is_on = act.get("is_on", False)
    color = action_color(kind, is_on, override=act.get("color"))
    solara.Button(
        "",
        icon_name=action_icon(kind, is_on),
        on_click=act["on_click"],
        icon=True,
        text=True,
        x_small=True,
        color=color,
        disabled=act.get("disabled", False),
        loading=act.get("loading", False),
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
        actions_width: Actions column width; defaults to the width the busiest
            row actually needs so the flexible content columns keep the rest.
    """
    collapsed, set_collapsed = solara.use_state(False)

    widths = [c.get("width", "minmax(0,1fr)") for c in columns]
    labels = [c["label"] for c in columns]
    if show_actions:
        widths.append(actions_width or actions_width_for(rows))
        labels.append(t("common.actions"))
    grid = grid_style(widths)

    with solara.Column(style="gap:0;width:100%;"):
        solara.Style(CHIP_CSS)
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
