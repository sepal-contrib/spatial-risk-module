"""Allocation-runs list for the Toolbox: saved runs + in-flight job overlay.

Renders ``allocation_runner.allocation_rows`` through the shared ``ProductTable``,
so it looks and behaves like every other product list in the app.
"""

import logging

import solara

from gui.i18n import t
from gui.widget.product_table import ProductTable
from gui.widget.text_style import MUTED

logger = logging.getLogger("spatial_risk")

# Hectare figures are compared down the column, so they get tabular figures:
# proportional digits make "312.4" and "1249.6" misalign at the decimal point.
_NUM = "font-variant-numeric:tabular-nums;"


def _run_meta(row):
    """'<source> · <years> yr · <date>' line under a run's name."""
    parts = [row.get("source") or t("toolbox.allocation.source_external")]
    years = row.get("years_forecast")
    if years:
        parts.append(f"{years:g} {t('toolbox.allocation.unit_years')}")
    created = row.get("created_at")
    if created:
        parts.append(str(created)[:10])
    return " · ".join(parts)


def _name_cell(row):
    """Run name with its provenance meta line (mock's rich row)."""

    def _fn(row=row):
        with solara.Column(style="gap:0;"):
            solara.Text(row["name"])
            solara.Text(_run_meta(row), style=MUTED + "font-size:0.72rem;")

    return {"type": "render", "fn": _fn}


def _hectares_cell(value, unit_key):
    """Right-aligned hectare figure with tabular figures.

    A "render" cell rather than a "text" one: ProductTable's text cells expose
    only ``muted``/``size``, so this is the supported way to pin a style.
    """

    def _fn(value=value, unit_key=unit_key):
        solara.Text(
            f"{value:,.1f} {t(unit_key)}",
            style=_NUM + "white-space:nowrap;",
        )

    return {"type": "render", "fn": _fn}


@solara.component
def AllocationList(
    rows,
    on_delete,
    on_open=None,
    on_toggle_density=None,
    density_on_map=frozenset(),
    on_edit=None,
    on_dismiss=None,
    toggling_keys=frozenset(),
):
    """Allocation runs table: one row per saved run plus in-flight jobs.

    Args:
        rows: output of ``allocation_runner.allocation_rows``.
        on_delete: callback(run_key) — delete a saved run (confirmed by the tile).
        on_open: callback(run_key) — open the run's read-only details dialog
            (the info action button); None omits the button. Job rows never
            get one.
        on_toggle_density: callback(row) — show/hide the density raster; None
            when there is no map to draw on.
        density_on_map: set of layer keys currently on the map.
        on_edit: callback(row) — reopen the allocation form prefilled with a
            failed job's submission entry so the user can fix and rerun.
        on_dismiss: callback(job_id) — discard a job row that is no longer
            running.
        toggling_keys: density layer keys whose map toggle is running; that
            row's map button is an hourglass that ignores clicks until the layer lands.
    """
    table_rows = []
    for r in rows:
        if r["kind"] == "job":
            job_actions = []
            # Only failed jobs that recorded their submission can be re-edited.
            if r["status"] == "failed" and r.get("entry") and on_edit is not None:
                job_actions.append(
                    {"kind": "edit", "on_click": lambda *_, rr=r: on_edit(rr)}
                )
            # Never on a running row: dismissing it only hides the row, and the
            # worker would still finish and register its run.
            if r["status"] != "running" and on_dismiss is not None:
                job_actions.append(
                    {
                        "kind": "dismiss",
                        "on_click": lambda *_, i=r["job_id"]: on_dismiss(i),
                    }
                )
            table_rows.append(
                {
                    "key": r["key"],
                    "cells": [
                        {"type": "text", "value": r["name"]},
                        {"type": "text", "value": "—", "muted": True},
                        {"type": "status", "status": r["status"]},
                    ],
                    "actions": job_actions,
                    "error": r.get("error"),
                }
            )
            continue

        actions = []
        if on_open is not None:
            actions.append(
                {"kind": "open", "on_click": lambda *_, k=r["key"]: on_open(k)}
            )
        # Only runs that actually wrote a density raster can toggle one.
        if on_toggle_density is not None and r.get("density_map_path"):
            from gui.scripts.density_map import density_layer_key

            actions.append(
                {
                    "kind": "map_toggle",
                    "on_click": lambda *_, rr=r: on_toggle_density(rr),
                    "is_on": density_layer_key(r["key"]) in density_on_map,
                    "loading": density_layer_key(r["key"]) in toggling_keys,
                }
            )
        actions.append(
            {"kind": "delete", "on_click": lambda *_, k=r["key"]: on_delete(k)}
        )

        warnings = r.get("warnings") or []

        table_rows.append(
            {
                "key": r["key"],
                "cells": [
                    _name_cell(r),
                    _hectares_cell(r["annual_ha"], "toolbox.allocation.unit_ha_yr"),
                    _hectares_cell(r["total_ha"], "toolbox.allocation.unit_ha"),
                ],
                "actions": actions,
                # Warnings ride the error slot: it is the only full-width line a
                # grid row has, and an unallocated-classes warning must be read.
                "error": warnings[0] if warnings else None,
            }
        )

    ProductTable(
        title=t("toolbox.allocation.title"),
        columns=[
            {"label": t("toolbox.allocation.field_name"), "width": "minmax(0,2fr)"},
            {"label": t("toolbox.allocation.result_annual"), "width": "minmax(0,1fr)"},
            {"label": t("toolbox.allocation.result_total"), "width": "minmax(0,1fr)"},
        ],
        rows=table_rows,
        empty_text=t("toolbox.allocation.empty"),
    )
