"""Registered sample-sets table widget (registry products + session-job overlay)."""

import logging
from typing import Callable, Optional

import solara

from gui.i18n import t
from gui.scripts.product_rows import (
    ACTIVE_SAMPLING_STATUSES,
    format_sample_points,
    sample_rows,
)
from gui.widget.product_table import ProductTable

logger = logging.getLogger("spatial_risk")


@solara.component
def SampleSetList(
    project,
    sampling_jobs=None,
    on_map=frozenset(),
    on_toggle_map: Optional[Callable[[str], None]] = None,
    on_remove: Optional[Callable[[str], None]] = None,
    on_dismiss: Optional[Callable[[str], None]] = None,
    on_open: Optional[Callable[[str], None]] = None,
    pending=frozenset(),
):
    """Samples table: one row per registered sample plus in-flight/failed jobs.

    Args:
        project: solara.Reactive[Project] — source of project.samples.
        sampling_jobs: solara.Reactive[list] | None — transient session jobs.
        on_map: set of sample keys currently shown on the map.
        on_toggle_map: callback(sample_key) — toggle a sample's map layer.
        on_remove: callback(sample_key) — delete a registered sample.
        on_dismiss: callback(job_id) — discard a failed job row.
        on_open: callback(sample_key) — open the read-only details dialog for a
            registered sample (info action button).
        pending: set of sample keys with a map layer in flight.
    """
    p = project.value
    jobs = sampling_jobs.value if sampling_jobs is not None else []
    data = sample_rows(p, jobs)

    rows = []
    for r in data:
        actions = []
        if r["kind"] == "sample":
            key = r["key"]
            alloc = f" / {r['allocation']}" if r["allocation"] else ""
            points = format_sample_points(
                r["n_total"],
                r["class_counts"],
                r["strategy"],
                more_fmt=t("widgets.sample_set_list.more_strata"),
            )
            if on_open is not None:
                actions.append(
                    {"kind": "open", "on_click": lambda *_, k=key: on_open(k)}
                )
            if on_toggle_map is not None:
                actions.append(
                    {
                        "kind": "map_toggle",
                        "on_click": lambda *_, k=key: on_toggle_map(k),
                        "is_on": key in on_map,
                        "disabled": key in pending,
                    }
                )
            if on_remove is not None:
                actions.append(
                    {"kind": "delete", "on_click": lambda *_, k=key: on_remove(k)}
                )
        else:
            alloc = ""
            if r.get("n_total") is not None:
                # Known as soon as the points are drawn -- shown while the map
                # tiles are still being built.
                points = format_sample_points(
                    r["n_total"],
                    r.get("class_counts") or {},
                    r["strategy"],
                    more_fmt=t("widgets.sample_set_list.more_strata"),
                )
            else:
                points = "…" if r["status"] in ACTIVE_SAMPLING_STATUSES else "—"
            if r["status"] not in ACTIVE_SAMPLING_STATUSES and on_dismiss is not None:
                actions.append(
                    {
                        "kind": "dismiss",
                        "on_click": lambda *_, i=r["job_id"]: on_dismiss(i),
                    }
                )

        error = r.get("error")
        if r["status"] == "failed" and not error:
            error = t("widgets.sample_set_list.unknown_error")
        rows.append(
            {
                "key": r["key"],
                "cells": [
                    {"type": "text", "value": r["name"]},
                    {
                        "type": "text",
                        "value": f"{r['strategy']}{alloc}",
                        "size": "0.8rem",
                    },
                    {"type": "text", "value": points, "size": "0.8rem", "muted": True},
                    {"type": "status", "status": r["status"]},
                ],
                "actions": actions,
                "error": error,
            }
        )

    ProductTable(
        title=t("widgets.sample_set_list.title"),
        columns=[
            {"label": t("widgets.sample_set_list.col_name"), "width": "minmax(0,2fr)"},
            {
                "label": t("widgets.sample_set_list.col_strategy"),
                "width": "minmax(0,1fr)",
            },
            {"label": t("widgets.sample_set_list.col_points"), "width": "70px"},
            {"label": t("widgets.sample_set_list.col_status"), "width": "82px"},
        ],
        rows=rows,
        empty_text=t("widgets.sample_set_list.empty"),
    )
