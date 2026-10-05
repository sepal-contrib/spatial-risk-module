"""Predictions list for the Inference tab: registry products + session-job overlay."""

import solara

from gui.i18n import t
from gui.scripts.product_rows import inference_rows
from gui.widget.product_table import ProductTable


@solara.component
def InferenceOutputList(
    project,
    inference_jobs,
    preds_on_map=None,
    on_toggle_map=None,
    on_dismiss=None,
    on_delete=None,
    on_edit=None,
    on_open=None,
    toggling_keys: frozenset = frozenset(),
):
    """Predictions table: one row per registered prediction group plus jobs.

    Args:
        project: solara.Reactive[Project] — source of project.predictions.
        inference_jobs: solara.Reactive[list] — transient session job dicts.
        preds_on_map: solara.Reactive[set] — row keys currently on the map.
        on_toggle_map: callback(row) — add/remove ALL the row's rasters.
        on_dismiss: callback(job_id) — discard a failed job row.
        on_delete: callback(row) — delete ALL the row's registered rasters
            (confirmed by the tile).
        on_edit: callback(row) — reopen the Predict dialog prefilled with a
            failed job's submission entry so the user can fix and rerun.
        on_open: callback(row) — open the read-only provenance dialog for a
            registered prediction (the info action button); None omits the
            button. Job rows never get one: a run with no registered output
            has nothing to explain yet.
        toggling_keys: row keys whose map toggle is running; that row's map
            button is an hourglass that ignores clicks until its rasters are on the map.
    """
    p = project.value
    data = inference_rows(p, inference_jobs.value)
    on_map = preds_on_map.value if preds_on_map is not None else set()

    rows = []
    for r in data:
        actions = []
        if r["kind"] == "prediction":
            if on_open is not None:
                actions.append(
                    {"kind": "open", "on_click": lambda *_, rr=r: on_open(rr)}
                )
            if on_toggle_map is not None:
                actions.append(
                    {
                        "kind": "map_toggle",
                        "on_click": lambda *_, rr=r: on_toggle_map(rr),
                        "is_on": r["key"] in on_map,
                        "loading": r["key"] in toggling_keys,
                    }
                )
            if on_delete is not None:
                actions.append(
                    {"kind": "delete", "on_click": lambda *_, rr=r: on_delete(rr)}
                )
        elif r["status"] != "running":
            # Only jobs that recorded their submission entry can be re-edited.
            if r["status"] == "failed" and r.get("entry") and on_edit is not None:
                actions.append(
                    {"kind": "edit", "on_click": lambda *_, rr=r: on_edit(rr)}
                )
            if on_dismiss is not None:
                actions.append(
                    {
                        "kind": "dismiss",
                        "on_click": lambda *_, i=r["job_id"]: on_dismiss(i),
                    }
                )

        error = r.get("error")
        if r["status"] == "failed" and not error:
            error = t("widgets.inference_output_list.unknown_error")
        rows.append(
            {
                "key": r["key"],
                "cells": [
                    {"type": "text", "value": r["name"]},
                    {"type": "chip", "value": r["model_key"], "color": "primary"},
                    {
                        "type": "text",
                        "value": r["dataset_name"],
                        "size": "0.8rem",
                        "muted": True,
                    },
                    {"type": "status", "status": r["status"]},
                ],
                "actions": actions,
                "error": error,
            }
        )

    ProductTable(
        title=t("widgets.inference_output_list.predictions_title"),
        columns=[
            {
                "label": t("widgets.inference_output_list.col_name"),
                "width": "minmax(0,2fr)",
            },
            {"label": t("widgets.inference_output_list.col_model"), "width": "90px"},
            {
                "label": t("widgets.inference_output_list.col_dataset"),
                "width": "minmax(0,1fr)",
            },
            {"label": t("widgets.inference_output_list.col_status"), "width": "95px"},
        ],
        rows=rows,
        empty_text=t("widgets.inference_output_list.empty"),
    )
