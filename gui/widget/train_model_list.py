"""Trained-models list for the Train tab: registry products + session-job overlay."""

import logging

import solara

from gui.i18n import t
from gui.scripts.product_rows import train_rows
from gui.widget.product_table import W_CHIP, W_NAME, W_STATUS, ProductTable

logger = logging.getLogger("spatial_risk")


@solara.component
def TrainModelList(project, train_jobs, model_labels, on_dismiss, on_delete, on_open):
    """Models table: one row per registered model plus in-flight/failed jobs.

    Args:
        project: solara.Reactive[Project] — source of project.models.
        train_jobs: solara.Reactive[list] — transient session job dicts.
        model_labels: dict model_type -> human label.
        on_dismiss: callback(job_id) — discard a failed/cancelled job row.
            A running job offers no action: training cannot be interrupted.
        on_delete: callback(model_key) — delete a registered model (confirmed
            by the tile).
        on_open: callback(model_key) — open the read-only details dialog for a
            registered model (info action button).
    """
    p = project.value
    data = train_rows(p, train_jobs.value, model_labels)

    rows = []
    for r in data:
        if r["kind"] == "model":
            actions = [
                {"kind": "open", "on_click": lambda *_, k=r["key"]: on_open(k)},
                {"kind": "delete", "on_click": lambda *_, k=r["key"]: on_delete(k)},
            ]
        elif r["status"] == "running":
            # Training cannot be interrupted (the worker has no cancel hook), so
            # a Cancel button only hid the row while the fit kept running.
            actions = []
        else:
            actions = [
                {"kind": "dismiss", "on_click": lambda *_, i=r["job_id"]: on_dismiss(i)}
            ]

        error = r.get("error")
        if r["status"] == "failed" and not error:
            error = t("widgets.train_model_list.unknown_error")
        rows.append(
            {
                "key": r["key"],
                "cells": [
                    {
                        "type": "text",
                        "value": r["name"],
                        "subtitle": f"{r['dataset_name']} · {r['sample_name']}",
                    },
                    {"type": "chip", "value": r["model_label"], "color": "primary"},
                    {"type": "status", "status": r["status"]},
                ],
                "actions": actions,
                "error": error,
            }
        )

    ProductTable(
        title=t("widgets.train_model_list.models_title"),
        columns=[
            {"label": t("widgets.train_model_list.col_name"), "width": W_NAME},
            {"label": t("widgets.train_model_list.col_type"), "width": W_CHIP},
            {"label": t("widgets.train_model_list.col_status"), "width": W_STATUS},
        ],
        rows=rows,
        empty_text=t("widgets.train_model_list.empty"),
    )
