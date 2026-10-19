"""Registered datasets table widget."""

import logging
from typing import Callable, Optional

import solara

from gui.i18n import t
from gui.widget.product_table import W_COUNT, W_NAME, W_STATUS, W_YEAR, ProductTable

logger = logging.getLogger("spatial_risk")


@solara.component
def DatasetList(
    project,
    on_edit: Optional[Callable[[str], None]] = None,
    on_remove: Optional[Callable[[str], None]] = None,
    on_duplicate: Optional[Callable[[str], None]] = None,
    on_open: Optional[Callable[[str], None]] = None,
):
    """Table of registered datasets with duplicate, edit and remove actions.

    ``on_open`` adds a details action that opens the read-only dataset dialog.
    """
    p = project.value
    datasets = (p.datasets if p is not None else {}) or {}

    rows = []
    for key, ds in datasets.items():
        actions = []
        if on_open is not None:
            actions.append({"kind": "open", "on_click": lambda *_, k=key: on_open(k)})
        if on_duplicate is not None:
            actions.append(
                {"kind": "duplicate", "on_click": lambda *_, k=key: on_duplicate(k)}
            )
        if on_edit is not None:
            actions.append({"kind": "edit", "on_click": lambda *_, k=key: on_edit(k)})
        if on_remove is not None:
            actions.append(
                {"kind": "delete", "on_click": lambda *_, k=key: on_remove(k)}
            )
        rows.append(
            {
                "key": key,
                "cells": [
                    {
                        "type": "text",
                        "value": key,
                        "subtitle": t(
                            "widgets.dataset_list.subtitle",
                            target=ds.target.name if ds.target else "—",
                            n=len(ds.features),
                        ),
                    },
                    {"type": "text", "value": str(len(ds.features)), "muted": True},
                    {
                        "type": "text",
                        "value": str(ds.year) if ds.year else "—",
                        "muted": True,
                    },
                    {"type": "status", "status": "ready"},
                ],
                "actions": actions,
            }
        )

    ProductTable(
        title=t("widgets.dataset_list.title"),
        columns=[
            {"label": t("widgets.dataset_list.col_name"), "width": W_NAME},
            {"label": t("widgets.dataset_list.col_feats"), "width": W_COUNT},
            {"label": t("widgets.dataset_list.col_year"), "width": W_YEAR},
            {"label": t("widgets.product_table.col_status"), "width": W_STATUS},
        ],
        rows=rows,
        empty_text=t("widgets.dataset_list.empty"),
    )
