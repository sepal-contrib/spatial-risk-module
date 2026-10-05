"""Registered datasets table widget."""

import logging
from typing import Callable, Optional

import solara

from gui.i18n import t
from gui.widget.product_table import ProductTable

logger = logging.getLogger("spatial_risk")


@solara.component
def DatasetList(
    project,
    on_edit: Optional[Callable[[str], None]] = None,
    on_remove: Optional[Callable[[str], None]] = None,
    on_duplicate: Optional[Callable[[str], None]] = None,
):
    """Table of registered datasets with duplicate, edit and remove actions."""
    p = project.value
    datasets = (p.datasets if p is not None else {}) or {}

    rows = []
    for key, ds in datasets.items():
        actions = []
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
                    {"type": "text", "value": key},
                    {
                        "type": "chip",
                        "value": ds.target.name if ds.target else "—",
                        "color": "error",
                    },
                    {"type": "chip", "value": str(len(ds.features))},
                    {
                        "type": "text",
                        "value": str(ds.year) if ds.year else "—",
                        "muted": True,
                        "size": "0.85rem",
                    },
                ],
                "actions": actions,
            }
        )

    ProductTable(
        title=t("widgets.dataset_list.title"),
        columns=[
            {"label": t("widgets.dataset_list.col_name"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.dataset_list.col_target"), "width": "minmax(0,1fr)"},
            {"label": t("widgets.dataset_list.col_feats"), "width": "60px"},
            {"label": t("widgets.dataset_list.col_year"), "width": "60px"},
        ],
        rows=rows,
        empty_text=t("widgets.dataset_list.empty"),
    )
