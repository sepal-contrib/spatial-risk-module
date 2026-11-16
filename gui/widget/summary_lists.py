"""Render-only, read-only summary tables for the Project Summary popup.

Each renderer takes the ``project`` *reactive* (not the plain value) and reads
``p = project.value`` at the top, shapes it via a Solara-free helper, and
renders a read-only ProductTable (``show_actions=False``) with the one-line
stats banner.

Taking the reactive (rather than a plain ``Project``) is load-bearing: the app
replaces ``project`` via ``project.set(p.model_copy())`` on every mutation, and
``model_copy()`` is *shallow*, so the new Project compares ``==`` to the
previous one reacton last rendered. Passing that value as a prop would trip
reacton's prop-equality bailout and freeze the table at its first snapshot;
subscribing to the reactive here re-renders on every set instead.
"""

import solara

from gui.i18n import t
from gui.scripts.product_rows import format_sample_points
from gui.scripts.summary_helpers import (
    dataset_rows,
    evaluation_rows,
    model_rows,
    prediction_rows,
    processed_variable_rows,
    raw_variable_rows,
    sample_rows,
)
from gui.widget.product_table import ProductTable


def _summary_table(title, banner, columns, rows, empty_text):
    ProductTable(
        title=title,
        columns=columns,
        rows=rows,
        empty_text=empty_text,
        show_actions=False,
        banner=banner,
    )


@solara.component
def RawVariablesSummary(project):
    """Summary table of the project's raw (source) variables."""
    p = project.value
    if p is None:
        return
    stats, data = raw_variable_rows(p)
    rows = [
        {
            "key": str(r["name"]),
            "cells": [
                {
                    "type": "text",
                    "value": str(r["name"]),
                    "chips": (
                        [
                            {
                                "value": t("widgets.summary_lists.chip_base"),
                                "color": "info",
                                "outlined": False,
                            }
                        ]
                        if r["is_base"]
                        else []
                    ),
                },
                {"type": "chip", "value": str(r["data_type"]), "color": "primary"},
                {"type": "text", "value": str(r["raster_type"]), "muted": True},
                {"type": "text", "value": str(r["year"]), "muted": True},
            ],
        }
        for r in data
    ]
    _summary_table(
        t("widgets.summary_lists.raw_vars_title"),
        t(
            "widgets.summary_lists.vars_banner",
            total=stats["total"],
            vector=stats["vector"],
            raster=stats["raster"],
        ),
        [
            {"label": t("widgets.summary_lists.col_name"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.summary_lists.col_type"), "width": "90px"},
            {"label": t("widgets.summary_lists.col_raster_type"), "width": "110px"},
            {"label": t("widgets.summary_lists.col_year"), "width": "70px"},
        ],
        rows,
        t("widgets.summary_lists.raw_vars_empty"),
    )


@solara.component
def ProcessedVariablesSummary(project):
    """Summary table of the harmonized and derived layers."""
    p = project.value
    if p is None:
        return
    stats, data = processed_variable_rows(p)
    rows = [
        {
            "key": str(r["name"]),
            "cells": [
                {"type": "text", "value": str(r["name"])},
                {"type": "chip", "value": str(r["source"])},
                {"type": "text", "value": str(r["raster_type"]), "muted": True},
                {"type": "text", "value": str(r["year"]), "muted": True},
            ],
        }
        for r in data
    ]
    _summary_table(
        t("widgets.summary_lists.processed_vars_title"),
        t(
            "widgets.summary_lists.vars_banner",
            total=stats["total"],
            vector=stats["vector"],
            raster=stats["raster"],
        ),
        [
            {"label": t("widgets.summary_lists.col_name"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.summary_lists.col_source"), "width": "120px"},
            {"label": t("widgets.summary_lists.col_raster_type"), "width": "110px"},
            {"label": t("widgets.summary_lists.col_year"), "width": "70px"},
        ],
        rows,
        t("widgets.summary_lists.processed_vars_empty"),
    )


@solara.component
def DatasetsSummary(project):
    """Summary table of the project's datasets."""
    p = project.value
    if p is None:
        return
    stats, data = dataset_rows(p)
    rows = [
        {
            "key": str(r["name"]),
            "cells": [
                {"type": "text", "value": str(r["name"])},
                {"type": "chip", "value": str(r["target_name"]), "color": "error"},
                {"type": "chip", "value": str(r["feature_count"])},
                {"type": "text", "value": str(r["year"]), "muted": True},
            ],
        }
        for r in data
    ]
    _summary_table(
        t("widgets.summary_lists.datasets_title"),
        t("widgets.summary_lists.datasets_banner", total=stats["total"]),
        [
            {"label": t("widgets.summary_lists.col_name"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.summary_lists.col_target"), "width": "minmax(0,1fr)"},
            {"label": t("widgets.summary_lists.col_feats"), "width": "70px"},
            {"label": t("widgets.summary_lists.col_year"), "width": "70px"},
        ],
        rows,
        t("widgets.summary_lists.datasets_empty"),
    )


@solara.component
def SamplesSummary(project):
    """Summary table of the project's sample sets."""
    p = project.value
    if p is None:
        return
    stats, data = sample_rows(p)
    rows = []
    for r in data:
        alloc = (
            f" / {r['allocation']}"
            if r["allocation"] and r["allocation"] != "—"
            else ""
        )
        points_str = format_sample_points(
            r["n_total"],
            r["class_counts"],
            r["strategy"],
            more_fmt=t("widgets.summary_lists.more_strata"),
        )
        rows.append(
            {
                "key": str(r["name"]),
                "cells": [
                    {"type": "text", "value": str(r["name"])},
                    {
                        "type": "text",
                        "value": f"{r['strategy']}{alloc}",
                        "size": "0.8rem",
                    },
                    {
                        "type": "text",
                        "value": points_str,
                        "size": "0.8rem",
                        "muted": True,
                    },
                    {"type": "text", "value": str(r["seed"]), "muted": True},
                ],
            }
        )
    _summary_table(
        t("widgets.summary_lists.samples_title"),
        t(
            "widgets.summary_lists.samples_banner",
            total=stats["total"],
            points=stats["points"],
        ),
        [
            {"label": t("widgets.summary_lists.col_name"), "width": "minmax(0,2fr)"},
            {
                "label": t("widgets.summary_lists.col_strategy"),
                "width": "minmax(0,1fr)",
            },
            {"label": t("widgets.summary_lists.col_points_classes"), "width": "150px"},
            {"label": t("widgets.summary_lists.col_seed"), "width": "60px"},
        ],
        rows,
        t("widgets.summary_lists.samples_empty"),
    )


@solara.component
def ModelsSummary(project):
    """Summary table of the project's trained models."""
    p = project.value
    if p is None:
        return
    stats, data = model_rows(p)
    rows = [
        {
            "key": str(r["name"]),
            "cells": [
                {
                    "type": "text",
                    "value": str(r["name"]),
                    "chips": (
                        [
                            {
                                "value": t("widgets.summary_lists.chip_trained"),
                                "color": "success",
                                "outlined": False,
                            }
                        ]
                        if r["trained"]
                        else []
                    ),
                },
                {"type": "chip", "value": str(r["model_type"]), "color": "primary"},
                {"type": "text", "value": str(r["year"]), "muted": True},
                {"type": "text", "value": str(r["n_samples"]), "muted": True},
                {"type": "text", "value": str(r["deviance"]), "muted": True},
                {
                    "type": "text",
                    "value": str(r["params"]),
                    "muted": True,
                    "size": "0.78rem",
                },
            ],
        }
        for r in data
    ]
    _summary_table(
        t("widgets.summary_lists.models_title"),
        t(
            "widgets.summary_lists.models_banner",
            total=stats["total"],
            trained=stats["trained"],
        ),
        [
            {"label": t("widgets.summary_lists.col_name"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.summary_lists.col_type"), "width": "70px"},
            {"label": t("widgets.summary_lists.col_year"), "width": "60px"},
            {"label": t("widgets.summary_lists.col_samples"), "width": "80px"},
            {"label": t("widgets.summary_lists.col_deviance"), "width": "90px"},
            {
                "label": t("widgets.summary_lists.col_params"),
                "width": "minmax(0,1.2fr)",
            },
        ],
        rows,
        t("widgets.summary_lists.models_empty"),
    )


@solara.component
def PredictionsSummary(project):
    """Summary table of the project's predictions."""
    p = project.value
    if p is None:
        return
    stats, data = prediction_rows(p)
    rows = [
        {
            "key": f"{r['model_key']}_{i}",
            "cells": [
                {"type": "text", "value": str(r["model_key"])},
                {"type": "chip", "value": str(r["dataset_name"])},
                {"type": "text", "value": str(r["year"]), "muted": True},
                {"type": "text", "value": str(r["window"]), "muted": True},
                {"type": "text", "value": "✓" if r["active"] else "—", "muted": True},
            ],
        }
        for i, r in enumerate(data)
    ]
    _summary_table(
        t("widgets.summary_lists.predictions_title"),
        t(
            "widgets.summary_lists.predictions_banner",
            total=stats["total"],
            active=stats["active"],
        ),
        [
            {"label": t("widgets.summary_lists.col_model"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.summary_lists.col_dataset"), "width": "minmax(0,1fr)"},
            {"label": t("widgets.summary_lists.col_year"), "width": "70px"},
            {"label": t("widgets.summary_lists.col_window"), "width": "70px"},
            {"label": t("widgets.summary_lists.col_active"), "width": "60px"},
        ],
        rows,
        t("widgets.summary_lists.predictions_empty"),
    )


@solara.component
def EvaluationsSummary(project):
    """Summary table of the project's evaluation runs."""
    p = project.value
    if p is None:
        return
    stats, data = evaluation_rows(p)
    rows = [
        {
            "key": str(r["name"]),
            "cells": [
                {"type": "text", "value": str(r["name"])},
                {"type": "chip", "value": str(r["n_predictions"])},
                {
                    "type": "text",
                    "value": str(r["csizes"]),
                    "size": "0.8rem",
                    "muted": True,
                },
                {
                    "type": "text",
                    "value": str(r["metrics"]),
                    "size": "0.8rem",
                    "muted": True,
                },
                {
                    "type": "text",
                    "value": str(r["created_at"]),
                    "size": "0.78rem",
                    "muted": True,
                },
            ],
        }
        for r in data
    ]
    _summary_table(
        t("widgets.summary_lists.evaluations_title"),
        t("widgets.summary_lists.evaluations_banner", total=stats["total"]),
        [
            {"label": t("widgets.summary_lists.col_name"), "width": "minmax(0,2fr)"},
            {"label": t("widgets.summary_lists.col_pred_count"), "width": "60px"},
            {
                "label": t("widgets.summary_lists.col_cell_sizes"),
                "width": "minmax(0,1fr)",
            },
            {"label": t("widgets.summary_lists.col_metrics"), "width": "minmax(0,1fr)"},
            {"label": t("widgets.summary_lists.col_created"), "width": "minmax(0,1fr)"},
        ],
        rows,
        t("widgets.summary_lists.evaluations_empty"),
    )
