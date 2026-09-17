"""Source and Derived variable list widgets."""

from typing import Callable, Optional

import solara

from gui.i18n import t
from gui.scripts.map_helpers import is_mappable
from gui.scripts.product_rows import derived_rows
from gui.scripts.variable_identity import is_base_raster
from gui.widget.product_table import ProductTable


def derived_source_key(p, var_name, fallback, year=None):
    """Raw-variable key a derived name traces back to.

    Change layers are named ``{op}_{source}_...`` — strip the operation prefix
    so they resolve to their start layer instead of "unknown".

    A name can belong to several raw variables (one temporal layer, one entry
    per year), and a reprojected copy keeps its source's ``year``, so prefer the
    candidate whose year matches. Change layers carry no year — their span lives
    in ``tags`` — and fall back to the first name match.
    """
    base = var_name
    for prefix in ("loss_", "gain_"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    matches = [
        k for k, raw_var in p.raw_variables.items() if base.startswith(raw_var.name)
    ]
    if year is not None:
        for key in matches:
            if getattr(p.raw_variables[key], "year", None) == year:
                return key
    return matches[0] if matches else fallback


@solara.component
def SourceVariableList(
    project,
    on_remove: Callable[[str], None],
    on_edit: Optional[Callable[[str], None]] = None,
    on_toggle_map: Optional[Callable[[str], None]] = None,
    vars_on_map=None,
    on_download: Optional[Callable[[str], None]] = None,
    downloading_keys: frozenset = frozenset(),
):
    """Table of source (raw) variables with download/map/edit/remove actions.

    Cloud-backed variables (GEEVar) show a "cloud" chip and, when
    ``on_download`` is given, a per-row download button. ``downloading_keys``
    are the keys whose download is running: their button spins and is
    disabled; every other row stays clickable (downloads run in parallel).
    """
    p = project.value
    raw_variables = (p.raw_variables if p is not None else {}) or {}
    on_map = vars_on_map.value if vars_on_map is not None else set()

    rows = []
    for key, var in raw_variables.items():
        is_base = is_base_raster(p, var)
        data_type_label = (
            var.data_type if isinstance(var.data_type, str) else var.data_type.value
        )
        is_cloud = type(var).__name__ == "GEEVar"

        status_chip = (
            {
                "value": t("widgets.variable_list.chip_cloud"),
                "icon": "mdi-cloud-outline",
                "color": "warning",
            }
            if is_cloud
            else {"value": t("widgets.variable_list.chip_local"), "color": "success"}
        )

        actions = []
        if on_download is not None and is_cloud:
            actions.append(
                {
                    "kind": "download",
                    "on_click": lambda *_, k=key: on_download(k),
                    "loading": key in downloading_keys,
                    "disabled": key in downloading_keys,
                }
            )
        if on_toggle_map is not None and is_mappable(var):
            actions.append(
                {
                    "kind": "map_toggle",
                    "on_click": lambda *_, k=key: on_toggle_map(k),
                    "is_on": key in on_map,
                }
            )
        if on_edit is not None:
            actions.append({"kind": "edit", "on_click": lambda *_, k=key: on_edit(k)})
        actions.append({"kind": "delete", "on_click": lambda *_, k=key: on_remove(k)})

        name_chips = (
            [
                {
                    "value": t("widgets.variable_list.chip_base"),
                    "color": "info",
                    "outlined": False,
                }
            ]
            if is_base
            else []
        )
        rows.append(
            {
                "key": key,
                "cells": [
                    {"type": "text", "value": var.name, "chips": name_chips},
                    {
                        "type": "chips",
                        "items": [
                            {"value": data_type_label, "color": "primary"},
                            status_chip,
                        ],
                    },
                    {
                        "type": "text",
                        "value": str(var.year) if var.year else "—",
                        "muted": True,
                    },
                ],
                "actions": actions,
            }
        )

    ProductTable(
        title=t("widgets.variable_list.source_title"),
        columns=[
            {
                "label": t("widgets.variable_list.source_col_name"),
                "width": "minmax(0,2fr)",
            },
            {"label": t("widgets.variable_list.source_col_type"), "width": "150px"},
            {"label": t("widgets.variable_list.source_col_year"), "width": "44px"},
        ],
        rows=rows,
        empty_text=t("widgets.variable_list.source_empty"),
    )


@solara.component
def DerivedVariableList(
    project,
    on_remove: Optional[Callable[[str], None]] = None,
    keys: Optional[list] = None,
    on_toggle_map: Optional[Callable[[str], None]] = None,
    derived_on_map=None,
    title: Optional[str] = None,
    jobs=None,
    on_dismiss: Optional[Callable[[str], None]] = None,
):
    """Derived (processed) variables, plus the layers still being generated.

    ``keys`` restricts the product rows to those registry keys (None = all).
    ``derived_on_map`` is the reactive set of keys currently drawn on the map
    (see ``gui/tile/derived_map.py``), which drives the toggle state.

    ``jobs`` is the reactive list of session job dicts for submissions still
    running — the same overlay the Train/Sampling/Inference tabs use, so a
    derived layer is a row from the moment its form is submitted instead of
    appearing out of nowhere minutes later. A job row has no product to act on
    (and the GDAL pass behind it is not cancellable), so it carries no actions
    until it fails, when ``on_dismiss`` lets the user clear it.
    """
    p = project.value
    if p is None:
        return
    data = derived_rows(p, jobs.value if jobs is not None else None, keys)
    if not data:
        return
    on_map = derived_on_map.value if derived_on_map is not None else set()
    unknown_source = t("widgets.variable_list.derived_source_unknown")

    rows = []
    for r in data:
        actions = []
        if r["kind"] == "variable":
            var = p.processed_variables[r["key"]]
            source_name = derived_source_key(
                p, var.name, unknown_source, year=getattr(var, "year", None)
            )
            if on_toggle_map is not None and is_mappable(var):
                actions.append(
                    {
                        "kind": "map_toggle",
                        "on_click": lambda *_, k=r["key"]: on_toggle_map(k),
                        "is_on": r["key"] in on_map,
                    }
                )
            if on_remove is not None:
                actions.append(
                    {"kind": "delete", "on_click": lambda *_, k=r["key"]: on_remove(k)}
                )
        else:
            # The output has no registry entry yet, so the source is resolved
            # from the name the job will register under.
            source_name = derived_source_key(p, r["name"], unknown_source)
            if r["status"] != "running" and on_dismiss is not None:
                actions.append(
                    {
                        "kind": "dismiss",
                        "on_click": lambda *_, i=r["job_id"]: on_dismiss(i),
                    }
                )

        error = r.get("error")
        if r["status"] == "failed" and not error:
            error = t("widgets.variable_list.derived_unknown_error")
        rows.append(
            {
                "key": r["key"],
                "cells": [
                    {"type": "text", "value": r["name"], "size": "0.9rem"},
                    {"type": "chip", "value": source_name},
                    {"type": "status", "status": r["status"]},
                ],
                "actions": actions,
                "error": error,
            }
        )

    ProductTable(
        title=title or t("widgets.variable_list.derived_title"),
        columns=[
            {
                "label": t("widgets.variable_list.derived_col_name"),
                "width": "minmax(0,2fr)",
            },
            {"label": t("widgets.variable_list.derived_col_source"), "width": "120px"},
            {"label": t("widgets.variable_list.derived_col_status"), "width": "90px"},
        ],
        rows=rows,
        empty_text="",
    )


def harmonization_row_status(
    key: str, var, status, running_keys, is_cloud: bool
) -> str:
    """ProductTable status token for one raw variable in the Step 3 list.

    Precedence: a run rewriting this layer beats everything; a cloud-backed
    layer is "not downloaded" whatever the grid check says (it has no local
    file to check); a None ``status`` means the off-thread check has not
    resolved yet.
    """
    if key in (running_keys or ()):
        return "running"
    if is_cloud:
        return "not_downloaded"
    if status is None:
        return "checking"
    return "pending" if key in status.pending else "harmonized"


@solara.component
def HarmonizationVariableList(
    project,
    status,
    on_harmonize: Callable[[str], None],
    running_keys=None,
    harmonize_disabled: bool = False,
    on_toggle_map: Optional[Callable[[str], None]] = None,
    derived_on_map=None,
    on_remove: Optional[Callable[[str], None]] = None,
):
    """Every harmonizable source variable with its harmonization status.

    The Step 3 counterpart of ``SourceVariableList``: one row per raw variable
    Run would touch (rasters and active vectors), a Status column fed by
    ``harmonization_status`` (``status``; None while it is still being
    computed off-thread), and a per-row harmonize action that is enabled only
    on pending rows — the same shape as the per-row download in Step 2.

    ``running_keys`` are the raw keys a run is currently rewriting (spinner on
    those rows); every harmonize button is disabled while ``harmonize_disabled``.
    Map toggle and remove act on the harmonized *output*, so they are shown
    only once one is registered; their callbacks receive the processed key.
    """
    from spatialrisk.harmonization import is_harmonizable, output_key

    p = project.value
    raw_variables = (p.raw_variables if p is not None else {}) or {}
    processed = (p.processed_variables if p is not None else {}) or {}
    on_map = derived_on_map.value if derived_on_map is not None else set()
    running = set(running_keys or ())

    rows = []
    for key, var in raw_variables.items():
        if not is_harmonizable(var):
            continue
        is_cloud = type(var).__name__ == "GEEVar"
        row_status = harmonization_row_status(key, var, status, running, is_cloud)
        out_key = output_key(var)
        output = processed.get(out_key)
        data_type_label = (
            var.data_type if isinstance(var.data_type, str) else var.data_type.value
        )

        actions = [
            {
                "kind": "harmonize",
                "on_click": lambda *_, k=key: on_harmonize(k),
                "loading": key in running,
                "disabled": harmonize_disabled
                or row_status not in ("pending", "not_downloaded"),
            }
        ]
        if output is not None and on_toggle_map is not None and is_mappable(output):
            actions.append(
                {
                    "kind": "map_toggle",
                    "on_click": lambda *_, k=out_key: on_toggle_map(k),
                    "is_on": out_key in on_map,
                }
            )
        if output is not None and on_remove is not None:
            actions.append(
                {"kind": "delete", "on_click": lambda *_, k=out_key: on_remove(k)}
            )

        name_chips = (
            [
                {
                    "value": t("widgets.variable_list.chip_base"),
                    "color": "info",
                    "outlined": False,
                }
            ]
            if is_base_raster(p, var)
            else []
        )
        rows.append(
            {
                "key": key,
                "cells": [
                    {"type": "text", "value": var.name, "chips": name_chips},
                    {"type": "chip", "value": data_type_label, "color": "primary"},
                    {
                        "type": "text",
                        "value": str(var.year) if getattr(var, "year", None) else "—",
                        "muted": True,
                    },
                    {"type": "status", "status": row_status},
                ],
                "actions": actions,
            }
        )

    ProductTable(
        title=t("widgets.variable_list.harmonization_title"),
        columns=[
            {
                "label": t("widgets.variable_list.source_col_name"),
                "width": "minmax(0,2fr)",
            },
            # Lean fixed widths: the panel is ~450px wide and the Name column
            # takes whatever is left, so every fixed px here is a truncated
            # name (see list-grid alignment: no max-content columns).
            {"label": t("widgets.variable_list.source_col_type"), "width": "58px"},
            {"label": t("widgets.variable_list.source_col_year"), "width": "40px"},
            {
                "label": t("widgets.variable_list.harmonization_col_status"),
                "width": "108px",
            },
        ],
        rows=rows,
        empty_text=t("widgets.variable_list.source_empty"),
    )
