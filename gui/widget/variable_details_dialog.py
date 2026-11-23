"""Read-only details of one Harmonization row.

The field selection is a plain function (``harmonized_details``) so it is
tested without rendering. It reads only what the project already holds: no
file is opened, so rendering never waits on the disk.
"""

from typing import Callable, List, NamedTuple, Optional, Tuple

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.variable_labels import year_label
from gui.widget.details_fields import ro_field
from spatialrisk.harmonization import (
    expected_raster_type,
    lacks_categorical_levels,
    output_key,
)

#: The rule ``LocalRasterVar.reproject_and_match`` applies when no method is
#: passed: nearest for categorical, bilinear for continuous.
DEFAULT_RESAMPLING = {"categorical": "nearest", "continuous": "bilinear"}
#: Its fallback when the raster type is unknown.
FALLBACK_RESAMPLING = "nearest"

_K = "widgets.variable_details."


class VariableDetails(NamedTuple):
    """What the dialog shows for one row."""

    name: str
    rows: List[Tuple[str, object]]


def _plain(value):
    """``RasterType.continuous`` -> ``"continuous"`` (str enums and plain strs)."""
    return getattr(value, "value", value)


def resampling_label(raw, output) -> Optional[str]:
    """The resampling method of a harmonized layer.

    The method stored on the output when there is one. An output written
    before it was stored, or a row not harmonized yet, shows the method the
    rule picks for its raster type, marked as that default. None for a vector
    (it is rasterized, not resampled).
    """
    stored = getattr(output, "resampling", None) if output is not None else None
    if stored:
        return str(stored)
    if _plain(getattr(raw, "data_type", None)) == "vector":
        return None
    raster_type = _plain(
        getattr(output, "raster_type", None)
        if output is not None
        else expected_raster_type(raw)
    )
    method = DEFAULT_RESAMPLING.get(raster_type)
    if method is None:
        return t(_K + "resampling_default_untyped", method=FALLBACK_RESAMPLING)
    return t(_K + "resampling_default", method=method, raster_type=raster_type)


def matches_reference_grid(key, raw, output, status):
    """Whether ``key``'s harmonized output still sits on the reference grid.

    ``status`` is the verdict the Harmonization list shows. True when the row
    is current and False when the reference moved. A row that is outdated for
    another reason (its raster type changed, or its categories are missing)
    returns that reason instead. None when nothing is harmonized or the check
    has not answered.
    """
    if output is None or status is None:
        return None
    if key in (getattr(status, "current", None) or ()):
        return True
    if key not in (getattr(status, "pending", None) or ()):
        return None
    if getattr(output, "raster_type", None) != expected_raster_type(raw):
        return t(_K + "grid_reason_raster_type")
    if lacks_categorical_levels(output):
        return t(_K + "grid_reason_categories")
    return False


def harmonized_details(project, key, status=None) -> Optional[VariableDetails]:
    """The details of the Harmonization row ``key`` (a raw variable key).

    None when the project or the variable is gone.
    """
    raw = (getattr(project, "raw_variables", None) or {}).get(key) if key else None
    if raw is None:
        return None
    processed = getattr(project, "processed_variables", None) or {}
    output = processed.get(output_key(raw))
    path = getattr(output, "path", None)
    rows = [
        (t(_K + "source_layer"), key),
        (t(_K + "target_crs"), getattr(output, "default_crs", None)),
        (t(_K + "resolution"), getattr(output, "default_resolution", None)),
        (t(_K + "resampling"), resampling_label(raw, output)),
        (t(_K + "matches_grid"), matches_reference_grid(key, raw, output, status)),
        (
            t(_K + "processing_steps"),
            list(getattr(output, "processing_history", None) or []) or None,
        ),
        (t(_K + "file_path"), str(path) if path else None),
    ]
    name = year_label(getattr(raw, "name", None) or key, getattr(raw, "year", None))
    return VariableDetails(name, rows)


@solara.component
def VariableDetailsDialog(
    project, key: Optional[str], on_close: Callable[[], None], status=None
):
    """Read-only view of one Harmonization row, with a Close button.

    Args:
        project: the project reactive (the dialog reads its current value).
        key: the row's raw variable key, or None while the dialog is closed.
        on_close: () -> None; clears the tile's selected key.
        status: the ``HarmonizationStatus`` the Harmonization list shows.
    """
    details = harmonized_details(project.value, key, status)
    with rv.Dialog(
        v_model=details is not None,
        on_v_model=lambda v: None if v else on_close(),
        max_width="560px",
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t(_K + "title", name=details.name if details else ""))
            with rv.CardText():
                if details is not None:
                    with solara.Column(style="gap:4px;"):
                        for label, value in details.rows:
                            ro_field(label, value)
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=lambda: on_close(),
                    text=True,
                    small=True,
                )
