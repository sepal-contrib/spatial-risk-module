"""Project-borders picker for the allocation form.

Mirrors the rate-table field's shape — a method select, that method's own
input, and a hint naming what the run will use. The widget only records the
choice as a ``BordersSelection``; ``allocation_runner.resolve_borders_file``
turns it into one canonical vector file when the run starts.

Five methods, not three: pysepal's ``AdminLevelSelector`` derives its cascade
depth from the method string, so each admin level is its own option.
"""

from pathlib import Path

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.components.inputs import (
    AdminLevelSelector,
    AssetSelectComponent,
    FileInputComponent,
)

from gui.i18n import t
from gui.scripts.allocation_runner import BordersSelection
from gui.scripts.aoi_io import admin_code_chain
from gui.widget.text_style import MUTED, TIGHT_FIELD, FieldHint

_VECTOR_EXTENSIONS = [".gpkg", ".shp", ".geojson", ".json"]
_HINT = MUTED + "font-size:0.75rem;"

#: (method, label key), in dropdown order: the admin cascade coarse-to-fine,
#: then the two bring-your-own methods.
_METHODS = [
    ("ADMIN0", "toolbox.allocation.borders_method_admin0"),
    ("ADMIN1", "toolbox.allocation.borders_method_admin1"),
    ("ADMIN2", "toolbox.allocation.borders_method_admin2"),
    ("FILE", "toolbox.allocation.borders_method_file"),
    ("ASSET", "toolbox.allocation.borders_method_asset"),
]

_ADMIN_METHODS = ("ADMIN0", "ADMIN1", "ADMIN2")


def _hint_text(selection) -> str:
    """What the run will use, in the user's words."""
    empty = t("toolbox.allocation.borders_hint_empty")
    if selection is None:
        return empty
    if selection.method == "FILE":
        return Path(selection.file_path).name if selection.file_path else empty
    if selection.method in _ADMIN_METHODS:
        if not selection.admin_code:
            return empty
        return t("toolbox.allocation.borders_hint_admin", code=selection.admin_code)
    asset = selection.asset or {}
    asset_id = asset.get("asset_id")
    if not asset_id:
        return empty
    # Mirrors allocation_runner._validate_borders: a column filter with no
    # value picked yet is not a runnable selection, so the hint must not
    # claim the asset is ready.
    column = asset.get("column")
    if column not in (None, "ALL") and asset.get("value") is None:
        return empty
    return asset_id


@solara.component
def _AdminCascade(method, admin_code, on_value):
    """Render pysepal's admin cascade, restored from the stored admin code.

    pysepal restores the cascade only from ``codes``, the whole chain with
    level 0 first. ``value`` is output only, because one code cannot name its
    parents. A selection stores just the leaf, so the chain is read from
    pygaul's local parquet, with no Earth Engine.

    The chain is read once, when the cascade mounts. The allocation form
    remounts the picker on every prefill, so each stored code still gets its
    own lookup. Changing ``codes`` while the cascade is mounted would overwrite
    what the user picked. For example, picking a new country clears the code,
    and the chain of a cleared code is empty.

    Args:
        method: "ADMIN0", "ADMIN1" or "ADMIN2".
        admin_code: the stored leaf GAUL code, or None.
        on_value: callback(BordersSelection) — the picker's own setter.
    """
    level = _ADMIN_METHODS.index(method)
    codes = solara.use_memo(
        lambda: (admin_code_chain(admin_code, level) if admin_code else None) or (),
        dependencies=[],
    )

    def on_codes(picked):
        # A user pick, published from here. The cascade stays silent when its
        # code does not change, and after the None dropped below it would say
        # nothing about a pick that clears the code.
        leaf = picked[level] if len(picked) > level else None
        on_value(BordersSelection(method=method, admin_code=leaf))

    def on_code(code):
        # Only a mount or a method change reaches this without on_codes. A
        # None then either means a chain that could not be read, which would
        # wipe the prefilled selection, or repeats what set_method stored.
        if code is not None:
            on_value(BordersSelection(method=method, admin_code=code))

    # gee=False: pysepal reads the code list from pygaul's local parquet and
    # fetches geometry from the FAO GAUL WFS, so admin borders need no Earth
    # Engine at all.
    AdminLevelSelector(
        method=method,
        gee=False,
        value=admin_code,
        on_value=on_code,
        codes=codes,
        on_codes=on_codes,
    )


@solara.component
def BordersPicker(value, on_value, sepal_client=None):
    """Collect the project borders as a BordersSelection.

    Args:
        value: the current ``BordersSelection``, or None.
        on_value: callback(BordersSelection | None) — the form's setter.
        sepal_client: passed through to the file picker.
    """
    method = (value.method if value else None) or "FILE"

    def set_method(new):
        # Each method owns its own payload, so switching drops the previous
        # one rather than carrying a stale admin code into an ASSET selection.
        on_value(BordersSelection(method=new or "FILE"))

    with solara.Div(classes=[TIGHT_FIELD]):
        rv.Select(
            label=t("toolbox.allocation.field_borders"),
            items=[{"text": t(key), "value": m} for m, key in _METHODS],
            item_text="text",
            item_value="value",
            v_model=method,
            on_v_model=set_method,
            dense=True,
            outlined=True,
        )

    with solara.Div(classes=[TIGHT_FIELD]):
        if method == "FILE":
            FileInputComponent(
                label=t("toolbox.allocation.field_borders_file"),
                value=(value.file_path if value else "") or "",
                on_value=lambda p: on_value(
                    BordersSelection(method="FILE", file_path=str(p)) if p else None
                ),
                sepal_client=sepal_client,
                root="",
                extensions=_VECTOR_EXTENSIONS,
                clearable=True,
            )
        elif method in _ADMIN_METHODS:
            _AdminCascade(
                method=method,
                admin_code=(value.admin_code if value else None),
                on_value=on_value,
            )
        else:
            # TABLE only: an IMAGE asset is not a border. A TABLE may still be
            # points or lines — the resolver rejects those.
            AssetSelectComponent(
                types=["TABLE"],
                value=(value.asset if value else None),
                on_value=lambda asset: on_value(
                    BordersSelection(method="ASSET", asset=asset)
                ),
            )

    FieldHint(children=[solara.Text(_hint_text(value), style=_HINT)])
