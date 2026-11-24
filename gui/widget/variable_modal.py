"""Add / Edit Variable modal for the Variables step."""

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.components.inputs import (
    AssetSelectComponent,
    FileInputComponent,
)

from gui.i18n import t
from gui.scripts import variable_classes as vc
from gui.scripts import variable_palettes as palettes
from gui.scripts import variable_probes as probes
from gui.scripts.legend_data import Label, format_number, variable_spec_from_vis
from gui.scripts.predefined import (
    active_param_specs,
    aoi_param_defaults,
    default_vis,
    resolved_entry,
    validate_entry,
    vis_override,
)
from gui.scripts.predefined_variables import (
    GROUP_ORDER,
    PREDEFINED_CATALOGUE,
    advanced_param_specs,
    basic_param_specs,
    build_predefined_name,
    changed_advanced_count,
    coerce_param_values,
    default_param_values,
    param_specs,
)
from gui.widget.artifact_name_field import ArtifactNameField
from gui.widget.band_select import (
    band_choices,
    band_status,
    bands_block_save,
    use_band_list,
)
from gui.widget.dialog_nudge import NUDGE_CSS, use_outside_click_nudge
from gui.widget.enter_key import ENTER_DEFAULT
from gui.widget.text_style import MUTED
from gui.widget.vis_style_fields import (
    ClassList,
    LegendPreview,
    PaletteMenu,
    RampSelect,
)
from spatialrisk.variables.models import (
    DataType,
    RasterizationMethod,
    RasterType,
)

VAR_TYPES = ["LocalRasterVar", "GEEVar", "LocalVectorVar"]
SOURCES = ["predefined", "custom"]

# Stored discriminator value -> i18n key for its user-facing display label.
# The internal values above are persisted and branched on throughout the app,
# so we only translate them for display and never change the stored strings.
_SOURCE_LABEL_KEYS = {
    "predefined": "vars.modal.source_predefined",
    "custom": "vars.modal.source_user",
}
_TYPE_LABEL_KEYS = {
    "LocalRasterVar": "vars.modal.type_local_raster",
    "GEEVar": "vars.modal.type_gee",
    "LocalVectorVar": "vars.modal.type_local_vector",
}

_RASTER_EXTENSIONS = [".tif", ".tiff", ".vrt", ".nc"]
# The visualization fields stack with one gap; their own details are hidden.
_VIS_FIELDS = "display:flex;flex-direction:column;gap:10px;margin-bottom:12px;"
_VECTOR_EXTENSIONS = [".geojson", ".gpkg", ".shp", ".json"]
# The Advanced and Display sections: a toggle row, then an indented block.
_TOGGLE = "justify-content:flex-start;padding:0 4px;margin:4px 0;text-transform:none;"
_SECTION = (
    "padding-left:8px;margin-bottom:8px;border-left:2px solid rgba(128,128,128,0.35);"
)


def _predefined_items():
    """Dropdown items grouped by theme, built at render time for the locale.

    Vuetify renders ``{"header": ...}`` items as non-selectable subheaders.
    """
    items = []
    for group in GROUP_ORDER:
        keys = [k for k, m in PREDEFINED_CATALOGUE.items() if m.get("group") == group]
        if not keys:
            continue
        items.append({"header": t(f"vars.predefined_group.{group}")})
        items.extend(
            {"text": t(PREDEFINED_CATALOGUE[k]["label_key"]), "value": k} for k in keys
        )
    return items


def _source_items():
    """Source dropdown items — friendly labels over the stored values."""
    return [{"text": t(_SOURCE_LABEL_KEYS[s]), "value": s} for s in SOURCES]


def _type_items():
    """Variable-type dropdown items — friendly labels over the class names."""
    return [{"text": t(_TYPE_LABEL_KEYS[v]), "value": v} for v in VAR_TYPES]


def custom_vis(raster_type: str, classes, ramp: str, invert: bool, span=None):
    """The ``vis_params`` a custom raster layer submits for its raster type.

    Categorical: its class list (a colour per value; a hidden or unlisted
    value is transparent). Continuous: the named ramp, reversed when inverted,
    pinned to ``span`` or, without one, stretched by the renderer.
    """
    if raster_type == RasterType.categorical.value:
        return {"classes": [dict(c) for c in classes]}
    return palettes.continuous_vis(ramp, invert, span)


def _own_palette(default):
    """A catalogue palette offered as the "default" ramp, or None.

    A palette that is a preset shows as that preset: None here.
    """
    palette = (default or {}).get("palette")
    if not palette or palettes.ramp_choice({"palette": palette})[0]:
        return None
    return palette


def _display_state(vis, kind, default_palette=None, default=None):
    """The style fields' state for ``vis``, a saved or default style.

    ``kind`` and ``values`` are the raster type and the class values of the
    default (``default``, else ``vis``) the fields are seeded for: a style
    seeded for another one is reseeded, never submitted.
    """
    vis = vis or {}
    values = _class_values(default or vis)
    classes = []
    if kind == RasterType.categorical.value:
        # Also reads a legacy ``["ffffff", colour]`` 0..1 mask as its classes.
        classes = vc.from_vis(vis) or []
        named = {c["value"]: c for c in (default or {}).get("classes") or []}
        if not vis.get("classes") and _class_values({"classes": classes}) == values:
            # A legacy mask has no names: the catalogue's translated ones.
            classes = [
                vc.make(
                    c["value"],
                    c["color"],
                    named[c["value"]]["label"],
                    c["visible"],
                    named[c["value"]].get("label_key"),
                )
                for c in classes
            ]
    ramp, invert = palettes.ramp_choice(vis, default_palette)
    pinned = "min" in vis and "max" in vis
    return {
        "kind": kind,
        "values": values,
        "classes": classes,
        "ramp": ramp or ("default" if default_palette else palettes.DEFAULT_RAMP),
        "invert": invert,
        "vmin": _number_text(vis["min"]) if pinned else "",
        "vmax": _number_text(vis["max"]) if pinned else "",
    }


def _class_values(vis):
    return frozenset(c["value"] for c in (vis or {}).get("classes") or [])


def _style_vis(style, categorical, default_palette=None):
    """The ``vis_params`` the style fields show, None while they cannot be drawn."""
    if categorical:
        return {"classes": [dict(c) for c in style["classes"]]}
    valid, span = probes.parse_span(style["vmin"], style["vmax"])
    # A "default" ramp whose layer now has a preset palette waits for its reseed.
    if not valid or style["ramp"] not in palettes.ramp_keys(default_palette):
        return None
    return palettes.continuous_vis(
        style["ramp"], style["invert"], span, default_palette
    )


def _number_text(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


_NO_DISPLAY = _display_state(None, None)


def _type_display(var_type: str) -> str:
    """Friendly label for a variable-type discriminator (falls back to raw)."""
    key = _TYPE_LABEL_KEYS.get(var_type)
    return t(key) if key else var_type


def _raw_params(values):
    """Form state for param values: lists stay lists (multichoice), the rest text.

    Stringifying a list would turn ``["large", "major"]`` into the text
    ``"['large', 'major']"`` and break the multiple select. None (an unset
    interval year) becomes blank, not the text "None".
    """
    return {
        k: ("" if v is None else list(v) if isinstance(v, (list, tuple)) else str(v))
        for k, v in values.items()
    }


def _resolvable_params(predefined_key, params_raw):
    """The form's choice values that are valid options.

    Dynamic catalogue fields (years, raster type, scale...) depend on choice
    params (period, source, metric). While a number field is mid-edit and the
    form does not coerce, the year list must still follow the choices on
    screen rather than jump back to the defaults.
    """
    out = {}
    for spec in param_specs(predefined_key):
        value = params_raw.get(spec["key"])
        if spec.get("type", "int") == "choice" and value in spec["options"]:
            out[spec["key"]] = value
    return out


def _parse_year(year):
    """The year field's value as an int, or None when blank."""
    try:
        return int(str(year).strip())
    except (TypeError, ValueError):
        return None


def _aoi_seeded_defaults(key):
    """Catalogue defaults for ``key``, overlaid with what the selected AOI suggests.

    Specs with an ``aoi_default`` hook (the MapBiomas ``region``) pick the
    collection that covers the AOI's country. The hook reads the AOI's method
    and name and the static registry only: no Earth Engine call in this
    event handler.
    """
    from gui.store.state_manager import app_state

    values = default_param_values(key)
    if key:
        values.update(aoi_param_defaults(key, app_state.aoi_result.value))
    return values


@solara.component
def VariableModal(
    open_: solara.Reactive[bool],
    on_add: Callable,
    on_save: Optional[Callable[[str, dict], None]] = None,
    editing_key: Optional[str] = None,
    initial_entry: Optional[dict] = None,
    sepal_client=None,
    existing_keys=frozenset(),
    gee_interface=None,
    aoi_ee: Optional[Callable] = None,
):
    """Modal dialog for adding or editing a variable.

    ``gee_interface`` and ``aoi_ee`` (a callable returning the AOI as an ee
    object) serve the value probes of a custom Earth Engine layer; without
    them those two buttons are disabled.
    """
    # --- state ---
    source, set_source = solara.use_state(SOURCES[0])
    predefined_key, set_predefined_key = solara.use_state("")
    var_type, set_var_type = solara.use_state(VAR_TYPES[0])
    name, set_name = solara.use_state("")
    year, set_year = solara.use_state("")
    # Raw (string) values for the selected predefined layer's declared params,
    # keyed by param key. Coerced to ints on submit.
    params_raw, set_params_raw = solara.use_state({})
    # Advanced parameters section: collapsed on layer selection, opened on edit
    # when the saved values differ from the defaults.
    advanced_open, set_advanced_open = solara.use_state(False)
    file_path, set_file_path = solara.use_state("")
    asset_id, set_asset_id = solara.use_state("")
    scale, set_scale = solara.use_state("")
    # Picked band of a multi-band raster: a name for GEE, a 1-based index for a file.
    band, set_band = solara.use_state(None)
    # Categorical first: most user layers are 0/1 masks.
    raster_type, set_raster_type = solara.use_state(RasterType.categorical.value)
    rasterization_method, set_rasterization_method = solara.use_state(
        RasterizationMethod.binary.value
    )
    # Visualization of a custom raster layer. Continuous: a ramp, optionally
    # pinned to Min/Max (both empty = stretched by the renderer). Categorical:
    # a class list, which starts as the 0/1 mask most custom layers are.
    ramp, set_ramp = solara.use_state(palettes.DEFAULT_RAMP)
    invert, set_invert = solara.use_state(False)
    vmin, set_vmin = solara.use_state("")
    vmax, set_vmax = solara.use_state("")
    seed = vc.default_classes(t("legend.class.absent"), t("legend.class.present"))
    classes, set_classes = solara.use_state(seed)
    # A predefined layer's display style: the fields, seeded from the catalogue
    # default or the saved override, and whether the user changed them since.
    display, set_display = solara.use_state(_NO_DISPLAY)
    display_open, set_display_open = solara.use_state(False)
    display_touched, set_display_touched = solara.use_state(False)

    def _start_classes(kind, method):
        """The classes a new layer of this kind starts from."""
        if kind != "LocalVectorVar":
            return seed
        if method == RasterizationMethod.unique.value:
            return []  # one class per feature: found in the file, not seeded
        # A vector mask is drawn over the basemap: its background starts hidden.
        return [{**seed[0], "visible": False}, seed[1]]

    # What a probe refused or failed with, shown under its button.
    stretch_error, set_stretch_error = solara.use_state(None)
    classes_error, set_classes_error = solara.use_state(None)
    zero_one, set_zero_one = solara.use_state(False)
    # The raster the classes belong to: another raster starts from the seed.
    probed = solara.use_ref(None)
    error, set_error = solara.use_state(None)

    is_edit = editing_key is not None

    # Bands of the picked raster (custom raster layers only). Hooks: unconditional.
    band_origin = "gee" if var_type == "GEEVar" else "file"
    band_ref = ""
    if source == "custom" and var_type == "GEEVar":
        band_ref = asset_id.strip()
    elif source == "custom" and var_type == "LocalRasterVar":
        band_ref = file_path.strip()
    bands_task = use_band_list(gee_interface, band_origin, band_ref)
    band_items = band_choices(bands_task, band_ref)
    band_values = [value for _, value in band_items]

    def _default_band():
        # A new list starts on its first band; an edit's saved band is kept.
        if band_values and band not in band_values:
            set_band(band_values[0])

    solara.use_effect(_default_band, [tuple(band_values)])
    # Only a multi-band source records a band; one band is the file as is.
    picked_band = band if len(band_values) > 1 and band in band_values else None

    # Typed param values when the form is valid; dynamic catalogue fields
    # (years, raster type, scale...) are resolved for them.
    _values, _bad_param = (
        coerce_param_values(predefined_key, params_raw)
        if predefined_key
        else ({}, None)
    )
    _typed = (
        _values
        if _bad_param is None
        else _resolvable_params(predefined_key, params_raw)
    )

    # Derived: catalogue entry for the selected predefined variable, with its
    # param-dependent fields evaluated for the values on screen.
    cat = (
        resolved_entry(predefined_key, _typed)
        if predefined_key in PREDEFINED_CATALOGUE
        else None
    )

    display_kind = cat["raster_type"] if cat is not None else None
    display_default = default_vis(predefined_key, _typed) if cat is not None else None
    own_palette = _own_palette(display_default)
    display_categorical = display_kind == RasterType.categorical.value
    display_vis = (
        _style_vis(display, display_categorical, own_palette)
        if display_default is not None and display["kind"] == display_kind
        else None
    )

    # A non-temporal predefined layer never carries a year, even if one is
    # left in the state from a previously selected temporal layer.
    _year = (
        year if source != "predefined" or (cat is not None and cat["temporal"]) else ""
    )

    # Static availability check (never Earth Engine): a year or month past the
    # source's last complete month, or an impossible combination (FRP from
    # FIRMS), shows inline next to the fields and blocks submit.
    _unavailable = (
        validate_entry(predefined_key, _values, _parse_year(_year))
        if cat is not None and _bad_param is None
        else None
    )
    availability_error = t(_unavailable[0], **_unavailable[1]) if _unavailable else None

    # Storage key the entry will land under (mirrors variables_tile.entry_key).
    # A predefined layer's params are baked into its name (forest_gfc_tc30), so
    # the preview only shows the suffix once the values are valid.
    if source == "predefined":
        _key_name = (
            build_predefined_name(predefined_key, _values)
            if predefined_key and _bad_param is None
            else predefined_key
        )
    else:
        _key_name = name.strip()
    storage_key = (
        f"{_key_name}_{_year}"
        if (_key_name and _year and str(_year).strip())
        else _key_name
    )
    key_exists = (
        bool(storage_key)
        and storage_key in existing_keys
        and storage_key != editing_key
    )

    def reset():
        set_source(SOURCES[0])
        set_predefined_key("")
        set_params_raw({})
        set_advanced_open(False)
        set_var_type(VAR_TYPES[0])
        set_name("")
        set_year("")
        set_file_path("")
        set_asset_id("")
        set_scale("")
        set_band(None)
        set_raster_type(RasterType.categorical.value)
        set_rasterization_method(RasterizationMethod.binary.value)
        set_ramp(palettes.DEFAULT_RAMP)
        set_invert(False)
        set_vmin("")
        set_vmax("")
        set_classes(seed)
        set_display(_NO_DISPLAY)
        set_display_open(False)
        set_display_touched(False)
        _clear_probes()
        probed.current = None
        set_error(None)

    def _clear_probes():
        set_stretch_error(None)
        set_classes_error(None)
        set_zero_one(False)

    def prefill_from_initial():
        if not open_.value or initial_entry is None:
            return
        set_source(initial_entry.get("source", "custom"))
        set_predefined_key(initial_entry.get("predefined_key", ""))
        initial_key = initial_entry.get("predefined_key", "")
        # Defaults under the saved values: a legacy or partial ``params``
        # still seeds every knob it lacks.
        initial_params = {
            **default_param_values(initial_key),
            **(initial_entry.get("params") or {}),
        }
        set_params_raw(_raw_params(initial_params))
        set_advanced_open(changed_advanced_count(initial_key, initial_params) > 0)
        set_var_type(initial_entry.get("type", VAR_TYPES[0]))
        set_name(initial_entry.get("name", ""))
        set_year(str(initial_entry.get("year") or ""))
        set_file_path(initial_entry.get("path", ""))
        set_asset_id(initial_entry.get("asset_id", ""))
        set_scale(initial_entry.get("scale", ""))
        set_band(initial_entry.get("band"))
        set_raster_type(initial_entry.get("raster_type", RasterType.categorical.value))
        set_rasterization_method(
            initial_entry.get("rasterization_method", RasterizationMethod.binary.value)
        )
        vis = initial_entry.get("vis_params") or {}
        saved_ramp, saved_invert = palettes.ramp_choice(vis)
        set_ramp(saved_ramp or palettes.DEFAULT_RAMP)
        set_invert(saved_invert)
        # A greyscale ramp pinned to 0..1 has a mask's shape: go by the type.
        continuous = initial_entry.get("raster_type") == RasterType.continuous.value
        saved = (
            None
            if continuous
            else vc.from_vis(vis, t("legend.class.absent"), t("legend.class.present"))
        )
        set_classes(
            saved
            or _start_classes(
                initial_entry.get("type"), initial_entry.get("rasterization_method")
            )
        )
        # A saved mask's 0..1 pin belongs to its classes, not to a ramp.
        pinned = continuous and "min" in vis and "max" in vis
        set_vmin(_number_text(vis["min"]) if pinned else "")
        set_vmax(_number_text(vis["max"]) if pinned else "")
        # A predefined layer stores only an override: one opens the section.
        override = bool(vis) and initial_key in PREDEFINED_CATALOGUE
        if initial_key in PREDEFINED_CATALOGUE:
            default = default_vis(initial_key, initial_params)
            set_display(
                _display_state(
                    vis or default,
                    resolved_entry(initial_key, initial_params)["raster_type"],
                    _own_palette(default),
                    default,
                )
            )
        set_display_open(override)
        set_display_touched(override)
        _clear_probes()
        saved_ref = initial_entry.get("asset_id") or initial_entry.get("path")
        probed.current = (saved_ref, initial_entry.get("band")) if saved_ref else None
        set_error(None)

    solara.use_effect(prefill_from_initial, [open_.value])

    def _reseed_display():
        # Another layer or params: the fields follow the new default unless
        # the user changed them. A raster type flip always reseeds, as class
        # colours cannot style a continuous layer (Design 6), and so do new
        # class values (their colours would be stale) and a "default" ramp
        # whose layer now has a preset palette.
        if cat is None:
            return
        moved = display["values"] != _class_values(display_default)
        stale = moved or (display["ramp"] == "default" and own_palette is None)
        if display_touched and display["kind"] == display_kind and not stale:
            return
        set_display(_display_state(display_default, display_kind, own_palette))
        set_display_touched(False)

    solara.use_effect(
        _reseed_display, [predefined_key, display_kind, repr(display_default)]
    )

    def _on_display(field, value):
        set_display(lambda prev: {**prev, field: value})
        set_display_touched(True)

    def _reset_display():
        set_display(_display_state(display_default, display_kind, own_palette))
        set_display_touched(False)

    # ---- value probes (gui.scripts.variable_probes) ----
    is_gee = var_type == "GEEVar"
    is_raster = var_type in ("LocalRasterVar", "GEEVar")
    is_vector = var_type == "LocalVectorVar"
    unique = is_vector and rasterization_method == RasterizationMethod.unique.value
    # A vector layer is rasterized to classes: 0/1, or one value per feature.
    categorical = is_vector or raster_type == RasterType.categorical.value
    start = _start_classes(var_type, rasterization_method)
    ref = (asset_id if is_gee else file_path).strip()
    can_probe = bool(ref) and (
        not is_gee or (gee_interface is not None and aoi_ee is not None)
    )

    def _on_new_raster():
        # Classes found on one raster (or band) are never kept on another. An
        # emptied picker is not another raster: the asset field empties as it
        # mounts; nor is a first band, set once the band list is read.
        if not ref:
            return
        prev = probed.current
        if prev is not None and prev != (ref, band) and prev != (ref, None):
            set_classes(start)
            _clear_probes()
        probed.current = (ref, band)

    solara.use_effect(_on_new_raster, [ref, band])

    def _probe(gee_probe, file_probe, kind: str, source: str, picked=None):
        # Blocking: use_task runs a plain function on a worker thread.
        extra = {} if picked is None else {"band": picked}
        if kind == "GEEVar":
            out = gee_probe(gee_interface, source, aoi_ee(), **extra)
        else:
            out = file_probe(source, **extra)
        return {**out, "ref": (source, picked)}

    stretch_task = solara.lab.use_task(
        lambda kind, source, picked: _probe(
            probes.gee_stretch, probes.file_stretch, kind, source, picked
        ),
        dependencies=None,
    )
    classes_task = solara.lab.use_task(
        lambda kind, source, picked: _probe(
            probes.gee_classes,
            probes.vector_classes if kind == "LocalVectorVar" else probes.file_classes,
            kind,
            source,
            picked,
        ),
        dependencies=None,
    )
    busy = stretch_task.pending or classes_task.pending

    def _probe_result(task, set_probe_error):
        """A finished probe's reply for the raster still picked, else None."""
        if task.error:
            set_probe_error(t("vars.modal.probe_failed", error=str(task.exception)))
            return None
        out = task.value if task.finished else None
        if not out or out["ref"] != (ref, picked_band) or not open_.value:
            return None  # another raster was picked, or the dialog closed
        if "error" in out:
            set_probe_error(t(out["error"]))
            return None
        return out

    def _on_stretched():
        out = _probe_result(stretch_task, set_stretch_error)
        if out:
            set_vmin(_number_text(round(out["span"][0], 6)))
            set_vmax(_number_text(round(out["span"][1], 6)))

    solara.use_effect(
        _on_stretched, [stretch_task.pending, stretch_task.finished, stretch_task.error]
    )

    def _on_classes_found():
        out = _probe_result(classes_task, set_classes_error)
        if not out:
            return
        # A find adds the new values; the listed classes keep their colours,
        # labels and visibility. The untouched seed is not a list to keep.
        listed = [] if classes == start else classes
        found = vc.classes_from_values(
            out["values"],
            palette=vc.palette_of(listed) or palettes.DEFAULT_QUALITATIVE,
            known=out["known"],
        )
        set_zero_one(not listed and vc.is_zero_one(out["values"]) and not out["known"])
        set_classes(vc.merge_found(listed, found))

    solara.use_effect(
        _on_classes_found,
        [classes_task.pending, classes_task.finished, classes_task.error],
    )

    def _start_probe(task, set_probe_error):
        if busy:  # a double click: one probe at a time
            return
        set_probe_error(None)
        task(var_type, ref, picked_band)

    def _pick_kind(kind=None, method=None):
        # Another kind of layer starts from its own classes.
        kind, method = kind or var_type, method or rasterization_method
        set_var_type(kind)
        set_rasterization_method(method)
        set_classes(_start_classes(kind, method))
        _clear_probes()

    def on_cancel():
        reset()
        open_.set(False)

    def on_select_predefined(key):
        """Select a predefined layer and seed its params with catalogue defaults.

        The previous layer's values must not leak across.
        """
        set_predefined_key(key or "")
        defaults = _aoi_seeded_defaults(key or "")
        set_params_raw(_raw_params(defaults))
        set_advanced_open(False)
        set_display_open(False)
        set_display_touched(False)
        if key in PREDEFINED_CATALOGUE and _parse_year(year) is not None:
            new_cat = resolved_entry(key, defaults)
            if new_cat["temporal"] and _parse_year(year) not in new_cat["years"]:
                set_year("")

    def _submit_predefined():
        if not predefined_key:
            set_error(t("vars.modal.error_no_predefined_selected"))
            return
        # Resolved: whether a year is asked can depend on a param (protected
        # areas' timing). Only temporal layers take the Year select.
        cat_entry = cat
        yr = _parse_year(year) if cat_entry["temporal"] else None
        if cat_entry["temporal"] and yr is None:
            set_error(t("vars.modal.error_year_required"))
            return
        values, bad_param = coerce_param_values(predefined_key, params_raw)
        if bad_param is not None:
            kind = bad_param.get("type", "int")
            if bad_param.get("error_key"):
                set_error(t(bad_param["error_key"]))
            elif kind == "year":
                set_error(
                    t("vars.modal.error_param_year", label=t(bad_param["label_key"]))
                )
            elif kind == "multichoice":
                set_error(
                    t(
                        "vars.modal.error_param_multichoice",
                        label=t(bad_param["label_key"]),
                    )
                )
            elif kind == "choice":
                set_error(
                    t(
                        "vars.modal.error_param_choice",
                        label=t(bad_param["label_key"]),
                        options=", ".join(bad_param["options"]),
                    )
                )
            else:
                set_error(
                    t(
                        "vars.modal.error_param_range",
                        label=t(bad_param["label_key"]),
                        min=bad_param["min"],
                        max=bad_param["max"],
                    )
                )
            return
        # Already shown inline (availability alert); refuse without repeating it.
        if validate_entry(predefined_key, values, yr) is not None:
            return
        # A year left over from another layer or method is not in this list.
        if yr is not None and yr not in resolved_entry(predefined_key, values)["years"]:
            set_error(
                t(
                    "vars.modal.error_param_year",
                    label=t("vars.modal.predefined_year_label"),
                )
            )
            return
        resolved = resolved_entry(predefined_key, values)
        default = default_vis(predefined_key, values)
        vis_params = None
        # Fields seeded for the other raster type are dropped (Design 6).
        if default is not None and display["kind"] == resolved["raster_type"]:
            as_classes = resolved["raster_type"] == RasterType.categorical.value
            class_error = as_classes and vc.validate(display["classes"])
            if class_error:
                set_error(t(class_error))
                return
            current = _style_vis(display, as_classes, _own_palette(default))
            if current is None:
                set_error(t("vars.modal.err_range"))
                return
            vis_params = vis_override(default, current)
        entry = {
            "source": "predefined",
            "type": "GEEVar",
            # Params are baked into the name so two settings of the same layer
            # are distinct variables; predefined_key stays the catalogue key.
            "name": build_predefined_name(predefined_key, values),
            "year": yr,
            "data_type": DataType.raster,
            # Resolved: e.g. active_fire is categorical for "any" detection but
            # continuous for fire days.
            "raster_type": RasterType(resolved["raster_type"]),
            "predefined_key": predefined_key,
            "params": values,
            # Only what differs from the catalogue default; None follows it.
            "vis_params": vis_params,
        }
        reset()
        open_.set(False)
        if is_edit and on_save is not None:
            on_save(editing_key, entry)
        else:
            on_add(entry)

    def _submit_custom():
        if not name.strip():
            set_error(t("vars.modal.error_name_required"))
            return
        span = None
        # A "unique" vector layer may go unstyled (more features than classes).
        if categorical and (classes or not unique):
            class_error = vc.validate(classes)
            if class_error:
                set_error(t(class_error))
                return
        elif is_raster:
            valid, span = probes.parse_span(vmin, vmax)
            if not valid:
                set_error(t("vars.modal.err_range"))
                return
        yr = int(year) if year and str(year).strip() else None
        entry = {
            "source": "custom",
            "type": var_type,
            "name": name.strip(),
            "year": yr,
        }
        if var_type == "LocalRasterVar":
            entry["path"] = (
                Path(file_path.strip())
                if file_path.strip()
                else Path(f"/tmp/{name.strip()}.tif")
            )
            entry["band"] = picked_band
            entry["raster_type"] = RasterType(raster_type)
            entry["data_type"] = DataType.raster
            entry["vis_params"] = custom_vis(raster_type, classes, ramp, invert, span)
        elif var_type == "GEEVar":
            entry["path"] = (
                asset_id.strip()
                if asset_id.strip()
                else f"projects/dummy/assets/{name.strip()}"
            )
            entry["default_scale"] = float(scale) if scale.strip() else None
            entry["band"] = picked_band
            # to_local_raster refuses to convert without a raster type.
            entry["raster_type"] = RasterType(raster_type)
            entry["data_type"] = DataType.raster
            entry["vis_params"] = custom_vis(raster_type, classes, ramp, invert, span)
        elif var_type == "LocalVectorVar":
            entry["path"] = (
                Path(file_path.strip())
                if file_path.strip()
                else Path(f"/tmp/{name.strip()}.geojson")
            )
            entry["rasterization_method"] = RasterizationMethod(rasterization_method)
            entry["data_type"] = DataType.vector
            if classes:
                entry["vis_params"] = {"classes": [dict(c) for c in classes]}
        reset()
        open_.set(False)
        if is_edit and on_save is not None:
            on_save(editing_key, entry)
        else:
            on_add(entry)

    def on_submit():
        if source == "predefined":
            _submit_predefined()
        else:
            _submit_custom()

    def _probe_button(label_key, hint_key, icon, task, set_probe_error, probe_error):
        solara.Button(
            label=t(label_key),
            icon_name=icon,
            on_click=lambda: _start_probe(task, set_probe_error),
            outlined=True,
            color="primary",
            small=True,
            disabled=busy or not can_probe,
            style="align-self:flex-start;",
        )
        solara.Text(t(hint_key), style=MUTED + "font-size:0.8rem;")
        if probe_error:
            solara.Text(probe_error, classes=["error--text"])

    def _find_probe():
        if not is_vector or unique:  # a vector mask is always 0 and 1
            _probe_button(
                "vars.modal.find_classes",
                "vars.modal.find_features_hint"
                if is_vector
                else "vars.modal.find_classes_hint",
                "mdi-magnify",
                classes_task,
                set_classes_error,
                classes_error,
            )
        if zero_one:
            solara.Text(
                t("vars.modal.zero_one_note"), style=MUTED + "font-size:0.8rem;"
            )

    def _vis_fields():
        """A custom layer's visualization, under its raster type or method."""
        setters = {
            "classes": set_classes,
            "ramp": set_ramp,
            "invert": set_invert,
            "vmin": set_vmin,
            "vmax": set_vmax,
        }
        _style_fields(
            {
                "classes": classes,
                "ramp": ramp,
                "invert": invert,
                "vmin": vmin,
                "vmax": vmax,
            },
            lambda field, value: setters[field](value),
            categorical,
            disabled=busy,
            list_key=ref,
            raster_type=raster_type,
            name=name,
            find_probe=_find_probe,
            stretch_probe=lambda: _probe_button(
                "vars.modal.stretch",
                "vars.modal.stretch_hint",
                "mdi-arrow-expand-horizontal",
                stretch_task,
                set_stretch_error,
                stretch_error,
            ),
        )

    def _display_section():
        _render_display_section(
            display_default is not None,
            vis_override(display_default, display_vis) is not None,
            display_open,
            set_display_open,
            lambda: _style_fields(
                display,
                _on_display,
                display_categorical,
                default_palette=own_palette,
                list_key=predefined_key,
                raster_type=display_kind,
            ),
            _reset_display,
        )

    title = t("vars.modal.title_edit") if is_edit else t("vars.modal.title_add")
    submit_label = t("vars.modal.submit_save") if is_edit else t("common.add")

    # Same dismissal contract as the shared CreationDialog frame: `persistent`
    # keeps a click aimed at an open v-select menu from taking the form with
    # it, and the ESC handler restores the keyboard dismissal it disables (via
    # on_cancel, so the form resets — the old outside-click close did not).
    # Not closed, but not ignored either: every outside click nudges the
    # .v-dialog element (content_class, not the card: see dialog_nudge).
    # Holds hooks — call it unconditionally.
    nudge, on_click_outside = use_outside_click_nudge(open_.value)
    dialog = rv.Dialog(
        v_model=open_.value,
        on_v_model=open_.set,
        max_width="560px",
        eager=True,
        persistent=True,
        no_click_animation=True,
        content_class=nudge,
    )
    # rv.use_event is a hook — call it unconditionally.
    rv.use_event(dialog, "click:outside", on_click_outside)
    rv.use_event(dialog, "keydown.esc", lambda *_: on_cancel())
    with dialog:
        with rv.Card():
            solara.Style(NUDGE_CSS)
            with rv.CardTitle():
                solara.Text(title)

            with rv.CardText():
                # ---- Source toggle (first field) ----
                rv.Select(
                    label=t("vars.modal.source_label"),
                    items=_source_items(),
                    v_model=source,
                    on_v_model=set_source,
                    dense=True,
                    outlined=True,
                    hint=t("vars.modal.source_hint"),
                    persistent_hint=True,
                )

                if source == "predefined":
                    _render_predefined_fields(
                        predefined_key,
                        on_select_predefined,
                        year,
                        set_year,
                        cat,
                        storage_key,
                        key_exists,
                        params_raw,
                        set_params_raw,
                        advanced_open,
                        set_advanced_open,
                        availability_error=availability_error,
                        display_section=_display_section,
                    )
                else:
                    _render_custom_fields(
                        var_type,
                        lambda kind: _pick_kind(kind=kind),
                        name,
                        set_name,
                        year,
                        set_year,
                        file_path,
                        set_file_path,
                        asset_id,
                        set_asset_id,
                        scale,
                        set_scale,
                        raster_type,
                        set_raster_type,
                        rasterization_method,
                        lambda method: _pick_kind(method=method),
                        sepal_client=sepal_client,
                        storage_key=storage_key,
                        key_exists=key_exists,
                        vis_fields=_vis_fields,
                        band_field=lambda: _render_band_field(
                            bands_task, band_ref, band_items, picked_band, set_band
                        ),
                    )

                if error:
                    rv.Alert(type="error", dense=True, children=[error])

            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"), on_click=on_cancel, text=True, small=True
                )
                solara.Button(
                    submit_label,
                    on_click=on_submit,
                    color="primary",
                    # a band list still being read, or unreadable: nothing to save
                    disabled=source == "custom"
                    and var_type in ("LocalRasterVar", "GEEVar")
                    and bands_block_save(bands_task, band_ref),
                    small=True,
                    icon_name="mdi-plus",
                    classes=[ENTER_DEFAULT],  # Enter presses it
                )


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------


def _render_predefined_fields(
    predefined_key,
    on_select_predefined,
    year,
    set_year,
    cat,
    storage_key,
    key_exists,
    params_raw,
    set_params_raw,
    advanced_open,
    set_advanced_open,
    availability_error=None,
    display_section=lambda: None,
):
    """Fields shown when source == 'predefined'.

    ``display_section`` draws the layer's Display section (the modal owns its
    state).
    """
    rv.Select(
        label=t("vars.modal.predefined_variable_label"),
        items=_predefined_items(),
        v_model=predefined_key,
        on_v_model=on_select_predefined,
        dense=True,
        outlined=True,
    )

    if cat:
        # Data-source blurb: where the layer comes from and how it is generated
        # (catalogue link included). Rendered inline so it updates per selection.
        if cat.get("description_key"):
            with solara.Div(
                style=MUTED + "font-size:0.85rem;padding:0 4px 4px 4px;",
            ):
                solara.Markdown(t(cat["description_key"]))

        # Year — selectable list for temporal variables
        if cat["temporal"]:
            available_years = cat.get("years", [])
            rv.Select(
                label=t("vars.modal.predefined_year_label"),
                items=available_years,
                v_model=int(year) if year and str(year).strip() else None,
                on_v_model=lambda v: set_year(str(v) if v else ""),
                dense=True,
                outlined=True,
            )

        # Layer-specific knobs declared by the catalogue, rendered generically
        # (no per-layer branching here). Basic knobs first; advanced ones sit
        # in a collapsed section.
        # Conditional params (depends_on) ruled out by their controller are
        # hidden; their values are reset to the default on submit. The
        # always-present div keeps later siblings (and their click handlers)
        # in place when a field comes and goes.
        active = {s["key"] for s in active_param_specs(predefined_key, params_raw)}
        basic = basic_param_specs(predefined_key)
        # Interval layers (deforestation_gfc, ...): start and end year side by
        # side in place of the Year select (they are non-temporal).
        with rv.Html(tag="div", style_="display:flex;gap:8px;"):
            for spec in basic:
                if spec.get("type") == "year":
                    with rv.Html(tag="div", style_="flex:1;min-width:0;"):
                        _render_param_field(spec, params_raw, set_params_raw)
        with rv.Html(tag="div"):
            for spec in basic:
                if spec["key"] in active and spec.get("type") != "year":
                    _render_param_field(spec, params_raw, set_params_raw)
        advanced = [
            s for s in advanced_param_specs(predefined_key) if s["key"] in active
        ]
        # Always-present boxes: opening a section must not shift the next.
        with rv.Html(tag="div"):
            if advanced:
                _render_advanced_section(
                    predefined_key,
                    advanced,
                    params_raw,
                    set_params_raw,
                    advanced_open,
                    set_advanced_open,
                )
        with rv.Html(tag="div"):
            display_section()

        with rv.Html(tag="div"):
            if availability_error:
                rv.Alert(
                    type="error",
                    dense=True,
                    class_="sr-availability-error",
                    children=[availability_error],
                )

        # Read-only info fields
        rv.TextField(
            label=t("vars.modal.predefined_var_type_label"),
            v_model=_type_display(cat.get("var_type", "GEEVar")),
            dense=True,
            outlined=True,
            disabled=True,
        )
        rv.TextField(
            label=t("vars.modal.predefined_raster_type_label"),
            v_model=cat["raster_type"],
            dense=True,
            outlined=True,
            disabled=True,
        )

    if storage_key:
        solara.Text(
            t(
                "widgets.artifact_name.exists_warning"
                if key_exists
                else "widgets.artifact_name.saved_as",
                key=storage_key,
            ),
            style=MUTED + "font-size:0.8rem;padding:0 4px 4px;",
        )


def _render_param_field(spec, params_raw, set_params_raw):
    """One catalogue param as a form field (int, choice or multichoice).

    The field shows exactly what will be submitted: no catalogue-default
    fallback for display (seeding defaults is ``on_select_predefined``'s job),
    and every update is functional so two quick edits build on the latest
    state rather than this render's snapshot.
    """
    pkey = spec["key"]
    kind = spec.get("type", "int")
    if kind == "year":
        raw = str(params_raw.get(pkey, "") or "").strip()
        rv.Select(
            label=t(spec["label_key"]),
            items=list(spec["options"]),
            # Blank (None) until picked: interval years have no default.
            v_model=int(raw) if raw.isdigit() else None,
            on_v_model=lambda v, k=pkey: set_params_raw(
                lambda prev: {**prev, k: "" if v in (None, "") else str(v)}
            ),
            dense=True,
            outlined=True,
            hint=t(spec["hint_key"]),
            persistent_hint=True,
        )
        return
    if kind == "multichoice":
        rv.Select(
            label=t(spec["label_key"]),
            items=[
                {
                    "text": t(spec["option_label_key_prefix"] + o["value"]),
                    "value": o["value"],
                }
                for o in spec["options"]
            ],
            v_model=list(params_raw.get(pkey) or []),
            on_v_model=lambda v, k=pkey: set_params_raw(
                lambda prev: {**prev, k: list(v or [])}
            ),
            multiple=True,
            chips=True,
            small_chips=True,
            dense=True,
            outlined=True,
            hint=t(spec["hint_key"]),
            persistent_hint=True,
        )
        return
    if kind == "choice":
        rv.Select(
            label=t(spec["label_key"]),
            items=[
                {"text": t(spec["option_label_key_prefix"] + o), "value": o}
                for o in spec["options"]
            ],
            v_model=params_raw.get(pkey, ""),
            on_v_model=lambda v, k=pkey: set_params_raw(
                lambda prev: {**prev, k: v or ""}
            ),
            dense=True,
            outlined=True,
            hint=t(spec["hint_key"]),
            persistent_hint=True,
        )
        return
    rv.TextField(
        label=t(spec["label_key"]),
        v_model=str(params_raw.get(pkey, "")),
        on_v_model=lambda v, k=pkey: set_params_raw(lambda prev: {**prev, k: v}),
        dense=True,
        outlined=True,
        type="number",
        hint=t(spec["hint_key"]),
        persistent_hint=True,
    )


def _render_advanced_section(
    predefined_key, specs, params_raw, set_params_raw, advanced_open, set_advanced_open
):
    """Collapsed "Advanced parameters" block: a toggle row, then the specs.

    The toggle is a ``solara.Button`` — ``rv.Btn(on_click=)`` drops clicks.
    The label carries "(n changed)" so a collapsed section still shows that
    non-default values are in play.
    """
    n_changed = changed_advanced_count(predefined_key, params_raw)
    label = t("vars.modal.advanced_toggle")
    if n_changed:
        label = f"{label} {t('vars.modal.advanced_changed', n=n_changed)}"
    solara.Button(
        label,
        icon_name="mdi-chevron-up" if advanced_open else "mdi-chevron-down",
        on_click=lambda: set_advanced_open(not advanced_open),
        text=True,
        small=True,
        style=_TOGGLE,
    )
    if not advanced_open:
        return
    defaults = _raw_params({s["key"]: s["default"] for s in specs})
    with solara.Div(style=_SECTION):
        with rv.Html(tag="div"):
            for spec in specs:
                _render_param_field(spec, params_raw, set_params_raw)
        solara.Button(
            t("vars.modal.advanced_reset"),
            icon_name="mdi-restore",
            on_click=lambda: set_params_raw(lambda prev: {**prev, **defaults}),
            text=True,
            x_small=True,
        )


def _render_display_section(editable, changed, is_open, set_open, fields, on_reset):
    """Collapsed "Display" block: a toggle row, then the style fields.

    Built like the Advanced section. "(changed)" marks a style that differs
    from the catalogue default. A layer with random colours shows a note.
    """
    label = t("vars.modal.display_toggle")
    if changed:
        label = f"{label} {t('vars.modal.display_changed')}"
    solara.Button(
        label,
        icon_name="mdi-chevron-up" if is_open else "mdi-chevron-down",
        on_click=lambda: set_open(not is_open),
        text=True,
        small=True,
        style=_TOGGLE,
    )
    if not is_open:
        return
    with solara.Div(style=_SECTION):
        if not editable:
            solara.Text(
                t("vars.modal.display_random_note"), style=MUTED + "font-size:0.8rem;"
            )
            return
        fields()
        solara.Button(
            t("vars.modal.display_reset"),
            icon_name="mdi-restore",
            on_click=on_reset,
            text=True,
            x_small=True,
        )


def _style_fields(
    style,
    on_field,
    categorical,
    *,
    disabled=False,
    default_palette=None,
    list_key="",
    raster_type="",
    name="",
    find_probe=None,
    stretch_probe=None,
):
    """Class list and palette menu, or ramp and Min/Max; then the legend preview.

    ``style`` has ``_display_state``'s fields and ``on_field(field, value)``
    sets one. The probe callables (custom layers) draw the find and stretch
    buttons.
    """
    classes = style["classes"]
    with rv.Html(tag="div", style_=_VIS_FIELDS):
        if categorical:
            if find_probe:
                find_probe()
            # Keyed on the raster: a clear also empties "Add a value".
            ClassList(
                classes=classes,
                on_classes=lambda value: on_field("classes", value),
                disabled=disabled,
            ).key(f"classes|{list_key}")
            with rv.Html(tag="div"):
                PaletteMenu(
                    value=vc.palette_of(classes),
                    on_value=lambda key: on_field("classes", vc.recolor(classes, key)),
                    disabled=disabled or not classes,
                )
        else:
            ramp = style["ramp"]
            RampSelect(
                value=ramp if ramp in palettes.ramp_keys(default_palette) else None,
                on_value=lambda value: on_field("ramp", value),
                invert=style["invert"],
                on_invert=lambda value: on_field("invert", value),
                disabled=disabled,
                default_palette=default_palette,
            )
            with rv.Html(tag="div", style_="display:flex;gap:12px;"):
                for label_key, field in (
                    ("vars.modal.range_min", "vmin"),
                    ("vars.modal.range_max", "vmax"),
                ):
                    rv.TextField(
                        label=t(label_key),
                        v_model=style[field],
                        on_v_model=lambda value, f=field: on_field(f, value),
                        type="number",
                        dense=True,
                        outlined=True,
                        hide_details=True,
                        disabled=disabled,
                        style_="flex:1;min-width:0;",
                    )
            if stretch_probe:
                stretch_probe()
        _legend_preview(
            _style_vis(style, categorical, default_palette),
            categorical,
            raster_type,
            name,
        )


def _legend_preview(vis, categorical, raster_type, name=""):
    """The legend the map will show for ``vis``, built the way the map builds it."""
    if vis is None or (categorical and not vc.visible_classes(vis["classes"])):
        return
    kind = "custom_classes" if categorical else "custom_palette"
    mid = format_number((vis["min"] + vis["max"]) / 2) if "min" in vis else None
    var = SimpleNamespace(name=name, raster_type=raster_type, vis_params=vis)
    spec = variable_spec_from_vis(vis, kind, var, Label(literal=name))
    LegendPreview(spec=spec, mid=mid)


def _render_custom_fields(
    var_type,
    set_var_type,
    name,
    set_name,
    year,
    set_year,
    file_path,
    set_file_path,
    asset_id,
    set_asset_id,
    scale,
    set_scale,
    raster_type,
    set_raster_type,
    rasterization_method,
    set_rasterization_method,
    sepal_client,
    storage_key,
    key_exists,
    vis_fields=None,
    band_field=lambda: None,
):
    """Fields shown when source == 'custom'.

    ``vis_fields`` draws the layer's visualization fields under its raster
    type or rasterization method (the modal owns their state).
    """
    ArtifactNameField(
        value=name,
        on_input=set_name,
        storage_key=storage_key,
        exists=key_exists,
        label=t("vars.modal.custom_name_label"),
    )
    rv.Select(
        label=t("vars.modal.custom_type_label"),
        items=_type_items(),
        v_model=var_type,
        on_v_model=set_var_type,
        dense=True,
        outlined=True,
        hint=t("vars.modal.custom_type_hint"),
        persistent_hint=True,
    )
    rv.TextField(
        label=t("vars.modal.custom_year_label"),
        v_model=year,
        on_v_model=set_year,
        dense=True,
        outlined=True,
        type="number",
        hint=t("vars.modal.custom_year_hint"),
        persistent_hint=True,
    )
    if var_type in ("LocalRasterVar", "LocalVectorVar"):
        FileInputComponent(
            label=t("vars.modal.custom_file_label"),
            value=file_path,
            on_value=lambda picked: set_file_path(picked or ""),
            sepal_client=sepal_client,
            root="",
            extensions=(
                _RASTER_EXTENSIONS
                if var_type == "LocalRasterVar"
                else _VECTOR_EXTENSIONS
            ),
            clearable=True,
        )
    if var_type == "LocalRasterVar":
        band_field()
    if var_type == "GEEVar":
        _GeeAssetField(asset_id=asset_id, on_asset_id=set_asset_id)
        band_field()
        rv.TextField(
            label=t("vars.modal.custom_scale_label"),
            v_model=scale,
            on_v_model=set_scale,
            dense=True,
            outlined=True,
            type="number",
            hint=t("vars.modal.custom_scale_hint"),
            persistent_hint=True,
        )
    if var_type in ("LocalRasterVar", "GEEVar"):
        rv.Select(
            label=t("vars.modal.custom_raster_type_label"),
            items=[r.value for r in RasterType],
            v_model=raster_type,
            on_v_model=set_raster_type,
            dense=True,
            outlined=True,
            hint=t("vars.modal.custom_raster_type_hint"),
            persistent_hint=True,
        )
        if vis_fields is not None:
            vis_fields()
    if var_type == "LocalVectorVar":
        rv.Select(
            label=t("vars.modal.custom_rasterization_method_label"),
            items=[r.value for r in RasterizationMethod],
            v_model=rasterization_method,
            on_v_model=set_rasterization_method,
            dense=True,
            outlined=True,
            hint=t("vars.modal.custom_rasterization_method_hint"),
            persistent_hint=True,
        )
        if vis_fields is not None:
            vis_fields()


def _render_band_field(task, ref, items, picked, set_band):
    """Band select of a multi-band raster; nothing for a single band."""
    # One wrapper whatever shows inside: a status line or select appearing on
    # its own shifts the fields below, and their buttons lose their handlers.
    with rv.Html(tag="div"):
        band_status(task, ref)
        if len(items) > 1:
            rv.Select(
                label=t("vars.modal.band_label"),
                items=[{"text": text, "value": value} for text, value in items],
                # never a value that is not an item: a new list renders once
                # before its default lands
                v_model=picked,
                on_v_model=set_band,
                dense=True,
                outlined=True,
                hint=t("vars.modal.band_hint", count=len(items)),
                persistent_hint=True,
            )


@solara.component
def _GeeAssetField(asset_id: str, on_asset_id: Callable[[str], None]):
    """Render pysepal's asset selector, seeded once with the stored asset id.

    The selector lists the user's assets and checks whatever is typed against
    Earth Engine. It lists IMAGE assets only, because a TABLE is not a raster
    layer. It publishes ``{asset_id, type, column, value}``, and the modal
    keeps only the id.

    ``value`` is two-way in pysepal 4, and a change from outside replaces the
    selector's draft. Passing the modal's own id back would count as such a
    change after every pick, since ``{"asset_id": id}`` never equals the
    published dict. So the seed is taken once, when the field mounts. Both
    ways out of the dialog run ``reset()``, which unmounts this field, so each
    opening mounts it again after the prefill has landed.

    Args:
        asset_id: the stored asset id, or "" for a new layer.
        on_asset_id: callback(str) — the modal's asset-id setter.
    """
    seed = solara.use_memo(
        lambda: {"asset_id": asset_id} if asset_id else None, dependencies=[]
    )
    AssetSelectComponent(
        types=["IMAGE"],
        value=seed,
        on_value=lambda sel: on_asset_id((sel or {}).get("asset_id") or ""),
    )
