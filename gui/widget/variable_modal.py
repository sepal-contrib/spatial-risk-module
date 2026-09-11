"""Add / Edit Variable modal for the Variables step."""

from pathlib import Path
from typing import Callable, Optional

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.components.inputs import (
    AssetSelectComponent,
    FileInputComponent,
)

from gui.i18n import t
from gui.scripts import variable_palettes as palettes
from gui.scripts.predefined_variables import (
    PREDEFINED_CATALOGUE,
    build_predefined_name,
    coerce_param_values,
    default_param_values,
    param_specs,
)
from gui.widget.artifact_name_field import ArtifactNameField
from gui.widget.text_style import MUTED
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
_VECTOR_EXTENSIONS = [".geojson", ".gpkg", ".shp", ".json"]


def _predefined_items():
    """Build dropdown items at render time so labels respect the active locale."""
    return [
        {"text": t(meta["label_key"]), "value": key}
        for key, meta in PREDEFINED_CATALOGUE.items()
    ]


def _source_items():
    """Source dropdown items — friendly labels over the stored values."""
    return [{"text": t(_SOURCE_LABEL_KEYS[s]), "value": s} for s in SOURCES]


def _type_items():
    """Variable-type dropdown items — friendly labels over the class names."""
    return [{"text": t(_TYPE_LABEL_KEYS[v]), "value": v} for v in VAR_TYPES]


def custom_vis(raster_type: str, presence_hex: str, ramp: str, invert: bool):
    """The ``vis_params`` a custom raster layer submits for its raster type.

    Categorical: white -> the presence colour, pinned to 0..1. Continuous: the
    named ramp, reversed when inverted, stretched by the renderer.
    """
    if raster_type == RasterType.categorical.value:
        return palettes.categorical_vis(presence_hex)
    return palettes.continuous_vis(ramp, invert)


def _type_display(var_type: str) -> str:
    """Friendly label for a variable-type discriminator (falls back to raw)."""
    key = _TYPE_LABEL_KEYS.get(var_type)
    return t(key) if key else var_type


@solara.component
def VariableModal(
    open_: solara.Reactive[bool],
    on_add: Callable,
    on_save: Optional[Callable[[str, dict], None]] = None,
    editing_key: Optional[str] = None,
    initial_entry: Optional[dict] = None,
    sepal_client=None,
    existing_keys=frozenset(),
):
    """Modal dialog for adding or editing a variable."""
    # --- state ---
    source, set_source = solara.use_state(SOURCES[0])
    predefined_key, set_predefined_key = solara.use_state("")
    var_type, set_var_type = solara.use_state(VAR_TYPES[0])
    name, set_name = solara.use_state("")
    year, set_year = solara.use_state("")
    # Raw (string) values for the selected predefined layer's declared params,
    # keyed by param key. Coerced to ints on submit.
    params_raw, set_params_raw = solara.use_state({})
    file_path, set_file_path = solara.use_state("")
    asset_id, set_asset_id = solara.use_state("")
    scale, set_scale = solara.use_state("")
    # Categorical first: most user layers are 0/1 masks.
    raster_type, set_raster_type = solara.use_state(RasterType.categorical.value)
    rasterization_method, set_rasterization_method = solara.use_state(
        RasterizationMethod.binary.value
    )
    # Palette pick for custom raster layers (see gui.scripts.variable_palettes).
    presence_hex, set_presence_hex = solara.use_state(f"#{palettes.DEFAULT_PRESENCE}")
    ramp, set_ramp = solara.use_state(palettes.DEFAULT_RAMP)
    invert, set_invert = solara.use_state(False)
    error, set_error = solara.use_state(None)

    is_edit = editing_key is not None

    # Derived: catalogue entry for the selected predefined variable
    cat = PREDEFINED_CATALOGUE.get(predefined_key) if predefined_key else None

    # Storage key the entry will land under (mirrors variables_tile.entry_key).
    # A predefined layer's params are baked into its name (forest_gfc_tc30), so
    # the preview only shows the suffix once the values are valid.
    if source == "predefined":
        _values, _bad_param = coerce_param_values(predefined_key, params_raw)
        _key_name = (
            build_predefined_name(predefined_key, _values)
            if predefined_key and _bad_param is None
            else predefined_key
        )
    else:
        _key_name = name.strip()
    storage_key = (
        f"{_key_name}_{year}"
        if (_key_name and year and str(year).strip())
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
        set_var_type(VAR_TYPES[0])
        set_name("")
        set_year("")
        set_file_path("")
        set_asset_id("")
        set_scale("")
        set_raster_type(RasterType.categorical.value)
        set_rasterization_method(RasterizationMethod.binary.value)
        set_presence_hex(f"#{palettes.DEFAULT_PRESENCE}")
        set_ramp(palettes.DEFAULT_RAMP)
        set_invert(False)
        set_error(None)

    def prefill_from_initial():
        if not open_.value or initial_entry is None:
            return
        set_source(initial_entry.get("source", "custom"))
        set_predefined_key(initial_entry.get("predefined_key", ""))
        initial_key = initial_entry.get("predefined_key", "")
        initial_params = initial_entry.get("params") or default_param_values(
            initial_key
        )
        set_params_raw({k: str(v) for k, v in initial_params.items()})
        set_var_type(initial_entry.get("type", VAR_TYPES[0]))
        set_name(initial_entry.get("name", ""))
        set_year(str(initial_entry.get("year") or ""))
        set_file_path(initial_entry.get("path", ""))
        set_asset_id(initial_entry.get("asset_id", ""))
        set_scale(initial_entry.get("scale", ""))
        set_raster_type(initial_entry.get("raster_type", RasterType.categorical.value))
        set_rasterization_method(
            initial_entry.get("rasterization_method", RasterizationMethod.binary.value)
        )
        vis = initial_entry.get("vis_params")
        saved_hex = palettes.presence_color(vis)
        saved_ramp, saved_invert = palettes.ramp_choice(vis)
        set_presence_hex(saved_hex or f"#{palettes.DEFAULT_PRESENCE}")
        set_ramp(saved_ramp or palettes.DEFAULT_RAMP)
        set_invert(saved_invert)
        set_error(None)

    solara.use_effect(prefill_from_initial, [open_.value])

    def on_cancel():
        reset()
        open_.set(False)

    def on_select_predefined(key):
        """Select a predefined layer and seed its params with catalogue defaults.

        The previous layer's values must not leak across.
        """
        set_predefined_key(key or "")
        set_params_raw({k: str(v) for k, v in default_param_values(key or "").items()})

    def _submit_predefined():
        if not predefined_key:
            set_error(t("vars.modal.error_no_predefined_selected"))
            return
        cat_entry = PREDEFINED_CATALOGUE[predefined_key]
        yr = int(year) if year and str(year).strip() else None
        if cat_entry["temporal"] and yr is None:
            set_error(t("vars.modal.error_year_required"))
            return
        values, bad_param = coerce_param_values(predefined_key, params_raw)
        if bad_param is not None:
            if bad_param.get("type", "int") == "choice":
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
        entry = {
            "source": "predefined",
            "type": "GEEVar",
            # Params are baked into the name so two settings of the same layer
            # are distinct variables; predefined_key stays the catalogue key.
            "name": build_predefined_name(predefined_key, values),
            "year": yr,
            "data_type": DataType.raster,
            "raster_type": RasterType(cat_entry["raster_type"]),
            "predefined_key": predefined_key,
            "params": values,
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
        is_raster = var_type in ("LocalRasterVar", "GEEVar")
        if (
            is_raster
            and raster_type == RasterType.categorical.value
            and not palettes.is_hex(presence_hex)
        ):
            set_error(t("vars.modal.error_hex_invalid"))
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
            entry["raster_type"] = RasterType(raster_type)
            entry["data_type"] = DataType.raster
            entry["vis_params"] = custom_vis(raster_type, presence_hex, ramp, invert)
        elif var_type == "GEEVar":
            entry["path"] = (
                asset_id.strip()
                if asset_id.strip()
                else f"projects/dummy/assets/{name.strip()}"
            )
            entry["default_scale"] = float(scale) if scale.strip() else None
            # to_local_raster refuses to convert without a raster type.
            entry["raster_type"] = RasterType(raster_type)
            entry["data_type"] = DataType.raster
            entry["vis_params"] = custom_vis(raster_type, presence_hex, ramp, invert)
        elif var_type == "LocalVectorVar":
            entry["path"] = (
                Path(file_path.strip())
                if file_path.strip()
                else Path(f"/tmp/{name.strip()}.geojson")
            )
            entry["rasterization_method"] = RasterizationMethod(rasterization_method)
            entry["data_type"] = DataType.vector
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

    title = t("vars.modal.title_edit") if is_edit else t("vars.modal.title_add")
    submit_label = (
        t("vars.modal.submit_save") if is_edit else t("vars.modal.submit_add")
    )

    # Same dismissal contract as the shared CreationDialog frame: `persistent`
    # keeps a click aimed at an open v-select menu from taking the form with
    # it, and the ESC handler restores the keyboard dismissal it disables (via
    # on_cancel, so the form resets — the old outside-click close did not).
    dialog = rv.Dialog(
        v_model=open_.value,
        on_v_model=open_.set,
        max_width="560px",
        eager=True,
        persistent=True,
        no_click_animation=True,
    )
    # rv.use_event is a hook — call it unconditionally.
    rv.use_event(dialog, "keydown.esc", lambda *_: on_cancel())
    with dialog:
        with rv.Card():
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
                    )
                else:
                    _render_custom_fields(
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
                        sepal_client=sepal_client,
                        storage_key=storage_key,
                        key_exists=key_exists,
                        presence_hex=presence_hex,
                        set_presence_hex=set_presence_hex,
                        ramp=ramp,
                        set_ramp=set_ramp,
                        invert=invert,
                        set_invert=set_invert,
                    )

                if error:
                    rv.Alert(type_="error", dense=True, children=[error])

            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"), on_click=on_cancel, text=True, small=True
                )
                solara.Button(
                    submit_label,
                    on_click=on_submit,
                    color="primary",
                    small=True,
                    icon_name="mdi-plus",
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
):
    """Fields shown when source == 'predefined'."""
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

        # Layer-specific knobs declared by the catalogue (e.g. forest_gfc's
        # tree-cover threshold). Rendered generically, like MODEL_REGISTRY
        # params in the model dialog — no per-layer branching here.
        for spec in param_specs(predefined_key):
            pkey = spec["key"]
            if spec.get("type", "int") == "choice":
                rv.Select(
                    label=t(spec["label_key"]),
                    items=[
                        {"text": t(spec["option_label_key_prefix"] + o), "value": o}
                        for o in spec["options"]
                    ],
                    # Same no-fallback contract as the TextField below: show
                    # exactly what will be submitted.
                    v_model=params_raw.get(pkey, ""),
                    on_v_model=lambda v, k=pkey: set_params_raw(
                        lambda prev: {**prev, k: v or ""}
                    ),
                    dense=True,
                    outlined=True,
                    hint=t(spec["hint_key"]),
                    persistent_hint=True,
                )
                continue
            rv.TextField(
                label=t(spec["label_key"]),
                # No catalogue-default fallback: the field must show exactly
                # what will be submitted. coerce_param_values reads params_raw
                # directly, so displaying a default the state does not hold
                # would show a valid number while submit rejected it as blank.
                # Seeding with the defaults is the job of on_select_predefined.
                v_model=str(params_raw.get(pkey, "")),
                # Functional update: two params edited in quick succession must
                # each build on the latest state, not on this render's snapshot.
                on_v_model=lambda v, k=pkey: set_params_raw(
                    lambda prev: {**prev, k: v}
                ),
                dense=True,
                outlined=True,
                type="number",
                hint=t(spec["hint_key"]),
                persistent_hint=True,
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
    presence_hex="",
    set_presence_hex=None,
    ramp=palettes.DEFAULT_RAMP,
    set_ramp=None,
    invert=False,
    set_invert=None,
):
    """Fields shown when source == 'custom'."""
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
            on_value=set_file_path,
            sepal_client=sepal_client,
            root="",
            extensions=(
                _RASTER_EXTENSIONS
                if var_type == "LocalRasterVar"
                else _VECTOR_EXTENSIONS
            ),
            clearable=True,
        )
    if var_type == "GEEVar":
        # pysepal's selector lists the user's assets and validates whatever is
        # typed against Earth Engine; IMAGE only, a TABLE is not a raster layer.
        # It publishes {asset_id, type, column, value} and treats ``value`` as
        # output-only, so an edited layer's id is restored through ``initial``
        # (snapshotted once at mount). The dialog is persistent and both ways
        # out run reset(), so the selector always mounts after the prefill.
        AssetSelectComponent(
            types=["IMAGE"],
            initial={"asset_id": asset_id} if asset_id else None,
            on_value=lambda sel: set_asset_id((sel or {}).get("asset_id") or ""),
        )
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
        _render_palette_fields(
            raster_type,
            presence_hex,
            set_presence_hex,
            ramp,
            set_ramp,
            invert,
            set_invert,
        )
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


_SWATCH_STYLE = (
    "min-width:32px;width:32px;height:32px;border-radius:50%;padding:0;"
    "background-color:{color};"
)
# Ramp cards: a gradient bar over its name, in a bordered card; the selected
# card is ringed and bold (the mockup's look).
_RAMP_GRID = (
    "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));"
    "gap:8px;width:100%;"
)
_RAMP_CARD = (
    "display:block;height:auto;min-width:0;width:100%;padding:6px;"
    "text-transform:none;letter-spacing:normal;font-weight:400;text-align:left;"
    "border:1px solid rgba(128,128,128,0.5);border-radius:4px;background:transparent;"
)
_RAMP_CARD_SELECTED = "box-shadow:0 0 0 2px currentColor;font-weight:600;"
_RAMP_BAR = (
    "display:block;height:14px;border-radius:3px;margin-bottom:6px;"
    "border:1px solid rgba(0,0,0,0.1);background:{gradient};"
)
# Selection ring drawn on a wrapper, not the button: Vuetify's elevation
# classes own the button's box-shadow, and its focus rules its outline.
_WRAP_STYLE = (
    "display:inline-block;margin:0 8px 8px 0;border-radius:{radius};padding:2px;"
)
_WRAP_SELECTED = "box-shadow:0 0 0 2px currentColor;"
# Outlined box with a floating legend, the look of the dialog's outlined fields.
_FIELDSET_STYLE = (
    "border:1px solid rgba(128,128,128,0.6);border-radius:4px;"
    "padding:8px 12px 4px;margin:4px 0 0;min-width:0;"
)
_LEGEND_STYLE = "padding:0 4px;font-size:0.75rem;opacity:0.75;"


def _render_palette_fields(
    raster_type,
    presence_hex,
    set_presence_hex,
    ramp,
    set_ramp,
    invert,
    set_invert,
):
    """Palette pick under the raster-type select for custom raster layers.

    Categorical: unlabelled presence swatches (absence stays white) plus a hex
    field that always shows the stored value — a swatch fills it, typing a
    valid hex deselects the swatches. Continuous: the named ramps with an
    invert switch. Buttons go through ``solara.Button`` (a raw ``rv.Btn``
    drops clicks); the ``sr-swatch`` / ``sr-ramp`` classes and the
    ``--selected`` modifier are what tests key on. The selected swatch also
    carries a check mark, so the choice reads without relying on the ring.
    """
    is_categorical = raster_type == RasterType.categorical.value
    legend = t(
        "vars.modal.presence_color_label" if is_categorical else "vars.modal.ramp_label"
    )
    with rv.Html(tag="fieldset", style_=_FIELDSET_STYLE):
        rv.Html(tag="legend", children=[legend], style_=_LEGEND_STYLE)
        if is_categorical:
            current = presence_hex.strip().lstrip("#").lower()
            with solara.Row(gap="0", style="flex-wrap:wrap;align-items:center;"):
                for color in palettes.PRESENCE_COLORS:
                    selected = color == current
                    with rv.Html(
                        tag="div",
                        style_=_WRAP_STYLE.format(radius="50%")
                        + (_WRAP_SELECTED if selected else ""),
                    ):
                        solara.Button(
                            label="",
                            icon_name="mdi-check" if selected else None,
                            on_click=lambda *_, c=color: set_presence_hex(f"#{c}"),
                            style=_SWATCH_STYLE.format(color=f"#{color}"),
                            classes=["sr-swatch"]
                            + (["sr-swatch--selected"] if selected else []),
                            elevation=0,
                            dark=True,
                        )
            rv.TextField(
                label=t("vars.modal.presence_hex_label"),
                v_model=presence_hex,
                on_v_model=set_presence_hex,
                dense=True,
                outlined=True,
                hint=t("vars.modal.presence_hex_hint"),
                persistent_hint=True,
                style_="max-width:180px;",
                class_="mt-2",
            )
            return

        with rv.Html(tag="div", style_=_RAMP_GRID):
            for key in palettes.RAMPS:
                selected = key == ramp
                solara.Button(
                    # One full-width block: v-btn__content is a centred flex
                    # row, so two direct children would sit side by side.
                    children=[
                        rv.Html(
                            tag="span",
                            style_="display:block;width:100%;",
                            children=[
                                rv.Html(
                                    tag="span",
                                    style_=_RAMP_BAR.format(
                                        gradient=palettes.ramp_css(key, invert)
                                    ),
                                ),
                                rv.Html(
                                    tag="span",
                                    children=[t(f"vars.modal.ramp_{key}")],
                                ),
                            ],
                        )
                    ],
                    on_click=lambda *_, k=key: set_ramp(k),
                    style=_RAMP_CARD + (_RAMP_CARD_SELECTED if selected else ""),
                    classes=["sr-ramp"] + (["sr-ramp--selected"] if selected else []),
                    text=True,
                )
        rv.Switch(
            label=t("vars.modal.ramp_invert_label"),
            v_model=invert,
            on_v_model=set_invert,
            dense=True,
            hide_details=True,
            class_="mt-0 mb-1",
        )
