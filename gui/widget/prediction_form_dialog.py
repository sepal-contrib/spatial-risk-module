"""New Prediction dialog for the Inference step.

One dialog for both ways a prediction is created: running a trained model on a
dataset, or importing a raster produced outside the app. A Source select swaps
the input fields; the name field is shared. Import keeps the backend's
non-destructive duplicate policy (a taken name gets a ``-2`` suffix), so its
name preview shows the *resolved* key instead of asking to replace.
"""

from pathlib import Path
from typing import Callable

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.components.inputs import FileInputComponent

from gui.i18n import t
from gui.scripts.artifact_names import (
    default_pred_name,
    mask_name_token,
    prediction_name_exists,
    sanitize_key,
)
from gui.scripts.inference_runner import (
    is_ml_family,
    is_mw_family,
    mask_layer_candidates,
    mw_window_options,
    suggested_mask_layer,
)
from gui.scripts.model_registry import MODEL_REGISTRY
from gui.scripts.prediction_import import resolve_import_key, sanitize_import_name
from gui.scripts.product_rows import prediction_row_key
from gui.widget.artifact_name_field import ArtifactNameField, use_artifact_name
from gui.widget.creation_dialog import _ADVANCED_PANEL_CSS, CreationDialog
from gui.widget.details_fields import ro_field
from gui.widget.model_form_dialog import model_label
from spatialrisk.far_helpers import strip_categorical_levels

# Raster file types accepted for a local prediction import. GeoTIFF only:
# the import is warped and rewritten band-wise, which needs a real raster
# file rather than a VRT stack or a NetCDF subdataset.
_IMPORT_RASTER_EXTENSIONS = [".tif", ".tiff"]

# Select value for the explicit "no mask" choice. A sentinel rather than ""
# because "" is the field's *unset* state, which validation refuses: predicting
# unmasked must be a decision, never a default.
NO_MASK = "__no_mask__"


def _source_items():
    return [
        {"text": t("tiles.inference.source_model"), "value": "model"},
        {"text": t("tiles.inference.source_import"), "value": "import"},
    ]


def _value_scale_items():
    return [
        {
            "text": t("widgets.prediction_import_modal.scale_probability"),
            "value": "probability",
        },
        {"text": t("widgets.prediction_import_modal.scale_risk"), "value": "risk"},
    ]


@solara.component
def PredictionFormDialog(
    project, open_, on_submit: Callable[[dict], None], sepal_client=None, prefill=None
):
    """Prediction form in the shared CreationDialog frame.

    on_submit(entry) receives {"kind": "model", "model_key", "dataset_key",
    "name"} or {"kind": "import", "name", "path", "value_scale"}; the tile owns
    the job row and the worker. A model entry for an ML family (GLM/RF/ICAR)
    also carries "mask_layer" — the storage key of the project raster to mask
    with (any processed raster, dataset membership not required), or None for
    an explicit "no mask".

    Args:
        project: solara.Reactive[Project].
        open_: solara.Reactive[bool].
        on_submit: callback receiving the entry dict described above.
        sepal_client: SEPAL client backing the import file picker.
        prefill: optional solara.Reactive holding a previously submitted entry
            dict (or None). While the dialog is open with a non-empty prefill,
            the fields are seeded from it — this is how a failed run reopens
            for editing. The tile owns clearing it before a fresh "New" open.
    """
    p = project.value

    source, set_source = solara.use_state("model")

    # --- model mode state
    model_keys = sorted(p.models.keys()) if p and p.models else []
    selected_model, set_selected_model = solara.use_state("")
    dataset_keys = sorted(p.datasets.keys()) if p and p.datasets else []
    selected_dataset, set_selected_dataset = solara.use_state("")

    # --- mask layer (ML families only)
    # GLM/RF/ICAR mask their output with a project raster the user assigns —
    # any processed raster qualifies (it need not be a feature of the selected
    # dataset), and "no mask" is a valid explicit choice, so the algorithms
    # carry no assumption about the mask's origin. The candidate list and the
    # forest-layer suggestion both come from the runner module so the dialog
    # can never offer a layer the runner would reject.
    mask_layer, set_mask_layer = solara.use_state("")
    ml_family = source == "model" and is_ml_family(selected_model)
    mask_candidates = mask_layer_candidates(p) if ml_family else []
    show_mask = ml_family

    # A prefilled mask choice must survive the re-seed that fires when the
    # prefilled model changes the candidate set — the seed consumes it instead
    # of overwriting it with the forest suggestion.
    pending_mask = solara.use_ref(None)

    def seed_mask_layer():
        """Suggest the sole forest layer; anything ambiguous starts unset."""
        if pending_mask.current is not None and ml_family:
            set_mask_layer(pending_mask.current)
            pending_mask.current = None
            return
        set_mask_layer(suggested_mask_layer(p) if ml_family else "")

    # Re-seeds whenever the candidate set changes (a model-family switch) and
    # never in between, so a deliberate pick survives re-renders.
    solara.use_effect(seed_mask_layer, [tuple(mask_candidates)])

    # --- window sizes (MW family only)
    # MW inference fans out one map per trained window; the user narrows the
    # run here (e.g. re-run only the window that won evaluation). Seeded with
    # every trained window so the default equals the historic all-windows run.
    selected_windows, set_selected_windows = solara.use_state([])
    mw_family = source == "model" and is_mw_family(selected_model)
    window_options = mw_window_options(p, selected_model) if mw_family else []
    pending_windows = solara.use_ref(None)

    def seed_windows():
        """Select every trained window, unless a prefill narrowed the run."""
        if pending_windows.current is not None and mw_family:
            keep = [w for w in pending_windows.current if w in window_options]
            set_selected_windows(keep or list(window_options))
            pending_windows.current = None
            return
        set_selected_windows(list(window_options))

    # selected_model is a dependency on purpose: two MW models can expose the
    # SAME option tuple, and switching between them must still reseed to
    # all-selected — a subset chosen for model A must not leak into model B.
    solara.use_effect(seed_windows, [selected_model, tuple(window_options)])

    # --- import mode state
    file_path, set_file_path = solara.use_state("")
    # The scale the file is on, not a display choice: every import is rescaled
    # onto the app's 1..65535 contract. "" is the unset state validate()
    # refuses — guessing it would silently rescale the wrong way.
    value_scale, set_value_scale = solara.use_state("")

    # Name tracks the mode's suggestion until the user edits it: model mode
    # suggests "model__dataset", import mode the sanitized file stem.
    if source == "model":
        # The mask is part of the run's identity, so it discriminates the
        # suggested name too: without it, two ML runs over one (model,
        # dataset) that differ only by mask both landed on the same name and
        # the second read as an overwrite of the first. Only once a mask has
        # actually been chosen — an unset field has nothing to say yet.
        mask_token = (
            mask_name_token(None if mask_layer == NO_MASK else mask_layer)
            if ml_family and mask_layer
            else None
        )
        suggestion = default_pred_name(selected_model, selected_dataset, mask_token)
    else:
        suggestion = (
            sanitize_import_name(Path(str(file_path)).stem) if file_path else ""
        )
    name_value, on_name_input, reset_name = use_artifact_name(suggestion)

    def seed_from_prefill():
        """Seed every field from the prefill entry each time the dialog opens.

        Keyed on the open flag as well as the entry so re-editing the same
        failed job after a cancel (same entry, fields reset on close) seeds
        again. A fresh open with the prefill cleared is a no-op.
        """
        entry = prefill.value if prefill is not None else None
        if not open_.value or not entry:
            return
        if entry.get("kind") == "import":
            set_source("import")
            set_file_path(entry.get("path", ""))
            set_value_scale(entry.get("value_scale", ""))
        else:
            set_source("model")
            set_selected_model(entry.get("model_key", ""))
            set_selected_dataset(entry.get("dataset_key", ""))
            if "mask_layer" in entry:
                # None was the explicit "no mask" submission — show the sentinel.
                pending_mask.current = entry["mask_layer"] or NO_MASK
            if "windows" in entry:
                pending_windows.current = list(entry["windows"])
        on_name_input(entry.get("name", ""))

    solara.use_effect(
        seed_from_prefill,
        [open_.value, prefill.value if prefill is not None else None],
    )

    clean = sanitize_key(name_value)
    # An MW run registers one prediction per selected window, suffixed
    # ``_w<size>`` (see BaseRiskModel._register_prediction), so both the name
    # preview and the "already taken" check work on the fanned-out keys.
    if source == "model" and mw_family and clean:
        mw_keys = [f"{clean}_w{w}" for w in sorted(selected_windows)]
        exists = any(prediction_name_exists(p, k) for k in mw_keys)
        preview_key = ", ".join(mw_keys) or clean
    else:
        mw_keys = []
        exists = prediction_name_exists(p, clean)
        preview_key = clean
    # Import never replaces: preview the key the import would actually get.
    # ".tif" regardless of the source suffix, because the adapter always writes
    # a GeoTIFF and disambiguates against that name — previewing a ".tiff"
    # source's own suffix would show a key the import never assigns.
    resolved_import_key = (
        resolve_import_key(p, name_value.strip(), ".tif")
        if p is not None and name_value.strip()
        else ""
    )

    def reset():
        pending_mask.current = None
        pending_windows.current = None
        set_selected_windows([])
        set_source("model")
        set_selected_model("")
        set_selected_dataset("")
        set_mask_layer("")
        set_file_path("")
        set_value_scale("")
        reset_name()

    def on_source(v):
        # Re-arm the suggestion so the name follows the new mode's default.
        set_source(v)
        reset_name()

    def validate():
        if p is None:
            return t("tiles.inference.error_no_project")
        if source == "import":
            # First, because nothing the user types in this form can fix it:
            # every import is warped onto the base raster's geobox, so without
            # one there is no grid to import onto. Caught here rather than in
            # the worker so no job is ever queued that can only fail.
            if getattr(p, "base_raster", None) is None:
                return t("widgets.prediction_import_modal.error_base_raster_required")
            if not file_path or not str(file_path).strip():
                return t("widgets.prediction_import_modal.error_select_raster")
            if not name_value.strip():
                return t("widgets.prediction_import_modal.error_enter_name")
            if value_scale not in ("probability", "risk"):
                return t("widgets.prediction_import_modal.error_scale_required")
            try:
                from spatialrisk.predictions.import_raster import (
                    ImportRasterError,
                    inspect_raster,
                )

                inspect_raster(file_path)  # metadata only: safe in a handler
            except ImportRasterError as exc:
                return str(exc)
            return None
        if not selected_model or selected_model not in p.models:
            return t("tiles.inference.error_invalid_model")
        if not selected_dataset or selected_dataset not in p.datasets:
            return t("tiles.inference.error_invalid_dataset")
        if ml_family and not mask_layer:
            return t("tiles.inference.error_mask_required")
        # Gated on window_options: an MW model exposing none (untrained legacy
        # entry, empty win_size_list) renders no field and keeps today's
        # behaviour — apply() runs all windows or raises, exactly as before.
        if mw_family and window_options and not selected_windows:
            return t("tiles.inference.error_windows_required")
        if not clean:
            return t("tiles.inference.error_name_required")
        return None

    def will_replace():
        if source == "import":
            return None  # duplicate imports suffix instead of replacing
        if mw_family and mw_keys:
            # Report every colliding key, not just the first: one run can
            # overwrite several _w<size> maps and the confirmation must name
            # all of them.
            taken = [k for k in mw_keys if prediction_name_exists(p, k)]
            return ", ".join(taken) if taken else None
        return clean if prediction_name_exists(p, clean) else None

    def launch():
        if source == "import":
            on_submit(
                {
                    "kind": "import",
                    "name": name_value.strip(),
                    "path": str(file_path),
                    "value_scale": value_scale,
                }
            )
        else:
            entry = {
                "kind": "model",
                "model_key": selected_model,
                "dataset_key": selected_dataset,
                "name": clean,
            }
            if ml_family:
                # The user's assignment: a project-raster key, or None for the
                # explicit "no mask" choice (validate() guarantees one of the
                # two). Benchmark families resolve their own layers and never
                # carry the key.
                entry["mask_layer"] = None if mask_layer == NO_MASK else mask_layer
            if mw_family and window_options:
                # No key at all when the model exposes no options — the runner
                # then passes windows=None and apply() runs its usual set.
                entry["windows"] = sorted(selected_windows)
            on_submit(entry)

    with CreationDialog(
        open_=open_,
        title=t("tiles.inference.dialog_title"),
        create_label=t("tiles.inference.run_button"),
        validate=validate,
        will_replace=will_replace,
        launch=launch,
        on_close=reset,
        replace_message=lambda k: t(
            "tiles.inference.confirm_overwrite_message", name=k
        ),
    ):
        rv.Select(
            label=t("tiles.inference.source_label"),
            items=_source_items(),
            item_text="text",
            item_value="value",
            v_model=source,
            on_v_model=on_source,
            dense=True,
            outlined=True,
        )
        if source == "model":
            rv.Select(
                label=t("tiles.inference.model_select_label"),
                items=model_keys,
                v_model=selected_model,
                on_v_model=set_selected_model,
                dense=True,
                outlined=True,
                no_data_text=t("tiles.inference.model_select_no_data"),
                hint=t("tiles.inference.model_select_hint"),
                persistent_hint=True,
            )
            rv.Select(
                label=t("tiles.inference.dataset_select_label"),
                items=dataset_keys,
                v_model=selected_dataset,
                on_v_model=set_selected_dataset,
                dense=True,
                outlined=True,
                no_data_text=t("tiles.inference.dataset_select_no_data"),
                hint=t("tiles.inference.dataset_select_hint"),
                persistent_hint=True,
            )
            if mw_family:
                rv.Select(
                    label=t("tiles.inference.windows_label"),
                    items=[
                        {"text": f"{w}×{w}", "value": w}  # noqa: RUF001
                        for w in window_options
                    ],
                    item_text="text",
                    item_value="value",
                    v_model=selected_windows,
                    on_v_model=set_selected_windows,
                    multiple=True,
                    chips=True,
                    small_chips=True,
                    deletable_chips=True,
                    dense=True,
                    outlined=True,
                    class_="multi-chips",
                    hint=t("tiles.inference.windows_hint"),
                    persistent_hint=True,
                )
            if show_mask:
                rv.Select(
                    label=t("tiles.inference.mask_layer_label"),
                    items=[
                        {"text": t("tiles.inference.mask_layer_none"), "value": NO_MASK}
                    ]
                    + [{"text": n, "value": n} for n in mask_candidates],
                    item_text="text",
                    item_value="value",
                    v_model=mask_layer,
                    on_v_model=set_mask_layer,
                    dense=True,
                    outlined=True,
                    clearable=True,
                    hint=t("tiles.inference.mask_layer_hint"),
                    persistent_hint=True,
                )
        else:
            FileInputComponent(
                label=t("widgets.prediction_import_modal.label_file"),
                value=file_path,
                on_value=set_file_path,
                sepal_client=sepal_client,
                root="",
                extensions=_IMPORT_RASTER_EXTENSIONS,
                clearable=True,
            )
            rv.Select(
                label=t("widgets.prediction_import_modal.label_scale"),
                items=_value_scale_items(),
                item_text="text",
                item_value="value",
                v_model=value_scale or None,
                on_v_model=lambda v: set_value_scale(v or ""),
                dense=True,
                outlined=True,
            )
            with solara.Div(style="font-size:0.85em;opacity:0.85;margin-top:4px;"):
                solara.Markdown(
                    f"**{t('widgets.prediction_import_modal.spec_intro')}**\n\n"
                    f"- {t('widgets.prediction_import_modal.spec_format')}\n"
                    f"- {t('widgets.prediction_import_modal.spec_values')}\n"
                    f"- {t('widgets.prediction_import_modal.spec_nodata')}\n"
                    f"- {t('widgets.prediction_import_modal.spec_zero')}\n"
                    f"- {t('widgets.prediction_import_modal.spec_warp')}\n"
                )
        ArtifactNameField(
            value=name_value,
            on_input=on_name_input,
            storage_key=preview_key if source == "model" else resolved_import_key,
            exists=exists if source == "model" else False,
            label=t("tiles.inference.pred_name_label"),
        )


def _group_predictions(project, row_key):
    """Every registered prediction belonging to one outputs-list row.

    A row groups a whole run: an MW run registers one Prediction per window
    under the same name, and they share all provenance but the raster path.
    Sorted by window so a multi-output run lists its files in a stable order.
    """
    predictions = (getattr(project, "predictions", None) or {}) if project else {}
    group = [
        pred for pred in predictions.values() if prediction_row_key(pred) == row_key
    ]
    return sorted(group, key=lambda p: (p.window is None, p.window or 0))


def _is_import(pred) -> bool:
    """Whether this prediction came from the raster-import flow.

    An import registers no model, so it has no model snapshot to explain. That
    absence *is* the signal — no separate "kind" flag is stored on disk, and
    adding one would leave every already-registered import unclassified.
    """
    return not getattr(pred, "model_snapshot", None)


@solara.component
def PredictionDetailsDialog(project, row_key, on_close: Callable[[], None]):
    """Read-only provenance for one prediction row: how it was produced.

    Everything shown was frozen onto the Prediction when its raster was written
    (see ``BaseRiskModel._register_prediction``), so a run stays explainable
    after the model that made it is retrained, renamed or deleted — reading the
    live model instead would quietly report today's config for yesterday's map.

    Args:
        project: solara.Reactive[Project].
        row_key: outputs-list row key to explain, or None (dialog closed).
        on_close: () -> None; clears the tile's selected key.
    """
    p = project.value
    group = _group_predictions(p, row_key) if row_key else []
    pred = group[0] if group else None

    with rv.Dialog(
        v_model=pred is not None,
        on_v_model=lambda v: None if v else on_close(),
        max_width="720px",
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("tiles.inference.details_title", key=row_key or ""))
            with rv.CardText():
                solara.Style(_ADVANCED_PANEL_CSS)
                if pred is not None:
                    _details_body(pred, group)
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=lambda: on_close(),
                    text=True,
                    small=True,
                )


def _details_body(pred, group):
    """The dialog's field stack for one prediction group."""
    imported = _is_import(pred)

    with solara.Column(style="gap:4px;"):
        ro_field(
            t("tiles.inference.source_label"),
            t("tiles.inference.source_import")
            if imported
            else t("tiles.inference.source_model"),
        )
        ro_field(t("tiles.inference.details_created"), pred.created_at)

        if imported:
            solara.Text(t("tiles.inference.details_imported_note"))
            _import_section(pred)
        else:
            _model_section(pred)
            _dataset_section(pred)

        _output_section(group)


def _import_section(pred):
    """Where an imported raster came from, as frozen at import time.

    Only an import adapted by the value-scale flow has source provenance; one
    registered before it can report nothing but the palette it was given.
    """
    params = pred.run_params or {}
    if not params.get("value_scale"):
        # Pre-adaptation import: only its display palette is known.
        ro_field(t("tiles.inference.details_palette_label"), pred.display_palette)
        return
    ro_field(
        t("tiles.inference.details_value_scale"),
        t(f"widgets.prediction_import_modal.scale_{params['value_scale']}"),
    )
    ro_field(t("tiles.inference.details_source_path"), params.get("source_path"))
    ro_field(t("tiles.inference.details_source_crs"), params.get("source_crs"))
    res = params.get("source_resolution") or []
    ro_field(
        t("tiles.inference.details_source_resolution"),
        " × ".join(f"{v:g}" for v in res) if res else None,  # noqa: RUF001
    )
    ro_field(t("tiles.inference.details_source_dtype"), params.get("source_dtype"))
    rng = params.get("source_range") or []
    ro_field(
        t("tiles.inference.details_source_range"),
        " – ".join(f"{v:g}" for v in rng) if rng else None,  # noqa: RUF001
    )


def _model_section(pred):
    """Model identity, run-time choices, then config behind the advanced panel."""
    snapshot = pred.model_snapshot or {}
    model_type = snapshot.get("model_type")
    registry = MODEL_REGISTRY.get(model_type)

    solara.Markdown(t("tiles.inference.details_model_header"))
    ro_field(
        t("tiles.inference.model_select_label"),
        model_label(model_type) if registry else (model_type or pred.model_key),
    )
    ro_field(t("tiles.train.model_name_label"), snapshot.get("name"))

    # Run-time choices are arguments to apply(), absent from the snapshot above.
    # Only rendered when the family actually took one: a blank Mask row on an
    # MW run would imply a choice the algorithm never offered.
    run_params = pred.run_params or {}
    if "mask_layer" in run_params:
        ro_field(
            t("tiles.inference.mask_layer_label"),
            run_params["mask_layer"] or t("tiles.inference.mask_layer_none"),
        )
    if run_params.get("windows"):
        ro_field(t("tiles.inference.windows_label"), run_params["windows"])

    formula = snapshot.get("formula")
    param_defs = [
        pd
        for pd in (registry["params"] if registry else [])
        if pd.get("group", "params") == "params"
    ]
    if not (formula or param_defs):
        return
    with rv.ExpansionPanels(flat=True, class_="advanced-params"):
        with rv.ExpansionPanel():
            with rv.ExpansionPanelHeader():
                solara.Text(t("tiles.train.advanced_parameters_header"))
            with rv.ExpansionPanelContent():
                if formula:
                    # levels=[...] is a fit-time safety net, noise to the reader.
                    ro_field(
                        t("tiles.train.formula_label"),
                        strip_categorical_levels(formula),
                    )
                for pd in param_defs:
                    # No registry default fallback: a param absent from the
                    # snapshot was not recorded, and printing today's default
                    # would invent provenance. format_value renders None as —.
                    ro_field(t(pd["label_key"]), snapshot.get(pd["key"]))


def _dataset_section(pred):
    """The dataset the model was applied over, as frozen at prediction time."""
    snapshot = pred.dataset_snapshot or {}
    solara.Markdown(t("tiles.inference.details_dataset_header"))
    ro_field(
        t("tiles.inference.dataset_select_label"),
        snapshot.get("name") or pred.dataset_name,
    )
    ro_field(t("tiles.inference.details_target"), snapshot.get("target_name"))
    ro_field(t("tiles.inference.details_features"), snapshot.get("feature_names"))


def _output_section(group):
    """One row per raster the run wrote (MW writes one per window)."""
    solara.Markdown(t("tiles.inference.details_output_header"))
    for pred in group:
        label = (
            t("tiles.inference.details_output_window", n=pred.window)
            if pred.window is not None
            else t("tiles.inference.details_output_label")
        )
        ro_field(label, str(pred.path))
