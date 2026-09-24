# gui/scripts/inference_runner.py
"""Family-aware adapter: turn a (model, dataset) into the correct model.apply() call.

This is the single place where the ML-vs-JNR-vs-MW apply() signature divergence
lives. apply() auto-registers Prediction(s) on the project, so this returns nothing
but raises clear errors on missing preconditions.
"""

import logging
from pathlib import Path

from spatialrisk.evaluation import interval_from_target

logger = logging.getLogger("spatial_risk")

_ML_FOLDER = {"glm": "glm_model", "rf": "rf_model", "icar": "icar_model"}

# Catalogue key of the Hansen layer used to *suggest* a mask in the Predict
# dialog. A processed variable carries the *variable* name, which bakes in the
# layer's parameters ("forest_gfc_tc30"), so suggestions are found by resolving
# the name back to this key — never by comparing against it. The mask itself is
# generic: any processed raster in the project (or none) can be assigned.
_FOREST_CATALOGUE_KEY = "forest_gfc"


def is_ml_family(model_key):
    """Whether *model_key* names a family whose apply() takes a mask layer."""
    return str(model_key or "").split("_")[0] in _ML_FOLDER


def is_mw_family(model_key):
    """Whether *model_key* names the moving-window family."""
    return str(model_key or "").split("_")[0] == "mw"


def mw_window_options(project, model_key):
    """Sorted trained window sizes of an MW model, for the Predict dialog.

    A fitted model exposes its actual trained windows (``ldefrate_files``
    keys); an unfitted one falls back to its configured ``win_size_list``.
    Non-MW keys and unknown models return [] so callers can render nothing.
    """
    if not is_mw_family(model_key):
        return []
    model = (getattr(project, "models", None) or {}).get(model_key)
    if model is None:
        return []
    trained = [
        int(k) for k in getattr(model, "ldefrate_files", {}) or {} if str(k).isdigit()
    ]
    if trained:
        return sorted(trained)
    return sorted(int(w) for w in getattr(model, "win_size_list", None) or [])


def missing_model_variables(project, model_key, dataset_key):
    """Variables the ML model's formula uses that the dataset has no feature for.

    Lets the Predict dialog refuse a doomed run up front instead of letting
    patsy fail mid-inference with ``NameError: name 'x' is not defined``.
    Returns [] for non-ML families (they resolve their own layers) and for
    anything it cannot check — apply() re-validates regardless.
    """
    if not is_ml_family(model_key):
        return []
    model = (getattr(project, "models", None) or {}).get(model_key)
    dataset = (getattr(project, "datasets", None) or {}).get(dataset_key)
    check = getattr(model, "missing_variables", None)
    if check is None or dataset is None:
        return []
    return check(dataset)


def _raster_variables(project):
    """The project's processed raster variables, keyed by storage key."""
    variables = getattr(project, "processed_variables", None) or {}
    return {
        key: var
        for key, var in variables.items()
        if getattr(var, "data_type", None) == "raster"
    }


def mask_layer_candidates(project):
    """Storage keys of *project* rasters assignable as the ML mask layer.

    Every processed raster in the project qualifies — the mask is a plain
    1=keep/0=suppress raster and is deliberately NOT restricted to the
    prediction dataset's features, so a run can mask with a layer that is not
    part of the dataset. This is the single source the Predict dialog lists
    from, so the dialog can never offer a layer the runner would reject.
    """
    return sorted(_raster_variables(project))


def suggested_mask_layer(project):
    """The layer key the Predict dialog seeds its mask select with.

    Masking to forest-at-period-start is the usual deforestation setup, so a
    sole Hansen-derived layer (``forest_gfc_tc<threshold>``, or the legacy bare
    ``forest_gfc``) is suggested. Zero or several such layers return "" — the
    choice is then the user's, because only they know which forest definition
    (or no mask at all) a run is meant to use.
    """
    # Imported inside the function: this module has no gui.scripts imports at
    # module scope, and keeping it that way avoids an import cycle.
    from gui.scripts.predefined_variables import resolve_predefined

    forest = [
        key
        for key, var in _raster_variables(project).items()
        if resolve_predefined(getattr(var, "name", key))[0] == _FOREST_CATALOGUE_KEY
    ]
    return forest[0] if len(forest) == 1 else ""


def _resolve_mask(project, mask_layer):
    """Path of the project raster an ML run masks with, or None for no mask.

    ``mask_layer`` is the user's explicit choice from the Predict dialog;
    empty means "no mask" (predict over the full stack). Nothing is ever
    resolved on the user's behalf here — the dialog owns the forest suggestion.
    """
    if not mask_layer:
        return None
    variable = _raster_variables(project).get(mask_layer)
    if variable is None:
        raise ValueError(
            f"Mask layer '{mask_layer}' is not a processed raster of this "
            f"project. Available layers: {mask_layer_candidates(project)}."
        )
    return variable.path


def _run_params_for(family, mask_layer, windows):
    """Run-time choices worth freezing onto the prediction, by family.

    Only the arguments a family actually honours are recorded — MW resolves its
    own layers, so storing a mask for it would be a lie the details dialog
    would then repeat. ``None`` is kept for the ML mask because "no mask" is an
    explicit user choice, not missing information.
    """
    if family in _ML_FOLDER:
        return {"mask_layer": mask_layer or None}
    if family == "mw" and windows:
        return {"windows": list(windows)}
    return {}


def run_inference(
    project, model_key, dataset_name, name=None, mask_layer=None, windows=None
):
    """Run inference for one registered model on one dataset.

    Parameters
    ----------
    name : str, optional
        User-chosen name for this run's prediction output(s). When given, each
        run gets its own output subfolder (so distinct names don't clobber files
        on disk) and the prediction(s) are keyed/labelled by it (see
        ``BaseRiskModel._register_prediction``). The caller is responsible for
        passing a path-safe token. When omitted, the legacy provenance-derived
        paths and keys are used unchanged.
    mask_layer : str, optional
        Storage key of the processed raster variable to mask with (ML families
        only), as assigned in the Predict dialog. Any project raster qualifies
        — it need not be a feature of *dataset_name*. Omitted or blank means
        no mask: the prediction covers the full raster stack. Ignored by the
        JNR/MW families, which resolve their own layers.
    windows : list[int], optional
        MW family only: subset of trained window sizes to run; None = all.
        Ignored by other families.

    Raises ValueError if preconditions are missing (no dataset target, a mask
    layer not in the project's processed rasters, unresolvable time interval
    for benchmark models).
    """
    model = project.models[model_key]
    dataset = project.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    if getattr(dataset, "target", None) is None:
        raise ValueError(f"Dataset '{dataset_name}' has no target set.")

    family = model_key.split("_")[0]
    model.project = project
    model.dataset = dataset
    # Hand the name to _register_prediction. Set unconditionally (None when not
    # provided) so a stale name from a prior named run on the same model instance
    # can't leak into a later unnamed run; None keeps the provenance-derived key.
    model._pending_pred_name = name or None
    # Run-time choices that shape the output but live outside the model config,
    # so the prediction's provenance can report them. Set unconditionally for
    # the same reason as the name above: a stale mask from a prior run on this
    # model instance must not leak into a later unmasked one.
    model._pending_run_params = _run_params_for(family, mask_layer, windows)

    if family in _ML_FOLDER:
        mask = _resolve_mask(project, mask_layer)
        subfolder = name or (getattr(model, "name", None) or model_key)
        out_dir = Path(getattr(project.folders, _ML_FOLDER[family])) / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{dataset_name}.tif"
        model.apply(out, dataset, mask, 0)
        return

    ti = interval_from_target(dataset.target.name)
    if ti is None:
        raise ValueError(
            f"Cannot derive time_interval from target '{dataset.target.name}'."
        )

    if family == "jnr":
        out_dir = Path(project.folders.rmj_bm) / (name or dataset_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"prob_bm_{dataset_name}.tif"
        model.apply(out, dataset, time_interval=ti, deforate_model=None)
        return

    if family == "mw":
        out_folder = Path(project.folders.rmj_mw)
        if name:
            out_folder = out_folder / name
        model.apply(
            dataset, time_interval=ti, output_folder=out_folder, windows=windows
        )
        return

    raise ValueError(
        f"Unknown model family '{model_key.split('_')[0]}' for key '{model_key}'."
    )
