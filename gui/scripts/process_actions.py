"""Pure orchestration for the Process tab (no Solara).

Mirrors notebooks/2.process_factory.ipynb: download GEE layers to local,
set a reprojected base raster, then reproject/match + rasterize, and apply
edge/dist post-processing.
"""

import logging
from pathlib import Path
from typing import List

from spatialrisk.harmonization import harmonization_status

logger = logging.getLogger("spatial_risk")


def _is_geevar(var) -> bool:
    """True if a variable still needs downloading (GEE-backed, not local)."""
    return type(var).__name__ == "GEEVar"


def existing_download_targets(project, keys=None) -> List[tuple]:
    """(key, path) for every pending GEEVar whose target file is already there.

    What the Variables tile needs to decide whether to ask before downloading:
    with ``overwrite=False`` each of these files would be reused as-is, which is
    right when it is the layer you already fetched and wrong when it is a stale
    or half-written one. ``keys`` restricts the check the same way
    ``materialize_raw_layers`` restricts the download (None = all pending).
    """
    targets = []
    for key, var in (project.raw_variables if project is not None else {}).items():
        if not _is_geevar(var) or (keys is not None and key not in keys):
            continue
        try:
            path = var.expected_local_path
        except Exception:  # pragma: no cover - a var with no project/name yet
            logger.debug("no expected path for %s", key, exc_info=True)
            continue
        if path.exists():
            targets.append((key, path))
    return targets


def materialize_raw_layers(
    project, keys=None, on_progress=None, overwrite: bool = False
) -> List[str]:
    """Download raw GEEVars to local vars, replacing them in raw_variables.

    Raster GEEVars -> to_local_raster(); vector GEEVars -> to_local_vector().
    Idempotent: already-local variables are skipped. ``keys`` restricts the
    download to those raw-variable keys (None = all pending). Returns the list
    of keys that were materialized.

    ``overwrite`` re-exports layers whose file is already on disk; the default
    reuses them (see ``existing_download_targets`` for the files that affects).

    ``on_progress(layer_key, layer_idx, n_layers, tiles_done, tiles_total)``
    reports download progress: once with zero tile counts as each layer starts,
    then per completed geedim tile. Vector and already-on-disk layers produce
    only the start event (they have no tile bar).
    """
    from spatialrisk.gee.progress import geedim_tile_progress
    from spatialrisk.log_utils import log_progress
    from spatialrisk.variables.models import DataType

    # Snapshot pairs: to_local_*().add_as_raw() mutates raw_variables in place.
    pending = [
        (k, v)
        for k, v in list(project.raw_variables.items())
        if _is_geevar(v) and (keys is None or k in keys)
    ]
    if pending:
        logger.info("Downloading %d GEE layer(s)…", len(pending))

    n_layers = len(pending)
    materialized: List[str] = []
    for idx, (key, var) in enumerate(
        log_progress(pending, "Downloading layer", label=lambda kv: kv[0])
    ):
        if on_progress is not None:
            on_progress(key, idx, n_layers, 0, 0)
        tile_cb = (
            (
                lambda done, total, _k=key, _i=idx: on_progress(
                    _k, _i, n_layers, done, total
                )
            )
            if on_progress is not None
            else lambda done, total: None
        )
        with geedim_tile_progress(tile_cb):
            if var.data_type == DataType.vector:
                local = var.to_local_vector(overwrite=overwrite)
            else:
                local = var.to_local_raster(overwrite=overwrite)
        # Single-image GEEVars return one var; lists are flattened defensively.
        locals_ = local if isinstance(local, list) else [local]
        for lv in locals_:
            lv.add_as_raw(auto_save=False)
        materialized.append(key)

    if materialized:
        logger.info("Downloaded %d layer(s).", len(materialized))
    return materialized


def auto_utm_epsg(path) -> str:
    """Compute the UTM EPSG for a raster path via calculate_utm_rioxarray."""
    from spatialrisk.geo_utils import calculate_utm_rioxarray

    epsg = calculate_utm_rioxarray(Path(path))
    epsg = str(epsg)
    return epsg if epsg.startswith("EPSG:") else f"EPSG:{epsg}"


def base_raster_resolution(var) -> "float | None":
    """Native pixel resolution (m) of a raw raster var, for pre-filling the base field.

    Prefers the var's recorded scale (``default_resolution`` / ``default_scale``, in
    metres). Falls back to the GeoTIFF's native pixel size, converting degrees -> metres
    for geographic CRSs. Returns ``None`` when nothing is available.
    """
    res = getattr(var, "default_resolution", None) or getattr(
        var, "default_scale", None
    )
    if res:
        return float(res)
    path = getattr(var, "path", None)
    if path is None:
        return None
    import rasterio

    with rasterio.open(path) as src:
        xres = abs(src.res[0])
        if src.crs is not None and src.crs.is_geographic:
            xres *= 111320.0  # approx metres per degree of longitude at the equator
    return float(xres)


def set_base_raster(project, base_key: str, epsg: str, resolution: float):
    """Reproject the chosen raw raster to `epsg`/`resolution` and set it as base."""
    base = project.raw_variables[base_key]
    reprojected = base.reproject(target_epsg=epsg, resolution=resolution)
    reprojected.use_as_base_raster()
    return reprojected


def run_processing(project, keys=None) -> dict:
    """Harmonize the raw variables that are not already on the base grid.

    ``keys`` restricts the run to those raw-variable keys (the per-row
    harmonize button in Step 3); None means every pending layer. Layers outside
    ``keys`` are reported as skipped whatever their status.

    Incremental by design: with N layers already aligned, adding one variable
    used to cost N+1 reprojections. ``harmonization_status`` decides what is
    still pending — see ``spatialrisk/harmonization.py`` for the three
    conditions. Re-deriving an aligned layer is a no-op in output terms, so
    skipping it is safe; a changed reference raster invalidates every layer's
    grid and they all re-run automatically.

    To force one layer through again, remove its harmonized output from the
    list (``remove_processed_variable``): that drops the registry entry, which
    is condition one, so the layer is pending on the next run.

    Status is read *after* downloading: a GEEVar has no local file to compare
    until it is materialized, and a freshly downloaded file is newer than any
    prior output, so it lands in ``pending`` on its own. A *skipped* download
    (the file was already there) is the exception, which is one of the reasons
    the nothing-pending branch saves too — see the comment there.

    Returns ``{"processed": [...], "skipped": [...]}`` — raw-variable keys.
    Requires base_raster to be set.
    """
    if project.base_raster is None:
        raise ValueError("Set a base raster before running processing.")
    if keys is None:
        materialize_raw_layers(project)
    else:
        materialize_raw_layers(project, list(keys))

    status = harmonization_status(project)
    pending = list(status.pending)
    skipped = list(status.current)
    if keys is not None:
        wanted = set(keys)
        skipped += [k for k in pending if k not in wanted]
        pending = [k for k in pending if k in wanted]
    if not pending:
        # Unconditional: two kinds of in-memory-only change reach this branch,
        # and before Step 3 became incremental the save at the end of every run
        # persisted both.
        #  - materialize_raw_layers replaced GEEVars with local vars using
        #    add_as_raw(auto_save=False). Not hypothetical: GEEVar
        #    .to_local_raster skips the download when the file already exists
        #    (gee_var.py:164), so the "new" local file can carry an old mtime
        #    and read as current.
        #  - the Variables tile mutates raw_variables in memory only (add, edit
        #    and remove all just write the dict). A user who removes a source
        #    variable and then presses Run with nothing pending would get the
        #    removal back on the next load.
        project.save()
        logger.info(
            "All %d layer(s) are already harmonized — nothing to do.",
            len(skipped),
        )
        return {"processed": [], "skipped": skipped}

    logger.info(
        "Harmonizing %d layer(s); %d already aligned.",
        len(pending),
        len(skipped),
    )
    logger.info("Reprojecting & matching pending raw variables…")
    project.reproject_and_match_all(source="raw", keys=pending)
    logger.info("Rasterizing pending raw variables…")
    project.rasterize_all(source="raw", keys=pending)
    project.save()
    logger.info("Processing complete.")
    return {"processed": pending, "skipped": skipped}


def apply_post_processing(project, processed_key: str, step: str):
    """Apply edge/dist to a processed variable and register the result."""
    logger.info("Applying %s to %s…", step, processed_key)
    var = project.processed_variables[processed_key]
    derived = var.apply_post_processing(step)
    derived.add_as_processed()
    logger.info("%s complete for %s.", step, processed_key)
    return derived


def postprocess_output_keys(project) -> List[str]:
    """Registry keys of processed variables produced by the Post-process step.

    Change layers carry a "change" tag; edge/dist outputs record the step in
    ``processing_history`` (variable-name suffix kept as a fallback for legacy
    variables saved before ``processing_history`` existed).
    """
    from spatialrisk.variables.models import PostProcessing

    steps = tuple(s.value for s in PostProcessing)
    suffixes = tuple(f"_{s}" for s in steps)
    keys = []
    for key, var in project.processed_variables.items():
        tags = getattr(var, "tags", None) or []
        history = getattr(var, "processing_history", None) or []
        name = getattr(var, "name", key)
        if (
            "change" in tags
            or any(s in steps for s in history)
            or name.endswith(suffixes)
        ):
            keys.append(key)
    return keys


def processing_output_keys(project) -> List[str]:
    """Registry keys of processed variables produced by the Process step.

    Everything in ``processed_variables`` that the Post-process step did not
    produce — i.e. the reprojected/matched rasters and rasterized vectors that
    ``run_processing`` writes.
    """
    postprocess = set(postprocess_output_keys(project))
    return [k for k in project.processed_variables if k not in postprocess]


def remove_processed_variable(
    project, key: str, map_=None, legend_port=None, delete_file: bool = False
) -> bool:
    """Unregister a processed variable and drop its map layer and legend.

    Serves both lists that render ``processed_variables`` — Harmonization
    outputs and Derived layers — so removal behaves identically in either tile.
    By default this only unregisters: the raster stays on disk, so re-running
    harmonization or the derived op simply re-registers it. ``delete_file``
    (the dialog's opt-in) also removes the raster, through
    ``Project.delete_variable_files`` — which refuses anything outside the
    project folder or still used by another variable.
    ``legend_port`` is threaded through rather than imported (this module is
    Solara-free); None disables legend withdrawal.

    Returns True when an entry was actually removed.
    """
    from gui.tile.derived_map import drop_derived_from_map

    if project is None or key not in project.processed_variables:
        return False
    try:
        # Before the del: the file plan is resolved from the registry entry.
        if delete_file:
            project.delete_variable_files(key)
    finally:
        # A file we could not unlink is not a reason to keep a layer the user
        # asked to drop: the entry goes either way, and the error still reaches
        # the caller (the tiles toast it).
        del project.processed_variables[key]
        drop_derived_from_map(key, map_, legend_port)
    return True


def change_layer_candidates(project) -> List[str]:
    """Sorted keys of processed temporal raster vars (change-detection inputs).

    Any two of these can be paired — same source or cross-source; both must be
    presence masks (1 = present, 0 = absent) of the same phenomenon, which is
    the user's responsibility.
    """
    from spatialrisk.variables.models import DataType

    return sorted(
        k
        for k, v in project.processed_variables.items()
        if getattr(v, "data_type", None) != DataType.vector
        and getattr(v, "year", None) is not None
    )


def _check_same_grid(start_var, end_var) -> None:
    """Raise if the two processed rasters are not on the same grid.

    Post-alignment they always should be; a mismatch means one predates the
    current base raster — differencing it would produce garbage.
    """
    import rasterio

    with rasterio.open(start_var.path) as a, rasterio.open(end_var.path) as b:
        if a.crs != b.crs or a.transform != b.transform or a.shape != b.shape:
            raise ValueError(
                f"'{start_var.name}' and '{end_var.name}' are not on the same "
                "grid — re-run Process so both layers are aligned to the base "
                "raster."
            )


def change_output_name(project, op: str, start_key: str, end_key: str):
    """Registry name generate_change_var will use, or None while invalid.

    Single source of truth for the change-layer naming convention — the
    Post-process dialog previews it and generate_change_var registers it.
    """
    start = project.processed_variables.get(start_key)
    end = project.processed_variables.get(end_key)
    if start is None or end is None:
        return None
    y1 = getattr(start, "year", None)
    y2 = getattr(end, "year", None)
    if y1 is None or y2 is None or y1 >= y2:
        return None
    if start.name == end.name:
        return f"{op}_{start.name}_{y1}_{y2}"
    return f"{op}_{start.name}_{y1}_{end.name}_{y2}"


def postprocess_output_name(project, pp_key: str, step: str):
    """Name apply_post_processing will register.

    Source variable name + step suffix (mirrors
    LocalRasterVar._create_post_var's new_var_name).
    """
    var = project.processed_variables.get(pp_key)
    if var is None:
        return None
    return f"{var.name}_{step}"


def generate_change_var(project, op: str, start_key: str, end_key: str):
    """Generate a loss/gain change layer from two aligned processed masks.

    Output convention: 1 = event, 0 = stable, 255 = nodata. Registers the
    result as a static processed variable and saves the project. Idempotent:
    an existing variable (or output file) is reused, and reuse of an existing
    variable does NOT save the project.
    """
    from spatialrisk.variables import LocalRasterVar
    from spatialrisk.variables.models import RasterType

    if op not in ("loss", "gain"):
        raise ValueError(f"op must be 'loss' or 'gain', got {op!r}")
    if start_key == end_key:
        raise ValueError("Choose two different layers.")

    start = project.processed_variables.get(start_key)
    end = project.processed_variables.get(end_key)
    if start is None:
        raise ValueError(f"Processed variable '{start_key}' not found.")
    if end is None:
        raise ValueError(f"Processed variable '{end_key}' not found.")

    y1 = getattr(start, "year", None)
    y2 = getattr(end, "year", None)
    if y1 is None or y2 is None:
        raise ValueError("Both layers must be temporal (have a year).")
    if y1 >= y2:
        raise ValueError("The start layer's year must be earlier than the end layer's.")

    name = change_output_name(project, op, start_key, end_key)

    existing = project.processed_variables.get(name)
    if existing is not None:
        logger.info("Change layer '%s' already exists — reusing it.", name)
        return existing

    _check_same_grid(start, end)

    out_path = Path(project.folders.processed_data_folder) / f"{name}.tif"
    if not out_path.exists():
        from spatialrisk.processing import process_change_xarray

        logger.info("Generating %s layer '%s'…", op, name)
        process_change_xarray(str(start.path), str(end.path), str(out_path), op=op)

    var = LocalRasterVar(
        name=name,
        path=out_path,
        raster_type=RasterType.categorical,
        project=project,
        tags=[op, "change", f"{y1}_{y2}"],
    )
    var.add_as_processed(auto_save=False)
    project.save()
    logger.info("Change layer '%s' registered.", name)
    return var
