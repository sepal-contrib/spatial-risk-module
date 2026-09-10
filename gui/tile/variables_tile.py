"""Step 2 — Variables tile."""

import asyncio
import logging

import ee
import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.layer_labels import raw_layer_label
from gui.scripts.map_helpers import add_vector_on_map, is_mappable
from gui.scripts.notify_bridge import (
    ERROR_TOAST_TIMEOUT,
    layer_progress_reporter,
    tracked_job,
)
from gui.scripts.solara_threads import publish_if_current, to_thread_in_context
from gui.scripts.variable_identity import is_base_raster
from gui.scripts.variable_map import add_raster_var_on_map
from gui.store.project_writers import writing
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.help import InfoButton
from gui.widget.variable_list import SourceVariableList
from gui.widget.variable_modal import VariableModal

# Unused directly, but required in scope: pydantic v2's model_rebuild() below
# resolves the Optional["Project"] forward reference using this module's
# namespace, so removing the import would break model construction.
from spatialrisk.project import Project  # noqa: F401
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.local_vector_var import LocalVectorVar

logger = logging.getLogger("spatial_risk")

LocalRasterVar.model_rebuild()
GEEVar.model_rebuild()
LocalVectorVar.model_rebuild()

# Keys of source variables currently displayed on the map (drives the toggle state).
vars_on_map = solara.reactive(set())


def _map_layer_key(key: str) -> str:
    """Unique map-layer key for a source variable."""
    return f"var_{key}"


def _minmax(image, var, gee_interface):
    """Compute (min, max) of the image over its AOI, or None if unavailable.

    Uses the *blocking* ``gee_interface.get_info`` so the session call is
    scheduled onto the GEE interface's own event loop (see ``_add_gee_layer``).
    Must therefore run off the Solara loop — i.e. inside a worker thread.
    """
    aoi = getattr(var, "aoi", None)
    if aoi is None or gee_interface is None:
        return None
    try:
        geom = aoi if isinstance(aoi, ee.Geometry) else aoi.geometry()
        stats = gee_interface.get_info(
            image.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geom,
                scale=getattr(var, "default_scale", None) or 100,
                maxPixels=1e8,
                bestEffort=True,
            )
        )
        mins = [v for k, v in stats.items() if k.endswith("_min") and v is not None]
        maxs = [v for k, v in stats.items() if k.endswith("_max") and v is not None]
        if mins and maxs:
            return min(mins), max(maxs)
    except Exception:
        logger.debug("min/max over AOI failed", exc_info=True)
    return None


def _grayscale_vis(image, var, gee_interface):
    """Grayscale palette stretched to the image's min/max over its AOI.

    Falls back to a bare grayscale palette (GEE's default 0-1 stretch) if the
    min/max can't be computed.
    """
    vis = {"palette": ["000000", "ffffff"]}
    mm = _minmax(image, var, gee_interface)
    if mm:
        vis["min"], vis["max"] = mm
    return vis


def _styled_layer(image, var, gee_interface):
    """Choose visualization for a source variable.

    A variable carrying its own ``vis_params`` (custom layers, picked in the
    modal) renders with exactly that.
    Predefined catalogue variables carry their own visualization spec (keyed by
    name): a ``random_visualizer`` flag (random RGB per class) or a ``vis_params``
    dict whose palette is stretched dynamically when no min/max is given.

    Everything else falls back to by-``raster_type`` defaults: categorical (incl.
    binary masks) -> black/white palette (0=black, 1=white); continuous ->
    grayscale stretched to the image's min/max.

    Returns (image_to_add, vis_params, render_kind). GEE variables are never
    post-process outputs (post-processing runs on downloaded rasters), so its
    render_kind vocabulary is the catalogue subset: "custom_palette",
    "random_visualizer", "catalogue_palette", "categorical_fallback",
    "continuous_fallback".
    Synchronous — any GEE stretch it computes goes through the blocking
    interface, so call it from a worker thread.
    """
    from gui.scripts.predefined_variables import (
        PREDEFINED_CATALOGUE,
        resolve_predefined,
    )

    # A palette the user picked in the Variables modal wins over the catalogue
    # and the raster-type fallbacks. Applied verbatim: a categorical pick is
    # pinned to 0..1, a ramp is palette-only and GEE stretches it.
    own = getattr(var, "vis_params", None) or {}
    if own.get("palette"):
        return image, dict(own), "custom_palette"

    # The name may carry a param suffix (forest_gfc_tc30), so resolve it back to
    # the catalogue key rather than looking the raw name up.
    cat_key, _params = resolve_predefined(getattr(var, "name", "") or "")
    cat = PREDEFINED_CATALOGUE.get(cat_key) if cat_key else None
    if cat:
        if cat.get("random_visualizer"):
            return image.randomVisualizer(), {}, "random_visualizer"
        vis = cat.get("vis_params")
        if vis:
            vis = dict(vis)
            if "min" not in vis or "max" not in vis:
                mm = _minmax(image, var, gee_interface)
                if mm:
                    vis.setdefault("min", mm[0])
                    vis.setdefault("max", mm[1])
            return image, vis, "catalogue_palette"

    rt = getattr(var, "raster_type", None)
    rt = rt.value if hasattr(rt, "value") else (str(rt) if rt is not None else "")
    if rt == "categorical":
        vis = {"palette": ["000000", "ffffff"], "min": 0, "max": 1}
        return image, vis, "categorical_fallback"
    return image, _grayscale_vis(image, var, gee_interface), "continuous_fallback"


def _add_gee_layer(map_, image, var, name: str, layer_key: str):
    """Style and add a GEE image layer to ``map_`` (blocking; run in a thread).

    ``image`` may be None for an asset-id GEEVar that has not been downloaded
    yet: the image is then built here, on the worker thread, through
    ``GEEVar.resolve_images`` (``ee.Image`` needs the initialised client).

    Returns ``(vis, render_kind)`` — plain data the caller uses to build the
    layer's legend on the main thread. Legend text must not be resolved here:
    ``t()`` reads a session-scoped reactive translator, and this runs on a
    worker thread.

    Uses the GEE interface's *synchronous* API (``add_ee_layer`` / ``get_info``),
    which schedules the underlying eeclient session calls onto the interface's
    own private event loop. The async map API (``add_ee_layer_async``) cannot be
    used here: awaited on Solara's event loop it touches session locks bound to
    the interface's loop and raises "bound to a different event loop". Offloading
    this blocking call with ``asyncio.to_thread`` keeps Solara's loop free, the
    same pattern the local raster/vector branches use.
    """
    if image is None:
        image = var.resolve_images()[0]
    styled_image, vis, render_kind = _styled_layer(image, var, map_.gee_interface)
    map_.add_ee_layer(styled_image, vis, name=name, key=layer_key, use_map_vis=False)
    return vis, render_kind


def _variable_to_entry(key: str, var, project) -> dict:
    """Reconstruct a modal entry dict from an existing variable object."""
    from gui.scripts.predefined_variables import resolve_predefined

    vtype = type(var).__name__

    # Predefined GEE variables hold an ee.Image (in gee_images) and carry no
    # local path / asset id — they are rebuilt from the catalogue by key. Round-
    # trip them as a predefined entry so editing re-fetches the image instead of
    # dropping gee_images and stringifying path=None into the literal "None"
    # (which then fails GEEVar validation on save). The name may carry a
    # parameter suffix (forest_gfc_tc30); resolve_predefined splits it back into
    # the catalogue key plus the values, which prefill the modal's param fields.
    cat_key, cat_params = resolve_predefined(var.name)
    if vtype == "GEEVar" and not var.path and cat_key is not None:
        return {
            "source": "predefined",
            "type": "GEEVar",
            "name": var.name,
            "predefined_key": cat_key,
            "params": cat_params,
            "year": str(var.year) if var.year else "",
        }

    entry = {
        "source": "custom",
        "type": vtype,
        "name": var.name,
        "year": str(var.year) if var.year else "",
    }
    if getattr(var, "vis_params", None):
        entry["vis_params"] = dict(var.vis_params)
    if vtype == "LocalRasterVar":
        entry["path"] = str(var.path)
        entry["raster_type"] = (
            var.raster_type.value
            if hasattr(var.raster_type, "value")
            else str(var.raster_type)
        )
    elif vtype == "GEEVar":
        entry["asset_id"] = str(var.path) if var.path else ""
        entry["scale"] = (
            str(var.default_scale) if getattr(var, "default_scale", None) else ""
        )
        if var.raster_type is not None:
            entry["raster_type"] = (
                var.raster_type.value
                if hasattr(var.raster_type, "value")
                else str(var.raster_type)
            )
    elif vtype == "LocalVectorVar":
        entry["path"] = str(var.path)
        entry["rasterization_method"] = (
            var.rasterization_method.value
            if hasattr(var.rasterization_method, "value")
            else str(var.rasterization_method)
        )
    return entry


def entry_key(entry: dict) -> str:
    """Storage key a modal entry will land under in raw_variables.

    Mirrors the key on_add computes from the built variable (name_year, or bare
    name when year is empty) — used to detect a duplicate BEFORE building the
    variable, which for predefined entries would already fetch the GEE image.
    """
    year = entry.get("year")
    return f"{entry['name']}_{year}" if year else entry["name"]


def _build_variable(entry: dict, project):
    """Instantiate the correct variable class from a modal entry dict."""
    if entry.get("source") == "predefined":
        return _build_predefined(entry, project)

    common = dict(
        name=entry["name"],
        year=entry.get("year"),
        project=project,
        # Palette picked in the modal (custom raster layers); None otherwise.
        vis_params=entry.get("vis_params"),
    )
    vtype = entry["type"]
    if vtype == "LocalRasterVar":
        return LocalRasterVar(
            path=entry["path"],
            raster_type=entry.get("raster_type"),
            data_type=entry["data_type"],
            **common,
        )
    if vtype == "GEEVar":
        return GEEVar(
            path=entry["path"],
            default_scale=entry.get("default_scale"),
            raster_type=entry.get("raster_type"),
            data_type=entry["data_type"],
            # Download clips the export to ``aoi.geometry()``; without it a
            # custom asset only fails later, at download time.
            aoi=_current_aoi_ee(),
            **common,
        )
    if vtype == "LocalVectorVar":
        return LocalVectorVar(
            path=entry["path"],
            rasterization_method=entry.get("rasterization_method"),
            data_type=entry["data_type"],
            **common,
        )
    raise ValueError(f"Unknown variable type: {vtype}")


def _current_aoi_ee():
    """The selected AOI as an ee object, or raise if the AOI step is not done."""
    from gui.scripts.predefined_variables import resolve_aoi_ee
    from gui.store.state_manager import app_state

    aoi_result = app_state.aoi_result.value
    if aoi_result is None:
        raise ValueError("No AOI selected — complete the AOI step first.")
    return resolve_aoi_ee(aoi_result)


def _build_predefined(entry: dict, project):
    """Build a GEEVar from a predefined catalogue entry."""
    from gui.scripts.predefined_variables import PREDEFINED_CATALOGUE

    key = entry["predefined_key"]
    cat = PREDEFINED_CATALOGUE[key]

    aoi_ee = _current_aoi_ee()
    year = entry.get("year")
    # Declared params (e.g. forest_gfc's tree_cover_threshold) travel as kwargs;
    # entries for unparameterised layers carry none and call through unchanged.
    image = cat["get_image"](aoi_ee, year, **entry.get("params") or {})

    return GEEVar(
        name=entry["name"],
        data_type=entry["data_type"],
        raster_type=entry.get("raster_type"),
        gee_images=[image],
        aoi=aoi_ee,
        project=project,
        year=year,
        # Export/prefill scale for sources whose native resolution is far from
        # the 30 m download fallback (e.g. ERA5-Land at ~11 km).
        default_scale=cat.get("default_scale"),
    )


def _var_legend(key: str, var, *, vis=None, render_kind=None):
    """The legend a source-variable layer publishes while it is on the map.

    ``vis`` / ``render_kind`` come from ``_add_gee_layer`` for GEE-backed
    layers; a local raster resolves its own style instead. Vector layers
    publish nothing and never reach here.
    """
    from gui.scripts.legend_data import (
        Label,
        variable_spec_from_style,
        variable_spec_from_vis,
    )
    from gui.scripts.legend_registry import LayerLegend
    from gui.scripts.predefined_variables import (
        PREDEFINED_CATALOGUE,
        resolve_predefined,
    )
    from gui.scripts.variable_styles import resolve_variable_style

    cat_key, _params = resolve_predefined(getattr(var, "name", "") or "")
    cat = PREDEFINED_CATALOGUE.get(cat_key) if cat_key else None
    label = (
        Label(key=cat["label_key"])
        if cat and cat.get("label_key")
        else Label(literal=key)
    )

    if vis is not None:
        spec = variable_spec_from_vis(
            vis, render_kind or "continuous_fallback", var, label
        )
    else:
        spec = variable_spec_from_style(resolve_variable_style(var), var, label)

    return LayerLegend(layer_id=_map_layer_key(key), label=label, spec=spec)


def _drop_from_map(key: str, map_, legend_port=None):
    """Remove a variable's layer from the map and forget its on-map state.

    The single removal chokepoint for source variables — toggle-off, delete,
    replace and edit all route through it, so a legend cannot outlive its
    layer. ``legend_port`` may be None (e.g. in tests without one) — that is
    a no-op, not a crash.
    """
    if map_ is not None:
        map_.remove_layer(_map_layer_key(key), none_ok=True)
    if legend_port is not None:
        legend_port.unregister(_map_layer_key(key))
    if key in vars_on_map.value:
        remaining = set(vars_on_map.value)
        remaining.discard(key)
        vars_on_map.set(remaining)


@solara.component
def VariablesTile(project, map_=None, sepal_client=None, legend_port=None):
    """Variables step: add, inspect, and process variables.

    Args:
        project: Reactive holding the current Project (or None).
        map_: SepalMap instance used by the per-variable "show on map" toggle.
        sepal_client: SEPAL client passed through to the Add Variable modal.
        legend_port: LegendPort for publishing/withdrawing source-variable
            legends; None disables legend publication (e.g. in tests without
            one).
    """
    modal_open = solara.use_reactive(False)
    editing_key, set_editing_key = solara.use_state(None)
    pending_toggle = solara.use_reactive(None)
    # Key of the variable being downloaded, or None for a bulk download.
    pending_download = solara.use_reactive(None)
    notifications = use_notifications()

    @solara.lab.use_task(dependencies=None, raise_error=False, prefer_threaded=True)
    async def download_task():
        """Materialize GEE-backed variables to local files (all, or one key).

        Runs on a worker thread (prefer_threaded) so the UI stays responsive;
        progress is driven by download_task.pending.
        """
        p = project.value
        if p is None:
            return
        key = pending_download.value
        keys = [key] if key is not None else None
        var = p.raw_variables.get(key) if key is not None else None
        title = (
            t("notifications.task_download_one", name=getattr(var, "name", key))
            if key is not None
            else t("notifications.task_download_all")
        )

        def _var_name(k):
            return getattr(p.raw_variables.get(k), "name", None) or k

        def _tracked_download():
            # Entered on the pool thread so the per-layer log lines feed this
            # tracker; to_thread_in_context supplies the kernel context the
            # tracker's bus updates need to reach the browser.
            with tracked_job(
                notifications,
                title,
                error_format=lambda exc: t("tiles.variables.error_download", exc=exc),
            ) as task:
                on_progress = layer_progress_reporter(
                    task,
                    format_title=lambda k, i, n: t(
                        "notifications.task_download_layer",
                        i=i + 1,
                        n=n,
                        name=_var_name(k),
                    ),
                    format_detail=lambda k, done, total: t(
                        "notifications.task_download_tiles",
                        name=_var_name(k),
                        done=done,
                        total=total,
                    ),
                )
                process_actions.materialize_raw_layers(p, keys, on_progress=on_progress)
                p.save()

        with writing(p.project_name):
            try:
                await to_thread_in_context(_tracked_download)
            except Exception:
                logger.exception("download failed")  # toast raised by tracked_job
            publish_if_current(project, p)

    def on_download(key=None):
        """Download one variable (key) or all pending GEE variables (None)."""
        pending_download.set(key)
        download_task()

    @solara.lab.use_task(dependencies=None, raise_error=False)
    async def _apply_map_toggle():
        """Add or remove a variable's layer on the map.

        Every layer-add is offloaded to a worker thread. GEE-backed layers use
        the GEE interface's blocking API (via ``_add_gee_layer``) so the session
        calls run on the interface's own event loop; the async map API crashes
        with "bound to a different event loop" when awaited on Solara's loop.
        Local raster/vector layers use the blocking ``add_raster_var_on_map`` /
        ``add_vector_on_map`` helpers the same way. Downloaded rasters keep the
        palette they had as a GEE layer (see ``add_raster_var_on_map``) instead of
        rendering grayscale.
        """
        key = pending_toggle.value
        if key is None or map_ is None:
            return
        p = project.value
        var = p.raw_variables.get(key) if p is not None else None
        if var is None or not is_mappable(var):
            return
        try:
            if key in vars_on_map.value:
                _drop_from_map(key, map_, legend_port)
                return

            images = getattr(var, "gee_images", None)
            layer_key = _map_layer_key(key)
            label = raw_layer_label(key)
            generation = legend_port.generation() if legend_port is not None else None
            legend = None
            if images or type(var).__name__ == "GEEVar":
                # An asset-id GEEVar has no image before download; the worker
                # resolves it (see _add_gee_layer).
                vis, render_kind = await asyncio.to_thread(
                    _add_gee_layer,
                    map_,
                    images[0] if images else None,
                    var,
                    label,
                    layer_key,
                )
                legend = _var_legend(key, var, vis=vis, render_kind=render_kind)
            elif type(var).__name__ == "LocalVectorVar":
                await asyncio.to_thread(
                    add_vector_on_map, map_, str(var.path), label, layer_key
                )
            else:  # LocalRasterVar — reuse the palette it had as a GEE layer
                await asyncio.to_thread(
                    add_raster_var_on_map,
                    map_,
                    str(var.path),
                    var=var,
                    layer_name=label,
                    key=layer_key,
                    fit_bounds=False,
                )
                legend = _var_legend(key, var)

            # A project switch during the await means this layer is stale
            # (see the InferenceTile add branch for the same guard, kept
            # outside `finally` there because a `return` inside `finally`
            # would discard an in-flight exception) — take it back off
            # rather than publish a legend for it.
            if legend_port is not None and legend_port.generation() != generation:
                map_.remove_layer(layer_key, none_ok=True)
                return

            vars_on_map.set(set(vars_on_map.value) | {key})
            if legend is not None and legend_port is not None:
                legend_port.register(legend)
        except Exception as exc:
            logger.exception("map toggle failed for %s", key)
            notifications.error(
                t("tiles.variables.error_toggle_map", key=key, exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )

    def on_toggle_map(key: str):
        """Trigger the async toggle task for one source variable."""
        if map_ is None:
            return
        pending_toggle.set(key)
        _apply_map_toggle()

    pending_add, set_pending_add = solara.use_state(None)

    def _do_add(entry: dict):
        p = project.value
        if p is None:
            logger.warning("on_add: project is None")
            notifications.error(
                t("tiles.variables.error_no_project"),
                timeout=ERROR_TOAST_TIMEOUT,
            )
            return
        try:
            var = _build_variable(entry, p)
            key = f"{var.name}_{var.year}" if var.year else var.name
            # Replacing an existing entry needs the same cleanup as an edit:
            # drop the stale map layer and reset the base raster if this was
            # its source (the replacement starts cloud-backed again).
            old = p.raw_variables.pop(key, None)
            if old is not None:
                if is_base_raster(p, old):
                    p.base_raster = None
                    notifications.warning(
                        t("tiles.variables.error_base_raster_reset", name=old.name),
                        timeout=ERROR_TOAST_TIMEOUT,
                    )
                _drop_from_map(key, map_, legend_port)
            p.raw_variables[key] = var
            logger.debug(
                "Added var '%s', raw_variables now: %s",
                key,
                list(p.raw_variables.keys()),
            )
            project.set(p.model_copy())
        except Exception as exc:
            logger.exception("on_add failed")
            notifications.error(
                t("tiles.variables.error_add_variable", exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )

    def on_add(entry: dict):
        logger.debug("on_add called: %s", entry)
        p = project.value
        if p is None:
            logger.warning("on_add: project is None")
            notifications.error(
                t("tiles.variables.error_no_project"),
                timeout=ERROR_TOAST_TIMEOUT,
            )
            return
        # Duplicate key (e.g. re-adding a predefined variable that was already
        # downloaded): confirm before silently clobbering it.
        if entry_key(entry) in p.raw_variables:
            set_pending_add(entry)
            return
        _do_add(entry)

    def on_edit_open(key: str):
        set_editing_key(key)
        modal_open.set(True)

    def on_save(old_key: str, new_entry: dict):
        p = project.value
        if p is None:
            return
        try:
            old_var = p.raw_variables.pop(old_key, None)
            if is_base_raster(p, old_var):
                p.base_raster = None
                notifications.warning(
                    t("tiles.variables.error_base_raster_reset", name=old_var.name),
                    timeout=ERROR_TOAST_TIMEOUT,
                )
            # The key may change on edit — drop the stale layer so it doesn't linger.
            _drop_from_map(old_key, map_, legend_port)
            var = _build_variable(new_entry, p)
            new_key = f"{var.name}_{var.year}" if var.year else var.name
            p.raw_variables[new_key] = var
            set_editing_key(None)
            project.set(p.model_copy())
        except Exception as exc:
            logger.exception("on_save failed")
            notifications.error(
                t("tiles.variables.error_save_variable", exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )

    pending_remove, set_pending_remove = solara.use_state(None)

    def _do_remove(key: str):
        p = project.value
        if p is None:
            return
        removed = p.raw_variables.pop(key, None)
        if is_base_raster(p, removed):
            p.base_raster = None
            notifications.warning(
                t("tiles.variables.error_base_raster_removed", name=removed.name),
                timeout=ERROR_TOAST_TIMEOUT,
            )
        _drop_from_map(key, map_, legend_port)
        project.set(p.model_copy())

    p = project.value
    pending_geevars = (
        [k for k, v in p.raw_variables.items() if type(v).__name__ == "GEEVar"]
        if p
        else []
    )

    with solara.Column(style="gap: 16px;"):
        with solara.Row(style="gap:4px;align-items:center;"):
            solara.Text(t("tiles.variables.description"))
            InfoButton(t("tiles.variables.info_header"), t("tiles.variables.info_md"))

        # Action bar
        solara.Button(
            t("tiles.variables.add_variable_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=lambda: modal_open.set(True),
        )

        # Source variable list (ProductTable renders its own collapsible header)
        SourceVariableList(
            project=project,
            on_remove=set_pending_remove,
            on_edit=on_edit_open,
            on_toggle_map=on_toggle_map if map_ is not None else None,
            vars_on_map=vars_on_map,
            on_download=on_download,
            download_pending=download_task.pending,
            downloading_key=pending_download.value if download_task.pending else None,
        )

        # Download-all button, below the list
        solara.Button(
            t("tiles.variables.download_button", count=len(pending_geevars)),
            icon_name="mdi-cloud-download-outline",
            color="primary",
            outlined=True,
            small=True,
            on_click=lambda: on_download(None),
            loading=download_task.pending and pending_download.value is None,
            disabled=download_task.pending or not pending_geevars,
        )
        if download_task.pending:
            solara.ProgressLinear(True)

    editing_entry = (
        _variable_to_entry(editing_key, p.raw_variables[editing_key], p)
        if editing_key and p and editing_key in p.raw_variables
        else None
    )
    VariableModal(
        open_=modal_open,
        on_add=on_add,
        on_save=on_save,
        editing_key=editing_key,
        initial_entry=editing_entry,
        sepal_client=sepal_client,
        existing_keys=frozenset(p.raw_variables) if p else frozenset(),
    )

    _pending_var = (
        p.raw_variables.get(pending_remove) if (p and pending_remove) else None
    )
    _pending_is_base = is_base_raster(p, _pending_var)
    _confirm_msg = t(
        "tiles.variables.confirm_remove_message", name=pending_remove or ""
    )
    if _pending_is_base:
        _confirm_msg += " " + t("tiles.variables.confirm_remove_base_warning")
    ConfirmDialog(
        open=pending_remove is not None,
        on_cancel=lambda: set_pending_remove(None),
        on_confirm=lambda: (_do_remove(pending_remove), set_pending_remove(None)),
        title=t("tiles.variables.confirm_remove_title"),
        message=_confirm_msg,
        confirm_label=t("common.remove"),
    )

    # Duplicate-add confirmation — warns when the replaced variable was already
    # downloaded (status resets to cloud) or backs the base raster.
    _add_key = entry_key(pending_add) if pending_add else None
    _add_old = p.raw_variables.get(_add_key) if (p and _add_key) else None
    _replace_msg = t("tiles.variables.confirm_replace_message", key=_add_key or "")
    if _add_old is not None and type(_add_old).__name__ != "GEEVar":
        _replace_msg += " " + t("tiles.variables.confirm_replace_downloaded_warning")
    if is_base_raster(p, _add_old):
        _replace_msg += " " + t("tiles.variables.confirm_replace_base_warning")
    ConfirmDialog(
        open=pending_add is not None,
        on_cancel=lambda: set_pending_add(None),
        on_confirm=lambda: (_do_add(pending_add), set_pending_add(None)),
        title=t("tiles.variables.confirm_replace_title"),
        message=_replace_msg,
        confirm_label=t("common.replace"),
    )
