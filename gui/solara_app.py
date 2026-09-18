"""Spatial Risk — Solara GUI entry point.

Run locally:
    ./run_solara.sh gui/solara_app.py 8910
"""

import asyncio
import logging
from datetime import datetime

import reacton.ipyvuetify as rv
import solara
from pysepal import mapping as sm
from pysepal.logger import setup_logging
from pysepal.sepalwidgets.vue_app import LocaleSelect, MapApp, ThemeToggle
from pysepal.solara import (
    NotificationProvider,
    get_current_gee_interface,
    get_current_sepal_client,
    setup_sessions,
    setup_solara_server,
    setup_theme_colors,
    use_notifications,
    with_sepal_sessions,
)
from pysepal.solara.locale import resolve_locale_state
from solara.lab.components.theming import theme

from gui.i18n import get_translator, reset_translator, set_app_locale, t
from gui.scripts.aoi_io import load_aoi, persist_aoi
from gui.scripts.map_helpers import (
    add_satellite_basemap,
    clear_project_overlays,
    show_aoi_on_map,
    sync_draw_control_visibility,
)
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT, install_task_log_handler
from gui.scripts.project_io import (
    delete_project,
    list_project_infos,
    load_project,
    project_dir_size,
    save_project,
)
from gui.scripts.project_ui_helpers import (
    aoi_project_name,
    compute_app_title,
    format_last_saved,
    manage_projects_label,
    overwrite_needed,
    validate_project_name,
)
from gui.scripts.tile_proxy import borrow_localtileserver_prefix
from gui.store.project_writers import is_writing
from gui.store.state_manager import app_state
from gui.tile.aoi_tile import AoiTile
from gui.tile.dataset_tile import DatasetTile
from gui.tile.derived_map import derived_on_map
from gui.tile.evaluation_tile import EvaluationTile, eval_jobs
from gui.tile.inference_tile import InferenceTile, inference_jobs, preds_on_map
from gui.tile.postprocess_tile import PostProcessTile
from gui.tile.process_tile import ProcessTile
from gui.tile.sampling_tile import SamplingTile, samples_on_map, sampling_jobs
from gui.tile.summary_tile import ProjectSummaryTile
from gui.tile.toolbox_tile import ToolboxTile, allocation_jobs, density_on_map
from gui.tile.train_tile import TrainTile, train_jobs
from gui.tile.variables_tile import VariablesTile, vars_on_map
from gui.widget.manage_projects import ConfirmDeleteProjectDialog, ManageProjectsDialog
from gui.widget.pipeline_header import PipelineHeader
from gui.widget.text_style import MUTED
from spatialrisk.project import DATA_DIR, Project

# On SEPAL, reuse the raster tiles' jupyter-server-proxy route for the PMTiles
# vector-tile server (vectortileserver never autodetects one). Must run before
# any vectortileserver TileClient is built — they are only constructed lazily
# when a sample layer is added, so after imports is early enough.
borrow_localtileserver_prefix()

logger = setup_logging(logger_name="spatial_risk")
logger.setLevel(logging.DEBUG)
logger.debug("Spatial Risk app initialized")
logger.debug("Solara version: %s", solara.__version__)

# Forward INFO+ milestones from tracked background jobs into the pysepal
# notification task pill (see gui/scripts/notify_bridge.py).
install_task_log_handler()

setup_solara_server(extra_asset_locations=[])


@solara.lab.on_kernel_start
def on_kernel_start():
    """Reset per-kernel state and open the SEPAL sessions."""
    reset_translator()  # drop the cached translator; Page rebuilds it from the
    # session LocaleState (the browser resolves the locale, not the config file)
    return setup_sessions()


def _loopback_bridge_widget():
    """Return the jupyter-loopback CommBridge singleton (or None).

    Layer helpers (localtileserver, pmtiles_map) enable the bridge lazily from
    worker threads, but under voila a ``display()`` outside the initial cell
    execution never reaches the browser — the widget must be mounted in the
    app's own widget tree so its JS half installs the tile-URL interceptors
    before any local tile layer renders.
    """
    try:
        import jupyter_loopback

        return jupyter_loopback.enable_comm_bridge(display=False)
    except Exception:  # pragma: no cover - anywidget missing / exotic runtime
        logging.getLogger(__name__).warning(
            "jupyter-loopback comm bridge unavailable; local tile layers "
            "may not render under voila",
            exc_info=True,
        )
        return None


@solara.component
def ProjectPanel(on_close=None):
    """Current-project status + New / Load / Save controls (left drawer).

    ``on_close`` (optional) closes the hosting Project step-dialog; called once a
    project finishes loading so the user lands back on the map.
    """
    p = app_state.project.value
    dirty = app_state.project_dirty.value
    last_saved = app_state.last_saved.value
    notifications = use_notifications()

    # Dialog / transient UI state
    load_open, set_load_open = solara.use_state(False)
    new_open, set_new_open = solara.use_state(False)
    discard_open, set_discard_open = solara.use_state(False)
    overwrite_open, set_overwrite_open = solara.use_state(False)

    infos = solara.use_reactive([])  # list[ProjectInfo]
    scan_failed = solara.use_reactive(False)
    selected = solara.use_reactive(None)  # selected project name
    pending_delete = solara.use_reactive(None)  # ProjectInfo being deleted
    pending_size = solara.use_reactive(0)  # its size on disk, in bytes
    # Delete failures are owned here rather than read off delete_task.error /
    # .exception: a Task keeps its error until the *next* invoke, so the trash
    # button on project B would open B's confirmation with A's failure under it.
    delete_error = solara.use_reactive(None)

    load_error, set_load_error = solara.use_state(None)
    load_busy, set_load_busy = solara.use_state(False)

    new_name, set_new_name = solara.use_state("")

    def refresh_infos():
        """Rescan the saved projects.

        The single source of truth for both the dialog list and the empty-state
        button's count — a delete changes it, so the old use_memo(deps=[]) count
        went stale the moment one landed.
        """
        try:
            infos.set(list_project_infos(DATA_DIR))
            scan_failed.set(False)
        except Exception as exc:  # pragma: no cover - defensive
            # Keep the reason. Swallowing it left the user with a bare "No saved
            # projects yet" and nothing to act on; the dialog surfaces this text.
            infos.set([])
            scan_failed.set(True)
            set_load_error(str(exc))

    solara.use_effect(refresh_infos, [])
    saved_count = None if scan_failed.value else len(infos.value)

    def existing_names() -> list:
        return [i.name for i in list_project_infos(DATA_DIR)]

    # ---- Manage / Load ---------------------------------------------------
    def open_manage():
        set_load_error(None)
        selected.set(None)
        refresh_infos()
        set_load_open(True)

    def do_load():
        if delete_task.pending:
            return  # a delete is in flight; nothing else may be staged
        name = selected.value
        if not name:
            return
        set_load_busy(True)
        set_load_error(None)
        try:
            loaded = load_project(name)
            when = next((i.modified for i in infos.value if i.name == name), None)
            # Restore the saved AOI (sidecar geometry + metadata) so the map can
            # frame it and the downstream tabs unlock. Set before installing the
            # project so the load-zoom effect sees it on the same render.
            # restoring_project(): Solara can render BETWEEN these two sets, and
            # the attach-on-select effect must not persist the incoming AOI
            # into the outgoing project (see AppState.attach_current_aoi).
            with app_state.restoring_project():
                app_state.aoi_result.set(
                    load_aoi(DATA_DIR / loaded.project_name, loaded.aoi)
                )
                app_state.load_project_state(loaded, when)
            notifications.success(t("project.status_loaded", name=name))
            set_load_open(False)
            if on_close is not None:
                on_close()
        except Exception as exc:
            set_load_error(str(exc))
        finally:
            set_load_busy(False)

    # ---- Delete ----------------------------------------------------------
    def open_delete(info):
        """Row trash button: stage a target and price it."""
        if delete_task.pending:
            return  # a delete is in flight; nothing else may be staged
        delete_error.set(None)  # never carry the last target's failure into this one
        # Priced for this one project only — never per row, so the list stays
        # cheap even with a multi-GB project in it.
        pending_size.set(project_dir_size(info.name))
        pending_delete.set(info)

    def cancel_delete():
        pending_delete.set(None)

    target = pending_delete.value
    target_is_open = (
        target is not None and p is not None and p.project_name == target.name
    )
    # A background task is still saving into this folder. Deleting now would let
    # its auto-save re-create the folder (Project.save() does mkdir(exist_ok=True)).
    # Keyed by name, so it also catches a job orphaned by a project switch.
    target_busy = target is not None and is_writing(target.name)

    @solara.lab.use_task(dependencies=None, raise_error=False, prefer_threaded=True)
    async def delete_task():
        """Delete the staged project's folder, off the render thread.

        Mirrors process_tile.process_task: async body, blocking call handed to
        asyncio.to_thread, target passed via a reactive. A 3.2 GB rmtree on a
        network-backed SEPAL home takes seconds — blocking here would freeze the UI.

        The rmtree cannot be cancelled once it has started, so nothing here may
        assume the world stood still while it ran: everything after the await is
        reconciled against what is actually on disk.
        """
        info = pending_delete.value
        if info is None:
            return
        delete_error.set(None)  # a retry starts clean
        # The render-time is_writing() gate is a TOCTOU: a writer can register
        # between the last render and this click, and its auto-save would re-create
        # the folder right after the rmtree. Re-check on the kernel thread.
        if is_writing(info.name):
            delete_error.set(t("project.dialog_delete_busy"))
            return
        try:
            await asyncio.to_thread(delete_project, info.name)
        except Exception as exc:
            delete_error.set(str(exc))
        finally:
            # Reconcile against the disk, not against what we assumed: the delete
            # can fail or partially fail, and we may have awaited for seconds while
            # the user moved on. Skipping this block (an unguarded raise used to)
            # would strand the app holding a project whose folder is gone — the next
            # save would then re-create it as a manifest-only zombie.
            refresh_infos()
            if selected.value == info.name:
                selected.set(None)
            gone = not (DATA_DIR / info.name).exists()
            # Re-read the project: like the closure's `p`, anything read before the
            # await can be stale — the user may have loaded a different one since.
            live = app_state.project.value
            if gone and live is not None and live.project_name == info.name:
                app_state.close_project_state()
            if gone:
                notifications.success(t("project.status_deleted", name=info.name))
                pending_delete.set(None)  # closes the confirm dialog; manage stays open

    def confirm_delete():
        """Confirm button: drop a re-click while a delete is already in flight.

        ``TaskAsyncio.__call__`` sets ``pending`` synchronously on this thread, so
        this is a hard guard; the button's ``disabled`` only reaches the browser a
        round-trip later, so a real double-click does land twice. Re-invoking would
        cancel the in-flight task — which does NOT stop the rmtree — and skip its
        continuation entirely, leaving the app holding a project whose folder is gone.
        """
        if delete_task.pending:
            return
        delete_task()

    # ---- New ------------------------------------------------------------
    def open_new():
        if dirty and p is not None:
            set_discard_open(True)
        else:
            _open_new_dialog()

    def _open_new_dialog():
        set_discard_open(False)
        set_new_name("")
        set_new_open(True)

    def do_create():
        validation = validate_project_name(new_name, existing_names())
        if not validation.valid:
            return  # Create button is disabled in this state; no-op guard
        app_state.new_project_state(Project(project_name=validation.cleaned))
        # new_project_state bumps project_loaded_signal, so the shell's
        # on-switch effects clear the previous project's map overlays/tracking
        # and reset the (empty) Train/Inference job lists — no manual reset here.
        notifications.success(t("project.status_created", name=validation.cleaned))
        set_new_open(False)
        # Dismiss the whole Project popup too, not just the inner New dialog, so
        # a freshly created project returns the user to the map (mirrors do_load).
        if on_close is not None:
            on_close()

    def load_instead():
        name = validate_project_name(new_name, existing_names()).cleaned
        set_new_open(False)
        selected.set(name)
        refresh_infos()
        set_load_open(True)

    # ---- Save -----------------------------------------------------------
    def do_save():
        if delete_task.pending:
            return  # a delete is in flight; nothing else may be staged
        if p is None:
            notifications.error(
                t("project.error_no_project_to_save"), timeout=ERROR_TOAST_TIMEOUT
            )
            return
        if overwrite_needed(p.project_name, last_saved, existing_names()):
            set_overwrite_open(True)
            return
        _really_save()

    def _really_save():
        set_overwrite_open(False)
        if delete_task.pending:
            return  # a delete is in flight; nothing else may be staged
        if p is None:  # project was deleted while the overwrite dialog was open
            return
        try:
            # Persist the AOI alongside the project: geometry → aoi.geojson
            # sidecar, light metadata → project.aoi (saved into the manifest).
            # Passing the stored metadata lets persist_aoi refuse to replace a
            # saved AOI with an empty one — see gui/scripts/aoi_io.py.
            p.aoi = persist_aoi(
                DATA_DIR / p.project_name,
                app_state.aoi_result.value,
                p.aoi,
            )
            path = save_project(p)
            app_state.mark_saved(datetime.now())
            note = ""
            if not p.raw_variables:
                note = t("project.status_saved_note_no_vars")
            elif p.base_raster is None:
                note = t("project.status_saved_note_no_base")
            notifications.success(t("project.status_saved", path=path, note=note))
        except Exception as exc:
            notifications.error(str(exc), timeout=ERROR_TOAST_TIMEOUT)

    # ---- Status block ---------------------------------------------------
    with solara.Column(style="gap: 8px; padding: 8px;"):
        if p is None:
            # Empty state: one primary action (New), Load secondary, no Save —
            # a disabled Save here can never become useful, so it's omitted.
            rv.Icon(
                children=["mdi-map-marker-plus-outline"],
                style_="font-size: 40px; opacity: 0.5;",
            )
            solara.Text(t("project.empty_heading"), style="font-weight: 600;")
            solara.Text(
                t("project.empty_help"),
                style=MUTED,
            )
            solara.Button(
                t("project.button_new_project"),
                icon_name="mdi-plus",
                color="primary",
                small=True,
                block=True,
                on_click=open_new,
            )
            solara.Button(
                manage_projects_label(saved_count),
                icon_name="mdi-folder-open-outline",
                color="primary",
                outlined=True,
                small=True,
                block=True,
                # Only a *known* zero disables it. A failed scan (None) leaves the
                # button live, so the user can open the dialog and read why.
                disabled=saved_count == 0,
                on_click=open_manage,
            )
        else:
            with solara.Row(style="gap: 8px; align-items: center;"):
                solara.Text(p.project_name, style="font-weight: 600;")
                rv.Chip(
                    children=[
                        t("project.chip_unsaved") if dirty else t("project.chip_saved")
                    ],
                    color="warning" if dirty else "primary",
                    text_color="white",
                    x_small=True,
                )
            solara.Text(
                t(
                    "project.stats",
                    raw=len(p.raw_variables),
                    processed=len(p.processed_variables),
                    models=len(p.models),
                ),
                style=MUTED + "font-size: 12px;",
            )
            solara.Text(
                format_last_saved(last_saved, datetime.now()),
                style=MUTED + "font-size: 12px;",
            )
            with solara.Row(style="gap: 8px;"):
                solara.Button(
                    t("project.button_new"),
                    icon_name="mdi-plus",
                    color="primary",
                    outlined=True,
                    small=True,
                    on_click=open_new,
                )
                solara.Button(
                    t("project.button_manage"),
                    icon_name="mdi-folder-open-outline",
                    color="primary",
                    outlined=True,
                    small=True,
                    on_click=open_manage,
                )
                solara.Button(
                    t("project.button_save"),
                    icon_name="mdi-content-save-outline",
                    color="primary",
                    outlined=True,
                    small=True,
                    on_click=do_save,
                    # Same round-trip-lag caveat as the Load button above: cosmetic
                    # only. do_save's own delete_task.pending guard is what holds.
                    disabled=delete_task.pending,
                )

    # ---- New dialog -----------------------------------------------------
    validation = validate_project_name(new_name, existing_names()) if new_open else None
    with rv.Dialog(
        v_model=new_open, on_v_model=set_new_open, max_width="400px", eager=True
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("project.dialog_new_title"))
            with rv.CardText():
                rv.TextField(
                    label=t("project.dialog_new_name_label"),
                    v_model=new_name,
                    on_v_model=set_new_name,
                    dense=True,
                    outlined=True,
                    autofocus=True,
                )
                if validation and new_name and not validation.valid:
                    solara.Error(validation.error)
                if validation and validation.valid and validation.exists:
                    solara.Warning(
                        t("project.dialog_new_exists_warning", name=validation.cleaned)
                    )
                    solara.Button(
                        t("project.dialog_new_load_instead"),
                        text=True,
                        small=True,
                        on_click=load_instead,
                    )
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"),
                    on_click=lambda: set_new_open(False),
                    text=True,
                    small=True,
                )
                solara.Button(
                    t("common.create"),
                    on_click=do_create,
                    color="primary",
                    small=True,
                    disabled=not (validation and validation.valid),
                )

    # ---- Discard-unsaved confirm (New while dirty) ----------------------
    with rv.Dialog(
        v_model=discard_open, on_v_model=set_discard_open, max_width="380px", eager=True
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("project.dialog_discard_title"))
            with rv.CardText():
                solara.Text(
                    t("project.dialog_discard_message", name=p.project_name)
                    if p is not None
                    else t("project.dialog_discard_title")
                )
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"),
                    on_click=lambda: set_discard_open(False),
                    text=True,
                    small=True,
                )
                solara.Button(
                    t("project.dialog_discard_confirm"),
                    on_click=_open_new_dialog,
                    color="error",
                    small=True,
                )

    # ---- Overwrite confirm (Save over an existing project) --------------
    with rv.Dialog(
        v_model=overwrite_open,
        on_v_model=set_overwrite_open,
        max_width="380px",
        eager=True,
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("project.dialog_overwrite_title"))
            with rv.CardText():
                solara.Text(
                    t("project.dialog_overwrite_message", name=p.project_name)
                    if p is not None
                    else t("project.dialog_overwrite_title")
                )
            with rv.CardActions(style_="justify-content: flex-end; gap: 8px;"):
                solara.Button(
                    t("common.cancel"),
                    on_click=lambda: set_overwrite_open(False),
                    text=True,
                    small=True,
                )
                solara.Button(
                    t("project.dialog_overwrite_confirm"),
                    on_click=_really_save,
                    color="error",
                    small=True,
                )

    # ---- Manage dialog + delete confirmation ----------------------------
    # Both render at the panel's top level, never inside a list row.
    ManageProjectsDialog(
        open=load_open,
        infos=infos.value,
        selected=selected.value,
        current=p.project_name if p is not None else None,
        on_select=selected.set,
        on_load=do_load,
        on_delete=open_delete,
        on_cancel=lambda: set_load_open(False),
        # `or delete_task.pending` so the Load button also *looks* disabled while a
        # delete is in flight; the handler-side guard in do_load is what actually
        # holds (this is a render-time prop, so it lags by one round-trip).
        busy=load_busy or delete_task.pending,
        error=load_error,
    )

    ConfirmDeleteProjectDialog(
        open=target is not None,
        name=target.name if target is not None else "",
        size_bytes=pending_size.value,
        on_cancel=cancel_delete,
        on_confirm=confirm_delete,  # never the task itself: it is re-entrant
        is_open_project=target_is_open,
        writer_active=target_busy,
        busy=delete_task.pending,
        error=delete_error.value,
    )


@solara.component
def WorkflowTabs(map_, gee_interface, sepal_client=None):
    """Workflow panel: pipeline header (step map + navigation) over the step tiles.

    Step order/gating live in gui/store/workflow_steps.py.
    """
    active_tab, set_active_tab = solara.use_state(0)

    # TabsItems hides inactive tabs client-side without unmounting them, so
    # AoiView never gets to remove its draw control (toolbar + editable drawn
    # shape) from the shared map when the user moves to another step. Mirror
    # the tab state onto the map here. Also keyed on project_loaded_signal
    # (a load remounts AoiView, whose restore may seed the control back onto
    # the map while another tab is active) and on the AOI loading flag: the
    # restore auto-select re-seeds the control from its async task — after
    # the load-time effect run — and flips loading False right afterwards,
    # so that flip is what re-hides a task-time re-add on a non-AOI tab.
    dc_hidden = solara.use_ref(False)
    # Last project_loaded_signal this effect saw: the helper may only drop a
    # remembered hide on a project switch — any other re-run while away
    # (moving between two non-AOI tabs, a loading flip) must preserve it, or
    # the toolbar never comes back on returning to the AOI tab.
    last_load_signal = solara.use_ref(app_state.project_loaded_signal.value)

    def _sync_draw_control():
        signal = app_state.project_loaded_signal.value
        project_switched = signal != last_load_signal.current
        last_load_signal.current = signal
        dc_hidden.current = sync_draw_control_visibility(
            map_,
            aoi_active=active_tab == 0,
            was_hidden=dc_hidden.current,
            project_switched=project_switched,
        )

    solara.use_effect(
        _sync_draw_control,
        [
            active_tab,
            app_state.project_loaded_signal.value,
            app_state.loading.value,
        ],
    )

    PipelineHeader(
        active_step=active_tab,
        on_navigate=set_active_tab,
        project=app_state.project,
        aoi_result=app_state.aoi_result,
    )

    with rv.TabsItems(v_model=active_tab):
        with rv.TabItem():
            AoiTile(
                map_=map_,
                gee_interface=gee_interface,
                aoi_result=app_state.aoi_result,
                restore_signal=app_state.project_loaded_signal.value,
                loading=app_state.loading,
            )
        with rv.TabItem():
            VariablesTile(
                project=app_state.project,
                map_=map_,
                sepal_client=sepal_client,
                legend_port=app_state.legend_port,
            )
        with rv.TabItem():
            ProcessTile(
                project=app_state.project,
                processing=app_state.processing,
                map_=map_,
                legend_port=app_state.legend_port,
            )
        with rv.TabItem():
            PostProcessTile(
                project=app_state.project,
                map_=map_,
                legend_port=app_state.legend_port,
            )
        with rv.TabItem():
            DatasetTile(project=app_state.project)

        with rv.TabItem():
            SamplingTile(project=app_state.project, map_=map_)

        with rv.TabItem():
            TrainTile(project=app_state.project)

        with rv.TabItem():
            InferenceTile(
                project=app_state.project,
                map_=map_,
                sepal_client=sepal_client,
                legend_port=app_state.legend_port,
            )

        with rv.TabItem():
            EvaluationTile(project=app_state.project)


def legend_props(state, translate) -> dict:
    """Props for pysepal's LegendComponent, derived from the legend registry.

    Pure and translation-time: the registry stores language-neutral specs, so
    everything visible is resolved here, on every render. That is what makes a
    locale switch re-translate legends that are already on screen.
    """
    from dataclasses import asdict

    from gui.scripts.legend_data import resolve_label, to_legend_data

    legends = state.layer_legends.value
    selected = state.selected_legend.value
    current = next((e for e in legends if e.layer_id == selected), None)

    return {
        "legend_data": asdict(to_legend_data(current.spec, translate))
        if current
        else {},
        "selector_options": [
            {"value": e.layer_id, "text": resolve_label(e.label, translate)}
            for e in legends
        ],
        "selected": current.layer_id if current else "",
    }


@solara.component
def MapLegend(state):
    """Floating map legend for the layers the app has added.

    Mounted as a sibling of MapApp (pysepal's own template does the same): the
    component is fixed bottom-center at z-index 150, above the map's 0/100, so it
    overlays the map rather than being clipped by it.
    """
    from pysepal.solara.components.legend import LegendComponent

    props = legend_props(state, t)
    LegendComponent(
        legend_data=props["legend_data"],
        selector_options=props["selector_options"],
        selected=props["selected"],
        event_set_selected=state.selected_legend.set,
    )


@solara.component
@with_sepal_sessions(module_name="spatial_risk")
def Page():
    """Main application page using MapApp layout.

    Left drawer  — Project (load / save dialog)
    Center       — SepalMap (fullscreen, always visible)
    Right panel  — Workflow tabs: AOI / Variables / Dataset
    """
    setup_theme_colors()

    gee_interface = get_current_gee_interface()
    sepal_client = get_current_sepal_client()
    theme_toggle = solara.use_memo(lambda: ThemeToggle(), [])
    locale_state = resolve_locale_state()
    locale_select = solara.use_memo(
        lambda: LocaleSelect(translator=get_translator()), []
    )

    def _bind_locale():
        # Wired here — NOT in on_kernel_start — because @with_sepal_sessions
        # creates the session's LocaleState only when Page first renders;
        # kernel-start would bind the process fallback (Codex review P1).
        locale_select.bind_locale_state(locale_state)

        def handler(change):
            set_app_locale(change["new"])

        locale_state.observe(handler, "locale")
        return lambda: locale_state.unobserve(handler, "locale")

    solara.use_effect(_bind_locale, [id(locale_state)])

    def _observe_theme():
        def handler(e):
            return setattr(theme, "dark", e["new"])

        theme_toggle.observe(handler, "dark")
        return lambda: theme_toggle.unobserve(handler, "dark")

    solara.use_effect(_observe_theme, [])

    def create_map():
        map_ = sm.SepalMap(
            zoom=3,
            min_zoom=3,
            center=[0, 0],
            gee=True,
            gee_interface=gee_interface,
            theme_toggle=theme_toggle,
            fullscreen=True,
        )
        # Second basemap (hidden) so the layers control offers a satellite
        # imagery choice next to the theme-driven CartoDB base.
        add_satellite_basemap(map_)
        return map_

    sepal_map = solara.use_memo(create_map, [id(gee_interface)])

    def sync_project_from_aoi():
        aoi = app_state.aoi_result.value
        if aoi is None or app_state.project.value is not None:
            return
        from spatialrisk.project import Project

        name = aoi_project_name(aoi.name, datetime.now())
        app_state.project.set(Project(project_name=name))

    solara.use_effect(sync_project_from_aoi, [app_state.aoi_result.value])

    # Persist the AOI into the open project the moment it is selected — not
    # only on manual Save. Job completions save the manifest directly
    # (project.save()), and before this a workflow-driven project that was
    # never manually saved wrote every one of those manifests with aoi: null,
    # reloading with all its artifacts but no AOI. Identity deps: Project's
    # reactive compares by identity, and attach_aoi is idempotent (a load
    # re-running it with the just-restored AOI is a no-op). Routed through
    # attach_current_aoi, which refuses to write while a load is mid-swap —
    # a render CAN slip between do_load's two reactive sets, and the stale
    # (new AOI, old project) pair wrote one project's AOI into another.
    def persist_aoi_on_select():
        app_state.attach_current_aoi(data_dir=DATA_DIR)

    solara.use_effect(
        persist_aoi_on_select,
        [id(app_state.aoi_result.value), id(app_state.project.value)],
    )

    # On every project switch (load OR new), the signal below bumps so these
    # effects re-run. Read it here so Page re-renders (and the effects re-run).
    project_loaded_signal = app_state.project_loaded_signal.value

    # View reset. The SepalMap is shared across switches, so first drop the
    # previous project's overlay layers (variables, sample points, predictions,
    # old AOI; basemaps kept) and forget each tile's "on map" tracking, THEN
    # draw this project's AOI and frame it. Clearing MUST precede the redraw, so
    # this effect runs before the job-list reset effect below.
    def render_map_on_switch():
        clear_project_overlays(sepal_map)
        vars_on_map.set(set())
        derived_on_map.set(set())
        samples_on_map.set(set())
        preds_on_map.set(set())
        density_on_map.set(set())
        app_state.clear_legends()
        show_aoi_on_map(sepal_map, app_state.aoi_result.value)

    solara.use_effect(render_map_on_switch, [project_loaded_signal])

    # Session job lists are transient overlays; product rows derive from the
    # loaded project's registries at render time. Only the leftovers of the
    # previous project's runs need clearing on switch.
    def reset_jobs_on_load():
        train_jobs.set([])
        inference_jobs.set([])
        eval_jobs.set([])
        sampling_jobs.set([])
        allocation_jobs.set([])

    solara.use_effect(reset_jobs_on_load, [project_loaded_signal])

    def _seed_test_aoi():
        import os

        if os.getenv("SOLARA_TEST", "false").lower() != "true":
            return
        if app_state.aoi_result.value is not None:
            return
        import geopandas as gpd
        from pysepal.solara.components.aoi import AoiResult
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            {"name": ["San Marino"]},
            geometry=[box(12.403, 43.893, 12.517, 43.993)],
            crs="EPSG:4326",
        )
        logger.debug("SOLARA_TEST: seeding AOI with San Marino")
        app_state.aoi_result.set(AoiResult(method="DRAW", name="San Marino", gdf=gdf))

    # Test AOI seeding disabled for now — start from an empty project.
    # (SOLARA_TEST stays on; re-enable by uncommenting the line below.)
    # solara.use_effect(_seed_test_aoi, [])

    def _seed_test_variables():
        import os

        if os.getenv("SOLARA_TEST", "false").lower() != "true":
            return
        p = app_state.project.value
        aoi_result = app_state.aoi_result.value
        if p is None or p.raw_variables or aoi_result is None:
            return
        from gui.scripts.predefined_variables import (
            PREDEFINED_CATALOGUE,
            get_aoi_ee_feature,
        )
        from spatialrisk.variables.gee_var import GEEVar
        from spatialrisk.variables.models import DataType, RasterType

        GEEVar.model_rebuild()
        aoi_ee = get_aoi_ee_feature(aoi_result.gdf)

        # Non-temporal variables
        for name in ["altitude", "slope", "protected_area", "roads", "rivers", "subj"]:
            cat = PREDEFINED_CATALOGUE[name]
            image = cat["get_image"](aoi_ee)
            p.raw_variables[name] = GEEVar(
                name=name,
                data_type=DataType.raster,
                raster_type=RasterType(cat["raster_type"]),
                gee_images=[image],
                aoi=aoi_ee,
                project=p,
            )

        # Temporal variables
        for name, years in [
            ("forest_gfc", [2015, 2020, 2024]),
            ("towns", [2015, 2020]),
        ]:
            cat = PREDEFINED_CATALOGUE[name]
            for year in years:
                key = f"{name}_{year}"
                image = cat["get_image"](aoi_ee, year)
                p.raw_variables[key] = GEEVar(
                    name=name,
                    data_type=DataType.raster,
                    raster_type=RasterType(cat["raster_type"]),
                    gee_images=[image],
                    aoi=aoi_ee,
                    project=p,
                    year=year,
                )

        p.base_raster = p.raw_variables["altitude"]

        # Seed dummy processed variables so the Dataset tab is usable
        from pathlib import Path

        from spatialrisk.variables.local_raster_var import LocalRasterVar

        LocalRasterVar.model_rebuild()

        for name in ["altitude", "slope", "protected_area", "roads", "rivers", "subj"]:
            cat = PREDEFINED_CATALOGUE[name]
            p.processed_variables[name] = LocalRasterVar(
                name=name,
                path=Path(f"/tmp/processed/{name}.tif"),
                raster_type=RasterType(cat["raster_type"]),
                data_type=DataType.raster,
                project=p,
            )
        for name, years in [
            ("forest_gfc", [2015, 2020, 2024]),
            ("towns", [2015, 2020]),
        ]:
            cat = PREDEFINED_CATALOGUE[name]
            for yr in years:
                key = f"{name}_{yr}"
                p.processed_variables[key] = LocalRasterVar(
                    name=name,
                    path=Path(f"/tmp/processed/{key}.tif"),
                    raster_type=RasterType(cat["raster_type"]),
                    data_type=DataType.raster,
                    year=yr,
                    project=p,
                )

        logger.debug(
            "SOLARA_TEST: seeded %d raw + %d processed variables",
            len(p.raw_variables),
            len(p.processed_variables),
        )
        app_state.project.set(p.model_copy())

    # Test variable seeding disabled for now — Step 2 starts with no variables.
    # (SOLARA_TEST stays on; re-enable by uncommenting the line below.)
    # solara.use_effect(_seed_test_variables, [app_state.project.value])

    # Test model/prediction seeding removed: it injected a fake GLM model and a
    # prediction pointing at a nonexistent /tmp path into whatever real project
    # was open, which the next save then persisted.

    # Handle to the MapApp widget, captured after mount (see _capture_map_app
    # below), so we can imperatively close its step-dialog from inside a tile.
    map_app_ref = solara.use_ref(None)

    def close_project_dialog():
        """Close the Project step-dialog (mirrors MapApp's deactivate path)."""
        widget = map_app_ref.current
        if widget is not None:
            widget.step_open = False
            widget.current_step = None

    # Left drawer: Project controls + read-only Project Summary (both open as dialogs)
    steps_data = [
        {
            "id": 1,
            "name": t("app.step_project"),
            "icon": "mdi-folder-outline",
            "display": "dialog",
            "content": ProjectPanel(on_close=close_project_dialog),
        },
        {
            "id": 2,
            "name": t("app.step_project_summary"),
            "icon": "mdi-clipboard-text-outline",
            "display": "dialog",
            "content": ProjectSummaryTile(
                project=app_state.project,
                project_dirty=app_state.project_dirty,
                last_saved=app_state.last_saved,
            ),
            "width": 760,
        },
        {
            "id": 3,
            "name": t("app.step_tools"),
            "icon": "mdi-toolbox-outline",
            "display": "dialog",
            "content": ToolboxTile(
                project=app_state.project,
                map_=sepal_map,
                sepal_client=sepal_client,
                legend_port=app_state.legend_port,
            ),
            "width": 780,
        },
    ]

    # Right panel: workflow tabs
    right_panel_config = {
        "title": t("app.panel_workflow_title"),
        "icon": "mdi-tune",
        "width": 480,
        "toggle_icon": "mdi-chevron-right",
    }

    right_panel_content = [
        {
            "content": [
                WorkflowTabs(
                    map_=sepal_map,
                    gee_interface=gee_interface,
                    sepal_client=sepal_client,
                )
            ],
        },
    ]

    solara.Title(t("app.title"))

    # pysepal's step-dialog content area sets `overflow-y: auto` with no
    # `overflow-x`, which CSS resolves to `overflow-x: auto` — producing a
    # spurious horizontal scrollbar even when the panel fits. The dialog card
    # already clips with `overflow: hidden`, so pin the content's x-axis hidden.
    solara.Style(".dialog-content { overflow-x: hidden !important; }")

    app_title = compute_app_title(
        app_state.project.value, app_state.project_dirty.value
    )

    # jupyter-loopback comm bridge, mounted in the page from the first render
    # so tile-URL interception is live before any local tile layer is added
    # (see _loopback_bridge_widget for why lazy display() is not enough).
    loopback_bridge = solara.use_memo(_loopback_bridge_widget, [])
    if loopback_bridge is not None:
        solara.display(loopback_bridge)

    # Kernel-scoped notification bus + UI (toasts top-right, task pill
    # bottom-right). Mounted BEFORE the MapApp element so the bus exists by the
    # time the workflow tiles first render — their use_notifications() then
    # resolves a real Notifier instead of a first-render NoopNotifier.
    NotificationProvider()

    # Floating layer legend (bottom-center over the map). Mounted before MapApp,
    # like NotificationProvider, so it is present from the first render.
    MapLegend(app_state)

    map_app_el = MapApp.element(
        app_title=app_title,
        app_icon="mdi-tree",
        main_map=[sepal_map],
        steps_data=steps_data,
        initial_step=1,  # auto-open the Project dialog (step id 1) at startup
        theme_toggle=[theme_toggle],
        language_selector=[locale_select],
        right_panel_config=right_panel_config,
        right_panel_content=right_panel_content,
        right_panel_open=True,
        is_pinned=False,
        # roomier Project dialog (scroll fix is the .dialog-content style above)
        dialog_width=560,
        repo_url="https://github.com/sepal-contrib/spatial-risk-module",
        # no hosted docs yet — GitHub's rendered Sphinx index stands in (the
        # MapApp fallback of <repo>/blob/main/doc/en.rst does not exist here)
        docs_url="https://github.com/sepal-contrib/spatial-risk-module/blob/main/docs/source/index.rst",
    )

    # Grab the realized MapApp widget so close_project_dialog() can drive its
    # step-dialog traits. Recaptured each render (cheap, element is stable).
    def _capture_map_app():
        try:
            map_app_ref.current = solara.get_widget(map_app_el)
        except Exception:  # pragma: no cover - defensive (pre-mount)
            pass

    solara.use_effect(_capture_map_app, [map_app_el])


routes = [
    solara.Route(path="/", component=Page, label="Spatial Risk"),
]
