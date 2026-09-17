"""Step 4 — Derived layers tile (list-first; form lives in DerivedLayerDialog)."""

import logging

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.file_prompts import DeletePrompt, delete_prompt
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT, tracked_job
from gui.scripts.solara_threads import publish_if_current, to_thread_in_context
from gui.store.project_writers import writing
from gui.tile.derived_map import derived_on_map, use_derived_map_toggle
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.derived_layer_dialog import CHANGE_OPS, DerivedLayerDialog
from gui.widget.help import InfoButton
from gui.widget.variable_list import DerivedVariableList
from spatialrisk.variables.file_cleanup import FilePlan

logger = logging.getLogger("spatial_risk")


@solara.component
def PostProcessTile(project, map_=None, legend_port=None):
    """Derived layers: change detection (loss/gain) + edge/dist on harmonized vars.

    Args:
        project: Reactive holding the current Project (or None).
        map_: SepalMap instance used by the "show on map" toggle.
        legend_port: LegendPort for publishing/withdrawing derived-layer
            legends; None disables legend publication (e.g. in tests without
            one).
    """
    dialog_open = solara.use_reactive(False)
    pending_change = solara.use_reactive(None)
    pending_post = solara.use_reactive(None)
    notifications = use_notifications()
    on_toggle_map = use_derived_map_toggle(
        project, map_, notifications, legend_port=legend_port
    )
    pending_remove, set_pending_remove = solara.use_state(None)
    # Reset for every removal — see the same state in Step 3.
    delete_files, set_delete_files = solara.use_state(False)

    p = project.value

    def _ask_remove(key: str):
        set_delete_files(False)
        set_pending_remove(key)

    def _do_remove(key: str, also_delete: bool = False):
        """Unregister a derived layer; optionally delete its raster too."""
        try:
            removed = process_actions.remove_processed_variable(
                p, key, map_, legend_port, delete_file=also_delete
            )
        except Exception as exc:
            # The entry is gone either way (unlinked under a finally there).
            logger.exception("deleting the files of %s failed", key)
            notifications.error(
                t("tiles.variables.error_delete_files", exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )
            removed = True
        if removed:
            project.set(p.model_copy())

    @solara.lab.use_task(dependencies=None, raise_error=False, prefer_threaded=True)
    async def change_task():
        entry = pending_change.value
        if p is None or entry is None:
            return
        title = t("notifications.task_change", op=entry["op"])

        def _tracked_change():
            with tracked_job(
                notifications,
                title,
                error_format=lambda exc: t("tiles.postprocess.error_change", exc=exc),
            ):
                process_actions.generate_change_var(
                    p,
                    entry["op"],
                    entry["start_key"],
                    entry["end_key"],
                )

        with writing(p.project_name):
            try:
                await to_thread_in_context(_tracked_change)
            except Exception:
                logger.exception("change detection failed")  # toast from tracked_job
                return
            publish_if_current(project, p)

    @solara.lab.use_task(dependencies=None, raise_error=False, prefer_threaded=True)
    async def post_task():
        entry = pending_post.value
        if p is None or entry is None:
            return
        title = t(
            "notifications.task_postprocess",
            step=entry["op"],
            name=entry["pp_key"],
        )

        def _tracked_post():
            with tracked_job(
                notifications,
                title,
                error_format=lambda exc: t(
                    "tiles.postprocess.error_post_processing", exc=exc
                ),
            ):
                process_actions.apply_post_processing(p, entry["pp_key"], entry["op"])

        with writing(p.project_name):
            try:
                await to_thread_in_context(_tracked_post)
            except Exception:
                logger.exception("post-processing failed")  # toast from tracked_job
                return
            publish_if_current(project, p)

    def on_submit(entry):
        """Dialog-validated entry -> background change or edge/dist task.

        Neither op may run inline: solara executes widget callbacks inside the
        session's websocket message loop, so a GDAL proximity pass over a large
        AOI would freeze the whole UI until it returned.
        """
        if entry["op"] in CHANGE_OPS:
            pending_change.set(entry)
            change_task()
            return
        pending_post.set(entry)
        post_task()

    with solara.Column(style="gap:16px;"):
        with solara.Row(style="gap:4px;align-items:center;"):
            solara.Text(t("tiles.postprocess.description"))
            InfoButton(
                t("tiles.postprocess.info_header"), t("tiles.postprocess.info_md")
            )
        if p is None or not p.processed_variables:
            solara.Info(t("tiles.postprocess.error_no_processed"))
            return

        solara.Button(
            t("tiles.postprocess.new_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=lambda: dialog_open.set(True),
        )
        if change_task.pending or post_task.pending:
            solara.ProgressLinear(True)

        DerivedVariableList(
            project=project,
            keys=process_actions.postprocess_output_keys(p),
            on_toggle_map=on_toggle_map,
            derived_on_map=derived_on_map,
            on_remove=_ask_remove,  # opens the dialog; the tick decides the raster
        )

    DerivedLayerDialog(project=project, open_=dialog_open, on_submit=on_submit)
    _files = (
        delete_prompt(p, pending_remove)
        if (p is not None and pending_remove)
        else DeletePrompt(plan=FilePlan())
    )
    ConfirmDialog(
        open=pending_remove is not None,
        on_cancel=lambda: set_pending_remove(None),
        on_confirm=lambda: (
            _do_remove(pending_remove, delete_files),
            set_pending_remove(None),
        ),
        title=t("tiles.postprocess.confirm_remove_title"),
        message=t(
            "tiles.postprocess.confirm_remove_message", name=pending_remove or ""
        ),
        confirm_label=t("common.remove"),
        checkbox_label=_files.checkbox_label,
        checkbox_value=delete_files,
        on_checkbox=set_delete_files,
        details=_files.details,
        note=_files.note,
    )
