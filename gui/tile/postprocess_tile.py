"""Step 4 — Derived layers tile (list-first; form lives in DerivedLayerDialog)."""

import logging

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.inflight import InflightKeys
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT, tracked_job
from gui.scripts.solara_threads import publish_if_current, spawn_in_context
from gui.store.project_writers import writing
from gui.tile.derived_map import derived_on_map, use_derived_map_toggle
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.derived_layer_dialog import CHANGE_OPS, DerivedLayerDialog
from gui.widget.help import InfoButton
from gui.widget.variable_list import DerivedVariableList

logger = logging.getLogger("spatial_risk")

# Output names whose derived-layer job is currently running. Keyed by the
# registry name the job will write, so the same layer cannot be generated
# twice at once (two GDAL runs on one .tif) while different layers run in
# parallel.
derived_inflight = InflightKeys(key="derived_inflight")


def _run_derived_job(entry, output_name, p, project_reactive, notifier):
    """Background worker: one derived layer, from the operation to the republish.

    Everything that must happen *after* the work — the project republish that
    makes the new layer appear in the list — runs on this thread. A shared
    ``solara.lab.use_task`` used to own this flow, and submitting a second
    layer re-invoked it, which cancelled the first coroutine at its ``await``:
    the worker finished (raster written, registered, saved) but the republish
    after the ``await`` never ran, so the layer never appeared.
    """
    is_change = entry["op"] in CHANGE_OPS
    if is_change:
        title = t("notifications.task_change", op=entry["op"])
        error_key = "tiles.postprocess.error_change"
    else:
        title = t(
            "notifications.task_postprocess", step=entry["op"], name=entry["pp_key"]
        )
        error_key = "tiles.postprocess.error_post_processing"
    try:
        with writing(p.project_name):
            with tracked_job(
                notifier, title, error_format=lambda exc: t(error_key, exc=exc)
            ):
                if is_change:
                    process_actions.generate_change_var(
                        p, entry["op"], entry["start_key"], entry["end_key"]
                    )
                else:
                    process_actions.apply_post_processing(
                        p, entry["pp_key"], entry["op"]
                    )
            publish_if_current(project_reactive, p)
    except Exception:
        logger.exception("derived layer job failed")  # toast from tracked_job
    finally:
        derived_inflight.release(output_name)


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
    notifications = use_notifications()
    on_toggle_map = use_derived_map_toggle(
        project, map_, notifications, legend_port=legend_port
    )
    pending_remove, set_pending_remove = solara.use_state(None)

    p = project.value
    running = derived_inflight.value  # subscribes: progress bar follows the jobs

    def _do_remove(key: str):
        """Unregister a derived layer (the raster stays on disk)."""
        if process_actions.remove_processed_variable(p, key, map_, legend_port):
            project.set(p.model_copy())

    def on_submit(entry):
        """Dialog-validated entry -> one background worker per derived layer.

        Never inline: solara runs widget callbacks inside the session's
        websocket message loop, so a GDAL proximity pass over a large AOI
        would freeze the whole UI. One worker per submission (not a shared
        task) so a second layer never cancels the first one's continuation.
        """
        cur = project.value
        if cur is None:
            return
        if entry["op"] in CHANGE_OPS:
            name = process_actions.change_output_name(
                cur, entry["op"], entry["start_key"], entry["end_key"]
            )
        else:
            name = process_actions.postprocess_output_name(
                cur, entry["pp_key"], entry["op"]
            )
        if name is None:  # the dialog validated; defensive
            return
        if not derived_inflight.claim(name):
            notifications.error(
                t("tiles.postprocess.already_running", name=name),
                timeout=ERROR_TOAST_TIMEOUT,
            )
            return
        spawn_in_context(_run_derived_job, (entry, name, cur, project, notifications))

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
        if running:
            solara.ProgressLinear(True)

        DerivedVariableList(
            project=project,
            keys=process_actions.postprocess_output_keys(p),
            on_toggle_map=on_toggle_map,
            derived_on_map=derived_on_map,
            on_remove=set_pending_remove,
        )

    DerivedLayerDialog(project=project, open_=dialog_open, on_submit=on_submit)
    ConfirmDialog(
        open=pending_remove is not None,
        on_cancel=lambda: set_pending_remove(None),
        on_confirm=lambda: (_do_remove(pending_remove), set_pending_remove(None)),
        title=t("tiles.postprocess.confirm_remove_title"),
        message=t(
            "tiles.postprocess.confirm_remove_message", name=pending_remove or ""
        ),
        confirm_label=t("common.remove"),
    )
