"""Step 4 — Derived layers tile (list-first; form lives in DerivedLayerDialog)."""

import logging
import uuid

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.file_prompts import DeletePrompt, delete_prompt
from gui.scripts.inflight import InflightKeys
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT, tracked_job
from gui.scripts.solara_threads import publish_if_current, spawn_in_context, update_job
from gui.store.project_writers import writing
from gui.tile.derived_map import derived_on_map, use_derived_map_toggle
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.derived_layer_dialog import CHANGE_OPS, DerivedLayerDialog
from gui.widget.help import InfoButton
from gui.widget.variable_list import DerivedVariableList
from spatialrisk.variables.file_cleanup import FilePlan

logger = logging.getLogger("spatial_risk")

# Session job rows for layers still being generated (module-level, so they
# survive re-renders). A row lands here the moment the dialog validates and is
# superseded by the registry entry once the raster does — see
# gui/scripts/product_rows.derived_rows for the suppression contract.
derived_jobs = solara.reactive([])

# Registry keys a derived-layer worker currently owns, so the same layer cannot
# be generated twice at once (two GDAL runs writing one .tif) while different
# layers still run in parallel.
derived_inflight = InflightKeys(key="derived_inflight")


def dismiss_derived_job(job_id: str) -> None:
    """Drop one finished job row (failed runs only — a run cannot be cancelled)."""
    derived_jobs.set([j for j in derived_jobs.value if j["id"] != job_id])


def forget_derived_jobs_for(output_key: str) -> None:
    """Drop every job row that produced ``output_key``.

    Called when the layer is deleted, so a stale "completed" row cannot
    resurface once the registry entry that superseded it is gone — the same
    guard the Train tab applies when a model is deleted.
    """
    derived_jobs.set(
        [j for j in derived_jobs.value if j.get("output_key") != output_key]
    )


def _run_derived_job(entry, job_id, output_key, p, project_reactive, notifier):
    """Background worker: one derived layer, from the operation to the republish.

    Everything that must happen *after* the work — marking the row done and the
    project republish that turns it into a product row — runs on this thread. A
    shared ``solara.lab.use_task`` used to own this flow, and submitting a
    second layer re-invoked it, which cancelled the first coroutine at its
    ``await``: the worker finished (raster written, registered, saved) but the
    continuation never ran. That used to lose the layer silently; now it would
    strand its row on "running" forever.

    The row is marked completed *before* the republish so the two never overlap:
    publishing first would render the job row and its fresh product row side by
    side until the status caught up.
    """
    try:
        # Inside the try: anything that raises before the job opens — a bad
        # entry, a title lookup — must still reach the release below, or the
        # key stays claimed for the session (a row stuck on "running" that can
        # never be submitted again).
        is_change = entry["op"] in CHANGE_OPS
        if is_change:
            title = t("notifications.task_change", op=entry["op"])
            error_key = "tiles.postprocess.error_change"
        else:
            title = t(
                "notifications.task_postprocess", step=entry["op"], name=entry["pp_key"]
            )
            error_key = "tiles.postprocess.error_post_processing"
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
            update_job(derived_jobs, job_id, status="completed")
            publish_if_current(project_reactive, p)
    except Exception as exc:
        # tracked_job already toasted anything raised inside it; the row keeps
        # the message so the failure is still readable after the toast is gone.
        logger.exception("derived layer job failed")
        update_job(derived_jobs, job_id, status="failed", error=str(exc))
    finally:
        derived_inflight.release(output_key)


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
            forget_derived_jobs_for(key)
            project.set(p.model_copy())

    def on_submit(entry):
        """Dialog-validated entry -> a job row plus the worker that fills it.

        The row is published here, on the event-handler thread, so the layer is
        listed the instant the form is submitted instead of appearing minutes
        later when the raster lands. The work itself never runs inline: solara
        executes widget callbacks inside the session's websocket message loop,
        so a GDAL proximity pass over a large AOI would freeze the whole UI.
        """
        cur = project.value
        if cur is None:
            return
        output = process_actions.derived_output(cur, entry)
        if output is None:  # the dialog validated; defensive
            return
        if not derived_inflight.claim(output.key):
            notifications.error(
                t("tiles.postprocess.already_running", name=output.name),
                timeout=ERROR_TOAST_TIMEOUT,
            )
            return

        job_id = str(uuid.uuid4())[:8]
        derived_jobs.set(
            list(derived_jobs.value)
            + [
                {
                    "id": job_id,
                    "name": output.name,
                    "output_key": output.key,
                    "status": "running",
                    "error": None,
                }
            ]
        )
        try:
            spawn_in_context(
                _run_derived_job,
                (entry, job_id, output.key, cur, project, notifications),
            )
        except Exception as exc:
            # The worker's finally is what releases the claim, so a thread that
            # never starts would hold the key for the rest of the session.
            derived_inflight.release(output.key)
            update_job(derived_jobs, job_id, status="failed", error=str(exc))
            logger.exception("could not start the derived layer worker")
            # The row carries the reason, but nothing ran to toast it: a worker
            # that never started raises outside tracked_job's scope.
            failed_key = (
                "tiles.postprocess.error_change"
                if entry["op"] in CHANGE_OPS
                else "tiles.postprocess.error_post_processing"
            )
            notifications.error(t(failed_key, exc=exc), timeout=ERROR_TOAST_TIMEOUT)
            return
        logger.info("Derived layer '%s' started (job=%s)", output.name, job_id)

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

        # No tile-wide progress bar: each run carries its own status in the
        # list, the same way the Train, Sampling and Inference tabs report.
        DerivedVariableList(
            project=project,
            keys=process_actions.postprocess_output_keys(p),
            on_toggle_map=on_toggle_map,
            derived_on_map=derived_on_map,
            on_remove=_ask_remove,  # opens the dialog; the tick decides the raster
            jobs=derived_jobs,
            on_dismiss=dismiss_derived_job,
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
