# gui/tile/evaluation_tile.py
"""Step 8 — Evaluation tile.

Score user-selected maps against ONE explicitly-chosen truth (deforestation
raster + forest-at-start variable + interval), so maps from different datasets
are directly comparable. List-first: the form lives in EvaluationFormDialog.
Each run is saved to project.evaluations and shown in a list; click a row to
view its table. See evaluate_against_truth in spatialrisk/evaluation.py.
"""

import logging
import uuid
from datetime import datetime

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts.notify_bridge import tracked_job
from gui.scripts.solara_threads import publish_if_current, spawn_in_context, update_job
from gui.store.project_writers import writing
from gui.tile.evaluation_helpers import (
    build_evaluation_record,
    delete_evaluation_run,
    map_items,
    variable_items,
)
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.evaluation_form_dialog import EvaluationFormDialog
from gui.widget.evaluation_results import EvaluationResults, EvaluationTableDialog
from gui.widget.help import InfoButton

logger = logging.getLogger("spatial_risk")

# Module-level reactive shared across re-renders (transient per-run job status).
eval_jobs = solara.reactive([])


def _run_evaluation(
    job_id,
    project,
    prediction_keys,
    spec,
    recompute,
    created_at,
    csizes,
    metrics,
    notifier=None,
    task_title=None,
):
    """Background job: evaluate, build + register a record, re-render the list."""
    try:
        from spatialrisk.evaluation import evaluate_against_truth

        p = project.value
        if p is None:
            return  # project was closed/deleted while the job was queued
        with tracked_job(notifier, task_title or "Evaluating predictions"), writing(
            p.project_name
        ):
            df = evaluate_against_truth(
                p,
                prediction_keys=prediction_keys or None,
                defor_file=spec["defor_file"],
                forest_file=spec["forest_file"],
                time_interval=spec["time_interval"],
                truth_tag=spec["truth_tag"],
                csizes=tuple(csizes),
                recompute_defrate=recompute,
                auto_save=False,
                # The job id IS this run's id: artifacts land in their own
                # evaluation/<truth_tag>/<run_id>/ folder, so a later run
                # against the same truth cannot overwrite this record's data.
                run_id=job_id,
            )
            resolved = list(prediction_keys) or list(p.predictions.keys())
            record = build_evaluation_record(
                p,
                df,
                spec,
                resolved_keys=resolved,
                run_id=job_id,
                created_at=created_at,
                csizes=tuple(csizes),
                metrics=metrics,
            )
            p.add_evaluation(record, auto_save=False)
            p.save()
            publish_if_current(project, p)
            update_job(eval_jobs, job_id, status="completed")
            logger.info(
                "Evaluation saved as project.evaluations['%s'] (%d rows)",
                record.storage_key(),
                len(df),
            )
    except Exception as exc:
        logger.exception("Evaluation failed")
        update_job(eval_jobs, job_id, status="failed", error=str(exc))


@solara.component
def EvaluationTile(project):
    """Evaluation tab: list saved runs; the New evaluation dialog handles setup."""
    p = project.value

    n_predictions = len(map_items(p))

    dialog_open = solara.use_reactive(False)
    selected_eval, set_selected_eval = solara.use_state(None)
    pending_delete, set_pending_delete = solara.use_state(None)
    notifications = use_notifications()

    def on_submit(entry):
        """Create the job row and spawn the worker (dialog pre-validated)."""
        spec = entry["spec"]
        job_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat(timespec="seconds")
        eval_jobs.set(
            list(eval_jobs.value)
            + [
                {
                    "id": job_id,
                    "status": "running",
                    "error": None,
                    "truth_tag": spec["truth_tag"],
                    "n_maps": len(entry["prediction_keys"]) or n_predictions,
                    "created_at": created_at,
                }
            ]
        )
        spawn_in_context(
            _run_evaluation,
            (
                job_id,
                project,
                entry["prediction_keys"],
                spec,
                entry["recompute"],
                created_at,
                entry["csizes"],
                entry["metrics"],
                notifications,
                t("notifications.task_evaluation", tag=spec["truth_tag"]),
            ),
        )

    def on_delete(key):
        cur = project.value
        if cur is None:
            return
        # Commits the manifest FIRST and only then removes this run's artifact
        # folder, so a failed save can never leave a record pointing at files
        # that were already deleted.
        deleted, error = delete_evaluation_run(cur, key)
        if error:
            logger.error(
                "Evaluation '%s' could not be deleted: saving the "
                "project failed (%s). The run and its artifacts "
                "were kept.",
                key,
                error,
            )
        if not deleted:
            return
        project.set(cur.model_copy())
        if selected_eval == key:
            set_selected_eval(None)

    def on_dismiss(job_id):
        eval_jobs.set([j for j in eval_jobs.value if j["id"] != job_id])

    with solara.Column(style="gap: 16px;"):
        with solara.Row(style="gap:4px;align-items:center;"):
            solara.Text(t("tiles.evaluation.description"))
            InfoButton(t("tiles.evaluation.info_header"), t("tiles.evaluation.info_md"))

        if p is None or not variable_items(p):
            solara.Info(t("tiles.evaluation.error_no_variables"))
            return

        solara.Button(
            t("tiles.evaluation.new_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=lambda: dialog_open.set(True),
            disabled=n_predictions == 0,
        )
        if n_predictions == 0:
            solara.Info(t("tiles.evaluation.error_no_predictions"))

        EvaluationResults(
            eval_jobs=eval_jobs,
            project=project,
            on_open=set_selected_eval,
            on_delete=set_pending_delete,
            on_dismiss=on_dismiss,
        )
        ConfirmDialog(
            open=pending_delete is not None,
            on_cancel=lambda: set_pending_delete(None),
            on_confirm=lambda: (on_delete(pending_delete), set_pending_delete(None)),
            title=t("tiles.evaluation.confirm_delete_title"),
            message=t(
                "tiles.evaluation.confirm_delete_message", key=pending_delete or ""
            ),
            confirm_label=t("common.delete"),
        )
        EvaluationTableDialog(
            project=project,
            eval_key=selected_eval,
            on_close=lambda: set_selected_eval(None),
        )

    EvaluationFormDialog(project=project, open_=dialog_open, on_submit=on_submit)
