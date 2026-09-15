"""Step 7 — Inference tile."""

import asyncio
import logging
import uuid

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import plural, t
from gui.scripts import artifact_names as _artifact_names
from gui.scripts.notify_bridge import tracked_job
from gui.scripts.product_rows import job_row_key
from gui.scripts.solara_threads import publish_if_current, spawn_in_context, update_job
from gui.store.project_writers import writing
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.inference_output_list import InferenceOutputList
from gui.widget.prediction_form_dialog import (
    PredictionDetailsDialog,
    PredictionFormDialog,
)

# The naming helpers live in gui/scripts/artifact_names.py; they are re-exported
# under this tile's historical private names because other modules (and
# tests/test_inference_tile_wiring) still reach them through here. Bound as
# assignments rather than aliased imports so the linter can see they are used.
_default_pred_name = _artifact_names.default_pred_name
_prediction_name_exists = _artifact_names.prediction_name_exists
_sanitize_pred_name = _artifact_names.sanitize_key

logger = logging.getLogger("spatial_risk")


# Module-level reactives shared across re-renders
inference_jobs = solara.reactive([])
# Row keys (prediction name, else "{model_key}__{dataset_name}") of prediction
# groups currently shown on the map — works for predictions loaded from disk,
# not just same-session runs.
preds_on_map = solara.reactive(set())


def _pred_layer_key(storage_key: str) -> str:
    """Unique map-layer key for a registered prediction."""
    return f"pred_{storage_key}"


def _pred_legend(
    storage_key: str, model_key: str, display_palette, pred_name: str, multi: bool
):
    """The legend one prediction raster publishes while it is on the map.

    ``pred_name`` is the row's human display name; ``multi`` says whether the
    row carries more than one storage key. A single-raster row shows
    ``pred_name`` alone, a multi-raster row (several storage keys under one
    prediction group) appends the storage key so the dropdown can still tell
    them apart.
    """
    from gui.scripts.legend_data import Label, prediction_spec
    from gui.scripts.legend_registry import LayerLegend

    label_text = f"{pred_name} — {storage_key}" if multi else pred_name

    return LayerLegend(
        layer_id=_pred_layer_key(storage_key),
        label=Label(literal=label_text),
        spec=prediction_spec(model_key, display_palette),
    )


def _drop_pred_layers(row, map_, legend_port) -> None:
    """Remove every layer a prediction row put on the map, and its legends.

    The single removal chokepoint: the toggle's off-branch and the delete path
    both go through it, so a legend can never outlive its layer. ``legend_port``
    may be None (e.g. in tests without one) — that is a no-op, not a crash.
    """
    storage_keys = list(row.get("storage_keys", []))
    if map_ is not None:
        for storage_key in storage_keys:
            map_.remove_layer(_pred_layer_key(storage_key), none_ok=True)
    if legend_port is not None:
        legend_port.unregister(*[_pred_layer_key(k) for k in storage_keys])


def _run_inference(
    job_id,
    model_key,
    dataset_key,
    project,
    name=None,
    project_reactive=None,
    notifier=None,
    task_title=None,
    mask_layer=None,
    windows=None,
):
    """Run model inference in a background thread.

    ``mask_layer`` is the project raster an ML run masks with, as assigned
    in the Predict dialog; None or blank means no mask. ``windows`` is the
    MW family's subset of trained window sizes; None means every window.
    """
    try:
        with tracked_job(notifier, task_title or f"Predicting: {model_key}"), writing(
            project.project_name
        ):
            from gui.scripts.inference_runner import run_inference

            run_inference(
                project,
                model_key,
                dataset_key,
                name=name,
                mask_layer=mask_layer,
                windows=windows,
            )

            # Model.apply() registers and saves the prediction on ``project``,
            # but that mutation alone does not notify Solara subscribers. Publish
            # a fresh project reference so the Project Summary, inference output
            # list, and Evaluation tile immediately see the new prediction. The
            # guard avoids restoring a project that was closed/switched mid-run.
            publish_if_current(project_reactive, project)

            update_job(
                inference_jobs,
                job_id,
                status="completed",
                output_path="see project predictions",
            )
            logger.info(
                "Inference completed: %s on %s (name=%s)", model_key, dataset_key, name
            )

    except Exception as exc:
        logger.exception("Inference failed for %s on %s", model_key, dataset_key)
        update_job(inference_jobs, job_id, status="failed", error=str(exc))


def _run_import(
    job_id,
    src_path,
    name,
    palette,
    project,
    project_reactive,
    notifier=None,
    task_title=None,
):
    """Copy a local raster into the project as a Prediction (background thread).

    The copy can be large, so it runs off the render thread like inference does.
    On success the placeholder job is updated to the real (model_key, dataset_name)
    so the per-job map toggle resolves the registered raster, and the project is
    republished so the outputs list and Step 8 — Evaluation pick it up.
    """
    try:
        with tracked_job(notifier, task_title or f"Importing '{name}'"), writing(
            project.project_name
        ):
            from gui.scripts.prediction_import import import_prediction

            pred = import_prediction(
                project, src_path, name, palette=palette, auto_save=True
            )

            update_job(
                inference_jobs,
                job_id,
                status="completed",
                pred_name=name,
                model_key=pred.model_key,
                dataset_name=pred.dataset_name,
                output_path=str(pred.path),
            )
            publish_if_current(project_reactive, project)
            logger.info(
                "Imported prediction '%s' registered as %s", name, pred.model_key
            )

    except Exception as exc:
        logger.exception("Prediction import failed for %s", src_path)
        update_job(inference_jobs, job_id, status="failed", error=str(exc))


@solara.component
def InferenceTile(project, map_=None, sepal_client=None, legend_port=None):
    """Inference tab: select trained model and dataset, run prediction.

    Args:
        project: Reactive holding the current Project (or None).
        map_: SepalMap instance used by the per-prediction "add to map" toggle.
        sepal_client: SEPAL client backing the local-raster import file picker.
        legend_port: LegendPort for publishing/withdrawing prediction legends;
            None disables legend publication (e.g. in tests without one).
    """
    p = project.value

    dialog_open = solara.use_reactive(False)
    notifications = use_notifications()

    # Form messages
    form_error, set_form_error = solara.use_state(None)

    def _launch_import(name, path, palette, entry=None):
        """Spawn a background copy for a raster the dialog validated.

        The dialog enforced the required fields and the no-project guard; the
        guard is kept here as a race safety net (surfaced via the tile's form
        error, as the dialog is closed by the time this runs).
        """
        if p is None:
            set_form_error(t("tiles.inference.error_no_project"))
            return
        job_id = str(uuid.uuid4())[:8]
        # Placeholder job; _run_import fills in the real model_key on completion.
        inference_jobs.set(
            list(inference_jobs.value)
            + [
                {
                    "id": job_id,
                    "model_key": name,
                    "dataset_name": "imported",
                    "status": "running",
                    "error": None,
                    "output_path": None,
                    # Submission entry, kept so a failed row can be re-edited.
                    "entry": entry,
                }
            ]
        )
        spawn_in_context(
            _run_import,
            (
                job_id,
                path,
                name,
                palette,
                p,
                project,
                notifications,
                t("notifications.task_import", name=name),
            ),
        )
        logger.info("Import started: '%s' (job=%s)", name, job_id)

    def _launch_inference(
        model_key, dataset_key, name, mask_layer=None, windows=None, entry=None
    ):
        """Create the output job row and spawn the worker. Inputs pre-validated."""
        job_id = str(uuid.uuid4())[:8]
        job = {
            "id": job_id,
            "model_key": model_key,
            "dataset_name": dataset_key,
            "pred_name": name,
            "status": "running",
            "error": None,
            "output_path": None,
            # Submission entry, kept so a failed row can be re-edited.
            "entry": entry,
        }
        inference_jobs.set(list(inference_jobs.value) + [job])
        spawn_in_context(
            _run_inference,
            (
                job_id,
                model_key,
                dataset_key,
                p,
                name,
                project,
                notifications,
                t("notifications.task_inference", model=model_key, dataset=dataset_key),
                mask_layer,
                windows,
            ),
        )
        logger.info(
            "Inference started: %s on %s as '%s' (job=%s)",
            model_key,
            dataset_key,
            name,
            job_id,
        )

    # Failed-job editing: the pencil action reopens the dialog seeded with the
    # job's submission entry; submitting launches a fresh run and drops the old
    # failed row so the rerun replaces it instead of piling up next to it.
    prefill = solara.use_reactive(None)
    editing_job_id = solara.use_ref(None)

    def on_edit(row):
        editing_job_id.current = row["job_id"]
        prefill.set(row["entry"])
        dialog_open.set(True)

    def open_new_dialog():
        editing_job_id.current = None
        prefill.set(None)
        dialog_open.set(True)

    def on_submit(entry):
        if entry["kind"] == "import":
            _launch_import(entry["name"], entry["path"], entry["palette"], entry=entry)
        else:
            # mask_layer is absent for the JNR/MW families, which resolve
            # their own layers rather than masking with a project raster;
            # windows is MW-only and absent everywhere else.
            _launch_inference(
                entry["model_key"],
                entry["dataset_key"],
                entry["name"],
                entry.get("mask_layer"),
                entry.get("windows"),
                entry=entry,
            )
        if editing_job_id.current is not None:
            on_dismiss(editing_job_id.current)
            editing_job_id.current = None
            prefill.set(None)

    def _forget_on_map(row_key):
        remaining = set(preds_on_map.value)
        remaining.discard(row_key)
        preds_on_map.set(remaining)

    pending_toggle = solara.use_reactive(None)

    @solara.lab.use_task(dependencies=None, raise_error=False)
    async def _apply_pred_toggle():
        """Add/remove a prediction row's raster(s) on the map.

        The layer-add is offloaded to a worker thread (it builds overviews and a
        localtileserver tile client, both blocking) so Solara's event loop stays
        responsive. Removal is cheap and stays inline.
        """
        row = pending_toggle.value
        if row is None or map_ is None or p is None:
            return
        storage_keys = [k for k in row.get("storage_keys", []) if k in p.predictions]
        if not storage_keys:
            return
        row_key = row["key"]
        try:
            if row_key in preds_on_map.value:
                _drop_pred_layers(row, map_, legend_port)
                _forget_on_map(row_key)
            else:
                from gui.scripts.prediction_map import add_prediction_on_map

                generation = (
                    legend_port.generation() if legend_port is not None else None
                )
                added_any = False
                landed = []
                try:
                    for sk in storage_keys:
                        pred = p.predictions[sk]
                        await asyncio.to_thread(
                            add_prediction_on_map,
                            map_,
                            str(pred.path),
                            model_key=row["model_key"],
                            layer_name=sk,
                            key=_pred_layer_key(sk),
                            fit_bounds=False,
                            display_palette=getattr(pred, "display_palette", None),
                        )
                        added_any = True
                        landed.append((sk, getattr(pred, "display_palette", None)))
                finally:
                    # Mark the row on-map if ANY layer landed (even on partial
                    # failure) so toggle-off can remove all its keys; fire the
                    # reactive once, not per-iteration. This bookkeeping must
                    # run even when add_prediction_on_map raises, so it stays
                    # in `finally` — but the staleness check below must NOT
                    # live here: a `return` inside `finally` discards any
                    # exception propagating from the `try` body, which would
                    # silently swallow a real failure instead of letting it
                    # reach the outer `except Exception` handler.
                    if added_any:
                        preds_on_map.set(set(preds_on_map.value) | {row_key})

                # Reached only when the add loop above completed without
                # raising — an in-flight exception skips straight to the
                # outer `except Exception` below instead. No port to publish
                # through (e.g. a test render) means there is nothing left to
                # guard or publish here.
                if legend_port is not None and legend_port.generation() != generation:
                    # A project switch during the await clears the map;
                    # anything that landed afterwards is stale, so take it
                    # back off instead of publishing a legend for a layer
                    # nobody wants.
                    for sk, _palette in landed:
                        map_.remove_layer(_pred_layer_key(sk), none_ok=True)
                    _forget_on_map(row_key)
                elif legend_port is not None and added_any:
                    multi = len(storage_keys) > 1
                    legend_port.register(
                        *[
                            _pred_legend(
                                sk, row["model_key"], palette, row["name"], multi
                            )
                            for sk, palette in landed
                        ]
                    )
        except Exception as exc:
            logger.exception("prediction map toggle failed for row %s", row.get("key"))
            set_form_error(t("tiles.inference.error_map_toggle", exc=exc))

    def on_toggle_map(row):
        """Trigger the threaded add/remove task for a prediction row."""
        if map_ is None:
            return
        pending_toggle.set(row)
        _apply_pred_toggle()

    pending_delete, set_pending_delete = solara.use_state(None)  # row dict or None
    # Row key whose provenance dialog is open, or None. The tile owns it
    # so the list stays a pure renderer (same shape as Train/Sampling).
    details_key, set_details_key = solara.use_state(None)

    def on_dismiss(job_id):
        # Failed job rows only — never touches the prediction registry.
        inference_jobs.set([j for j in inference_jobs.value if j["id"] != job_id])

    def _delete_row(row):
        cur = project.value
        if cur is None:
            return
        row_key = row["key"]
        if map_ is not None and row_key in preds_on_map.value:
            _drop_pred_layers(row, map_, legend_port)
            _forget_on_map(row_key)
        deleted = False
        for sk in row.get("storage_keys", []):
            deleted = cur.delete_prediction(sk, auto_save=False) or deleted
        if deleted:
            cur.save()
            project.set(cur.model_copy())
        # Purge completed session jobs that produced this row, so a stale
        # "completed" job doesn't resurface once its registry group is gone.
        # A running/failed re-run of the same name is left alone.
        inference_jobs.set(
            [
                j
                for j in inference_jobs.value
                if job_row_key(j) != row_key or j.get("status") != "completed"
            ]
        )

    with solara.Column(style="gap: 16px;"):
        solara.Text(t("tiles.inference.description"))

        solara.Button(
            t("tiles.inference.new_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=open_new_dialog,
        )

        if form_error:
            rv.Alert(type_="error", dense=True, children=[form_error])

        if _apply_pred_toggle.pending:
            rv.ProgressLinear(indeterminate=True, color="primary")

        # Outputs list
        InferenceOutputList(
            project=project,
            inference_jobs=inference_jobs,
            preds_on_map=preds_on_map,
            on_toggle_map=on_toggle_map if map_ is not None else None,
            on_dismiss=on_dismiss,
            on_delete=set_pending_delete,
            on_edit=on_edit,
            on_open=lambda row: set_details_key(row["key"]),
        )

        _pending_count = (
            len(pending_delete.get("storage_keys", [])) if pending_delete else 0
        )
        ConfirmDialog(
            open=pending_delete is not None,
            on_cancel=lambda: set_pending_delete(None),
            on_confirm=lambda: (_delete_row(pending_delete), set_pending_delete(None)),
            title=t("tiles.inference.confirm_delete_predictions_title"),
            message=plural(
                _pending_count,
                "tiles.inference.confirm_delete_predictions_message_one",
                "tiles.inference.confirm_delete_predictions_message_other",
            ),
            confirm_label=t("common.delete"),
        )

    PredictionFormDialog(
        project=project,
        open_=dialog_open,
        on_submit=on_submit,
        sepal_client=sepal_client,
        prefill=prefill,
    )

    PredictionDetailsDialog(
        project=project,
        row_key=details_key,
        on_close=lambda: set_details_key(None),
    )
