"""Step 6 — Train tile."""

import logging
import uuid

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts.artifact_names import sanitize_key

# re-export: tests import from here
from gui.scripts.model_registry import (
    MODEL_KEYS,
    MODEL_REGISTRY,
)
from gui.scripts.notify_bridge import tracked_job
from gui.scripts.solara_threads import publish_if_current, spawn_in_context, update_job
from gui.store.project_writers import writing
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.model_form_dialog import (
    ModelDetailsDialog,
    ModelFormDialog,
    model_short_label,
)
from gui.widget.train_model_list import TrainModelList
from spatialrisk.evaluation import interval_from_target

logger = logging.getLogger("spatial_risk")

# Compat alias — the shared helper moved to gui/scripts/artifact_names.py
# (feeds _storage_key below).
_sanitize_name = sanitize_key


# Module-level reactive shared across re-renders
train_jobs = solara.reactive([])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _storage_key(model_key: str, name: str) -> str:
    """Project.models key for a model — mirrors BaseRiskModel's key formula."""
    return f"{model_key}_{name}" if name else model_key


def _update_job(job_id, *, skip_if_cancelled=True, **changes):
    """Immutably update a train job by id so the UI re-renders (see update_job)."""
    update_job(train_jobs, job_id, skip_if_cancelled=skip_if_cancelled, **changes)


def _parse_param(value: str, ptype: str):
    """Parse a string parameter value to the correct type."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if ptype == "int":
        return int(value)
    if ptype == "float":
        return float(value)
    return value


def build_fit_kwargs(model_key, dataset, project):
    """Family-specific fit() kwargs. ML models fit on the attached dataset (no args)."""
    if model_key == "mw":
        return {
            "time_interval": interval_from_target(dataset.target.name),
            "folder": project.folders.rmj_mw,
        }
    if model_key == "jnr":
        return {"folder": project.folders.rmj_bm}
    return {}


def _run_training(
    job_id,
    model_key,
    param_values,
    dataset,
    sample,
    project,
    project_reactive=None,
    model_name=None,
    notifier=None,
    task_title=None,
    formula=None,
):
    """Run model training in a background thread."""
    registry = MODEL_REGISTRY[model_key]
    model_cls = registry["class"]

    try:
        with tracked_job(notifier, task_title or f"Training {model_key}"), writing(
            project.project_name
        ):
            # Build model kwargs from param_values
            kwargs = {}
            for param_def in registry["params"]:
                key = param_def["key"]
                raw = param_values.get(key, param_def["default"])
                if key == "win_size_list" and isinstance(raw, str):
                    kwargs[key] = [int(x.strip()) for x in raw.split(",") if x.strip()]
                else:
                    kwargs[key] = _parse_param(
                        str(raw) if raw is not None else None, param_def["type"]
                    )

            if formula:
                # BaseRiskModel.formula: _prepare_samples prefers it over
                # auto-generation (iCAR appends "+ cell" internally on top).
                kwargs["formula"] = formula

            model = model_cls(**kwargs)
            # The user-chosen name drives both the project.models key and the pickle
            # filename, so it MUST be set before fit() (fit() calls save()).
            model.name = model_name or None
            model.dataset = dataset
            model.project = project
            if sample is not None:
                model.sample = sample
                model.sample_name = sample.name

            model.fit(**build_fit_kwargs(model_key, dataset, project))

            # Update job on success (immutably, so the UI actually re-renders).
            _update_job(
                job_id,
                status="completed",
                deviance=model.deviance,
                n_samples=model.n_samples,
            )

            # Register in the project under the name-derived key. The user already
            # confirmed any overwrite in the UI, so if the key is taken we delete the
            # superseded model (and its files) first to avoid orphaned pickles.
            storage_key = _storage_key(model_key, model_name)
            if storage_key in project.models:
                project.delete_model(storage_key, auto_save=False)
            model.register(project, key=storage_key, auto_save=True)
            _update_job(job_id, model_storage_key=model._model_key())

            # register() mutates project.models in place; publish a fresh copy on the
            # reactive so dependent tiles (Step 7 — Inference) re-render and list the
            # newly trained model. Skipped when the project was deleted or switched
            # out while training ran — see publish_if_current.
            publish_if_current(project_reactive, project)
            logger.info("Model %s trained and registered.", model_key)

    except Exception as exc:
        logger.exception("Training failed for %s", model_key)
        _update_job(job_id, status="failed", error=str(exc))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@solara.component
def TrainTile(project):
    """Train tab: list trained models; the New model dialog handles creation."""
    p = project.value

    dialog_open = solara.use_reactive(False)
    notifications = use_notifications()

    def on_submit(entry):
        """Create the job row and spawn the worker (dialog pre-validated)."""
        dataset = p.datasets[entry["dataset_key"]]
        sample = p.samples[entry["sample_key"]] if entry["sample_key"] else None
        job_id = str(uuid.uuid4())[:8]
        job = {
            "id": job_id,
            "model_name": entry["name"],
            "model_type": entry["model_key"],
            "model_label": model_short_label(entry["model_key"]),
            "dataset_name": entry["dataset_key"],
            "sample_name": entry["sample_key"] or None,
            "status": "running",
            "error": None,
            "deviance": None,
            "n_samples": None,
        }
        train_jobs.set(list(train_jobs.value) + [job])
        spawn_in_context(
            _run_training,
            (
                job_id,
                entry["model_key"],
                entry["params"],
                dataset,
                sample,
                p,
                project,
                entry["name"],
                notifications,
                t("notifications.task_training", name=entry["name"]),
                entry["formula"],
            ),
        )
        logger.info(
            "Training started: %s '%s' on dataset %s (job=%s)",
            entry["model_key"],
            entry["name"],
            entry["dataset_key"],
            job_id,
        )

    def on_cancel(job_id):
        _update_job(job_id, skip_if_cancelled=False, status="cancelled")

    def on_dismiss(job_id):
        # Failed/cancelled job rows only — never touches the model registry.
        train_jobs.set([j for j in train_jobs.value if j["id"] != job_id])

    pending_delete, set_pending_delete = solara.use_state(None)
    details_key, set_details_key = solara.use_state(None)

    def _delete_model(key):
        cur = project.value
        if cur is not None and key in cur.models:
            cur.delete_model(key, auto_save=True)
            project.set(cur.model_copy())
        # Purge session jobs that produced this model, so a stale "completed"
        # job row doesn't resurface once its registry entry is gone.
        train_jobs.set(
            [j for j in train_jobs.value if j.get("model_storage_key") != key]
        )

    with solara.Column(style="gap: 16px;"):
        solara.Text(t("tiles.train.description"))

        solara.Button(
            t("tiles.train.new_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=lambda: dialog_open.set(True),
        )

        # Trained models list
        TrainModelList(
            project=project,
            train_jobs=train_jobs,
            model_labels={k: model_short_label(k) for k in MODEL_KEYS},
            on_cancel=on_cancel,
            on_dismiss=on_dismiss,
            on_delete=set_pending_delete,
            on_open=set_details_key,
        )

        ConfirmDialog(
            open=pending_delete is not None,
            on_cancel=lambda: set_pending_delete(None),
            on_confirm=lambda: (
                _delete_model(pending_delete),
                set_pending_delete(None),
            ),
            title=t("tiles.train.confirm_delete_model_title"),
            message=t(
                "tiles.train.confirm_delete_model_message", key=pending_delete or ""
            ),
            confirm_label=t("common.delete"),
        )

    # A model registers only when its worker finishes, so p.models lags a
    # launch; the dialog rejects the storage key of a run still in flight.
    running_keys = frozenset(
        _storage_key(j["model_type"], j["model_name"])
        for j in train_jobs.value
        if j.get("status") == "running"
    )
    ModelFormDialog(
        project=project,
        open_=dialog_open,
        on_submit=on_submit,
        running_keys=running_keys,
    )
    ModelDetailsDialog(
        project=project, model_key=details_key, on_close=lambda: set_details_key(None)
    )
