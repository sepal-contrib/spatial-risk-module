"""Step 4 — Dataset tile (list-first; form lives in DatasetFormDialog)."""

import logging

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts.dataset_validation import choice_value
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.dataset_form_dialog import DatasetFormDialog
from gui.widget.dataset_list import DatasetList
from gui.widget.help import InfoButton
from spatialrisk.dataset import Dataset

logger = logging.getLogger("spatial_risk")


@solara.component
def DatasetTile(project):
    """Dataset step: list of registered datasets + New/Edit dialog."""
    p = project.value

    dialog_open = solara.use_reactive(False)
    editing_key, set_editing_key = solara.use_state(None)
    initial, set_initial = solara.use_state(None)
    notifications = use_notifications()
    pending_remove, set_pending_remove = solara.use_state(None)

    def on_new():
        set_editing_key(None)
        set_initial(None)
        dialog_open.set(True)

    def on_edit(key):
        if p is None or key not in p.datasets:
            return
        ds = p.datasets[key]
        set_editing_key(key)
        set_initial(
            {
                "name": ds.name or key,
                "target": (
                    choice_value(ds.target.name, getattr(ds.target, "year", None))
                    if ds.target
                    else ""
                ),
                "features": [
                    choice_value(f.name, getattr(f, "year", None)) for f in ds.features
                ],
            }
        )
        dialog_open.set(True)

    def _do_remove(key):
        if p is None or key not in p.datasets:
            return
        del p.datasets[key]
        # Persist the removal (matches delete_sample/delete_prediction).
        p.save()
        project.set(p.model_copy())

    def on_submit(entry, edit_key):
        """Build, validate and register the dataset described by `entry`."""
        if p is None:
            notifications.error(
                t("tiles.dataset.error_no_project"), timeout=ERROR_TOAST_TIMEOUT
            )
            return
        try:
            # Each temporal variable carries its own year (picked with it).
            ds = Dataset(project=p, name=entry["name"])
            ds.set_target(entry["target"], year=entry["target_year"])
            ds.set_features(entry["features"], years=entry["feature_years"])
            ds.validate()
            key = edit_key if edit_key else entry["name"]
            # Persist immediately so the dataset survives a reload without a
            # manual Save (matches add_sample/add_model/add_prediction).
            p.add_dataset(ds, key=key, auto_save=True)
            logger.debug(
                "Registered dataset '%s' with %d features", key, len(entry["features"])
            )
            # Leave editing_key/initial alone: the CreationDialog frame closes
            # itself right after this returns, and clearing them here re-rendered
            # the still-open dialog as "New dataset" for its fade-out. on_new /
            # on_edit set both before every open, so nothing stale survives.
            project.set(p.model_copy())
        except Exception as exc:
            logger.exception("dataset submit failed")
            # The dialog has closed by now, so a failure only the builder can
            # see (a layer missing on disk) is a toast, not a step-panel alert.
            notifications.error(
                t("tiles.dataset.error_registration_failed", exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )

    has_processed = p is not None and bool(p.processed_variables)

    with solara.Column(style="gap:16px;"):
        with solara.Row(style="gap:4px;align-items:center;"):
            solara.Text(t("tiles.dataset.description"))
            InfoButton(t("tiles.dataset.info_header"), t("tiles.dataset.info_md"))

        if not has_processed:
            solara.Info(t("tiles.dataset.error_no_processed"))
            return

        solara.Button(
            t("tiles.dataset.new_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=on_new,
        )

        DatasetList(project=project, on_edit=on_edit, on_remove=set_pending_remove)

        ConfirmDialog(
            open=pending_remove is not None,
            on_cancel=lambda: set_pending_remove(None),
            on_confirm=lambda: (_do_remove(pending_remove), set_pending_remove(None)),
            title=t("tiles.dataset.confirm_remove_title"),
            message=t(
                "tiles.dataset.confirm_remove_message", name=pending_remove or ""
            ),
            confirm_label=t("common.remove"),
        )

    DatasetFormDialog(
        project=project,
        open_=dialog_open,
        on_submit=on_submit,
        editing_key=editing_key,
        initial=initial,
    )
