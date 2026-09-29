"""New / Edit Dataset dialog for the Datasets step."""

from typing import Callable, Optional

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.artifact_names import suggest_name
from gui.scripts.dataset_validation import dataset_form_error, variable_choices
from gui.widget.artifact_name_field import ArtifactNameField, use_artifact_name
from gui.widget.creation_dialog import CreationDialog


@solara.component
def DatasetFormDialog(
    project,
    open_,
    on_submit: Callable[[dict, Optional[str]], None],
    editing_key: Optional[str] = None,
    initial: Optional[dict] = None,
):
    """Dataset form in the shared CreationDialog frame.

    Args:
        project: solara.Reactive[Project].
        open_: solara.Reactive[bool].
        on_submit: callback(entry, editing_key) — the tile builds/validates/
            registers the Dataset (mutation stays in the tile, contract #1/#7).
        editing_key: when set, the dialog opens prefilled for edit; the
            storage key is fixed (name field disabled) so models that
            reference the dataset by name are never orphaned by a rename.
        initial: the edited dataset's fields, used to prefill the form.
    """
    p = project.value
    is_edit = editing_key is not None

    # Select values from variable_choices: "name" or "name@year".
    target_value, set_target_value = solara.use_state("")
    feature_values, set_feature_values = solara.use_state([])

    existing = set(p.datasets) if p is not None and p.datasets else set()
    name_value, on_name_input, reset_name = use_artifact_name(
        editing_key if is_edit else suggest_name("dataset", existing)
    )
    clean = (name_value or "").strip()

    def reset():
        set_target_value("")
        set_feature_values([])
        reset_name()

    def prefill():
        if not open_.value or initial is None:
            return
        set_target_value(initial.get("target", ""))
        set_feature_values(list(initial.get("features", [])))

    solara.use_effect(prefill, [open_.value])

    # A temporal variable is offered once per year ("defor (2015)"), so picking
    # the variable picks its year too — each one can use a different year.
    choices = variable_choices(p) if p else []
    by_value = {c["value"]: c for c in choices}
    target = by_value.get(target_value)
    target_items = [{"text": c["text"], "value": c["value"]} for c in choices]
    feature_choices = [
        c for c in choices if target is None or c["name"] != target["name"]
    ]
    # The features the select shows as chips. v_model keeps a value the select
    # has no item for — a feature later picked as the target, or one removed
    # from the project since an edited dataset was saved — without rendering
    # it, so the user can neither see nor remove it: never submit it either.
    offered = {c["value"] for c in feature_choices}
    chosen = [by_value[v] for v in feature_values if v in offered]
    # One year per variable: the formula names each feature once, so once a
    # year of a variable is picked its other years are greyed out.
    taken = {c["name"]: c["value"] for c in chosen}
    feature_items = [
        {
            "text": c["text"],
            "value": c["value"],
            "disabled": taken.get(c["name"], c["value"]) != c["value"],
        }
        for c in feature_choices
    ]

    def validate():
        if p is None:
            return t("tiles.dataset.error_no_project")
        if not clean:
            return t("tiles.dataset.error_dataset_name_required")
        # Caught here, the modal stays open with the error instead of closing
        # and leaving it for the tile (#40).
        err = dataset_form_error(choices, target_value, [c["value"] for c in chosen])
        return t(err) if err else None

    def will_replace():
        if not is_edit and clean in existing:
            return clean
        return None

    def launch():
        on_submit(
            {
                "name": editing_key if is_edit else clean,
                "target": target["name"],
                "target_year": target["year"],
                "features": [c["name"] for c in chosen],
                "feature_years": {
                    c["name"]: c["year"] for c in chosen if c["year"] is not None
                },
            },
            editing_key,
        )

    with CreationDialog(
        open_=open_,
        title=(
            t("tiles.dataset.dialog_title_edit", key=editing_key)
            if is_edit
            else t("tiles.dataset.dialog_title_new")
        ),
        create_label=t("common.save")
        if is_edit
        else t("tiles.dataset.register_button"),
        validate=validate,
        will_replace=will_replace,
        launch=launch,
        on_close=reset,
        replace_message=lambda k: t("tiles.dataset.confirm_replace_message", key=k),
    ):
        rv.Select(
            label=t("tiles.dataset.target_variable_label"),
            items=target_items,
            v_model=target_value,
            on_v_model=set_target_value,
            dense=True,
            outlined=True,
            hint=t("tiles.dataset.target_hint"),
            persistent_hint=True,
        )
        rv.Select(
            label=t("tiles.dataset.feature_variables_label"),
            items=feature_items,
            v_model=feature_values,
            on_v_model=set_feature_values,
            multiple=True,
            dense=True,
            outlined=True,
            chips=True,
            small_chips=True,
            deletable_chips=True,
            class_="multi-chips",
            hint=t("tiles.dataset.features_hint"),
            persistent_hint=True,
        )
        ArtifactNameField(
            value=name_value,
            on_input=on_name_input,
            storage_key=clean,
            exists=will_replace() is not None,
            label=t("tiles.dataset.dataset_name_label"),
            disabled=is_edit,
        )
