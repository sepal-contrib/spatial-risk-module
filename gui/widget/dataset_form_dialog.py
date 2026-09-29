"""New / Edit Dataset dialog for the Datasets step."""

from typing import Callable, Optional

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.artifact_names import suggest_name
from gui.widget.artifact_name_field import ArtifactNameField, use_artifact_name
from gui.widget.creation_dialog import CreationDialog


def available_years(project, var_names):
    """Sorted intersection of years across the given temporal variable names."""
    if project is None or not var_names:
        return []
    year_sets = []
    for name in var_names:
        if project.is_temporal(name, source="processed"):
            years = project.get_variable_years(name, source="processed")
            if years:
                year_sets.append(set(years))
    if not year_sets:
        return []
    return sorted(set.intersection(*year_sets))


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

    target_name, set_target_name = solara.use_state("")
    feature_names, set_feature_names = solara.use_state([])
    year, set_year = solara.use_state(None)

    existing = set(p.datasets) if p is not None and p.datasets else set()
    name_value, on_name_input, reset_name = use_artifact_name(
        editing_key if is_edit else suggest_name("dataset", existing)
    )
    clean = (name_value or "").strip()

    def reset():
        set_target_name("")
        set_feature_names([])
        set_year(None)
        reset_name()

    def prefill():
        if not open_.value or initial is None:
            return
        set_target_name(initial.get("target", ""))
        set_feature_names(list(initial.get("features", [])))
        set_year(initial.get("year"))

    solara.use_effect(prefill, [open_.value])

    available_vars = p.list_unique_variable_names(source="processed") if p else []
    feature_options = [v for v in available_vars if v != target_name]
    selected_temporal = [target_name] + feature_names if target_name else feature_names
    years = available_years(p, selected_temporal) if p else []

    def validate():
        if p is None:
            return t("tiles.dataset.error_no_project")
        if not clean:
            return t("tiles.dataset.error_dataset_name_required")
        if not target_name:
            return t("tiles.dataset.error_target_required")
        if not feature_names:
            return t("tiles.dataset.error_features_required")
        return None

    def will_replace():
        if not is_edit and clean in existing:
            return clean
        return None

    def launch():
        on_submit(
            {
                "name": editing_key if is_edit else clean,
                "target": target_name,
                "features": list(feature_names),
                "year": year,
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
            items=available_vars,
            v_model=target_name,
            on_v_model=set_target_name,
            dense=True,
            outlined=True,
            hint=t("tiles.dataset.target_hint"),
            persistent_hint=True,
        )
        rv.Select(
            label=t("tiles.dataset.feature_variables_label"),
            items=feature_options,
            v_model=feature_names,
            on_v_model=set_feature_names,
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
        if years:
            rv.Select(
                label=t("tiles.dataset.year_label"),
                items=years,
                v_model=year,
                on_v_model=set_year,
                dense=True,
                outlined=True,
                hint=t("tiles.dataset.year_hint"),
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
