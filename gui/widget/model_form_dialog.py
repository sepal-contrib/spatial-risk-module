"""New Model dialog for the Train step."""

import asyncio
from typing import Callable

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.artifact_names import sanitize_key, suggest_version
from gui.scripts.formula_validation import validate_formula
from gui.scripts.model_registry import MODEL_KEYS, MODEL_REGISTRY
from gui.widget.artifact_name_field import ArtifactNameField, use_artifact_name
from gui.widget.creation_dialog import _ADVANCED_PANEL_CSS, CreationDialog
from gui.widget.details_fields import ro_field
from gui.widget.help import InfoPopup
from gui.widget.model_stats_panel import ModelStatsPanel
from spatialrisk.far_helpers import generate_patsy_formula, strip_categorical_levels


def model_label(key: str) -> str:
    """Resolve a model's display label at render time."""
    return t(MODEL_REGISTRY[key]["label_key"])


def model_short_label(key: str) -> str:
    """Compact type tag for list chips (RF, GLM, iCAR…).

    By catalog convention ``models.<key>.short`` — the full label overflows
    narrow table columns.
    """
    return t(f"models.{key}.short")


def model_labels() -> list:
    """Resolve all model display labels at render time."""
    return [model_label(k) for k in MODEL_KEYS]


def _default_params(model_key: str) -> dict:
    """Return default parameter values for a model."""
    return {p["key"]: p["default"] for p in MODEL_REGISTRY[model_key]["params"]}


def _make_param_component(model_key: str, group: str):
    """Factory: dedicated component for one model's params.

    See train_tile history: one component *type* per (model, group) so
    switching models does a clean unmount/mount instead of child-tree
    reconciliation.
    """
    param_defs = [
        p
        for p in MODEL_REGISTRY[model_key]["params"]
        if p.get("group", "params") == group
    ]
    is_variables = group == "variables"

    @solara.component
    def _Params(params: dict, set_params, feature_options=None):
        def _update(param_key, value):
            set_params({**params, param_key: value})

        for param_def in param_defs:
            pkey = param_def["key"]
            current = params.get(pkey, param_def["default"])
            ptype = param_def["type"]
            # Per-parameter help resolved by catalog convention (enforced by
            # tests/test_i18n.py).
            hint = t(f"models.{model_key}.params.{pkey}.hint")

            if is_variables:
                rv.Select(
                    label=t(param_def["label_key"]),
                    items=feature_options or [],
                    v_model=current or None,
                    on_v_model=lambda v, k=pkey: _update(k, v),
                    dense=True,
                    outlined=True,
                    clearable=True,
                    no_data_text=t("tiles.train.variables_select_no_data"),
                    hint=hint,
                    persistent_hint=True,
                )
            elif ptype == "select":
                rv.Select(
                    label=t(param_def["label_key"]),
                    items=param_def.get("items", []),
                    v_model=current,
                    on_v_model=lambda v, k=pkey: _update(k, v),
                    dense=True,
                    outlined=True,
                    hint=hint,
                    persistent_hint=True,
                )
            else:
                rv.TextField(
                    label=t(param_def["label_key"]),
                    v_model=str(current) if current is not None else "",
                    on_v_model=lambda v, k=pkey: _update(k, v),
                    dense=True,
                    outlined=True,
                    type_="number" if ptype in ("int", "float") else None,
                    hint=hint,
                    persistent_hint=True,
                )

    _Params.__name__ = f"Params_{model_key}_{group}"
    _Params.__qualname__ = f"Params_{model_key}_{group}"
    return _Params


PARAM_COMPONENTS = {k: _make_param_component(k, "params") for k in MODEL_KEYS}
VARIABLE_COMPONENTS = {k: _make_param_component(k, "variables") for k in MODEL_KEYS}
MODEL_HAS_VARIABLES = {
    k: any(p.get("group") == "variables" for p in MODEL_REGISTRY[k]["params"])
    for k in MODEL_KEYS
}
MODEL_HAS_PARAMS = {
    k: any(p.get("group", "params") == "params" for p in MODEL_REGISTRY[k]["params"])
    for k in MODEL_KEYS
}
MODEL_HAS_FORMULA = {k: MODEL_REGISTRY[k].get("has_formula", False) for k in MODEL_KEYS}


@solara.component
def ModelFormDialog(
    project, open_, on_submit: Callable[[dict], None], running_keys=frozenset()
):
    """Model form in the shared CreationDialog frame.

    on_submit(entry) receives {"model_key","name","params","dataset_key",
    "sample_key"}; the tile owns job creation and training.

    running_keys: storage keys of training jobs still in flight. A model
    registers only when its worker finishes, so ``project.models`` lags a
    launch — without this, the same name resubmitted while the first run
    is going would skip the overwrite confirm and train a second copy.
    """
    p = project.value

    selected_key, set_selected_key = solara.use_state(MODEL_KEYS[0])
    info_open, set_info_open = solara.use_state(False)
    all_params, set_all_params = solara.use_state(
        {k: _default_params(k) for k in MODEL_KEYS}
    )

    dataset_keys = sorted(p.datasets.keys()) if p and p.datasets else []
    selected_dataset, set_selected_dataset = solara.use_state(
        dataset_keys[0] if dataset_keys else ""
    )
    sample_keys = sorted(p.samples.keys()) if p and p.samples else []
    selected_sample, set_selected_sample = solara.use_state(
        sample_keys[0] if sample_keys else ""
    )

    models = p.models if p and p.models else {}
    name_value, on_name_input, reset_name = use_artifact_name(
        suggest_version(selected_key, set(models) | set(running_keys))
    )
    clean = sanitize_key(name_value)
    storage_key = f"{selected_key}_{clean}" if clean else selected_key

    registry = MODEL_REGISTRY[selected_key]
    needs_sample = registry.get("has_sampling", False)

    selected_ds_obj = (
        p.datasets.get(selected_dataset)
        if p and p.datasets and selected_dataset
        else None
    )
    feature_options = []
    if selected_ds_obj:
        feature_options = [v.name for v in selected_ds_obj.features]
        if (
            selected_ds_obj.target is not None
            and selected_ds_obj.target.name not in feature_options
        ):
            feature_options.append(selected_ds_obj.target.name)

    # One shared formula string for the three patsy models (they share the
    # same auto-formula); per-model copies would go stale on model switches.
    formula_text, set_formula_text = solara.use_state("")
    has_formula = MODEL_HAS_FORMULA[selected_key]
    # Bumped on every reset (Cancel/ESC/after-submit) so a reopen of the
    # unconditionally-mounted, eager dialog regenerates the prefill instead of
    # leaving formula_text stuck at "" (deps below would otherwise be
    # unchanged on reopen).
    prefill_nonce, set_prefill_nonce = solara.use_state(0)

    # include_levels=False keeps the displayed formula short (bare C(x), no
    # raster read); fit re-arms the level domains via inject_categorical_levels.
    # Still off the render path: extract/classify can grow I/O again.
    @solara.lab.use_task(
        dependencies=[selected_dataset, has_formula, prefill_nonce],
        raise_error=False,
        prefer_threaded=True,
    )
    async def prefill_formula():
        if selected_ds_obj is None or not has_formula:
            return None
        text = await asyncio.to_thread(
            generate_patsy_formula, selected_ds_obj, include_levels=False
        )
        return (selected_dataset, text)

    def _apply_prefill():
        if prefill_formula.error:
            # A stale success for a prior dataset must not linger once the
            # new dataset's generation fails.
            set_formula_text("")
            return
        res = prefill_formula.value
        if res is None:
            return
        ds_key, text = res
        # Identity check: a slow run for dataset A must not overwrite the
        # prefill after the user switched to dataset B.
        if ds_key == selected_dataset:
            set_formula_text(text)  # overwrites edits = regenerate-on-switch

    solara.use_effect(
        _apply_prefill,
        [prefill_formula.value, prefill_formula.error, selected_dataset, prefill_nonce],
    )

    def reset():
        reset_name()
        set_formula_text("")
        set_prefill_nonce(lambda n: n + 1)

    def validate():
        if p is None:
            return t("tiles.train.error_no_project")
        if not clean:
            return t("tiles.train.error_name_required")
        if storage_key in running_keys:
            return t("tiles.train.error_name_running", name=clean)
        if not selected_dataset or selected_dataset not in p.datasets:
            return t("tiles.train.error_invalid_dataset")
        if needs_sample and (not selected_sample or selected_sample not in p.samples):
            return t("tiles.train.error_invalid_sample")
        if has_formula:
            if prefill_formula.pending:
                return t("tiles.train.error_formula_generating")
            target_name = (
                selected_ds_obj.target.name
                if selected_ds_obj is not None and selected_ds_obj.target is not None
                else ""
            )
            feature_names = (
                [v.name for v in selected_ds_obj.features]
                if selected_ds_obj is not None
                else []
            )
            err = validate_formula(formula_text, target_name, feature_names)
            if err:
                key, kwargs = err
                return t(key, **kwargs)
        if MODEL_HAS_VARIABLES[selected_key]:
            model_params = all_params.get(selected_key, {})
            for pdef in registry["params"]:
                if pdef.get("group") != "variables":
                    continue
                val = model_params.get(pdef["key"])
                if not val:
                    return t(
                        "tiles.train.error_select_layer", label=t(pdef["label_key"])
                    )
                if val not in feature_options:
                    return t(
                        "tiles.train.error_layer_not_in_dataset",
                        label=t(pdef["label_key"]),
                        val=val,
                        dataset=selected_dataset,
                    )
        return None

    def will_replace():
        return storage_key if storage_key in models else None

    def launch():
        on_submit(
            {
                "model_key": selected_key,
                "name": clean,
                "params": all_params.get(selected_key, {}),
                "dataset_key": selected_dataset,
                "sample_key": selected_sample if needs_sample else "",
                "formula": formula_text if has_formula else None,
            }
        )

    def _set_model_params(new_params, mk=selected_key):
        set_all_params({**all_params, mk: new_params})

    ParamComponent = PARAM_COMPONENTS[selected_key]

    with CreationDialog(
        open_=open_,
        title=t("tiles.train.dialog_title"),
        create_label=t("tiles.train.train_button"),
        validate=validate,
        will_replace=will_replace,
        launch=launch,
        on_close=reset,
        replace_message=lambda k: t("tiles.train.confirm_overwrite_message", key=k),
    ):
        # The model help lives on the select's own icon rather than in a button
        # beside it: the field then owns its message strip like every other
        # field in the form, so the gap above the dataset select needs no
        # hand-tuned margin. It is the *prepend-inner* icon (the only one whose
        # click Vuetify stops from opening the menu) shifted to the right of
        # the box by the shared .field-info-icon CSS in creation_dialog.
        model_select = rv.Select(
            label=t("tiles.train.model_select_label"),
            items=[{"text": model_label(k), "value": k} for k in MODEL_KEYS],
            item_text="text",
            item_value="value",
            v_model=selected_key,
            on_v_model=set_selected_key,
            dense=True,
            outlined=True,
            prepend_inner_icon="mdi-information-outline",
            class_="field-info-icon",
            hint=t("tiles.train.model_select_hint"),
            persistent_hint=True,
        )
        # rv.use_event is a hook — call it unconditionally.
        rv.use_event(
            model_select,
            "click:prepend-inner",
            lambda *_: set_info_open(True),
        )
        InfoPopup(
            t(
                "tiles.train.model_description_header_for",
                label=model_label(selected_key),
            ),
            t(f"models.{selected_key}.summary_md")
            + "\n\n"
            + t(registry["description_key"]),
            info_open,
            set_info_open,
        )

        rv.Select(
            label=t("tiles.train.dataset_select_label"),
            items=dataset_keys,
            v_model=selected_dataset,
            on_v_model=set_selected_dataset,
            dense=True,
            outlined=True,
            no_data_text=t("tiles.train.dataset_select_no_data"),
            hint=t("tiles.train.dataset_select_hint"),
            persistent_hint=True,
        )
        if needs_sample:
            rv.Select(
                label=t("tiles.train.sample_select_label"),
                items=sample_keys,
                v_model=selected_sample,
                on_v_model=set_selected_sample,
                dense=True,
                outlined=True,
                no_data_text=t("tiles.train.sample_select_no_data"),
                hint=t("tiles.train.sample_select_hint"),
                persistent_hint=True,
            )

        if MODEL_HAS_VARIABLES[selected_key]:
            solara.Markdown(t("tiles.train.variables_header"))
            if not selected_dataset:
                solara.Info(t("tiles.train.variables_info_no_dataset"))
            elif not feature_options:
                solara.Info(
                    t(
                        "tiles.train.variables_info_no_features",
                        dataset=selected_dataset,
                    )
                )
            VarComponent = VARIABLE_COMPONENTS[selected_key]
            VarComponent(
                params=all_params.get(selected_key, {}),
                set_params=_set_model_params,
                feature_options=feature_options,
            )

        ArtifactNameField(
            value=name_value,
            on_input=on_name_input,
            storage_key=storage_key,
            exists=storage_key in models,
            label=t("tiles.train.model_name_label"),
        )

        # Every parameter has a working default, so tuning is progressive-
        # disclosed at the end of the form, collapsed by default.
        if MODEL_HAS_PARAMS[selected_key] or has_formula:
            with rv.ExpansionPanels(flat=True, class_="advanced-params"):
                with rv.ExpansionPanel():
                    with rv.ExpansionPanelHeader():
                        solara.Text(t("tiles.train.advanced_parameters_header"))
                    with rv.ExpansionPanelContent():
                        if has_formula:
                            if prefill_formula.pending:
                                formula_hint = t("tiles.train.formula_generating")
                            elif prefill_formula.error:
                                # .error is a bool; the message lives on .exception
                                formula_hint = str(prefill_formula.exception)
                            else:
                                formula_hint = t("tiles.train.formula_hint")
                            rv.Textarea(
                                label=t("tiles.train.formula_label"),
                                v_model=formula_text,
                                on_v_model=set_formula_text,
                                disabled=prefill_formula.pending,
                                dense=True,
                                outlined=True,
                                auto_grow=True,
                                rows=2,
                                class_="formula-field",
                                hint=formula_hint,
                                persistent_hint=True,
                            )
                        ParamComponent(
                            params=all_params.get(selected_key, {}),
                            set_params=_set_model_params,
                        )


@solara.component
def ModelDetailsDialog(project, model_key, on_close: Callable[[], None]):
    """Read-only view of a registered model, mirroring the New-model form.

    Shows the same fields the model was created with (type, dataset, sample,
    variables, name, advanced parameters) prefilled from the stored model —
    no Create action, just Close.

    Args:
        project: solara.Reactive[Project].
        model_key: project.models key to display, or None (dialog closed).
        on_close: () -> None; clears the tile's selected key.
    """
    p = project.value
    models = p.models if p and getattr(p, "models", None) else {}
    model = models.get(model_key) if model_key else None

    mk = getattr(model, "model_type", None)
    registry = MODEL_REGISTRY.get(mk)

    active_tab, set_active_tab = solara.use_state(0)

    with rv.Dialog(
        v_model=model is not None,
        on_v_model=lambda v: None if v else on_close(),
        max_width="720px",
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("tiles.train.details_title", key=model_key or ""))
            with rv.CardText():
                solara.Style(_ADVANCED_PANEL_CSS)
                if model is not None:
                    with rv.Tabs(
                        v_model=active_tab, on_v_model=set_active_tab, grow=False
                    ):
                        rv.Tab(children=[t("tiles.train.stats.tab_config")])
                        rv.Tab(children=[t("tiles.train.stats.tab_statistics")])
                    # pt-3: an outlined field's floating label and its
                    # fieldset legend sit ~6px *above* the box, and v-tabs-items
                    # starts flush against the tab strip — without this the
                    # first field's label ("Model") lands on the tab slider.
                    with rv.TabsItems(v_model=active_tab, class_="pt-3"):
                        with rv.TabItem():
                            with solara.Column(style="gap:4px;"):
                                ro_field(
                                    t("tiles.train.model_select_label"),
                                    model_label(mk) if registry else mk,
                                )
                                ro_field(
                                    t("tiles.train.dataset_select_label"),
                                    getattr(model, "dataset_name", None),
                                )
                                if registry and registry.get("has_sampling"):
                                    ro_field(
                                        t("tiles.train.sample_select_label"),
                                        getattr(model, "sample_name", None),
                                    )

                                var_defs = [
                                    pd
                                    for pd in (registry["params"] if registry else [])
                                    if pd.get("group") == "variables"
                                ]
                                if var_defs:
                                    solara.Markdown(t("tiles.train.variables_header"))
                                    for pd in var_defs:
                                        ro_field(
                                            t(pd["label_key"]),
                                            getattr(model, pd["key"], None),
                                        )

                                ro_field(
                                    t("tiles.train.model_name_label"),
                                    getattr(model, "name", None),
                                )

                                param_defs = [
                                    pd
                                    for pd in (registry["params"] if registry else [])
                                    if pd.get("group", "params") == "params"
                                ]
                                stored_formula = getattr(model, "formula", None)
                                if param_defs or stored_formula:
                                    with rv.ExpansionPanels(
                                        flat=True, class_="advanced-params"
                                    ):
                                        with rv.ExpansionPanel():
                                            with rv.ExpansionPanelHeader():
                                                solara.Text(
                                                    t(
                                                        "tiles.train.advanced_parameters_header"
                                                    )
                                                )
                                            with rv.ExpansionPanelContent():
                                                if stored_formula:
                                                    # levels=[...] is a fit-time safety
                                                    # net, noise to the reader.
                                                    ro_field(
                                                        t("tiles.train.formula_label"),
                                                        strip_categorical_levels(
                                                            stored_formula
                                                        ),
                                                    )
                                                for pd in param_defs:
                                                    ro_field(
                                                        t(pd["label_key"]),
                                                        getattr(
                                                            model,
                                                            pd["key"],
                                                            pd["default"],
                                                        ),
                                                    )
                        with rv.TabItem():
                            # This file owns the tab order, so it also owns the
                            # index — the panel is told whether it is on screen,
                            # never asked to recognise itself. Its charts need
                            # that: ipecharts measures its container only at DOM
                            # attach, and one attached in a hidden/transitioning
                            # tab stays width 0 forever (see gui/widget/echarts.py,
                            # and _ChartsTab in evaluation_results for the same
                            # wiring). Statistics is the second rv.Tab above.
                            ModelStatsPanel(model=model, visible=active_tab == 1)
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=lambda: on_close(),
                    text=True,
                    small=True,
                )
