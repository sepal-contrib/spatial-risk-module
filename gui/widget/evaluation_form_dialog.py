"""New evaluation dialog for the Evaluation step."""

from typing import Callable

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.tile.evaluation_helpers import (
    ALL_METRICS,
    build_truth_spec,
    default_forest_key,
    forest_key_for_truth,
    map_items,
    metric_items,
    missing_forest_for_truth,
    parse_csizes,
    parse_interval,
    variable_items,
)
from gui.widget.creation_dialog import CreationDialog
from gui.widget.text_style import MUTED


@solara.component
def EvaluationFormDialog(project, open_, on_submit: Callable[[dict], None]):
    """Evaluation form in the shared CreationDialog frame.

    Args:
        project: solara.Reactive[Project].
        open_: solara.Reactive[bool].
        on_submit: callback(entry) — the tile spawns the evaluation job.
            entry keys: spec, prediction_keys, csizes, metrics, recompute.

    Runs are keyed by a unique run_id, so there is no replace flow
    (will_replace always returns None).
    """
    p = project.value

    var_items = variable_items(p)
    pred_items = map_items(p)

    truth_key, set_truth_key = solara.use_state("")
    forest_key, set_forest_key = solara.use_state(default_forest_key(p) or "")
    interval, set_interval = solara.use_state("")
    selected_maps, set_selected_maps = solara.use_state([])
    csizes_text, set_csizes_text = solara.use_state("300")
    selected_metrics, set_selected_metrics = solara.use_state(list(ALL_METRICS))
    recompute, set_recompute = solara.use_state(True)

    def on_truth_change(key):
        set_truth_key(key)
        ti = parse_interval(p, key)
        set_interval(str(ti) if ti is not None else "")
        # An interval target names its forest-at-start layer (same threshold /
        # definition, start year): pick it when the project has it, else go
        # back to the default rather than keep the previous target's pair.
        set_forest_key(forest_key_for_truth(p, key) or default_forest_key(p) or "")

    def reset():
        set_truth_key("")
        set_forest_key(default_forest_key(p) or "")
        set_interval("")
        set_selected_maps([])
        set_csizes_text("300")
        set_selected_metrics(list(ALL_METRICS))
        set_recompute(True)

    def validate():
        spec, err = build_truth_spec(p, truth_key, forest_key, interval)
        if err:
            return err
        _, err = parse_csizes(csizes_text)
        if err:
            return err
        if not selected_metrics:
            return t("tiles.evaluation.error_select_metric")
        return None

    def launch():
        # Re-resolve the validated inputs (build_truth_spec/parse_csizes are
        # cheap and pure; validate() just ran them successfully).
        spec, _ = build_truth_spec(p, truth_key, forest_key, interval)
        csizes, _ = parse_csizes(csizes_text)
        on_submit(
            {
                "spec": spec,
                "prediction_keys": list(selected_maps),
                "csizes": csizes,
                "metrics": list(selected_metrics),
                "recompute": recompute,
            }
        )

    with CreationDialog(
        open_=open_,
        title=t("tiles.evaluation.dialog_title"),
        create_label=t("common.add"),
        validate=validate,
        will_replace=lambda: None,
        launch=launch,
        on_close=reset,
    ):
        rv.Select(
            label=t("tiles.evaluation.truth_label"),
            items=var_items,
            item_text="text",
            item_value="value",
            v_model=truth_key,
            on_v_model=on_truth_change,
            dense=True,
            outlined=True,
        )
        rv.Select(
            label=t("tiles.evaluation.forest_label"),
            items=var_items,
            item_text="text",
            item_value="value",
            v_model=forest_key,
            on_v_model=set_forest_key,
            dense=True,
            outlined=True,
        )
        # Always-present wrapper: a conditional sibling mid-dialog would drop
        # later click handlers.
        missing = missing_forest_for_truth(p, truth_key)
        with rv.Html(tag="div"):
            if missing:
                solara.Text(
                    t(
                        "tiles.evaluation.forest_for_truth_hint",
                        name=missing[0],
                        year=missing[1],
                    ),
                    style=MUTED + "font-size:0.8rem;padding:0 4px 8px;",
                )
        rv.TextField(
            label=t("tiles.evaluation.interval_label"),
            v_model=interval,
            on_v_model=set_interval,
            type="number",
            dense=True,
            outlined=True,
            hint=t("tiles.evaluation.interval_hint"),
            persistent_hint=True,
        )
        rv.Select(
            label=t("tiles.evaluation.maps_label"),
            items=pred_items,
            item_text="text",
            item_value="value",
            v_model=selected_maps,
            on_v_model=set_selected_maps,
            multiple=True,
            chips=True,
            small_chips=True,
            deletable_chips=True,
            dense=True,
            outlined=True,
            class_="multi-chips",
            no_data_text=t("tiles.evaluation.maps_no_data"),
        )
        rv.TextField(
            label=t("tiles.evaluation.csizes_label"),
            v_model=csizes_text,
            on_v_model=set_csizes_text,
            dense=True,
            outlined=True,
            hint=t("tiles.evaluation.csizes_hint"),
            persistent_hint=True,
        )
        rv.Select(
            label=t("tiles.evaluation.metrics_label"),
            items=metric_items(),
            item_text="text",
            item_value="value",
            v_model=selected_metrics,
            on_v_model=set_selected_metrics,
            multiple=True,
            chips=True,
            small_chips=True,
            deletable_chips=True,
            dense=True,
            outlined=True,
            class_="multi-chips",
        )
        rv.Switch(
            label=t("tiles.evaluation.recompute_label"),
            v_model=recompute,
            on_v_model=set_recompute,
        )
