"""New Derived Layer dialog for the Post-process step."""

from typing import Callable

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.process_actions import change_layer_candidates, derived_output
from gui.widget.creation_dialog import CreationDialog
from gui.widget.text_style import MUTED
from spatialrisk.variables.models import PostProcessing

CHANGE_OPS = ["loss", "gain"]
OPERATIONS = CHANGE_OPS + [s.value for s in PostProcessing]


@solara.component
def DerivedLayerDialog(project, open_, on_submit: Callable[[dict], None]):
    """Derived-layer form: operation-first, with an output-name preview.

    Derived layers are auto-named (the name encodes provenance), so the
    grammar's name slot is a read-only preview instead of an editable field.
    on_submit(entry) receives {"op","start_key","end_key","pp_key"}.
    """
    p = project.value

    op, set_op = solara.use_state("loss")
    start_key, set_start_key = solara.use_state("")
    end_key, set_end_key = solara.use_state("")
    pp_key, set_pp_key = solara.use_state("")

    is_change = op in CHANGE_OPS
    entry = {"op": op, "start_key": start_key, "end_key": end_key, "pp_key": pp_key}
    output = derived_output(p, entry) if p and (is_change or pp_key) else None
    output_name = output.name if output else None

    def reset():
        set_op("loss")
        set_start_key("")
        set_end_key("")
        set_pp_key("")

    def validate():
        if p is None:
            return t("tiles.postprocess.error_no_processed")
        if is_change:
            if not (start_key and end_key and start_key != end_key and output_name):
                return t("tiles.postprocess.output_preview_invalid")
        elif not pp_key:
            return t("tiles.postprocess.output_preview_invalid")
        return None

    def will_replace():
        # generate_change_var reuses an existing layer (idempotent) and
        # edge/dist overwrite their own derived output — no confirm step.
        return None

    def launch():
        on_submit(entry)

    with CreationDialog(
        open_=open_,
        title=t("tiles.postprocess.dialog_title"),
        create_label=t("tiles.postprocess.generate_button"),
        validate=validate,
        will_replace=will_replace,
        launch=launch,
        on_close=reset,
    ):
        rv.Select(
            label=t("tiles.postprocess.operation_label"),
            items=[
                {"text": t(f"tiles.postprocess.op_label_{o}"), "value": o}
                for o in OPERATIONS
            ],
            v_model=op,
            on_v_model=set_op,
            dense=True,
            outlined=True,
            hint=t(f"tiles.postprocess.op_hint_{op}"),
            persistent_hint=True,
        )
        if is_change:
            candidates = change_layer_candidates(p) if p else []
            rv.Select(
                label=t("tiles.postprocess.start_layer_label"),
                items=candidates,
                v_model=start_key,
                on_v_model=set_start_key,
                dense=True,
                outlined=True,
            )
            rv.Select(
                label=t("tiles.postprocess.end_layer_label"),
                items=candidates,
                v_model=end_key,
                on_v_model=set_end_key,
                dense=True,
                outlined=True,
            )
            solara.Text(
                t("tiles.postprocess.change_help_text"),
                style=MUTED + "font-size:0.8rem;font-style:italic;",
            )
        else:
            rv.Select(
                label=t("tiles.postprocess.processed_variable_label"),
                items=list(p.processed_variables.keys()) if p else [],
                v_model=pp_key,
                on_v_model=set_pp_key,
                dense=True,
                outlined=True,
                hint=t("tiles.postprocess.processed_variable_hint"),
                persistent_hint=True,
            )

        # Name slot of the grammar: read-only provenance-derived output name.
        solara.Text(
            t("tiles.postprocess.output_preview", key=output_name)
            if output_name
            else t("tiles.postprocess.output_preview_invalid"),
            style=MUTED + "font-size:0.8rem;",
        )
