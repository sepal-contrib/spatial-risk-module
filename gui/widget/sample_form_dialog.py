"""New Sample dialog for the Sampling step."""

from typing import Callable

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts.artifact_names import suggest_name
from gui.widget.artifact_name_field import ArtifactNameField, use_artifact_name
from gui.widget.creation_dialog import _ADVANCED_PANEL_CSS, CreationDialog
from gui.widget.details_fields import ro_field

SAMPLING_STRATEGIES = ["random", "stratified", "systematic"]
ALLOCATION_METHODS = ["equal", "proportional", "deforisk"]


def is_continuous_strata(p, strategy: str, raster_var: str) -> bool:
    """True when a continuous raster is about to be used as the strata variable.

    Stratified sampling treats every distinct pixel value as its own class, so a
    continuous variable (altitude, distance…) yields one stratum per value —
    almost never what the user means. Only an *explicitly* continuous
    ``raster_type`` warns: the field is optional, and treating "unset" as
    continuous would flag most variables and train users to ignore the warning.
    """
    if p is None or strategy != "stratified" or not raster_var:
        return False
    var = getattr(p, "processed_variables", {}).get(raster_var)
    raster_type = getattr(var, "raster_type", None)
    if raster_type is None:
        return False
    return getattr(raster_type, "value", raster_type) == "continuous"


def _systematic_modes():
    return [
        {"text": t("tiles.sampling.systematic_mode_n_samples"), "value": "n_samples"},
        {"text": t("tiles.sampling.systematic_mode_spacing"), "value": "spacing"},
    ]


@solara.component
def SampleFormDialog(
    project, open_, existing_names, running_names, on_submit: Callable[[dict], None]
):
    """Sample form in the shared CreationDialog frame.

    Args:
        project: solara.Reactive[Project].
        open_: solara.Reactive[bool].
        existing_names: frozenset — persisted sample names (replace-confirmable).
        running_names: frozenset — names of in-flight jobs (hard validation
            error: an unfinished sample cannot be replaced).
        on_submit: callback(entry) — the tile spawns the sampling job
            (deleting the replaced sample first when entry["replace"]).
    """
    p = project.value
    raster_keys = (
        sorted(
            k
            for k, v in p.processed_variables.items()
            if getattr(v, "data_type", None) == "raster"
        )
        if p
        else []
    )

    strategy, set_strategy = solara.use_state("random")
    raster_var, set_raster_var = solara.use_state(raster_keys[0] if raster_keys else "")
    mask_var, set_mask_var = solara.use_state("")
    allocation, set_allocation = solara.use_state("equal")
    adapt, set_adapt = solara.use_state(False)
    n_samples, set_n_samples = solara.use_state(10000)
    seed, set_seed = solara.use_state(1234)
    sys_mode, set_sys_mode = solara.use_state("n_samples")
    spacing_m, set_spacing_m = solara.use_state(1000)

    taken = set(existing_names) | set(running_names)
    name_value, on_name_input, reset_name = use_artifact_name(
        suggest_name(strategy, taken)
    )
    clean = (name_value or "").strip()

    def reset():
        set_strategy("random")
        set_mask_var("")
        set_allocation("equal")
        set_adapt(False)
        set_sys_mode("n_samples")
        reset_name()

    use_spacing = strategy == "systematic" and sys_mode == "spacing"

    def validate():
        if not clean:
            return t("tiles.sampling.error_name_required")
        if clean in running_names:
            return t("tiles.sampling.error_name_running", name=clean)
        if not raster_var or raster_var not in p.processed_variables:
            return t("tiles.sampling.error_invalid_raster")
        if mask_var and mask_var not in p.processed_variables:
            return t("tiles.sampling.error_invalid_mask")
        if use_spacing and (spacing_m is None or spacing_m <= 0):
            return t("tiles.sampling.error_invalid_spacing")
        # Every non-spacing mode (random, stratified, and systematic's own
        # "by count" sub-mode) needs a concrete positive count. Clearing the
        # field used to fall through to n_samples=None, which downstream
        # (random.py/stratified.py) means "every valid pixel" — on a
        # country-scale raster that is hundreds of GiB. Only systematic +
        # spacing_m is allowed to carry the intentional None.
        if not use_spacing and (n_samples is None or n_samples <= 0):
            return t("tiles.sampling.error_invalid_n_samples")
        return None

    def will_replace():
        return clean if clean in existing_names else None

    def launch():
        on_submit(
            {
                "name": clean,
                "strategy": strategy,
                "raster_var": raster_var,
                "mask_var": mask_var,
                "allocation": allocation,
                "adapt": adapt,
                "n_samples": None if use_spacing else n_samples,
                "spacing_m": spacing_m if use_spacing else None,
                "seed": seed,
                "replace": clean in existing_names,
            }
        )

    with CreationDialog(
        open_=open_,
        title=t("tiles.sampling.dialog_title"),
        create_label=t("tiles.sampling.generate_button"),
        validate=validate,
        will_replace=will_replace,
        launch=launch,
        on_close=reset,
        replace_message=lambda k: t("tiles.sampling.confirm_replace_message", key=k),
    ):
        rv.Select(
            label=t("tiles.sampling.strategy_label"),
            items=SAMPLING_STRATEGIES,
            v_model=strategy,
            on_v_model=set_strategy,
            dense=True,
            outlined=True,
            # Dynamic help: describes the currently selected sampling design.
            hint=t(f"tiles.sampling.strategy_hint_{strategy}"),
            persistent_hint=True,
        )
        raster_label = t(
            "tiles.sampling.raster_variable_label_strata"
            if strategy == "stratified"
            else "tiles.sampling.raster_variable_label_area"
        )
        rv.Select(
            label=raster_label,
            items=raster_keys,
            v_model=raster_var,
            on_v_model=set_raster_var,
            dense=True,
            outlined=True,
            hint=t("tiles.sampling.raster_variable_hint"),
            persistent_hint=True,
        )
        # Advisory only — the user may have a reason, so Generate stays enabled.
        if is_continuous_strata(p, strategy, raster_var):
            solara.Warning(
                t("tiles.sampling.warn_continuous_strata", name=raster_var),
                dense=True,
            )
        rv.Select(
            label=t("tiles.sampling.mask_variable_label"),
            items=[""] + raster_keys,
            v_model=mask_var,
            on_v_model=set_mask_var,
            dense=True,
            outlined=True,
            hint=t("tiles.sampling.mask_variable_hint"),
            persistent_hint=True,
        )

        if strategy == "stratified":
            rv.Select(
                label=t("tiles.sampling.allocation_label"),
                items=ALLOCATION_METHODS,
                v_model=allocation,
                on_v_model=set_allocation,
                dense=True,
                outlined=True,
                hint=t(f"tiles.sampling.allocation_hint_{allocation}"),
                persistent_hint=True,
            )
            if allocation == "deforisk":
                rv.Switch(
                    label=t("tiles.sampling.adapt_label"),
                    v_model=adapt,
                    on_v_model=set_adapt,
                    hint=t("tiles.sampling.adapt_hint"),
                    persistent_hint=True,
                )

        if strategy == "systematic":
            rv.Select(
                label=t("tiles.sampling.define_grid_label"),
                items=_systematic_modes(),
                item_text="text",
                item_value="value",
                v_model=sys_mode,
                on_v_model=set_sys_mode,
                dense=True,
                outlined=True,
            )

        if use_spacing:
            rv.TextField(
                label=t("tiles.sampling.spacing_label"),
                type_="number",
                v_model=str(spacing_m) if spacing_m is not None else "",
                on_v_model=lambda v: set_spacing_m(
                    int(round(float(v))) if v and v.strip() else None
                ),
                dense=True,
                outlined=True,
                step="1",
                hint=t("tiles.sampling.spacing_hint"),
                persistent_hint=True,
            )
        else:
            _n_hint = (
                t("tiles.sampling.n_samples_hint_deforisk")
                if strategy == "stratified" and allocation == "deforisk"
                else t("tiles.sampling.n_samples_hint")
            )
            rv.TextField(
                label=t("tiles.sampling.n_samples_label"),
                type_="number",
                v_model=str(n_samples) if n_samples is not None else "",
                on_v_model=lambda v: set_n_samples(int(v) if v and v.strip() else None),
                dense=True,
                outlined=True,
                hint=_n_hint,
                persistent_hint=True,
            )

        ArtifactNameField(
            value=name_value,
            on_input=on_name_input,
            storage_key=clean,
            exists=clean in existing_names,
            label=t("tiles.sampling.sample_name_label"),
        )

        # The seed has a working default, so it is progressive-disclosed at the
        # end of the form, collapsed by default (self-managed panels). Styling
        # comes from the shared .advanced-params CSS in CreationDialog.
        with rv.ExpansionPanels(flat=True, class_="advanced-params"):
            with rv.ExpansionPanel():
                with rv.ExpansionPanelHeader():
                    solara.Text(t("common.advanced"))
                with rv.ExpansionPanelContent():
                    rv.TextField(
                        label=t("tiles.sampling.seed_label"),
                        type_="number",
                        v_model=str(seed) if seed is not None else "",
                        on_v_model=lambda v: set_seed(
                            int(v) if v and v.strip() else None
                        ),
                        dense=True,
                        outlined=True,
                        hint=t("tiles.sampling.seed_hint"),
                        persistent_hint=True,
                    )


@solara.component
def SampleDetailsDialog(project, sample_key, on_close: Callable[[], None]):
    """Read-only view of a registered sample, mirroring the New-sample form.

    Shows the inputs the sample was generated with (design, rasters,
    allocation, count or spacing, name, seed) — no Create action, just Close.
    Generated results are not repeated here; the list's Points column owns them.

    Args:
        project: solara.Reactive[Project].
        sample_key: project.samples key to display, or None (dialog closed).
        on_close: () -> None; clears the tile's selected key.
    """
    p = project.value
    samples = p.samples if p and getattr(p, "samples", None) else {}
    sample = samples.get(sample_key) if sample_key else None

    strategy = getattr(sample, "strategy", None)
    allocation = getattr(sample, "allocation", None)
    # A sample is defined either by a point count or by a grid spacing — the
    # form's mode flag is not stored, so the surviving value is the mode.
    spacing_m = getattr(sample, "spacing_m", None)

    with rv.Dialog(
        v_model=sample is not None,
        on_v_model=lambda v: None if v else on_close(),
        max_width="560px",
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(t("tiles.sampling.details_title", key=sample_key or ""))
            with rv.CardText():
                solara.Style(_ADVANCED_PANEL_CSS)
                if sample is not None:
                    with solara.Column(style="gap:4px;"):
                        ro_field(t("tiles.sampling.strategy_label"), strategy)
                        ro_field(
                            t(
                                "tiles.sampling.raster_variable_label_strata"
                                if strategy == "stratified"
                                else "tiles.sampling.raster_variable_label_area"
                            ),
                            getattr(sample, "raster_var_name", None),
                        )
                        ro_field(
                            t("tiles.sampling.mask_variable_label"),
                            getattr(sample, "mask_var_name", None),
                        )

                        if strategy == "stratified":
                            ro_field(t("tiles.sampling.allocation_label"), allocation)
                            if allocation == "deforisk":
                                ro_field(
                                    t("tiles.sampling.adapt_label"),
                                    getattr(sample, "adapt", False),
                                )

                        if spacing_m is not None:
                            ro_field(
                                t("tiles.sampling.spacing_label"),
                                (
                                    int(spacing_m)
                                    if float(spacing_m).is_integer()
                                    else spacing_m
                                ),
                            )
                        else:
                            ro_field(
                                t("tiles.sampling.n_samples_label"),
                                getattr(sample, "n_samples", None),
                            )

                        ro_field(
                            t("tiles.sampling.sample_name_label"),
                            getattr(sample, "name", None),
                        )

                        # The seed is progressive-disclosed here exactly as in
                        # the form (self-managed panels, shared CSS).
                        with rv.ExpansionPanels(flat=True, class_="advanced-params"):
                            with rv.ExpansionPanel():
                                with rv.ExpansionPanelHeader():
                                    solara.Text(t("common.advanced"))
                                with rv.ExpansionPanelContent():
                                    ro_field(
                                        t("tiles.sampling.seed_label"),
                                        getattr(sample, "seed", None),
                                    )
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=lambda: on_close(),
                    text=True,
                    small=True,
                )
