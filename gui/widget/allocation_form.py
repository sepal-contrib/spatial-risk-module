"""Form dialog for a deforestation-allocation run.

Follows the form idioms of ``model_form_dialog.py``: a ``CreationDialog`` frame
owns the Create flow (validate → launch → close), the fields are ``rv`` inputs
bound through ``v_model``/``on_v_model``, and local paths are picked with
pysepal's ``FileInputComponent`` rather than typed into a bare text field.
"""

import logging
from pathlib import Path
from typing import Callable

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.components.inputs import FileInputComponent

from gui.i18n import t
from gui.scripts.allocation_runner import (
    AllocationForm,
    mask_items,
    preview_defrate_source,
    validate_form,
)
from gui.scripts.artifact_names import suggest_name
from gui.tile.evaluation_helpers import map_items
from gui.widget.artifact_name_field import use_artifact_name
from gui.widget.borders_picker import BordersPicker
from gui.widget.creation_dialog import CreationDialog
from gui.widget.details_fields import ro_field
from gui.widget.text_style import MUTED, TIGHT_FIELD, FieldHint

logger = logging.getLogger("spatial_risk")

_TABLE_EXTENSIONS = [".csv"]

_HINT = MUTED + "font-size:0.75rem;"

# Rate-table modes: resolve automatically from the prediction (default), or
# let the user pick a table file.
_DEFRATE_AUTO = "auto"
_DEFRATE_CUSTOM = "custom"

# Density-raster extents; "none" is the select's concrete stand-in for
# "no raster" (rv.Select wants a real v_model value, build_form maps it back).
_DENSITY_NONE = "none"
_DENSITY_PROJECT = "project"
_DENSITY_AOI = "aoi"


def _as_float(text):
    """Parse a numeric field, or None when blank or not a number."""
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


@solara.component
def DefrateResolutionHint(project_value, pred_key, override):
    """Eager preview of the rate table the run will use.

    The visible half of "auto-resolved": provenance chip + file name for a
    table that exists, "will be computed" for FAR families, and the resolver's
    reason when nothing resolves — shown at form time, not after the run.
    """
    src = preview_defrate_source(project_value, pred_key, user_path=override or None)

    if src.provenance == "unavailable":
        solara.Warning(src.caveat or "", dense=True)
        return

    will_compute = src.path is not None and not src.path.exists()
    with FieldHint():
        with solara.Row(style="gap:8px;align-items:center;flex-wrap:wrap;"):
            # 'Computed' needs no provenance chip: its description already says
            # where the table comes from. The other provenances name a file, so
            # the chip is what tells persisted/sibling/user tables apart.
            if src.provenance != "computed":
                provenance_key = (
                    f"toolbox.allocation.provenance_{src.provenance}".replace("-", "_")
                )
                rv.Chip(
                    small=True,
                    outlined=True,
                    color="primary",
                    children=[t(provenance_key)],
                )
            solara.Text(
                t("toolbox.allocation.defrate_will_compute")
                if will_compute
                else src.path.name,
                style=_HINT,
            )
    if src.caveat:  # the JNR observed-rates caveat
        solara.Warning(src.caveat, dense=True)


@solara.component
def AllocationFormDialog(
    open_,
    project,
    on_launch,
    on_close,
    sepal_client=None,
    running_names=frozenset(),
    prefill=None,
    active_names=frozenset(),
):
    """Collect the allocation inputs; hand a validated AllocationForm to on_launch.

    Args:
        open_: solara.Reactive[bool] — dialog visibility (owned by the tile).
        project: solara.Reactive[Project] — source of the prediction list.
        on_launch: callback(AllocationForm) — start the run.
        on_close: callback() — the tile closes the dialog.
        sepal_client: passed through to the file pickers.
        running_names: frozenset — names of every job currently in the tile's
            job list, running or failed (the call site does not filter by
            status; a failed job still holds its name — see
            gui/tile/toolbox_tile.py). Taken for the name suggestion only: run
            keys carry a run id, so a duplicate name never replaces anything.
        prefill: optional solara.Reactive holding the AllocationForm a failed
            job was launched with (or None). While the dialog is open with a
            non-empty prefill, every field is seeded from it — this is how the
            list's edit action reopens a failed run.
        active_names: frozenset — names of jobs still *running*. Unlike
            running_names these are rejected outright: a second click while
            the first launch is still going would otherwise start a second,
            identical run (run keys never collide, so nothing else stops it).
            Failed jobs are left out so the edit action can relaunch one under
            its own name.
    """
    p = project.value

    # Suggested-until-edited, like every other creation form. Saved runs and
    # in-flight jobs both hold a name; a run's storage key appends a run id, so
    # this counter is cosmetic — it only keeps two launches from reading alike.
    taken = {
        record.name for record in (getattr(p, "allocations", None) or {}).values()
    } | set(running_names)
    name_value, on_name_input, reset_name = use_artifact_name(
        suggest_name("allocation", taken)
    )

    pred_key, set_pred_key = solara.use_state(None)
    defrate_mode, set_defrate_mode = solara.use_state(_DEFRATE_AUTO)
    defrate_override, set_defrate_override = solara.use_state("")
    borders, set_borders = solara.use_state(None)
    mask, set_mask = solara.use_state("")
    juris_ha, set_juris_ha = solara.use_state("")
    years, set_years = solara.use_state("4")
    density_extent, set_density_extent = solara.use_state(_DENSITY_NONE)

    # Bumped on every seed so the borders picker remounts: AdminLevelSelector
    # only snapshots its `initial` restore seed at mount, so an admin code
    # pushed into an already-mounted picker would be silently ignored.
    borders_seed, set_borders_seed = solara.use_state(0)

    def seed_from_prefill():
        """Seed every field from the prefill entry each time the dialog opens.

        Keyed on the open flag as well as the entry so re-editing the same
        failed job after a cancel (same entry, fields reset on close) seeds
        again. A fresh open with the prefill cleared is a no-op.
        """
        entry = prefill.value if prefill is not None else None
        if not open_.value or entry is None:
            return
        # on_name_input, not a raw setter: it marks the field dirty, so the
        # retry keeps its name instead of jumping to the next allocation_<n>.
        on_name_input(entry.name or "")
        set_pred_key(entry.prediction_key)
        if entry.user_defrate_path:
            set_defrate_mode(_DEFRATE_CUSTOM)
            set_defrate_override(entry.user_defrate_path)
        else:
            set_defrate_mode(_DEFRATE_AUTO)
            set_defrate_override("")
        set_borders(entry.borders)
        set_mask(entry.mask_file or "")
        set_juris_ha(
            "" if entry.defor_juris_ha is None else f"{entry.defor_juris_ha:.12g}"
        )
        set_years(
            "" if entry.years_forecast is None else f"{entry.years_forecast:.12g}"
        )
        # Entries from before the extent existed carry a density_map bool.
        legacy_density = getattr(entry, "density_map", False)
        set_density_extent(
            getattr(entry, "density_extent", None)
            or (_DENSITY_AOI if legacy_density else _DENSITY_NONE)
        )
        set_borders_seed(borders_seed + 1)

    solara.use_effect(
        seed_from_prefill,
        [open_.value, prefill.value if prefill is not None else None],
    )

    custom_table = defrate_mode == _DEFRATE_CUSTOM

    def build_form():
        return AllocationForm(
            name=(name_value or "").strip(),
            prediction_key=pred_key,
            user_defrate_path=(
                str(defrate_override) if custom_table and defrate_override else None
            ),
            borders=borders,
            mask_file=str(mask) if mask else None,
            defor_juris_ha=_as_float(juris_ha),
            years_forecast=_as_float(years),
            density_extent=(
                None if density_extent == _DENSITY_NONE else density_extent
            ),
        )

    def validate():
        pending_name = (name_value or "").strip()
        if pending_name and pending_name in active_names:
            return t("toolbox.allocation.error_name_running", name=pending_name)
        if custom_table and not defrate_override:
            # Silent fallback to auto-resolution would betray the visible mode.
            return "Choose the rate-table file, or switch back to automatic resolution."
        return validate_form(p, build_form())

    def launch():
        on_launch(build_form())

    def reset():
        reset_name()
        set_pred_key(None)
        set_defrate_mode(_DEFRATE_AUTO)
        set_defrate_override("")
        set_borders(None)
        set_mask("")
        set_juris_ha("")
        set_years("4")
        set_density_extent(_DENSITY_NONE)
        on_close()

    items = map_items(p) if p is not None else []

    with CreationDialog(
        open_=open_,
        title=t("toolbox.allocation.title"),
        create_label=t("toolbox.allocation.run"),
        validate=validate,
        # History-safe storage keys (name + run id): runs never replace each
        # other, so the overwrite branch of the Create flow never triggers.
        will_replace=lambda: None,
        launch=launch,
        on_close=reset,
        max_width="620px",
    ):
        rv.TextField(
            label=t("toolbox.allocation.field_name"),
            v_model=name_value,
            on_v_model=on_name_input,
            dense=True,
            outlined=True,
        )

        if items:
            rv.Select(
                label=t("toolbox.allocation.field_riskmap"),
                items=items,
                item_text="text",
                item_value="value",
                v_model=pred_key,
                on_v_model=set_pred_key,
                dense=True,
                outlined=True,
                clearable=True,
            )
        else:
            solara.Info(t("toolbox.allocation.no_predictions"))

        # Mirrors the render condition of DefrateResolutionHint below: only
        # collapse the select's own messages row when a hint is actually
        # going to render under it, or the field ends up ~22px tighter than
        # the form's rhythm with nothing supplying that spacing back.
        defrate_hint_will_render = bool(pred_key) or (
            custom_table and bool(defrate_override)
        )
        with solara.Div(classes=[TIGHT_FIELD] if defrate_hint_will_render else []):
            rv.Select(
                label=t("toolbox.allocation.field_defrate"),
                items=[
                    {
                        "text": t("toolbox.allocation.defrate_option_auto"),
                        "value": _DEFRATE_AUTO,
                    },
                    {
                        "text": t("toolbox.allocation.defrate_option_custom"),
                        "value": _DEFRATE_CUSTOM,
                    },
                ],
                item_text="text",
                item_value="value",
                v_model=defrate_mode,
                on_v_model=lambda v: set_defrate_mode(v or _DEFRATE_AUTO),
                dense=True,
                outlined=True,
            )
        if custom_table:
            with solara.Div(classes=[TIGHT_FIELD]):
                FileInputComponent(
                    label=t("toolbox.allocation.field_defrate_override"),
                    value=defrate_override,
                    on_value=set_defrate_override,
                    sepal_client=sepal_client,
                    root="",
                    extensions=_TABLE_EXTENSIONS,
                    clearable=True,
                )
        if pred_key or (custom_table and defrate_override):
            DefrateResolutionHint(
                project_value=p,
                pred_key=pred_key,
                override=defrate_override if custom_table else "",
            )

        with solara.Div().key(f"borders-{borders_seed}"):
            BordersPicker(
                value=borders,
                on_value=set_borders,
                sepal_client=sepal_client,
            )

        # The mask is one of the project's processed rasters (Hansen forest &
        # co.), not a free file: everything the risk map was built from is
        # already registered, so offer exactly that.
        mask_choices = mask_items(p)
        if mask_choices:
            # field_mask_none (below) only renders while no mask is picked —
            # same reasoning as the rate-table select above.
            mask_hint_will_render = not mask
            with solara.Div(classes=[TIGHT_FIELD] if mask_hint_will_render else []):
                rv.Select(
                    label=t("toolbox.allocation.field_mask"),
                    items=mask_choices,
                    item_text="text",
                    item_value="value",
                    v_model=mask or None,
                    on_v_model=lambda v: set_mask(v or ""),
                    dense=True,
                    outlined=True,
                    clearable=True,
                )
        else:
            FieldHint(
                children=[
                    solara.Text(t("toolbox.allocation.field_mask_empty"), style=_HINT)
                ]
            )
        if not mask:
            FieldHint(
                children=[
                    solara.Text(t("toolbox.allocation.field_mask_none"), style=_HINT)
                ]
            )

        with solara.Row(style="gap:12px;"):
            rv.TextField(
                label=t("toolbox.allocation.field_juris_ha"),
                v_model=juris_ha,
                on_v_model=set_juris_ha,
                type="number",
                dense=True,
                outlined=True,
                style_="flex:2;",
            )
            rv.TextField(
                label=t("toolbox.allocation.field_years"),
                v_model=years,
                on_v_model=set_years,
                type="number",
                dense=True,
                outlined=True,
                style_="flex:1;",
            )

        rv.Select(
            label=t("toolbox.allocation.field_density"),
            items=[
                {
                    "text": t("toolbox.allocation.density_option_none"),
                    "value": _DENSITY_NONE,
                },
                {
                    "text": t("toolbox.allocation.density_option_project"),
                    "value": _DENSITY_PROJECT,
                },
                {
                    "text": t("toolbox.allocation.density_option_aoi"),
                    "value": _DENSITY_AOI,
                },
            ],
            item_text="text",
            item_value="value",
            v_model=density_extent,
            on_v_model=lambda v: set_density_extent(v or _DENSITY_NONE),
            dense=True,
            outlined=True,
        )


# --- read-only run details (ModelDetailsDialog / SampleDetailsDialog pattern) --

_SECTION = (
    MUTED + "font-size:0.7rem;font-weight:600;"
    "letter-spacing:0.08em;text-transform:uppercase;margin-top:4px;"
)

_BORDERS_METHOD_KEYS = {
    "ADMIN0": "toolbox.allocation.borders_method_admin0",
    "ADMIN1": "toolbox.allocation.borders_method_admin1",
    "ADMIN2": "toolbox.allocation.borders_method_admin2",
    "FILE": "toolbox.allocation.borders_method_file",
    "ASSET": "toolbox.allocation.borders_method_asset",
}


def _details_riskmap(record):
    """'<key> (prediction)' or '<file> (external risk map)'."""
    if record.prediction_key:
        return (
            f"{record.prediction_key} "
            f"({t('toolbox.allocation.riskmap_prediction')})"
        )
    if record.external_riskmap:
        return (
            f"{Path(record.external_riskmap).name} "
            f"({t('toolbox.allocation.source_external')})"
        )
    return None


def _details_defrate(record):
    """Provenance label, plus the file name when the user picked the table."""
    src = record.defrate_source or {}
    provenance = (src.get("provenance") or "").replace("-", "_")
    if not provenance:
        return None
    label = t(f"toolbox.allocation.provenance_{provenance}")
    path = src.get("path")
    if provenance == "user" and path:
        return f"{label} — {Path(path).name}"
    return label


def _details_borders(record):
    """'<method> — <code|file|asset>'; pre-picker runs fall back to the file."""
    src = record.borders_source or {}
    method = (src.get("method") or "").upper()
    key = _BORDERS_METHOD_KEYS.get(method)
    if key is None:
        return Path(record.borders_file).name if record.borders_file else None
    if method.startswith("ADMIN"):
        detail = src.get("admin_code")
    elif method == "FILE":
        detail = Path(src["file_path"]).name if src.get("file_path") else None
    else:
        detail = src.get("asset")
    label = t(key)
    return f"{label} — {detail}" if detail else label


@solara.component
def AllocationDetailsDialog(project, run_key, on_close: Callable[[], None]):
    """Read-only view of a saved allocation run, opened by clicking its row.

    Open iff ``run_key`` resolves to a record — a stale key (run deleted while
    selected) renders it closed, mirroring ModelDetailsDialog.

    Args:
        project: solara.Reactive[Project].
        run_key: project.allocations key to display, or None (dialog closed).
        on_close: () -> None; clears the tile's selected key.
    """
    p = project.value
    runs = (getattr(p, "allocations", None) or {}) if p is not None else {}
    record = runs.get(run_key) if run_key else None

    with rv.Dialog(
        v_model=record is not None,
        on_v_model=lambda v: None if v else on_close(),
        max_width="560px",
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(
                    t(
                        "toolbox.allocation.details_title",
                        name=record.name if record is not None else "",
                    )
                )
            with rv.CardText():
                if record is not None:
                    with solara.Column(style="gap:4px;"):
                        solara.Text(
                            t("toolbox.allocation.section_inputs"), style=_SECTION
                        )
                        ro_field(
                            t("toolbox.allocation.field_riskmap"),
                            _details_riskmap(record),
                        )
                        ro_field(
                            t("toolbox.allocation.field_defrate"),
                            _details_defrate(record),
                        )
                        ro_field(
                            t("toolbox.allocation.field_borders"),
                            _details_borders(record),
                        )
                        ro_field(
                            t("toolbox.allocation.field_mask"),
                            Path(record.mask_file).name
                            if record.mask_file
                            else t("toolbox.allocation.field_mask_none"),
                        )
                        with solara.Row(style="gap:8px;"):
                            with solara.Column(style="flex:1;"):
                                ro_field(
                                    t("toolbox.allocation.field_juris_ha"),
                                    f"{record.defor_juris_ha:,.1f}",
                                )
                            with solara.Column(style="flex:1;"):
                                ro_field(
                                    t("toolbox.allocation.field_years"),
                                    f"{record.years_forecast:g}",
                                )

                        solara.Text(
                            t("toolbox.allocation.section_results"), style=_SECTION
                        )
                        with solara.Row(style="gap:8px;"):
                            with solara.Column(style="flex:1;"):
                                ro_field(
                                    t("toolbox.allocation.result_annual"),
                                    f"{record.annual_ha:,.1f} "
                                    f"{t('toolbox.allocation.unit_ha_yr')}",
                                )
                            with solara.Column(style="flex:1;"):
                                ro_field(
                                    t("toolbox.allocation.result_total"),
                                    f"{record.total_ha:,.1f} "
                                    f"{t('toolbox.allocation.unit_ha')}",
                                )
                        ro_field(
                            t("toolbox.allocation.field_output_table"),
                            record.csv_path,
                        )
                        ro_field(
                            t("toolbox.allocation.field_created"),
                            (record.created_at or "")[:10] or None,
                        )
                        for warning in record.warnings or []:
                            rv.Html(
                                tag="span",
                                class_="error--text",
                                style_="font-size:0.8rem;",
                                children=[str(warning)],
                            )
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=on_close,
                    text=True,
                    small=True,
                )
