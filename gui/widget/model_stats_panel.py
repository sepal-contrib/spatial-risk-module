"""Statistics tab content for the model details dialog (Spec A §4).

Every number here is in-sample — the caveat strip states that first. Stats
come from model.stats when present; otherwise a background recovery attempt
runs via solara.use_task (an RF pickle is tens of MB and rebuilding design
info re-reads the training CSV — never on the render thread, see
solara-event-handlers-block-websocket-loop). Recovery is read-only: this
panel NEVER assigns model.stats or saves the project.

Card labels are resolved by catalog convention under ``tiles.train.stats``.
The lookup is built with an f-string, exactly like the per-parameter hints in
model_form_dialog: tests/test_i18n.py scans for literal translator calls with a
regex, and a literal prefix concatenated with a variable is picked up as if the
prefix itself were a key.
"""

import reacton.ipyvuetify as rv
import solara
import solara.lab
from pysepal.solara import use_theme_dark

from gui.i18n import t
from gui.scripts.model_stats_charts import dist_curve_option, importance_bars_option
from gui.scripts.model_stats_view import (
    DASH,
    categorical_references,
    coefficient_rows,
    glm_convergence_line,
    icar_convergence_summary,
    importance_entries,
    load_tab_dist,
    stat_cards,
)

# Number formatting is the view-model's job ("the widget never formats
# numbers"): reimplementing it here would resurrect the scientific-notation
# rendering A9 removed. The intercept lines and the OOB number are the two
# values that reach the widget raw, so they borrow the same formatter.
from gui.scripts.model_stats_view import _fmt as fmt_stat
from gui.widget.echarts import RENDERER_SVG, EChartsChart, theme_accent
from gui.widget.help import InfoButton
from gui.widget.product_table import BUSY_ICON, HOURGLASS_CSS, ProductTable
from gui.widget.text_style import MUTED

_CARD_STYLE = (
    "border:1px solid rgba(128,128,128,0.25);border-radius:4px;"
    "padding:8px 12px;min-width:110px;"
)

# Half the cell is available on each side of the centre line; 46% leaves the
# bar a little air at full scale.
_BAR_MAX_PCT = 46
# Only used when a table has no usable estimate to scale against (every row
# None or 0.0). The real scale is per-table — see _bar_scale.
_BAR_FALLBACK_SCALE = 0.5
# Sub-1% bars would floor to "0%" and read as absent rather than small.
_BAR_MIN_PCT = 1.0


@solara.component
def _StatCards(model, stats):
    """Header cards — one per non-None statistic (the view-model omits the rest).

    ``stats`` is passed explicitly rather than read off the model: on the
    recovery path it comes from the background task and is never written back.
    """
    cards = stat_cards(model, stats)
    with solara.Row(gap="8px", style="flex-wrap:wrap;"):
        for card in cards:
            with solara.Column(style=_CARD_STYLE, gap="0px"):
                # Weight and full-strength colour sit on the label, not the
                # value: the labels are the fixed scaffolding a reader scans to
                # find a statistic, while the values differ in length and glyph
                # shape enough to stand out on their own at the larger size.
                # The label takes no explicit colour so it inherits the card's
                # own — black on the light theme, white on the dark one — and
                # MUTED dims the value by opacity rather than naming a grey, so
                # both sides stay right in both themes (see text_style).
                solara.Text(
                    t(f"tiles.train.stats.{card['key']}"),
                    style="font-size:0.68rem;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:0.06em;",
                )
                solara.Text(
                    str(card["value"]),
                    style=(
                        # A flagged value keeps the error colour at full
                        # strength: dimming a warning is the one place this
                        # grey would cost meaning.
                        "font-size:1.0rem;font-weight:400;"
                        + ("color:var(--v-error-base);" if card.get("warn") else MUTED)
                    ),
                )


@solara.component
def ModelStatsPanel(model, visible=True):
    """Stat cards + family panel with pending/empty states, over a footnote.

    Args:
        model: the registered model to describe.
        visible: whether this panel's tab is the one on screen, forwarded to
            the family charts. ipecharts measures its container once, at DOM
            attach, and a view attached while its tab is hidden or still
            transitioning measures width 0 and never recovers (see
            gui/widget/echarts.py) — this flag is what makes the adapter
            schedule its post-transition resize. The caller owns the tab
            index, because the caller owns the tab order; the default suits a
            bare mount (e.g. a test), which counts as shown.
    """
    # Hooks run unconditionally — the task itself decides whether to work.
    # use_task lives in solara.lab; .pending/.finished/.value per the
    # solara-task-pending-and-error-gotchas memory (.error is a bool, the
    # exception is on .exception).
    # raise_error=False keeps stats_recovery's own contract intact end to end:
    # it promises the caller an empty state rather than an error, and the
    # default (True) would re-raise inside the render instead, taking the whole
    # details dialog down. An errored task is neither pending nor finished, so
    # it falls through to the empty state below.
    recovered = solara.lab.use_task(
        lambda: _recover(model),
        dependencies=[
            getattr(model, "name", None),
            getattr(model, "trained_at", None),
            getattr(model, "stats", None) is not None,
        ],
        raise_error=False,
    )
    stats = getattr(model, "stats", None)
    if stats is None and recovered.finished and recovered.value is not None:
        stats = recovered.value

    with solara.Column(gap="12px"):
        _StatCards(model=model, stats=stats)
        if stats is not None:
            _FamilyPanel(model=model, stats=stats, visible=visible)
        elif recovered.pending:
            with solara.Row(gap="8px"):
                solara.Style(HOURGLASS_CSS)
                rv.Icon(children=[BUSY_ICON], small=True, class_="sr-hourglass")
                solara.Text(t("tiles.train.stats.recovering"), style=MUTED)
        else:
            solara.Info(t("tiles.train.stats.empty_state"))
            if getattr(model, "model_type", "") == "icar":
                solara.Text(t("tiles.train.stats.icar_retrain_hint"), style=MUTED)
        # Footnote, not an alert: the caveat qualifies every number above but
        # is permanent and non-actionable, so an always-on info strip at the
        # top spent the panel's most prominent slot on text nobody re-reads.
        # The asterisk carries the reference; the message string owns it so
        # every locale keeps the marker.
        solara.Text(
            t("tiles.train.stats.caveat_training_fit"),
            style=MUTED + "font-size:0.72rem;line-height:1.3;",
        )


def _recover(model):
    """Best-effort disk recovery, off the render thread. Never writes."""
    if getattr(model, "stats", None) is not None:
        return None  # nothing to do — fit-time stats win
    from spatialrisk.mlmodels.stats_recovery import recover_stats

    return recover_stats(model)


@solara.component
def _FamilyPanel(model, stats, visible=True):
    """Family-specific body, dispatched on ``model.model_type``.

    An unknown family renders nothing rather than raising: the cards above are
    already family-agnostic, so a model type this panel has not learnt about
    still opens.
    """
    mt = getattr(model, "model_type", "")
    if mt == "glm":
        _GlmPanel(stats=stats)
    elif mt == "rf":
        _RfPanel(stats=stats, visible=visible)
    elif mt == "icar":
        _IcarPanel(stats=stats)
    elif mt in ("mw", "jnr"):
        _RmjPanel(stats=stats, visible=visible)


def _bar_scale(rows):
    """Estimate magnitude that fills the effect bar, per table.

    A fixed scale saturates: at the old 0.5 every |estimate| >= 0.5 drew at the
    same full width, so a real beta of 0.63 and one of 3.0 were pixel-identical
    — the column asserting they are equal effects when they are not. Scaling to
    the table's own largest magnitude keeps every bar distinct.

    The bar is a WITHIN-table comparator, not a cross-model one: the numeric
    estimate sits in the adjacent column for absolute reading, and two of these
    tables never appear side by side (one model per details dialog).
    """
    return (
        max((abs(r["estimate_raw"]) for r in rows if r["estimate_raw"]), default=0.0)
        or _BAR_FALLBACK_SCALE
    )


def _effect_bar(row, scale):
    """Centred effect bar for a coefficient row (the ``render`` CellSpec).

    Returns the cell callable ProductTable invokes inside the cell wrapper —
    see ``_render_cell`` in product_table.py. Positive estimates grow right
    from the centre line in the error colour (risk-raising), negative ones grow
    left in the success colour (protective). A credible interval that crosses
    zero is drawn grey: the sign is not resolved, so it must not be read as a
    direction.

    ``scale`` is the magnitude that fills the bar — see ``_bar_scale``.

    Arithmetic uses the row's ``*_raw`` floats — the sibling display strings are
    comma-grouped and would have to be parsed back.
    """
    est = row.get("estimate_raw")
    lo, hi = row.get("ci_low_raw"), row.get("ci_high_raw")

    def _cell():
        if est is None:
            solara.Text(DASH)
            return
        width = min(abs(est) / scale, 1.0) * _BAR_MAX_PCT
        # The width is rendered with :.0f, so anything under 0.5% would floor
        # to "0%" and be indistinguishable from a missing estimate. A nonzero
        # coefficient always gets a visible sliver.
        if est != 0:
            width = max(width, _BAR_MIN_PCT)
        crosses = lo is not None and hi is not None and lo < 0 < hi
        colour = (
            "grey"
            if crosses
            else ("var(--v-error-base)" if est > 0 else "var(--v-success-base)")
        )
        side = "left:50%;" if est > 0 else "right:50%;"
        with rv.Html(tag="div", style_="position:relative;height:9px;width:100%;"):
            rv.Html(
                tag="div",
                style_="position:absolute;left:50%;top:0;width:1px;height:9px;"
                "background:rgba(128,128,128,0.4);",
            )
            rv.Html(
                tag="div",
                style_=f"position:absolute;top:1px;height:7px;{side}"
                f"width:{width:.0f}%;background:{colour};border-radius:1px;",
            )

    return _cell


def _coefficient_table(columns, rows):
    """Read-only coefficient table shared by the GLM and iCAR panels."""
    ProductTable(
        title=t("tiles.train.stats.coef_header"),
        columns=columns,
        rows=rows,
        empty_text=t("tiles.train.stats.coef_empty"),
        show_actions=False,
    )


def _use_coefficient_rows(stats):
    """``(rows, scale, collapsed, split, set_split)`` for a coefficient panel.

    The GLM and iCAR panels differ only in their columns, so the level/variable
    toggle lives here rather than twice over. It mirrors the RF importance
    switch: the default view is one row per variable, the switch drills into a
    categorical's individual levels.

    ``scale`` is computed from the PER-LEVEL rows in both views, so flipping
    the switch never resizes the bars that stay on screen. This used to hold
    for free — the magnitude-picked row was always the global max — but the
    resolution-preferring selection can collapse a categorical to a smaller row
    than its noisiest level, and a per-view scale would then inflate every bar
    in the collapsed table and shrink them on toggle.

    ``collapsed`` is a ROW-COUNT comparison, not a list comparison. The two
    views rename a collapsed categorical differently (``subj (= 9)`` against
    ``subj = 9``), so comparing the lists would call every model with any
    categorical splittable — including one whose categorical contributes a
    single design column, where the switch would reveal nothing.

    A hook, so the caller must invoke it unconditionally at the top of its own
    component body (see reacton-first-render-i18n-use-event-keyerror: a hook
    behind a branch changes the hook count between renders).
    """
    split, set_split = solara.use_state(False)
    aggregated = coefficient_rows(stats)
    per_level = coefficient_rows(stats, aggregate=False)
    return (
        per_level if split else aggregated,
        _bar_scale(per_level),
        len(per_level) != len(aggregated),
        split,
        set_split,
    )


@solara.component
def _CoefficientNotes(stats, collapsed, split, set_split):
    """The level switch plus an info popup holding the explanatory notes.

    The notes used to render as always-on muted text under the table; they now
    sit behind a small info button (the standard ``InfoButton`` popup) so the
    table keeps its vertical space. The popup carries:

    - the strongest-contrast note, only when a categorical actually collapsed
      — otherwise it would describe a view that does not exist;
    - the reference-level line, whenever any categorical is present: every
      categorical estimate is a contrast AGAINST that level and no row names
      it, so without it the odds-ratio column is unreadable. It applies to
      BOTH views and even when nothing collapsed (a two-level categorical
      contributes a single row, no switch, whose odds ratio still needs its
      baseline).

    With nothing to explain and nothing to toggle, the whole row is silent.
    """
    refs = categorical_references(stats)
    parts = []
    if collapsed:
        parts.append(t("tiles.train.stats.coef_strongest_note"))
    if refs:
        parts.append(
            t(
                "tiles.train.stats.coef_reference_note",
                refs=", ".join(f"{variable} = {ref}" for variable, ref in refs),
            )
        )
    if not collapsed and not parts:
        return
    with solara.Row(gap="4px", style="align-items:center;flex-wrap:wrap;"):
        if collapsed:
            solara.Switch(
                label=t("tiles.train.stats.coef_split_toggle"),
                value=split,
                on_value=set_split,
            )
        if parts:
            InfoButton(
                title=t("tiles.train.stats.coef_info_title"),
                markdown="\n\n".join(parts),
            )


@solara.component
def _HeaderInfoButton(title, markdown):
    """``InfoButton`` wrapped so its click never reaches an enclosing toggle.

    Vuetify's ExpansionPanelHeader toggles on ANY click that bubbles to it,
    so a bare InfoButton in the header would open the popup AND flip the
    collapsible. The wrapper's ``click.stop`` listener (an ipyvue event
    modifier, same mechanism as ConfirmDialog's ``keydown.esc``) halts the
    bubble; the button's own handler has already fired by then.
    """
    with rv.Html(tag="div", style_="display:inline-flex;") as wrap:
        InfoButton(title=title, markdown=markdown)
    # rv.use_event is a hook — call it unconditionally at top level.
    rv.use_event(wrap, "click.stop", lambda *_: None)


@solara.component
def _GlmPanel(stats):
    """Coefficients with display-time odds ratios, intercepts, solver line."""
    vm_rows, scale, collapsed, split, set_split = _use_coefficient_rows(stats)
    rows = [
        {
            "key": r["name"],
            "cells": [
                {"type": "text", "value": r["name"]},
                {"type": "text", "value": r["estimate"]},
                {"type": "text", "value": r["odds_ratio"], "muted": True},
                {"type": "render", "fn": _effect_bar(r, scale)},
            ],
        }
        for r in vm_rows
    ]
    with solara.Column(gap="4px"):
        _coefficient_table(
            [
                {
                    "label": t("tiles.train.stats.col_predictor"),
                    "width": "minmax(0,2fr)",
                },
                {"label": t("tiles.train.stats.col_estimate"), "width": "90px"},
                {"label": t("tiles.train.stats.col_odds_ratio"), "width": "90px"},
                {"label": t("tiles.train.stats.col_effect"), "width": "minmax(0,1fr)"},
            ],
            rows,
        )
        _CoefficientNotes(
            stats=stats, collapsed=collapsed, split=split, set_split=set_split
        )
        # patsy's design 'Intercept' column and sklearn's own intercept_ are
        # two different numbers (spec §2.1); labelling them separately is what
        # stops a reader adding them up by accident.
        if stats.intercept_design is not None:
            solara.Text(
                t(
                    "tiles.train.stats.intercept_design",
                    value=fmt_stat(stats.intercept_design),
                ),
                style=MUTED,
            )
        if stats.intercept_fitted is not None:
            solara.Text(
                t(
                    "tiles.train.stats.intercept_fitted",
                    value=fmt_stat(stats.intercept_fitted),
                ),
                style=MUTED,
            )
        line = glm_convergence_line(stats)
        if line is not None:
            solara.Text(t("tiles.train.stats.converged_line", line=line), style=MUTED)


@solara.component
def _IcarPanel(stats):
    """Posterior summary table: estimate, SD and the 95% credible interval.

    A table rather than a forest plot on purpose: an ECharts whisker/forest
    series needs a JS ``renderItem`` callable, and callables cannot cross the
    widget wire. The centred effect bar carries the same shape information.

    Vrho is deliberately absent — it is a variance parameter, not a log-odds
    coefficient, so it neither belongs in this table nor takes the odds-ratio
    transform. It rides in ``stats.vrho`` and is already a stat card, as is the
    cell-level rho summary beside it.
    """
    vm_rows, scale, collapsed, split, set_split = _use_coefficient_rows(stats)
    conv = icar_convergence_summary(stats)
    coef_warn = conv is not None and conv["coef"] is not None and conv["coef"]["warn"]
    # The MCMC section's open state (ExpansionPanels index: 0 open, None
    # closed). Healthy chains start collapsed — background information — but a
    # coefficient warning invalidates the whole table above, so it must never
    # hide behind a closed panel. A slow Vrho alone stays closed: it is the
    # normal state of this sampler, not news. Hook: unconditional, before any
    # branch.
    mcmc_open, set_mcmc_open = solara.use_state(0 if coef_warn else None)
    rows = [
        {
            "key": r["name"],
            "cells": [
                {"type": "text", "value": r["name"]},
                {"type": "text", "value": r["estimate"]},
                {"type": "text", "value": r["std"], "muted": True},
                {"type": "text", "value": r["ci_low"], "muted": True},
                {"type": "text", "value": r["ci_high"], "muted": True},
                {"type": "render", "fn": _effect_bar(r, scale)},
            ],
        }
        for r in vm_rows
    ]
    with solara.Column(gap="4px"):
        _coefficient_table(
            [
                {
                    "label": t("tiles.train.stats.col_predictor"),
                    "width": "minmax(0,2fr)",
                },
                {"label": t("tiles.train.stats.col_estimate"), "width": "80px"},
                {"label": t("tiles.train.stats.col_std"), "width": "70px"},
                {"label": t("tiles.train.stats.col_ci_low"), "width": "80px"},
                {"label": t("tiles.train.stats.col_ci_high"), "width": "80px"},
                {"label": t("tiles.train.stats.col_effect"), "width": "minmax(0,1fr)"},
            ],
            rows,
        )
        _CoefficientNotes(
            stats=stats, collapsed=collapsed, split=split, set_split=set_split
        )
        solara.Text(t("tiles.train.stats.icar_ci_note"), style=MUTED)
        if conv is not None:
            # The numeric stand-in for deforisk's mcmc.pdf trace plots, judged
            # per group: only a badly mixed COEFFICIENT raises the warning —
            # Vrho mixes slowly on virtually every affordable run (deforisk's
            # chains did too, unshown), so it gets a neutral note instead of
            # condemning the table above. warning-base, not error-base — the
            # fit is suspect, not absent. The .advanced-params restyle comes
            # with the details dialog (see _ADVANCED_PANEL_CSS in
            # model_form_dialog).
            coef_style = "color:var(--v-warning-base);" if coef_warn else MUTED
            with rv.ExpansionPanels(
                flat=True,
                class_="advanced-params",
                v_model=mcmc_open,
                on_v_model=set_mcmc_open,
            ):
                with rv.ExpansionPanel():
                    with rv.ExpansionPanelHeader():
                        with solara.Row(gap="8px", style="align-items:center;"):
                            if coef_warn:
                                rv.Icon(
                                    children=["mdi-alert"], small=True, color="warning"
                                )
                            solara.Text(t("tiles.train.stats.icar_mcmc_header"))
                            _HeaderInfoButton(
                                title=t("tiles.train.stats.icar_mcmc_info_title"),
                                markdown=t("tiles.train.stats.icar_mcmc_info_md"),
                            )
                    with rv.ExpansionPanelContent():
                        with solara.Column(gap="4px"):
                            if conv["coef"] is not None:
                                solara.Text(
                                    t(
                                        "tiles.train.stats.icar_conv_coef_line",
                                        rhat=conv["coef"]["rhat"],
                                        ess=conv["coef"]["ess"],
                                    ),
                                    style=coef_style,
                                )
                                if coef_warn:
                                    solara.Text(
                                        t("tiles.train.stats.icar_conv_coef_warn"),
                                        style=coef_style,
                                    )
                            if conv["vrho"] is not None:
                                vrho_key = (
                                    "icar_conv_vrho_slow"
                                    if conv["vrho"]["slow"]
                                    else "icar_conv_vrho_line"
                                )
                                solara.Text(
                                    t(
                                        f"tiles.train.stats.{vrho_key}",
                                        rhat=conv["vrho"]["rhat"],
                                        ess=conv["vrho"]["ess"],
                                    ),
                                    style=MUTED,
                                )


@solara.component
def _RfPanel(stats, visible=True):
    """Impurity importances as bars, plus the OOB number.

    The 'exploratory' chip is load-bearing rather than decorative: impurity
    importance is biased toward continuous and high-cardinality predictors and
    supports no causal reading, and sklearn's oob_score_ is plain accuracy on
    the training sample, never validation.

    Stored importances are one row per design column, so a categorical's
    levels arrive as separate rows. The default view sums them into one bar
    per variable (the only total comparable to a continuous predictor); the
    split switch re-shapes the same stats into one bar per level to show
    which category carries the importance.
    """
    dark = use_theme_dark()  # hook: unconditional, before any branch
    split, set_split = solara.use_state(False)
    aggregated = importance_entries(stats)
    per_level = importance_entries(stats, aggregate=False)
    entries = per_level if split else aggregated
    # The bars take the app's live "primary" accent, so they match every
    # color="primary" control and follow a theme or palette change.
    option = importance_bars_option(entries, dark=dark, accent=theme_accent(dark))
    with solara.Column(gap="8px"):
        with solara.Row(gap="8px", style="align-items:center;"):
            solara.Text(t("tiles.train.stats.importance_header"))
            rv.Chip(
                children=[t("tiles.train.stats.exploratory_chip")],
                x_small=True,
                outlined=True,
            )
        if per_level != aggregated:
            solara.Switch(
                label=t("tiles.train.stats.importance_split_toggle"),
                value=split,
                on_value=set_split,
            )
        if option is not None:
            EChartsChart(
                option=option,
                identity="model-stats-importance",
                dark=dark,
                renderer=RENDERER_SVG,
                height=f"{max(28 * len(entries), 120)}px",
                visible=visible,
            )
        if stats.oob_accuracy is not None:
            solara.Text(
                t("tiles.train.stats.oob_line", value=fmt_stat(stats.oob_accuracy)),
                style=MUTED,
            )


@solara.component
def _RmjPanel(stats, visible=True):
    """MW/JNR distance curve, with the fitted threshold drawn on it.

    The curve comes from the tab_dist.csv the fit wrote. That file lives
    outside the project JSON, so it can be deleted independently — a missing
    or unreadable one degrades to a line of text (load_tab_dist returns None),
    never an error inside the dialog.
    """
    dark = use_theme_dark()  # hook: unconditional, before any branch
    # use_memo keeps the CSV read off every re-render (theme toggle, tab
    # switch); solara render bodies run on the websocket loop, so repeated
    # disk reads there are exactly what freezes the UI.
    path = getattr(stats, "tab_dist_path", None)
    rows = solara.use_memo(lambda: load_tab_dist(stats), [str(path)])
    option = dist_curve_option(
        rows,
        stats.dist_thresh,
        stats.perc_thresh,
        dark=dark,
        accent=theme_accent(dark),
    )
    with solara.Column(gap="8px"):
        solara.Text(t("tiles.train.stats.dist_curve_header"))
        if option is not None:
            EChartsChart(
                option=option,
                identity="model-stats-dist-curve",
                dark=dark,
                renderer=RENDERER_SVG,
                height="260px",
                visible=visible,
            )
        else:
            solara.Text(t("tiles.train.stats.tab_dist_missing"), style=MUTED)
