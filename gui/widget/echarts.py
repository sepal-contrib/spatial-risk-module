"""The app's single seam onto ipecharts (Apache ECharts for ipywidgets).

Every chart in the app renders through ``EChartsChart``. Callers hand over a
plain, serializable ECharts option ``dict`` and a renderer choice; nothing
outside this module needs to know that ipecharts exists, which keeps the chart
builders (``gui/scripts/*``) pure and unit-testable.

Why ``EChartsRawWidget`` and not ``EChartsWidget``: ipecharts ships both. Its
``EChartsWidget.option`` trait is ``Instance(Option)`` — a tree of 57 typed
sub-widget traits (``Title``, ``XAxis``, ``seriesitems.Bar``, …) that a caller
must assemble object by object. ``EChartsRawWidget`` overrides that trait with
a plain ``Dict``, taking the option verbatim. Both share one frontend view, so
the raw path costs nothing and lets the chart builders stay dict-only.

The pysepal ipecharts guide also shows a Solara-native factory,
``EChartsWidget.element(option=...)``, which would let reacton own the widget
lifecycle. The installed pin (``ipecharts>=1.4,<2``; 1.4.0 verified 2026-07-21)
exposes ``.element()`` on neither widget class, so this adapter keeps the
manual ``use_memo`` + ``use_effect(widget.close)`` lifecycle below (guarded by
the orphan-widget regression test). Migrate only after pinning a release that
actually ships ``.element()``, then re-verify: disposal on identity change and
teardown, hidden-tab attach sizing, option updates / interaction-state reset,
and large-scatter memory.

The pure half of the adapter — palette, theme colours, option shaping, renderer
names — lives in ``gui/scripts/echarts_options`` so that solara-free chart
builders can import it without dragging solara in. Import that module directly
rather than reaching through this one.
"""

import hashlib
import json
import sys
import threading
from pathlib import Path

import reacton.ipyvuetify as rv
import solara
import solara.lab

from gui.scripts.echarts_options import (
    DEFAULT_ACCENT,
    RENDERER_CANVAS,
    RENDERER_SVG,
    resolve_renderer,
    themed_option,
)

__all__ = [
    "DEFAULT_HEIGHT",
    "EChartsChart",
    "RENDERER_CANVAS",
    "RENDERER_SVG",
    "build_chart_widget",
    "frontend_asset_dir",
    "theme_accent",
]

# ECharts measures its canvas from the container, and a container with no
# height renders an empty chart — so the height is always a concrete pixel
# value. The width is left to stretch (see build_chart_widget).
DEFAULT_HEIGHT = "360px"

# ipecharts measures its container exactly once, at DOM attach — its frontend
# has no ResizeObserver. A view that attaches inside a hidden (display:none)
# or still-transitioning v-window-item measures width 0, falls back to
# echarts' 100px default, and never recovers. The frontend's `update_classes`
# handler ends in `chart.resize()` (verified in the 1.4.0 bundle), so toggling
# a marker DOM class from the kernel forces one client-side re-measure. The
# delay lets Vuetify's ~300 ms tab/dialog transition finish before measuring.
_RESIZE_NUDGE_CLASS = "sr-echarts-resize-nudge"
_RESIZE_NUDGE_DELAY = 0.5  # seconds


def theme_accent(dark=False):
    """The live Vuetify ``primary`` colour for one theme, as a colour string.

    The single place the charts read the app accent. Chart builders take it as
    their ``accent=`` argument and derive every series colour from it (see
    ``gui.scripts.echarts_options.accent_ramp``), so a chart is painted in the
    same green/gold as the Next button and every ``color="primary"`` control,
    and follows a palette change instead of freezing a hex of its own — which is
    what the charts' old hardcoded blues did.

    ``dark`` is the caller's already-resolved theme, not something read here:
    the app's source of truth is pysepal's kernel-scoped ``use_theme_dark()``,
    and every caller is a component that has hooked it (see the note on
    ``_ChartsTab`` for why ``solara.lab.theme.dark`` is the wrong one to read).

    Read per render, never captured at import: ``setup_theme_colors()`` writes
    these slots during app startup, long after this module is imported. An unset
    slot (a bare test harness, a theme that never got set up) falls back to
    ``DEFAULT_ACCENT`` rather than painting nothing.
    """
    themes = solara.lab.theme.themes
    return getattr(themes.dark if dark else themes.light, "primary", None) or (
        DEFAULT_ACCENT
    )


def build_chart_widget(
    option, *, dark=False, renderer=RENDERER_SVG, height=DEFAULT_HEIGHT
):
    """Build an ipecharts widget from a plain option ``dict``.

    Applies the app's chart theme (transparent background, theme ink) to a copy
    of ``option``, so the caller's dict is left untouched and can be reused
    across a light/dark re-render.

    ``renderer`` is explicit rather than inferred: pass ``RENDERER_SVG`` for the
    small metric bar charts and ``RENDERER_CANVAS`` for dense scatters. Only the
    caller knows how many points it is about to draw.

    Sizing: ``width="auto"`` makes ipecharts add its ``echarts-widget-auto-width``
    class (100% of the container), so the chart tracks the width of whatever
    card or dialog holds it. ``height`` stays a fixed pixel value.
    """
    from ipecharts import EChartsRawWidget

    resolved = resolve_renderer(renderer)
    return EChartsRawWidget(
        option=themed_option(option, dark=dark),
        renderer=resolved,
        # Dirty-rectangle repaints (guide recommendation): on canvas, hover
        # and zoom redraw only the damaged region instead of the full 200k
        # point cloud. A browser-side optimization only — it is NOT the large-
        # mode threshold, and the deployed-SEPAL performance gate still stands.
        use_dirty_rect=resolved == RENDERER_CANVAS,
        width="auto",
        height=height,
    )


def _option_digest(option):
    """Short, stable hash of an option dict's *contents*.

    ``EChartsChart`` keys its ``use_memo`` on this instead of on the dict
    itself: a dict is unhashable, and ``sort_keys`` makes the digest
    independent of key insertion order, so an option rebuilt from the same data
    never thrashes the widget.

    ``default=str`` is a safety valve, not a feature. An option value ``json``
    cannot serialize — a stray numpy scalar, a ``Path`` — must not raise from
    inside a render; it is hashed via its ``str()`` instead. For a value whose
    ``repr`` carries its address that digest moves on every render, so the
    chart is rebuilt each time: still correct, just no longer memoized. Chart
    builders should keep their options JSON-clean (``metric_bar_option`` has a
    test pinning exactly that).
    """
    payload = json.dumps(option, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@solara.component
def EChartsChart(
    option,
    identity="",
    *,
    dark=False,
    renderer=RENDERER_SVG,
    height=DEFAULT_HEIGHT,
    option_digest=None,
    visible=True
):
    """Render an ECharts option dict as a chart.

    Args:
        option: plain serializable ECharts option dict.
        identity: optional string folding in *extrinsic* rebuild triggers —
            see below. Callers with none can leave it out.
        dark: True for the app's dark theme.
        renderer: ``RENDERER_SVG`` (default, small bar charts) or
            ``RENDERER_CANVAS`` (high-point-count scatter).
        height: CSS pixel height of the chart container.
        option_digest: optional caller-supplied stand-in for the option hash —
            see "Skipping the hash" below. ``None`` (the default) keeps the
            adapter hashing the option itself.
        visible: whether the chart's container is currently shown. A caller
            whose chart lives in a tab/dialog passes the tab's active state;
            each time it is/becomes True the adapter schedules a resize nudge
            — see "Sizing" below. NOT part of the widget's identity: hiding
            never tears the chart down.

    The widget is built inside a ``use_memo`` keyed on a digest of ``option``
    plus the presentation inputs and ``identity``, so a re-render that changes
    none of them reuses the same widget (no chart teardown, no flicker), while
    a change to any of them builds a fresh one. Recreating rather than mutating
    keeps the option dict and the live chart from drifting apart, and leaves no
    trait observers behind: the widget owns all its own state, and the
    ``use_effect`` below closes the one it replaces so nothing outlives the
    render that dropped it.

    **Contract:** anything that changes what the chart *draws* is already
    covered — this component hashes the option itself, so a caller can rebuild
    its option dict every render and rely on the digest to decide. Callers do
    NOT need to enumerate the option's inputs anywhere.

    ``identity`` is for the opposite case: inputs that are NOT visible in the
    option yet must still force a fresh widget. One known kind:

    * **Live chart state.** Interaction state (legend toggles, zoom) lives in
      the chart instance, not in the option. Where reusing a widget across a
      change of subject would carry that state over — e.g. switching to another
      evaluation run that happens to draw an identical option — name the
      subject in ``identity``.

    Do NOT put attach-timing proxies (an active tab index) into ``identity``:
    that forces a rebuild of every chart on each tab switch, and the ones in
    the tab being LEFT re-attach inside a ``display:none`` container, measure
    width 0 and render squished — the exact bug the ``visible`` nudge below
    exists to prevent. A wrong ``identity`` otherwise costs at most an
    unnecessary rebuild; it can no longer render stale data.

    **Sizing.** ipecharts measures its container only at DOM attach and has no
    ResizeObserver, so a view attached while its tab was hidden or mid
    transition stays mis-sized forever. Whenever ``visible`` is/becomes True,
    a timer (``_RESIZE_NUDGE_DELAY``, past Vuetify's ~300 ms transition)
    toggles ``_RESIZE_NUDGE_CLASS`` on the widget: the frontend's
    ``update_classes`` handler calls ``chart.resize()``, re-measuring the now
    settled layout. The toggle alternates add/remove so every re-entry is a
    real trait change. Unmounting (or losing visibility) cancels a pending
    nudge.

    **Skipping the hash.** Hashing the option is cheap for a bar chart (1.2 ms
    at 1k values) and expensive for a dense scatter: re-measured 2026-07-21 on
    the scatter's real option, 2.9 ms at 2k points, 118 ms at 50k and 470 ms at
    200k — per render, in a dialog the user is interacting with. Those are
    order-of-magnitude figures from one dev machine that was also running a dev
    server, not constants (an earlier pass on the same code quoted 63/239 ms);
    what is stable is that the cost grows superlinearly with point count. A caller
    that already knows a cheap, complete identity for its option may pass it as
    ``option_digest``, and the adapter uses that INSTEAD of hashing (see
    ``gui.scripts.evaluation_echarts.pred_obs_chart_identity``, derived from one
    ``stat()``).

    That inverts the contract above, so it is opt-in and the caller takes on the
    obligation the adapter otherwise carries: **``option_digest`` must change
    whenever anything in ``option`` changes.** A digest that misses an input
    renders a stale chart with no error anywhere — precisely the failure mode
    the default path exists to prevent. Use it only where the option is a pure
    function of inputs the caller can name exhaustively and cheaply. The
    presentation inputs (``dark``, ``renderer``, ``height``) stay in the memo
    key either way, so a digest never has to account for them.
    """
    widget = solara.use_memo(
        lambda: build_chart_widget(option, dark=dark, renderer=renderer, height=height),
        [
            identity,
            _option_digest(option) if option_digest is None else option_digest,
            dark,
            renderer,
            height,
        ],
    )

    # The widget is built by `use_memo`, NOT by reacton, so reacton does not own
    # it and never disposes it: when the memo key moves (a tab switch alone
    # rebuilds every chart — see `identity`), the previous widget leaves the
    # tree while `ipywidgets`' module-level instance registry keeps a STRONG
    # reference to it for the kernel's lifetime, pinning its whole option — for
    # the scatter, every point row. Measured before this effect existed: 20 tab
    # cycles left 20 orphan widgets (+9 MB RSS at 2k points, +84 MB at 20k), and
    # closing the dialog freed none of them. Returning `widget.close` as the
    # effect's cleanup makes reacton close each widget when it is replaced and
    # when the component unmounts, which is the disposal the old
    # `solara.FigurePlotly` got for free by being reacton-owned.
    def _dispose_replaced_widget():
        return widget.close

    solara.use_effect(_dispose_replaced_widget, [widget])

    def _nudge_resize_when_shown():
        if not visible:
            return
        # The trait sync only reaches the browser from a context-bearing
        # thread; a plain threading.Timer thread has none, so capture the
        # kernel context here (render thread) and attach it there — the
        # spawn_in_context worker-thread pattern (gui/scripts/solara_threads).
        # Absent context (headless tests) the toggle still lands on the trait.
        try:
            from solara.server import kernel_context

            ctx = (
                kernel_context.get_current_context()
                if kernel_context.has_current_context()
                else None
            )
        except Exception:
            ctx = None

        def nudge():
            try:
                if ctx is not None:
                    from solara.server import kernel_context

                    if not kernel_context.has_current_context():
                        kernel_context.set_context_for_thread(
                            ctx, threading.current_thread()
                        )
                if _RESIZE_NUDGE_CLASS in widget._dom_classes:
                    widget.remove_class(_RESIZE_NUDGE_CLASS)
                else:
                    widget.add_class(_RESIZE_NUDGE_CLASS)
            except Exception:  # a late nudge on a closed widget must not raise
                pass

        timer = threading.Timer(_RESIZE_NUDGE_DELAY, nudge)
        timer.daemon = True
        timer.start()
        return timer.cancel

    solara.use_effect(_nudge_resize_when_shown, [widget, visible])

    # Handing the widget to a container's `children` is how this app already
    # mounts non-solara widgets (see the SepalMap in solara_app.Page). It puts
    # the exact widget instance in the tree; `solara.display` instead falls
    # back to IPython display when there is no notebook context, so the chart
    # never reaches the DOM under a plain render. The wrapper also supplies the
    # full-width block the chart's own auto-width class stretches to.
    rv.Html(tag="div", children=[widget], style_="width: 100%;")


def frontend_asset_dir():
    """Directory holding ipecharts' prebuilt JS, or None if it is missing.

    The widget's frontend (echarts itself included) must be served from the
    installed package: SEPAL deployments cannot rely on reaching a JS CDN.
    Returns the prebuilt labextension directory — either the copy inside the
    package (pip wheels) or the one under ``share/jupyter/labextensions``
    (conda packages, and pip after the data files are installed).
    """
    import ipecharts

    candidates = [
        Path(ipecharts.__file__).parent / "labextension",
        Path(sys.prefix) / "share" / "jupyter" / "labextensions" / "ipecharts",
    ]
    return next((c for c in candidates if (c / "package.json").is_file()), None)
