"""Unit tests for the shared ECharts adapter (Evaluation tile migration, Task 2).

Two modules under test, split along the app's layering rule:

* ``gui.scripts.echarts_options`` — pure, solara-free: the accent-derived
  palette, the theme ink/grid colours, option shaping, and renderer validation.
  Every later chart task builds its option dicts here, so this half must import
  without solara.
* ``gui.widget.echarts`` — the solara/ipecharts half: builds the widget, reads
  the live theme accent, and renders it. Nothing outside the widget layer may
  import it.

The palette assertions pin *relationships*, not hex values: the ramp's length,
determinism, direction and hue, and the fact that a single series is the accent
itself. Series colour is now derived from the app's Vuetify ``primary``, so
pinning literals would only re-freeze the drift this palette exists to remove —
the earlier ColorBrewer ``Blues`` ramp matched no theme and never followed one.
What must not drift is that a chart is painted in the app's own accent.
"""

import subprocess
import sys
from pathlib import Path

import ipecharts
import reacton

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Palette — every series colour derives from the app's "primary" accent
# --------------------------------------------------------------------------

# The two live values pysepal's setup_theme_colors writes, used as realistic
# inputs. The palette must work for ANY accent, so nothing below assumes these.
LIGHT_PRIMARY = "#5BB624"
DARK_PRIMARY = "#76591e"


def _hue_of(hex_color):
    """HSL hue of a '#rrggbb' string, in turns."""
    import colorsys

    rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(*rgb)[0]


def _luminance_of(hex_color):
    return sum(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _same_hue(a, b):
    """Hues equal within 8-bit rounding.

    A shade is computed in float HSL and rounded back to one byte per channel,
    which can nudge the recovered hue by a fraction of a percent — the ramp
    keeps the accent's hue, it just cannot store it exactly.
    """
    return abs(_hue_of(a) - _hue_of(b)) < 0.01


def test_single_series_is_the_accent_itself():
    """One cell size = one bar colour; shading would encode nothing."""
    from gui.scripts.echarts_options import accent_color, accent_ramp

    assert accent_ramp(1, LIGHT_PRIMARY) == [accent_color(LIGHT_PRIMARY)]
    assert accent_ramp(1, LIGHT_PRIMARY) == [LIGHT_PRIMARY.lower()]


def test_palette_length_matches_the_series_count():
    """One colour per series, for every count the charts can ask for."""
    from gui.scripts.echarts_options import accent_ramp

    for n in range(1, 9):
        assert len(accent_ramp(n, LIGHT_PRIMARY)) == n, f"n={n}"


def test_palette_is_deterministic():
    """Same inputs in, byte-identical list out — no sampling, no randomness."""
    from gui.scripts.echarts_options import accent_ramp

    for n in range(1, 9):
        assert accent_ramp(n, LIGHT_PRIMARY) == accent_ramp(n, LIGHT_PRIMARY)


def test_palette_follows_the_accent_it_is_given():
    """The whole point: change the app's primary, the bars move with it.

    Hue is what carries the accent's identity through the ramp — lightness is
    the dimension the ramp itself spends — so it is hue that must match.
    """
    from gui.scripts.echarts_options import accent_ramp

    for n in range(1, 6):
        for accent in (LIGHT_PRIMARY, DARK_PRIMARY, "#c0392b"):
            for shade in accent_ramp(n, accent):
                assert _same_hue(shade, accent), f"n={n} accent={accent} {shade}"


def test_palette_endpoints_are_stable_across_counts():
    """The ramp always spans the same light->dark ends, whatever n is."""
    from gui.scripts.echarts_options import accent_ramp

    ends = {(c[0], c[-1]) for c in (accent_ramp(n, LIGHT_PRIMARY) for n in range(2, 9))}
    assert len(ends) == 1


def test_palette_darkens_monotonically():
    """Darker = larger cell size. Guards the ramp's direction, not its values."""
    from gui.scripts.echarts_options import accent_ramp

    for n in range(2, 9):
        lums = [_luminance_of(c) for c in accent_ramp(n, LIGHT_PRIMARY)]
        assert lums == sorted(lums, reverse=True), f"n={n}: {lums}"


def test_palette_colors_are_all_hex_triplets():
    """Every entry is a '#rrggbb' string ECharts can take verbatim."""
    from gui.scripts.echarts_options import accent_ramp

    for n in range(1, 9):
        for c in accent_ramp(n, LIGHT_PRIMARY):
            assert len(c) == 7 and c[0] == "#"
            int(c[1:], 16)


def test_palette_rejects_a_non_positive_series_count():
    """Zero or fewer series is a caller bug, not an empty palette."""
    import pytest

    from gui.scripts.echarts_options import accent_ramp

    for bad in (0, -1):
        with pytest.raises(ValueError):
            accent_ramp(bad, LIGHT_PRIMARY)


def test_palette_falls_back_when_the_theme_slot_is_unusable():
    """An empty or non-hex theme slot must not take a render down."""
    from gui.scripts.echarts_options import DEFAULT_ACCENT, accent_color, accent_ramp

    for bad in (None, "", "primary", "#12345", 0):
        assert accent_color(bad) == DEFAULT_ACCENT.lower()
        assert len(accent_ramp(3, bad)) == 3


def test_accent_is_lifted_clear_of_the_dark_surface():
    """The dark theme's primary is a deep gold that sinks into its own surface.

    Hue is preserved — it is the same colour, only light enough to see.
    """
    from gui.scripts.echarts_options import accent_color

    on_dark = accent_color(DARK_PRIMARY, dark=True)
    assert _luminance_of(on_dark) > _luminance_of(DARK_PRIMARY)
    assert _same_hue(on_dark, DARK_PRIMARY)
    # The light theme's accent already reads on white and is left alone.
    assert accent_color(LIGHT_PRIMARY) == LIGHT_PRIMARY.lower()


def test_accent_fill_is_the_accent_at_the_given_opacity():
    """Translucent marks (the scatter cloud) are the accent, not a fixed blue."""
    from gui.scripts.echarts_options import accent_fill

    assert accent_fill(LIGHT_PRIMARY, 0.55) == "rgba(91, 182, 36, 0.55)"


# --------------------------------------------------------------------------
# Theme colours — the exact values the Plotly charts used
# --------------------------------------------------------------------------


def test_theme_colors_reuse_the_existing_app_values():
    """Ink and grid are the exact values the Plotly charts drew."""
    from gui.scripts.echarts_options import theme_colors

    assert theme_colors(dark=True) == {"ink": "#c3c2b7", "grid": "#33322f"}
    assert theme_colors(dark=False) == {"ink": "#52514e", "grid": "#e3e2dd"}


def test_theme_colors_defaults_to_light():
    """No ``dark`` argument means the light theme."""
    from gui.scripts.echarts_options import theme_colors

    assert theme_colors() == theme_colors(dark=False)


# --------------------------------------------------------------------------
# Option shaping
# --------------------------------------------------------------------------


def test_themed_option_makes_the_background_transparent():
    """The chart sits on the dialog's own surface in both themes."""
    from gui.scripts.echarts_options import themed_option

    assert themed_option({})["backgroundColor"] == "transparent"


def test_themed_option_applies_the_theme_ink_to_text():
    """Top-level text takes the theme's ink colour."""
    from gui.scripts.echarts_options import themed_option

    assert themed_option({}, dark=True)["textStyle"]["color"] == "#c3c2b7"
    assert themed_option({}, dark=False)["textStyle"]["color"] == "#52514e"


def test_themed_option_keeps_caller_text_style_keys():
    """The adapter owns the colour; the caller still owns font size etc."""
    from gui.scripts.echarts_options import themed_option

    out = themed_option({"textStyle": {"fontSize": 13, "color": "#ff0000"}})
    assert out["textStyle"]["fontSize"] == 13
    assert out["textStyle"]["color"] == "#52514e"


def test_themed_option_preserves_every_other_key():
    """The adapter adds two keys and touches nothing else the caller set."""
    from gui.scripts.echarts_options import themed_option

    option = {"series": [{"type": "bar", "data": [1, 2]}], "grid": {"left": 40}}
    out = themed_option(option)
    assert out["series"] == [{"type": "bar", "data": [1, 2]}]
    assert out["grid"] == {"left": 40}


def test_themed_option_does_not_mutate_the_callers_dict():
    """Later tasks reuse their option dicts across light/dark renders."""
    from gui.scripts.echarts_options import themed_option

    option = {"series": [], "textStyle": {"fontSize": 12}}
    themed_option(option, dark=True)
    assert option == {"series": [], "textStyle": {"fontSize": 12}}


# --------------------------------------------------------------------------
# Renderer policy — SVG for small bar charts, canvas for dense scatter
# --------------------------------------------------------------------------


def test_renderer_constants_are_the_echarts_spellings():
    """The constants are what ECharts itself expects, not app aliases."""
    from gui.scripts.echarts_options import RENDERER_CANVAS, RENDERER_SVG

    assert RENDERER_SVG == "svg"
    assert RENDERER_CANVAS == "canvas"


def test_resolve_renderer_accepts_both_supported_renderers():
    """Both supported names pass through unchanged."""
    from gui.scripts.echarts_options import (
        RENDERER_CANVAS,
        RENDERER_SVG,
        resolve_renderer,
    )

    assert resolve_renderer(RENDERER_SVG) == "svg"
    assert resolve_renderer(RENDERER_CANVAS) == "canvas"


def test_resolve_renderer_rejects_anything_else():
    """A typo'd renderer silently falls back to canvas in echarts — fail loud."""
    import pytest

    from gui.scripts.echarts_options import resolve_renderer

    for bad in ("SVG", "webgl", "", None):
        with pytest.raises(ValueError):
            resolve_renderer(bad)


# --------------------------------------------------------------------------
# Layering — the pure half must not need solara
# --------------------------------------------------------------------------


def test_option_module_imports_without_solara():
    """gui/scripts/* is solara-free by the app's layering rule.

    Run in a subprocess with 'solara' and 'ipecharts' blocked at import time:
    an accidental top-level import of either would raise ImportError.
    """
    code = (
        "import sys\n"
        "BLOCKED = ('solara', 'ipecharts', 'reacton', 'ipyvuetify', 'plotly')\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCKED:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from gui.scripts.echarts_options import accent_ramp, themed_option\n"
        "assert accent_ramp(1, '#5BB624') == ['#5bb624']\n"
        "assert themed_option({})['backgroundColor'] == 'transparent'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# --------------------------------------------------------------------------
# Widget construction — raw-dict path, headless
# --------------------------------------------------------------------------


def test_build_chart_widget_uses_the_raw_dict_widget_class():
    """EChartsRawWidget takes a plain dict; EChartsWidget needs typed Options."""
    from gui.widget.echarts import build_chart_widget

    widget = build_chart_widget({"series": []})
    assert isinstance(widget, ipecharts.EChartsRawWidget)
    assert isinstance(widget.option, dict)


def test_build_chart_widget_themes_the_option_it_hands_to_echarts():
    """The widget gets an already-themed option, not the caller's raw dict."""
    from gui.widget.echarts import build_chart_widget

    widget = build_chart_widget({"series": []}, dark=True)
    assert widget.option["backgroundColor"] == "transparent"
    assert widget.option["textStyle"]["color"] == "#c3c2b7"


def test_build_chart_widget_defaults_to_the_svg_renderer():
    """SVG is the default: most charts here are small bar charts."""
    from gui.widget.echarts import build_chart_widget

    assert build_chart_widget({}).renderer == "svg"


# --------------------------------------------------------------------------
# Accent — the charts' one link to the app's live theme
# --------------------------------------------------------------------------


def test_theme_accent_reads_the_live_vuetify_primary():
    """Not a captured constant: setup_theme_colors writes these at startup."""
    import solara.lab

    from gui.widget.echarts import theme_accent

    themes = solara.lab.theme.themes
    before = (themes.light.primary, themes.dark.primary)
    try:
        themes.light.primary = "#111111"
        themes.dark.primary = "#222222"
        assert theme_accent(dark=False) == "#111111"
        assert theme_accent(dark=True) == "#222222"
    finally:
        themes.light.primary, themes.dark.primary = before


def test_theme_accent_falls_back_when_the_slot_is_empty():
    """A bare harness (or a theme that never got set up) still draws."""
    import solara.lab

    from gui.scripts.echarts_options import DEFAULT_ACCENT
    from gui.widget.echarts import theme_accent

    themes = solara.lab.theme.themes
    before = themes.light.primary
    try:
        themes.light.primary = ""
        assert theme_accent() == DEFAULT_ACCENT
    finally:
        themes.light.primary = before


def test_build_chart_widget_takes_an_explicit_renderer():
    """The caller chooses, because only it knows the point count."""
    from gui.scripts.echarts_options import RENDERER_CANVAS
    from gui.widget.echarts import build_chart_widget

    widget = build_chart_widget({}, renderer=RENDERER_CANVAS)
    assert widget.renderer == "canvas"


def test_build_chart_widget_enables_dirty_rect_only_for_canvas():
    """Dirty-rectangle repaints are canvas-only, and pinned False for SVG.

    They only help (and only exist) on canvas; pinning the SVG case keeps the
    flag from silently riding along if the default renderer ever changes.
    """
    from gui.scripts.echarts_options import RENDERER_CANVAS, RENDERER_SVG
    from gui.widget.echarts import build_chart_widget

    option = {"series": [{"type": "bar", "data": [1, 2]}]}
    assert build_chart_widget(option, renderer=RENDERER_CANVAS).use_dirty_rect is True
    assert build_chart_widget(option, renderer=RENDERER_SVG).use_dirty_rect is False


def test_build_chart_widget_is_full_width_with_an_explicit_height():
    """width='auto' gets echarts-widget-auto-width (100%); height must be real.

    A zero-height container makes echarts render nothing at all, so the height
    is always a concrete pixel value rather than 'auto'.
    """
    from gui.widget.echarts import build_chart_widget

    widget = build_chart_widget({}, height="420px")
    assert widget.width == "auto"
    assert widget.height == "420px"


def test_build_chart_widget_has_a_default_pixel_height():
    """A container with no height draws nothing, so height is always concrete."""
    from gui.widget.echarts import build_chart_widget

    assert build_chart_widget({}).height.endswith("px")


# --------------------------------------------------------------------------
# Component — recreation, not mutation
# --------------------------------------------------------------------------


def _render_chart(**kwargs):
    from gui.widget.echarts import EChartsChart

    kwargs.setdefault("option", {"series": []})
    kwargs.setdefault("identity", "run-a")
    return reacton.render(EChartsChart(**kwargs), handle_error=False)


def _chart_widget(rc):
    return rc.find(ipecharts.EChartsRawWidget).widget


def test_chart_component_renders_headlessly():
    """The component mounts under reacton with no browser present."""
    _, rc = _render_chart()
    assert _chart_widget(rc) is not None


def test_chart_component_keeps_its_widget_when_nothing_identifying_changed():
    """Re-rendering must not thrash the widget (and the browser canvas).

    The props must actually differ between renders: reacton bails out of a
    re-render entirely when props are ``==``-equal, so passing the very same
    option would never re-run the component body and would pass this assertion
    even with no memoization at all. A tuple and a list are unequal in Python
    but serialize identically, so the body re-runs while the memo's option
    digest — the thing under test — stays put.
    """
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": [{"data": (1, 2)}]})
    first = _chart_widget(rc)
    rc.render(EChartsChart(option={"series": [{"data": [1, 2]}]}, identity="run-a"))
    assert _chart_widget(rc) is first


def test_chart_component_recreates_its_widget_when_the_option_changes():
    """The adapter owns staleness: a changed option rebuilds on its own.

    ``identity`` is deliberately held fixed. This is the assertion that makes
    the memo's option digest load-bearing — drop the digest from the ``use_memo``
    deps and ``use_memo`` hands back the widget built from ``[1]``, which is the
    silent-stale-chart failure the old design pushed onto every caller.
    """
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": [{"data": [1]}]})
    first = _chart_widget(rc)
    rc.render(EChartsChart(option={"series": [{"data": [2]}]}, identity="run-a"))
    second = _chart_widget(rc)
    assert second is not first
    assert second.option["series"] == [{"data": [2]}]


def test_chart_component_reuses_its_widget_across_a_key_order_change():
    """The digest sorts keys: an option rebuilt from the same data is the same.

    Chart builders construct their option dict fresh on every render, so a
    digest sensitive to insertion order would rebuild the widget forever.
    """
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": (), "grid": {"left": 8}})
    first = _chart_widget(rc)
    rc.render(
        EChartsChart(option={"grid": {"left": 8}, "series": []}, identity="run-a")
    )
    assert _chart_widget(rc) is first


def test_chart_component_survives_an_option_json_cannot_serialize():
    """A stray numpy scalar must not raise out of the digest mid-render.

    ``np.int64`` is not a Python ``int`` subclass, so ``json.dumps`` refuses it
    outright — exactly the value that leaks in when a builder forwards a pandas
    cell straight into an option.
    """
    import numpy as np

    from gui.widget.echarts import EChartsChart

    option = {"series": [{"data": [np.int64(3)]}], "grid": {"top": np.int64(52)}}
    _, rc = _render_chart(option=option)
    assert _chart_widget(rc) is not None
    # and it still discriminates: a different unserializable value rebuilds
    first = _chart_widget(rc)
    rc.render(
        EChartsChart(
            option={"series": [{"data": [np.int64(4)]}], "grid": {"top": np.int64(52)}},
            identity="run-a",
        )
    )
    assert _chart_widget(rc) is not first


# --------------------------------------------------------------------------
# option_digest — the caller-supplied identity escape hatch
# --------------------------------------------------------------------------


def test_chart_component_hashes_the_option_itself_by_default():
    """No digest supplied = the adapter's own hash decides. Unchanged behaviour.

    Pinned explicitly because every other caller relies on it: the default path
    must keep owning staleness so a builder can rebuild its dict every render.
    """
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": [{"data": [1]}]})
    first = _chart_widget(rc)
    rc.render(EChartsChart(option={"series": [{"data": [2]}]}, identity="run-a"))
    assert _chart_widget(rc) is not first


def test_a_caller_supplied_digest_replaces_the_option_hash():
    """The scatter's escape hatch: hashing 200k points costs ~470 ms a render.

    A caller that already knows a cheap identity for its option (file mtime, a
    run id) passes it as ``option_digest`` and the adapter skips json+sha1
    entirely. Proven by changing the option while holding the digest fixed: the
    widget must be REUSED, which can only happen if the option was never hashed.
    """
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": [{"data": [1]}]}, option_digest="points@1")
    first = _chart_widget(rc)
    rc.render(
        EChartsChart(
            option={"series": [{"data": [2]}]},
            identity="run-a",
            option_digest="points@1",
        )
    )
    assert _chart_widget(rc) is first
    # ...and the stale option was genuinely never applied
    assert _chart_widget(rc).option["series"] == [{"data": [1]}]


def test_a_changed_caller_digest_rebuilds_the_widget():
    """``option_digest`` stands in for the content hash, so moving it rebuilds."""
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": [{"data": [1]}]}, option_digest="points@1")
    first = _chart_widget(rc)
    rc.render(
        EChartsChart(
            option={"series": [{"data": [2]}]},
            identity="run-a",
            option_digest="points@2",
        )
    )
    second = _chart_widget(rc)
    assert second is not first
    assert second.option["series"] == [{"data": [2]}]


def test_the_caller_digest_does_not_override_the_presentation_inputs():
    """Theme and renderer stay the adapter's business, digest or not."""
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(option={"series": []}, option_digest="points@1", dark=False)
    first = _chart_widget(rc)
    rc.render(
        EChartsChart(
            option={"series": []}, identity="run-a", option_digest="points@1", dark=True
        )
    )
    second = _chart_widget(rc)
    assert second is not first
    assert second.option["textStyle"]["color"] == "#c3c2b7"


def test_chart_component_recreates_its_widget_when_the_identity_changes():
    """A new subject gets a fresh widget, so live chart state never carries over."""
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart()
    first = _chart_widget(rc)
    rc.render(EChartsChart(option={"series": []}, identity="run-b"))
    assert _chart_widget(rc) is not first


def test_chart_component_recreates_its_widget_when_the_theme_flips():
    """Theme lives in the option dict, so the widget must be rebuilt, not poked."""
    from gui.widget.echarts import EChartsChart

    _, rc = _render_chart(dark=False)
    first = _chart_widget(rc)
    rc.render(EChartsChart(option={"series": []}, identity="run-a", dark=True))
    second = _chart_widget(rc)
    assert second is not first
    assert second.option["textStyle"]["color"] == "#c3c2b7"


# --------------------------------------------------------------------------
# Disposal — a replaced widget must not survive in the kernel's registry
# --------------------------------------------------------------------------


def _live_chart_widgets():
    """Every EChartsRawWidget ipywidgets still holds a strong reference to.

    ``ipywidgets.widgets.widget._instances`` is the module-level registry behind
    the deprecated ``Widget.widgets`` property — a plain dict, so an entry is a
    STRONG reference that lives for the kernel's lifetime unless the widget is
    closed. Reached directly here to avoid the deprecation warning.
    """
    from ipywidgets.widgets.widget import _instances

    return [
        w
        for w in list(_instances.values())
        if isinstance(w, ipecharts.EChartsRawWidget)
    ]


def test_replaced_chart_widgets_are_closed_rather_than_orphaned():
    """A rebuilt chart must not leave its predecessor registered in the kernel.

    The widget is built inside ``use_memo``, so reacton does not own it and will
    not dispose it. Without an explicit close, every memo-key change (an
    ordinary tab switch — ``identity`` carries the active tab index) drops a
    widget out of the tree that the ipywidgets registry keeps alive forever,
    pinning its whole option: for the scatter, every point row. Measured with
    the disposal removed: 20 identity changes left 20 orphans (+9 MB RSS at 2k
    points, +84 MB at 20k) and ``rc.close()`` freed none of them.

    Counted as a delta against the widgets already registered by other tests in
    the session, so the assertion is about THIS component's churn only.
    """
    from gui.widget.echarts import EChartsChart

    before = len(_live_chart_widgets())
    _, rc = _render_chart(identity="cycle-0")
    for i in range(1, 21):
        rc.render(EChartsChart(option={"series": []}, identity=f"cycle-{i}"))
    live = len(_live_chart_widgets()) - before
    assert live == 1, f"{live} chart widgets alive after 20 rebuilds (want 1)"

    rc.close()
    assert len(_live_chart_widgets()) - before == 0


# --------------------------------------------------------------------------
# No CDN — the frontend must ship with the package
# --------------------------------------------------------------------------


def test_ipecharts_frontend_assets_resolve_from_the_local_install():
    """The labextension (incl. echarts itself) is on disk, not fetched at runtime.

    SEPAL deployments have no guaranteed egress to a JS CDN, so the widget's
    frontend must come from the installed package. Checks the prebuilt
    labextension directory the package declares, its webpack entry point, and
    that echarts is compiled into the bundle rather than listed as a remote.
    """
    import json

    from gui.widget.echarts import frontend_asset_dir

    asset_dir = frontend_asset_dir()
    assert asset_dir is not None and asset_dir.is_dir(), asset_dir

    pkg = json.loads((asset_dir / "package.json").read_text())
    entry = pkg["jupyterlab"]["_build"]["load"]
    assert (asset_dir / entry).is_file(), entry

    licenses = json.loads(
        (asset_dir / "static" / "third-party-licenses.json").read_text()
    )
    bundled = {p["name"] for p in licenses["packages"]}
    assert "echarts" in bundled, sorted(bundled)
