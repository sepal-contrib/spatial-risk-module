"""PipelineHeader tests.

Badge text logic, render smoke, and codebase-gotcha contracts (no
rv.Btn(on_click), anchored dropdown, reactives-as-props).
"""

import contextlib
import inspect
from types import SimpleNamespace

import reacton
import solara
import solara.lab


def _project_with(**counts):
    from spatialrisk.project import Project

    p = Project(project_name="t")
    for _ in range(counts.get("raw", 0)):
        p.raw_variables[f"v{_}"] = SimpleNamespace(data_type="raster")
    return p


def test_count_text_pluralizes_and_falls_back():
    """count_text pluralizes variable counts and falls back to 'nothing yet'."""
    from gui.store.workflow_steps import STEPS
    from gui.widget.pipeline_header import count_text

    aoi = SimpleNamespace(name="Acre")
    p = _project_with(raw=2)
    variables = STEPS[1]
    assert count_text(variables, p, aoi) == "2 variables"
    assert count_text(variables, _project_with(raw=1), aoi) == "1 variable"
    assert count_text(variables, _project_with(raw=0), aoi) == "nothing yet"
    # AOI step: the badge is the AOI name (a string count).
    assert count_text(STEPS[0], None, aoi) == "Acre"
    assert count_text(STEPS[0], None, None) == "nothing yet"


def test_render_smoke_empty_session():
    """PipelineHeader renders with no project and no AOI selected."""
    from gui.widget.pipeline_header import PipelineHeader

    project = solara.reactive(None, equals=lambda a, b: a is b)
    aoi = solara.reactive(None)
    box, rc = reacton.render(
        PipelineHeader(
            active_step=0,
            on_navigate=lambda i: None,
            project=project,
            aoi_result=aoi,
        ),
        handle_error=False,
    )
    rc.close()


def test_render_smoke_populated_project():
    """PipelineHeader renders with a project that has outputs and an AOI."""
    from gui.widget.pipeline_header import PipelineHeader

    project = solara.reactive(_project_with(raw=3), equals=lambda a, b: a is b)
    aoi = solara.reactive(SimpleNamespace(name="Acre"))
    box, rc = reacton.render(
        PipelineHeader(
            active_step=1,
            on_navigate=lambda i: None,
            project=project,
            aoi_result=aoi,
        ),
        handle_error=False,
    )
    rc.close()


@contextlib.contextmanager
def _theme_state(dark: bool):
    """Put the KERNEL theme state (pysepal's ``ThemeState``) in a given mode.

    Restores it after. Mirrors ``tests/test_evaluation.py::_dark_theme``: the
    header follows ``use_theme_dark()``, i.e. pysepal's resolved
    ``ThemeState.dark``, not solara's own ``use_dark_effective()``/theme
    traitlet.
    """
    from pysepal.solara import get_current_theme_state

    state = get_current_theme_state()
    before = (state.mode, state.dark)
    try:
        # set_mode already keeps `dark` aligned with a fixed mode.
        state.set_mode("dark" if dark else "light")
        yield state
    finally:
        state.mode, state.dark = before


def _render_header_styles():
    """Render PipelineHeader and return the ring and badge inline styles.

    The ring is the current-step segment's box-shadow; the badge is the
    "1 / 9" pill's background. Sentinel primaries on
    ``solara.lab.theme.themes`` keep the assertion immune to whatever
    ``setup_theme_colors()`` last wrote at import/startup time.
    """
    from gui.widget.pipeline_header import PipelineHeader

    project = solara.reactive(None, equals=lambda a, b: a is b)
    aoi = solara.reactive(None)
    box, rc = reacton.render(
        PipelineHeader(
            active_step=0,
            on_navigate=lambda i: None,
            project=project,
            aoi_result=aoi,
        ),
        handle_error=False,
    )
    ring = badge = None

    def walk(widget):
        nonlocal ring, badge
        style = getattr(widget, "style_", None) or ""
        if "box-shadow" in style:
            ring = style
        if "color: #ffffff" in style:  # the badge pill, uniquely
            badge = style
        for child in getattr(widget, "children", None) or []:
            walk(child)
        # solara.lab.Menu's activator is not a normal ipywidgets child: it is
        # its own VuetifyTemplate trait (see gui/widget/pipeline_header.py's
        # `as activator:` + `solara.lab.Menu(activator=activator, ...)`), so
        # the badge — inside that activator — is unreachable via `.children`
        # alone.
        for child in getattr(widget, "activator", None) or []:
            walk(child)

    walk(box)
    rc.close()
    return ring, badge


def test_header_ring_and_badge_follow_pysepal_theme_state():
    """The ring AND the badge pick the primary of the ACTIVE ThemeState mode.

    Not solara's own (unfed) ``dark_effective`` — the 2b32b93 regression.
    Sentinel primaries make the assertion immune to whatever
    ``setup_theme_colors()`` happened to write into the shared
    ``solara.lab.theme.themes`` object earlier in the process.
    """
    themes = solara.lab.theme.themes
    before_primaries = (themes.light.primary, themes.dark.primary)
    themes.light.primary = "#111111"
    themes.dark.primary = "#222222"
    try:
        with _theme_state(dark=False):
            ring, badge = _render_header_styles()
            assert "rgba(17, 17, 17, 0.55)" in ring
            assert "background: #111111" in badge

        with _theme_state(dark=True):
            ring, badge = _render_header_styles()
            assert "rgba(34, 34, 34, 0.55)" in ring
            assert "background: #222222" in badge
    finally:
        themes.light.primary, themes.dark.primary = before_primaries


def test_jump_menu_is_a_dropdown_not_a_modal():
    """The unified title activator opens a dropdown, not a modal.

    Anchored under it, not a centred modal — and its rows keep the working
    click primitives.
    """
    from gui.widget import pipeline_header

    src = inspect.getsource(pipeline_header)
    assert "solara.lab.Menu(" in src  # anchored dropdown
    assert "rv.Dialog(" not in src  # no modal popup
    # use_activator_width=False sends min-width="auto" to v-menu; Vuetify's
    # off-screen guard then computes NaN and stops clamping, so the dropdown
    # runs off the right edge of the viewport.
    assert "use_activator_width=False" not in src


def test_no_dead_click_patterns():
    """No rv.Btn(on_click=) (dead-click gotcha); rv.use_event(...) is used."""
    from gui.widget import pipeline_header

    src = inspect.getsource(pipeline_header)
    assert "rv.Btn(" not in src  # dead-click gotcha
    assert "rv.use_event(" in src  # segment + dropdown-row clicks
    # Reads reactives inside the component (prop-equality bailout).
    assert "project.value" in src and "aoi_result.value" in src


def test_unified_activator_replaces_all_steps_button():
    """Title + badge + count + caret are ONE menu activator.

    The separate "ALL STEPS" button and the "Step 7 of 9" subtitle are gone.
    """
    from gui.widget import pipeline_header

    src = inspect.getsource(pipeline_header)
    assert "workflow.step_badge" in src  # n/total badge in the activator
    assert "workflow.all_steps" not in src  # old button label gone
    assert "workflow.step_position" not in src  # old subtitle gone
    assert "mdi-menu-down" in src  # caret affordance kept
    assert "sr-step-jump:hover" in src  # hover tint on the activator
    assert 'class_="sr-step-jump"' in src  # ...and the class is on the activator
