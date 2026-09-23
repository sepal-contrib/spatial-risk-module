"""Import-level and wiring guards on the Solara app shell.
"""


def test_workflow_tabs_uses_pipeline_header():
    """The tab strip is the shared PipelineHeader, not a hand-rolled rv.Tabs."""
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.WorkflowTabs)
    assert "PipelineHeader(" in src
    assert "on_navigate=set_active_tab" in src
    # The old strip and its hand-maintained gating are gone.
    assert "rv.Tab(" not in src
    assert "disabled_flags" not in src


def test_workflow_tabs_hosts_all_tiles_in_registry_order():
    """One rv.TabItem per registry step, tiles in canonical order.

    the registry is the...
    """
    import inspect

    import gui.solara_app as app
    from gui.store.workflow_steps import STEPS

    src = inspect.getsource(app.WorkflowTabs)
    assert src.count("with rv.TabItem():") == len(STEPS)
    tiles = [
        "AoiTile",
        "VariablesTile",
        "ProcessTile",
        "PostProcessTile",
        "DatasetTile",
        "SamplingTile",
        "TrainTile",
        "InferenceTile",
        "EvaluationTile",
    ]
    positions = [src.index(t) for t in tiles]
    assert positions == sorted(positions), "tiles out of registry order"


def test_app_state_has_no_stale_current_step():
    """AppState carries no leftover current_step field."""
    from gui.store.state_manager import AppState

    assert not hasattr(AppState(), "current_step")


def test_train_tile_selects_dataset_and_sample():
    """Dataset/sample selection lives in ModelFormDialog (Task 7 moved the form out of.

    T...

    this regression guard now targets that module.
    """
    import inspect

    from gui.widget import model_form_dialog

    src = inspect.getsource(model_form_dialog)
    assert "selected_dataset" in src
    assert "has_sampling" in src
    assert "p.datasets" in src and "p.samples" in src
    # dataset no longer derived from the sample
    assert "sample_set.dataset_name" not in src


def test_page_resets_job_lists_on_load():
    """Switching projects clears the session job lists;.

    product rows derive from the registries at render time (no job_restore facade).
    """
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.Page)
    assert "reset_jobs_on_load" in src
    assert "build_train_jobs" not in src
    assert "build_inference_jobs" not in src
    assert "solara.use_effect(reset_jobs_on_load, [project_loaded_signal])" in src


def test_page_clears_map_overlays_on_switch():
    """Switching projects clears the previous project's overlay layers and the.

    per-tile...
    """
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.Page)
    assert "clear_project_overlays(sepal_map)" in src
    assert "vars_on_map.set(set())" in src
    assert "samples_on_map.set(set())" in src
    assert "preds_on_map.set(set())" in src
    assert "solara.use_effect(render_map_on_switch, [project_loaded_signal])" in src


def test_solara_app_imports_summary_tile():
    """The shell imports the Project Summary tile."""
    import gui.solara_app as app

    assert hasattr(app, "ProjectSummaryTile")


def test_page_wires_project_summary_step():
    """Project Summary is a left-rail dialog step, not a workflow tab."""
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.Page)
    assert 't("app.step_project_summary")' in src
    assert "ProjectSummaryTile(" in src
    # Left-drawer step opens as a modal dialog.
    assert '"display": "dialog"' in src


def test_workflow_tabs_wires_aoi_restore_signal():
    """The AOI restore signal reaches the tabs."""
    import inspect

    import gui.solara_app as solara_app

    src = inspect.getsource(solara_app.WorkflowTabs)
    assert "restore_signal=app_state.project_loaded_signal.value" in src


def test_aoi_tile_imports_pysepal_view():
    """The AOI tile builds on pysepal's AOI view."""
    # The vendored restore fork was upstreamed into pysepal (AoiView
    # restore-on-mount + AoiResult.asset); the tile must use the library.
    import inspect

    import gui.tile.aoi_tile as aoi_tile

    src = inspect.getsource(aoi_tile)
    assert "from pysepal.solara.components.aoi import AoiView" in src
    assert "gui.widget.aoi_view" not in src


def test_solara_app_installs_task_log_handler():
    """Job log lines only reach the notification pill through the bridge handler.

    boot...
    """
    import inspect

    import gui.solara_app as app

    assert "install_task_log_handler()" in inspect.getsource(app)


def test_page_mounts_notification_provider_before_the_map_app():
    """The pysepal NotificationProvider is the only notification UI (the custom.

    LogConso...

    It must mount before the MapApp element so the bus exists when the workflow
    tiles first render — pysepal 4's use_notifications() raises
    NotificationProviderError instead of quietly handing back a NoopNotifier
    when nothing is mounted yet.
    """
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.Page)
    assert "NotificationProvider()" in src
    assert src.index("NotificationProvider()") < src.index("MapApp.element(")
    assert "LogConsole" not in src


def test_page_sources_locale_from_pysepal():
    """Live language switching is pure wiring on pysepal 4's locale.

    MapApp mounts pysepal's own selector when given ``locales=``; that selector
    writes pysepal's kernel locale and ``use_app_locale()`` makes t() follow it.
    Nothing else asserts on either half, and the fork's ``LocaleState`` wiring
    must not creep back.
    """
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.Page)
    assert "use_app_locale()" in src
    assert "locales=app_available_locales()" in src
    assert "language_selector=" not in src
    assert "resolve_locale_state" not in inspect.getsource(app)


def test_toolbox_is_a_third_left_rail_entry():
    """The Toolbox sits beside Project and Project Summary in the left rail."""
    import inspect

    from gui.solara_app import Page

    src = inspect.getsource(Page.f)
    assert "ToolboxTile" in src
    assert "mdi-toolbox-outline" in src
    assert "app.step_tools" in src


def test_toolbox_is_not_a_workflow_step():
    """The Toolbox is not a workflow step: STEPS and its numbering are untouched."""
    from gui.store.workflow_steps import STEPS

    assert len(STEPS) == 9
    assert all(s.key != "toolbox" for s in STEPS)


def test_page_resets_allocation_jobs_and_density_on_load():
    """Switching projects clears in-flight allocations and their map layers."""
    import inspect

    from gui.solara_app import Page

    src = inspect.getsource(Page.f)
    assert "allocation_jobs.set([])" in src
    assert "density_on_map.set(set())" in src


def test_notification_area_is_gone():
    """The home-grown message bar was replaced by pysepal notifications."""
    import importlib.util

    # find_spec returns None for a missing submodule of an existing package —
    # no dependency on the test runner's working directory.
    assert importlib.util.find_spec("gui.widget.notification_area") is None


def test_solara_test_flag_removed():
    """The legacy dev flag is gone; PYSEPAL_DEV_AUTH replaces it."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "gui" / "solara_app.py"
    content = src.read_text()
    legacy = "SOLARA" + "_TEST"
    assert legacy not in content, f"{legacy} should not appear in source"


def test_pysepal_dev_auth_flag_present():
    """The app uses PYSEPAL_DEV_AUTH for dev mode activation."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "gui" / "solara_app.py"
    content = src.read_text()
    assert "PYSEPAL_DEV_AUTH" in content, "PYSEPAL_DEV_AUTH should appear in source"
    assert (
        "_DEV_AUTH_ARMED" in content
    ), "Module-level guard _DEV_AUTH_ARMED should exist"


def test_prime_dev_auth_guarded():
    """prime_dev_auth() call is guarded by _DEV_AUTH_ARMED to prevent RuntimeError."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "gui" / "solara_app.py"
    content = src.read_text()
    assert "prime_dev_auth()" in content, "prime_dev_auth() call must be present"
    # Verify it appears in a guarded context
    lines = content.split("\n")
    prime_dev_auth_line = None
    for i, line in enumerate(lines):
        if "prime_dev_auth()" in line:
            prime_dev_auth_line = i
            break
    assert prime_dev_auth_line is not None, "prime_dev_auth() not found"
    # Check that the previous lines contain the guard
    context = "\n".join(
        lines[max(0, prime_dev_auth_line - 3) : prime_dev_auth_line + 1]
    )
    assert (
        "_DEV_AUTH_ARMED" in context
    ), "prime_dev_auth() must be guarded by _DEV_AUTH_ARMED"


def test_page_shares_the_scope_theme_state_no_manual_toggle_wiring():
    """v4 owns the theme toggle itself; Page only shares its scope's ThemeState.

    No hand-rolled ThemeToggle memo, no _observe_theme effect mirroring the
    widget into solara.lab.theme — SepalMap and MapApp.element both take
    theme_state= directly, which pysepal 4 keeps in sync on its own.
    """
    import inspect

    import gui.solara_app as app

    src = inspect.getsource(app.Page.f)
    assert "theme_state=theme_state" in src
    assert "theme_toggle" not in src
    assert "_observe_theme" not in src
    assert "ThemeToggle" not in inspect.getsource(app)


def test_conftest_guards_dev_auth():
    """Test isolation: conftest sets PYSEPAL_DEV_AUTH to 0 to prevent logins."""
    import os

    assert (
        os.environ.get("PYSEPAL_DEV_AUTH") == "0"
    ), "conftest should set PYSEPAL_DEV_AUTH=0 for test isolation"
