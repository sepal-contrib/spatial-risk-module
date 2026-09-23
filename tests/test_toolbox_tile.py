"""ToolboxTile wiring (browserless, structural)."""

import inspect

import solara

from gui.tile import toolbox_tile


def test_module_exposes_job_and_map_reactives():
    """Both survive re-renders, so the shell can reset them on project switch."""
    assert isinstance(toolbox_tile.allocation_jobs, solara.Reactive)
    assert isinstance(toolbox_tile.density_on_map, solara.Reactive)
    assert toolbox_tile.allocation_jobs.value == []


def test_worker_runs_in_context_with_tracked_job_and_writing():
    """The worker follows the app's background-job contract."""
    src = inspect.getsource(toolbox_tile)
    assert "spawn_in_context" in src
    assert "tracked_job" in src
    assert "writing(" in src
    assert "update_job" in src


def test_tile_takes_project_reactive_not_app_state():
    """Tiles receive the project reactive directly (see the tile contract)."""
    sig = inspect.signature(toolbox_tile.ToolboxTile.f)
    assert list(sig.parameters)[0] == "project"
    assert "app_state" not in inspect.getsource(toolbox_tile)


def test_tile_mounts_with_a_project():
    """The tile renders end to end without a map or a client."""
    import reacton

    from gui.i18n import t
    from spatialrisk.project import Project

    t("common.cancel")  # warm the translator before the first render
    box, _rc = reacton.render(
        toolbox_tile.ToolboxTile(project=solara.reactive(Project(project_name="p")))
    )
    assert box is not None


# --- two-pane shell + latest-run card (mock fidelity) -------------------


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    children = list(getattr(widget, "children", []) or [])
    for slot in getattr(widget, "v_slots", None) or []:
        children.extend(slot.get("children", []) or [])
    for child in children:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _all_text(box):
    import ipyvuetify as vw

    out = []
    for cls in (vw.Html, vw.Btn, vw.Chip):
        for w in _find(box, cls):
            out.extend(str(c) for c in (w.children or []) if isinstance(c, str))
    return " ".join(out)


def test_tile_has_a_tool_registry():
    """Future tools are one registry entry away — the mock's tool list."""
    tools = toolbox_tile._TOOLS
    assert [tool["key"] for tool in tools] == ["allocation"]
    assert all("label_key" in tool and "icon" in tool for tool in tools)


def test_tile_renders_the_tool_list_pane():
    """The dialog shows the tool list beside the selected tool's panel."""
    from _notification_host import render_under_notifications

    from gui.i18n import t
    from spatialrisk.project import Project

    t("common.cancel")
    box, rc = render_under_notifications(
        lambda: toolbox_tile.ToolboxTile(
            project=solara.reactive(Project(project_name="p"))
        )
    )
    try:
        assert t("toolbox.tool_allocation") in _all_text(box)
    finally:
        rc.close()


def test_tile_has_no_latest_run_card():
    """Runs live in the table only; the headline card is gone (2026-07-30 spec)."""
    from _notification_host import render_under_notifications

    from gui.i18n import t
    from spatialrisk.allocations import AllocationRun
    from spatialrisk.project import Project

    t("common.cancel")
    project = Project(project_name="p")
    project.allocations["reserve_bbb22222"] = AllocationRun(
        name="reserve",
        run_id="bbb22222",
        created_at="2026-07-29T10:00:00",
        borders_file="/b.gpkg",
        defor_juris_ha=20000.0,
        years_forecast=4,
        annual_ha=312.4,
        total_ha=1249.6,
        out_dir="/out",
        csv_path="/out/defor_project.csv",
    )

    box, rc = render_under_notifications(
        lambda: toolbox_tile.ToolboxTile(project=solara.reactive(project))
    )

    try:
        assert "312.4" in _all_text(box)  # the run still renders, in the table
        src = inspect.getsource(toolbox_tile)
        assert "latest_result" not in src
        assert "AllocationResultCard" not in src
    finally:
        rc.close()


def test_body_has_no_duplicated_heading():
    """The body repeats neither the dialog's heading nor its subtitle.

    The dialog frame already says 'Tools'. (The tool's own description DOES
    render, under the panel-header title — see the header test.)
    """
    src = inspect.getsource(toolbox_tile)
    assert "InfoButton" in src
    assert "toolbox.allocation.description" in src  # the under-title blurb
    assert "toolbox.title" not in src  # no duplicated heading
    assert "toolbox.subtitle" not in src


def _rail_button(box):
    """The rail's tool button: the Btn whose icon is the tool's mdi icon."""
    import ipyvuetify as vw

    for btn in _find(box, vw.Btn):
        icons = [str(i.children[0]) for i in _find(btn, vw.Icon) if i.children]
        if toolbox_tile._TOOLS[0]["icon"] in icons:
            return btn
    return None


def test_rail_icon_exists_in_the_bundled_mdi_font():
    """jupyter-vuetify ships MDI ~4.9; a 5.x glyph renders as an empty button.

    mdi-earth-remove did exactly that (2026-07-30), so the registry is pinned
    to glyphs known to exist in the 4.x font.
    """
    known_4x = {"mdi-earth", "mdi-earth-off", "mdi-pine-tree", "mdi-tree", "mdi-axe"}
    for tool in toolbox_tile._TOOLS:
        assert tool["icon"] in known_4x, (
            f"{tool['icon']} is not on the known-4.x allowlist; verify it exists "
            "in the bundled MDI font before using it (see this test's docstring)"
        )


def test_rail_is_icon_only_with_primary_selection():
    """The tool rail mirrors the app drawer: icon button, primary when active."""
    from _notification_host import render_under_notifications

    from gui.i18n import t
    from spatialrisk.project import Project

    t("common.cancel")
    box, rc = render_under_notifications(
        lambda: toolbox_tile.ToolboxTile(
            project=solara.reactive(Project(project_name="p"))
        )
    )

    try:
        btn = _rail_button(box)
        assert btn is not None
        assert btn.icon  # icon-only, no text label on the button
        assert btn.color == "primary"  # the (only) tool is selected
        assert not [c for c in (btn.children or []) if isinstance(c, str) and c.strip()]
    finally:
        rc.close()


def test_panel_header_carries_title_description_and_info_button():
    """Header = tool title + description under it; details live in the popup."""
    import ipyvuetify as vw
    from _notification_host import render_under_notifications

    from gui.i18n import t
    from spatialrisk.project import Project

    t("common.cancel")
    box, rc = render_under_notifications(
        lambda: toolbox_tile.ToolboxTile(
            project=solara.reactive(Project(project_name="p"))
        )
    )

    try:
        # Rail buttons announce their tool name via a tooltip pinned to the
        # RIGHT: the rail hugs the dialog's left edge, so a bottom tooltip
        # clips there.
        tooltips = _find(box, vw.Tooltip)
        tool_name = t("toolbox.tool_allocation")
        assert len(tooltips) > 0, "No tooltips rendered"
        tooltip_texts = []
        for tooltip in tooltips:
            if tooltip.children:
                tooltip_texts.extend([str(c) for c in tooltip.children])
        assert tool_name in tooltip_texts, f"{tool_name} not in tooltip texts"
        assert any(tt.right for tt in tooltips), "rail tooltip must open to the right"

        # The header title AND the one-line description both render in the panel.
        flat = _all_text(box)
        assert tool_name in flat
        assert t("toolbox.allocation.description") in flat

        # The info popup carries the method details/references, not the short blurb.
        src = inspect.getsource(toolbox_tile)
        assert "InfoButton" in src
        assert 'markdown=t(tool["info_key"])' in src
    finally:
        rc.close()


def test_tile_mounts_the_details_dialog():
    """Row clicks land somewhere: the tile owns the selected key + dialog."""
    src = inspect.getsource(toolbox_tile)
    assert "AllocationDetailsDialog" in src
    assert "on_open" in src


def test_form_name_suggestion_skips_in_flight_jobs():
    """A running job's name is taken, so the form opens on allocation_2."""
    import ipyvuetify as vw
    from _notification_host import render_under_notifications

    from gui.i18n import t
    from spatialrisk.project import Project

    t("common.cancel")  # warm the translator before the first render
    previous = toolbox_tile.allocation_jobs.value
    toolbox_tile.allocation_jobs.set(
        [{"id": "j1", "name": "allocation_1", "status": "running", "error": None}]
    )
    try:
        box, rc = render_under_notifications(
            lambda: toolbox_tile.ToolboxTile(
                project=solara.reactive(Project(project_name="p"))
            )
        )
        try:
            label = t("toolbox.allocation.field_name")
            fields = [f for f in _find(box, vw.TextField) if f.label == label]
            assert fields, "the Run name field did not render"
            assert fields[0].v_model == "allocation_2"
        finally:
            rc.close()
    finally:
        toolbox_tile.allocation_jobs.set(previous)
