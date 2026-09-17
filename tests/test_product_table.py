"""Unit tests for the shared ProductTable widget's pure helpers."""


def test_action_icons_standardized():
    """Every action kind resolves to its one canonical icon."""
    from gui.widget.product_table import action_icon

    assert action_icon("map_toggle", is_on=False) == "mdi-map-plus"
    assert action_icon("map_toggle", is_on=True) == "mdi-map-minus"
    assert action_icon("edit") == "mdi-pencil-outline"
    assert action_icon("delete") == "mdi-delete-outline"
    assert action_icon("download") == "mdi-cloud-download-outline"
    assert action_icon("cancel") == "mdi-stop-circle"
    # The details action reads as "what is this / how was it made", not
    # "peek at a table" — several of the dialogs it opens show no table.
    assert action_icon("open") == "mdi-information-outline"
    assert action_icon("dismiss") == "mdi-close"


def test_action_colors_are_theme_tokens_never_literal_grey():
    """The off-state map toggle must not be a literal grey.

    "grey darken-1" renders nearly black on the dark theme, which reads as a
    *disabled* button — the icon is enabled and clickable. Returning None lets
    Vuetify use the theme's default text colour (white on dark, dark on light),
    matching the edit/delete icons beside it.
    """
    from gui.widget.product_table import action_color

    assert action_color("map_toggle", is_on=False) is None
    assert action_color("map_toggle", is_on=True) == "primary"
    assert action_color("edit") is None
    assert action_color("delete") is None
    assert action_color("download") == "primary"
    assert action_color("open") == "primary"
    # An explicit per-action override still wins.
    assert action_color("delete", override="error") == "error"


def test_grid_style_builds_template_columns():
    """The column list becomes a CSS grid template."""
    from gui.widget.product_table import grid_style

    style = grid_style(["minmax(0,1fr)", "90px", "112px"])
    assert "display:grid" in style
    assert "grid-template-columns:minmax(0,1fr) 90px 112px;" in style


def test_status_maps_cover_all_states():
    """No job status falls through to the unknown-status placeholder."""
    from gui.widget.product_table import STATUS_COLORS, STATUS_ICONS

    for status in ("running", "ready", "completed", "failed", "cancelled"):
        assert status in STATUS_ICONS
        assert status in STATUS_COLORS
    assert STATUS_ICONS["running"] == "mdi-loading mdi-spin"
    assert STATUS_ICONS["completed"] == "mdi-check-circle"
    # Status tones are Vuetify theme tokens, not literal colours, so they track
    # the app palette in light and dark ("cancelled" stays a neutral grey).
    assert STATUS_COLORS["failed"] == "error"
    assert STATUS_COLORS["ready"] == "success"


def test_product_table_renders_duplicate_row_keys():
    """Rendering must not require unique row keys.

    Rows may share a name (a temporal variable per year, summary rows keyed
    by display name). Regression: reacton raised KeyError "Duplicate key
    'forest_gfc'" on project load once rows were mounted with
    .key(row["key"]).
    """
    import solara

    from gui.i18n import t
    from gui.widget.product_table import ProductTable

    t("common.close")  # prime the catalog: first t() inside a first render
    # corrupts reacton's widget map (known harness artifact)

    rows = [
        {
            "key": "forest_gfc",
            "cells": [{"type": "text", "value": "forest_gfc"}],
            "actions": [],
            "error": None,
        }
        for _ in range(2)
    ]
    box, rc = solara.render(
        ProductTable(
            title="Vars", columns=[{"label": "Name"}], rows=rows, empty_text="none"
        ),
        handle_error=False,
    )
    rc.close()


def test_actions_width_fits_the_widest_row():
    """The Actions column is sized to the busiest row, not to the maximum.

    It used to be a flat 112px — room for four buttons even when a table
    only ever shows two, stealing ~50px from the ellipsised name column
    beside it.
    """
    from gui.widget.product_table import actions_width_for

    two = actions_width_for([{"actions": [1, 2]}, {"actions": [1]}])
    four = actions_width_for([{"actions": [1, 2]}, {"actions": [1, 2, 3, 4]}])
    assert two.endswith("px") and four.endswith("px")
    assert int(two[:-2]) < int(four[:-2])
    # Two x-small icon buttons (20px each) must still fit, with margin to spare.
    assert 48 <= int(two[:-2]) <= 72
    # A table whose rows carry no actions still needs room for the header label.
    assert int(actions_width_for([{"actions": []}])[:-2]) >= 48


def test_name_cell_carries_a_full_text_tooltip():
    """A truncated name must stay readable on hover."""
    from gui.widget.product_table import name_tooltip

    assert name_tooltip({"type": "text", "value": "a_very_long_name"}) == {
        "title": "a_very_long_name"
    }
    assert name_tooltip({"type": "status", "status": "ready"}) == {}


def test_harmonization_statuses_and_action_are_in_the_vocabulary():
    """Step 3's per-variable list reuses the status cell and action column."""
    from gui.i18n import t
    from gui.widget.product_table import (
        STATUS_COLORS,
        STATUS_ICONS,
        action_color,
        action_icon,
    )

    for status in ("harmonized", "pending", "not_downloaded", "checking"):
        assert status in STATUS_ICONS
        assert status in STATUS_COLORS
        assert t(f"widgets.product_table.status_{status}") != (
            f"widgets.product_table.status_{status}"
        )
    assert STATUS_COLORS["harmonized"] == "success"
    assert STATUS_COLORS["pending"] == "warning"
    assert STATUS_COLORS["not_downloaded"] == "warning"
    assert action_icon("harmonize") == "mdi-sync"
    assert action_color("harmonize") == "primary"
