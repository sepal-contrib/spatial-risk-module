"""The default display style of each catalogue layer and the override diff."""

import pytest
from _catalogue_checks import _combos

from gui.scripts.predefined import (
    PREDEFINED_CATALOGUE,
    default_vis,
    is_editable,
    resolved_entry,
    vis_override,
)
from gui.scripts.predefined.forest_change import LOSS_PALETTE

KEYS = sorted(PREDEFINED_CATALOGUE)


@pytest.mark.parametrize("key", KEYS)
def test_every_entry_has_a_default_or_is_random(key):
    """Every combination gives a style, or None only for a random visualizer."""
    for values in _combos(key):
        style = default_vis(key, values)
        if style is None:
            assert resolved_entry(key, values).get("random_visualizer"), (key, values)


def test_only_subj_is_not_editable():
    """Only the random-visualizer layer is not editable."""
    assert {k for k in KEYS if not is_editable(k, {})} == {"subj"}


def test_mask_default_uses_the_legend_class_keys():
    """A 0/1 mask is two classes labelled by the entry's legend keys."""
    classes = default_vis("deforestation_gfc", {})["classes"]
    assert [c["value"] for c in classes] == [0, 1]
    assert [c["color"] for c in classes] == LOSS_PALETTE
    assert tuple(c["label_key"] for c in classes) == (
        "legend.class.stable",
        "legend.class.loss",
    )


def test_class_labels_default():
    """A class table gives one class per label, each with its label_key."""
    params = {"mode": "groups"}
    entry = resolved_entry("mapbiomas_lulc", params)
    classes = default_vis("mapbiomas_lulc", params)["classes"]
    assert [c["value"] for c in classes] == sorted(
        c["value"] for c in entry["class_labels"]
    )
    by_value = {c["value"]: c for c in entry["class_labels"]}
    for c in classes:
        assert c["color"] == by_value[c["value"]]["color"].lstrip("#").lower()
        assert c["label_key"] == by_value[c["value"]]["label_key"]


def test_continuous_default_keeps_the_pinned_range():
    """A pinned range survives; an auto-stretched layer has a palette only."""
    slope = default_vis("slope", {})
    assert {"palette", "min", "max"} <= set(slope)
    precipitation = default_vis("precipitation", {})
    assert precipitation["palette"] and not {"min", "max"} & set(precipitation)


def test_override_is_none_when_unchanged():
    """The default, or the same colours spelled differently, stores nothing."""
    default = default_vis("deforestation_gfc", {})
    assert vis_override(default, default) is None
    shouted = {
        "classes": [
            {**c, "color": "#" + c["color"].upper()} for c in default["classes"]
        ]
    }
    assert vis_override(default, shouted) is None


def test_override_keeps_a_real_change():
    """One recoloured class is stored whole."""
    default = default_vis("deforestation_gfc", {})
    current = {
        "classes": [{**default["classes"][0], "color": "00ff00"}, default["classes"][1]]
    }
    assert vis_override(default, current) == current
