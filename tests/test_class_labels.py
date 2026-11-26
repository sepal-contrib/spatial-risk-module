"""Multi-class legends from a catalogue entry's ``class_labels`` (MapBiomas uses them).

The table is drawn through the same class-list machinery as a user's custom
classes (``gui.scripts.variable_classes``): GEE remaps the class values onto
the palette, local rasters go through the class VRT.
"""

import types

import pytest

from gui.scripts.legend_data import (
    Label,
    variable_spec_from_style,
    variable_spec_from_vis,
)
from gui.scripts.predefined import DYNAMIC_ATTRS, PREDEFINED_CATALOGUE, resolved_entry
from gui.scripts.variable_classes import from_class_labels
from gui.scripts.variable_styles import resolve_variable_style

TMF_COLORS = ["005a00", "64953c", "ffb432", "d2f064", "0064ff", "d2d2d2"]
KEY = "demo_classes"
TABLE = [
    {"value": v, "label_key": f"legend.class.demo_{v}", "color": c}
    for v, c in zip(range(1, 7), TMF_COLORS)
]


@pytest.fixture(autouse=True)
def demo(monkeypatch):
    """A six-class catalogue entry (the former JRC TMF state table)."""
    monkeypatch.setitem(
        PREDEFINED_CATALOGUE,
        KEY,
        {
            "raster_type": "categorical",
            "temporal": True,
            "years": [2020],
            "get_image": None,
            "class_labels": TABLE,
        },
    )


def _var(name=KEY, raster_type="categorical"):
    rt = type("RT", (), {"value": raster_type})()
    return types.SimpleNamespace(name=name, raster_type=rt, vis_params=None)


def test_class_labels_is_a_dynamic_attr():
    """D's resolved_entry resolves it like legend_class_keys (E relies on this)."""
    assert DYNAMIC_ATTRS[-1] == "class_labels"


def test_callable_class_labels_are_resolved(monkeypatch):
    """A params-dependent table comes back as a plain list."""
    table = [{"value": 7, "label_key": "legend.class.forest", "color": "00ff00"}]
    monkeypatch.setitem(PREDEFINED_CATALOGUE[KEY], "class_labels", lambda params: table)
    assert resolved_entry(KEY, {})["class_labels"] == table


def test_from_class_labels_sorts_and_keeps_the_label_key():
    """Non-contiguous codes (E's MapBiomas) keep their own colours, in value order."""
    classes = from_class_labels(
        [
            {"value": 30, "label_key": "a", "color": "ff0000"},
            {"value": 3, "label_key": "b", "color": "#0000ff"},
        ]
    )
    assert [(c["value"], c["color"], c["label"]) for c in classes] == [
        (3, "0000ff", "b"),
        (30, "ff0000", "a"),
    ]


def test_local_style_draws_the_class_table_through_the_class_vrt():
    """Same shape as a user class list: variable_map builds the VRT from it."""
    style = resolve_variable_style(_var())
    assert style["render_kind"] == "class_palette"
    assert [c["value"] for c in style["classes"]] == [1, 2, 3, 4, 5, 6]
    assert [c["color"] for c in style["classes"]] == TMF_COLORS
    assert (style["vmin"], style["vmax"], style["nodata"]) == (1, 6, 0)


class _RemapImage:
    """Records ``remap`` and returns a marker image."""

    def __init__(self):
        self.calls = []

    def remap(self, from_values, to_values):
        self.calls.append((list(from_values), list(to_values)))
        return "REMAPPED"


def test_gee_style_remaps_classes_onto_a_pinned_palette():
    """Remap to 0..n-1 so any class codes map exactly onto the palette."""
    from gui.tile.variables_tile import _styled_layer

    image = _RemapImage()
    styled, vis, kind = _styled_layer(image, _var(), None)
    assert kind == "class_palette"
    assert styled == "REMAPPED"
    assert image.calls == [([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5])]
    assert vis == {"palette": TMF_COLORS, "min": 0, "max": 5}


def test_legend_lists_every_class_with_its_colour():
    """GEE and local legends: one chip per class, in class order."""
    title = Label(literal="TMF")
    vis = {"palette": TMF_COLORS, "min": 0, "max": 5}
    for spec in (
        variable_spec_from_vis(vis, "class_palette", _var(), title),
        variable_spec_from_style(resolve_variable_style(_var()), _var(), title),
    ):
        assert spec.kind == "chips"
        assert spec.colors == tuple(f"#{c}" for c in TMF_COLORS)
        assert [label.key for label in spec.labels] == [c["label_key"] for c in TABLE]
