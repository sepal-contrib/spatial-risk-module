"""Processed-variable layers publish and withdraw legends.

Derived layers reach the registry through an explicit ``LegendPort`` (see
``gui/scripts/legend_registry.py``), never the ``app_state`` singleton — tiles
take reactives and handles as arguments per the project's tile contract.
"""

from gui.tile import derived_map


class _Var:
    """A minimal stand-in for a LocalRasterVar, enough for style resolution."""

    def __init__(self, name="forest_gfc_edge", history=("edge",), year=None):
        """Build a fake processed raster variable."""
        self.name = name
        self.year = year
        self.raster_type = "continuous"
        self.tags = []
        self.processing_history = list(history)
        self.path = "/tmp/x.tif"


def _fake_legend_port():
    """A minimal LegendPort double that records calls.

    Mirrors tests/test_density_map.py and tests/test_variables_legend.py's
    ``_fake_legend_port`` — tiles get an explicit handle, not the app_state
    singleton.
    """
    from gui.scripts.legend_registry import LegendPort

    registered = []
    unregistered = []
    return (
        LegendPort(
            register=lambda *legends: registered.extend(legends),
            unregister=lambda *ids: unregistered.extend(ids),
            generation=lambda: 0,
        ),
        registered,
        unregistered,
    )


def test_derived_legend_is_keyed_by_the_derived_layer_key():
    """_derived_legend keys its LayerLegend by the derived-layer map key."""
    legend = derived_map._derived_legend("proc1", _Var())
    assert legend.layer_id == derived_map.derived_layer_key("proc1")
    assert legend.spec.kind == "gradient"
    assert [dict(label.args).get("value") for label in legend.spec.labels] == [
        "30",
        "1000",
    ]


def test_derived_legend_label_names_the_year():
    """Two years of the same layer on the map get legends that tell them apart."""
    legend = derived_map._derived_legend("forest_gfc_edge_2010", _Var(year=2010))
    assert legend.label.literal == "forest_gfc_edge (2010)"
    assert derived_map._derived_legend("k", _Var()).label.literal == "forest_gfc_edge"


def test_drop_derived_from_map_unregisters_the_legend():
    """drop_derived_from_map removes the layer, the legend, and on-map state."""
    removed = []

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """Record the key removed from the fake map."""
            removed.append(key)

    port, registered, unregistered = _fake_legend_port()
    port.register(derived_map._derived_legend("proc1", _Var()))
    derived_map.derived_on_map.set({"proc1"})

    derived_map.drop_derived_from_map("proc1", FakeMap(), port)

    assert removed == [derived_map.derived_layer_key("proc1")]
    assert unregistered == [derived_map.derived_layer_key("proc1")]
    assert "proc1" not in derived_map.derived_on_map.value


def test_drop_derived_from_map_tolerates_a_missing_port():
    """A None legend_port is a no-op, not a crash — tiles can render without one."""
    removed = []

    class FakeMap:
        def remove_layer(self, key, none_ok=False):
            """Record the key removed from the fake map."""
            removed.append(key)

    derived_map.derived_on_map.set({"proc1"})

    derived_map.drop_derived_from_map("proc1", FakeMap(), None)

    assert removed == [derived_map.derived_layer_key("proc1")]
    assert "proc1" not in derived_map.derived_on_map.value
