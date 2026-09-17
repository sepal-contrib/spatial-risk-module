"""Harmonized and derived variable rows can be deleted.

Both lists render through DerivedVariableList over ``project.processed_variables``
(Harmonization writes one slice, Derived layers the other). Until now neither
tile passed ``on_remove``, so a wrong harmonization or a mistyped derived layer
could only be undone by starting the project over.

Deleting drops the registry entry and its map layer; like the source-variable
delete it does not touch the raster on disk.
"""

import inspect

import pytest


def _src(fn):
    return inspect.getsource(fn)


def test_process_tile_lists_harmonized_vars_with_remove():
    """Harmonized outputs get a remove button that opens the dialog."""
    from gui.tile.process_tile import ProcessTile

    src = _src(ProcessTile)
    assert "on_remove=set_pending_remove" in src  # opens the dialog, never deletes
    assert "ConfirmDialog" in src
    assert "remove_processed_variable" in src  # the layer goes with the entry


def test_postprocess_tile_lists_derived_vars_with_remove():
    """Derived layers get a remove button that opens the dialog."""
    from gui.tile.postprocess_tile import PostProcessTile

    src = _src(PostProcessTile)
    assert "on_remove=set_pending_remove" in src
    assert "ConfirmDialog" in src
    assert "remove_processed_variable" in src


@pytest.mark.parametrize(
    "key",
    [
        "tiles.process.confirm_remove_title",
        "tiles.process.confirm_remove_message",
        "tiles.postprocess.confirm_remove_title",
        "tiles.postprocess.confirm_remove_message",
    ],
)
def test_confirm_copy_resolves(key):
    """Every confirm string resolves to real copy."""
    from gui import i18n

    assert i18n.t(key, name="v1") != key


def test_remove_processed_variable_drops_entry_and_layer(monkeypatch):
    """The shared helper unregisters the variable and forgets its map layer."""
    from gui.scripts import process_actions
    from gui.tile import derived_map

    class _Project:
        def __init__(self):
            self.processed_variables = {"forest_2010": object(), "keep": object()}

    class _Map:
        def __init__(self):
            self.removed = []

        def remove_layer(self, key, none_ok=False):
            self.removed.append(key)

    project, map_ = _Project(), _Map()
    derived_map.derived_on_map.set({"forest_2010", "keep"})

    process_actions.remove_processed_variable(project, "forest_2010", map_)

    assert "forest_2010" not in project.processed_variables
    assert "keep" in project.processed_variables
    assert derived_map.derived_layer_key("forest_2010") in map_.removed
    assert derived_map.derived_on_map.value == {"keep"}


def test_remove_processed_variable_is_a_noop_for_unknown_key():
    """An unknown key changes nothing."""
    from gui.scripts import process_actions

    class _Project:
        processed_variables = {"a": object()}

    process_actions.remove_processed_variable(_Project(), "missing", None)
    assert set(_Project.processed_variables) == {"a"}
