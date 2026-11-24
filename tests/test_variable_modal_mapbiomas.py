"""Selecting a MapBiomas layer pre-selects the region from the AOI country."""

import types

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.store.state_manager import app_state  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402


def _select(rc, label_key):
    label = t(label_key)
    return next(s for s in rc.find(vw.Select).widgets if s.label == label)


@pytest.fixture
def aoi():
    """Set app_state.aoi_result to a stand-in (only method and name are read)."""
    previous = app_state.aoi_result.value

    def _set(method, name):
        app_state.aoi_result.set(
            None if method is None else types.SimpleNamespace(method=method, name=name)
        )

    yield _set
    app_state.aoi_result.set(previous)


def _render(initial_entry=None):
    _box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=lambda entry: None,
            initial_entry=initial_entry,
        ),
        handle_error=False,
    )
    return rc


@pytest.mark.parametrize(
    "method, name, region",
    [
        ("ADMIN1", "COL_Meta", "colombia"),
        ("ADMIN0", "GUY", "amazon"),  # Pan-Amazon fallback
        ("ADMIN1", "PER_Cusco", "peru"),
        ("DRAW", "my polygon", "brazil"),  # review focus 4: no country -> default
        ("ADMIN1", "ARG_Salta", "brazil"),  # no MapBiomas collection -> default
        (None, None, "brazil"),  # no AOI yet
    ],
)
def test_selecting_the_layer_seeds_the_region_from_the_aoi(aoi, method, name, region):
    """The region select follows the AOI country, else the spec default."""
    aoi(method, name)
    rc = _render()
    try:
        _select(rc, "vars.modal.predefined_variable_label").v_model = "mapbiomas_lulc"
        assert _select(rc, "vars.modal.param_mapbiomas_region").v_model == region
    finally:
        rc.close()


def test_editing_keeps_the_stored_region(aoi):
    """Reopening a saved layer never re-seeds from the AOI."""
    aoi("ADMIN1", "COL_Meta")
    rc = _render(
        {
            "source": "predefined",
            "predefined_key": "mapbiomas_lulc",
            "params": {"region": "peru", "mode": "mask", "groups": ["forest"]},
            "year": 2020,
        }
    )
    try:
        assert _select(rc, "vars.modal.param_mapbiomas_region").v_model == "peru"
    finally:
        rc.close()


def test_layers_without_aoi_hooks_keep_their_defaults(aoi):
    """A layer without ``aoi_default`` specs gets its plain catalogue defaults."""
    aoi("ADMIN1", "COL_Meta")
    rc = _render()
    try:
        _select(rc, "vars.modal.predefined_variable_label").v_model = "forest_gfc"
        label = t("vars.modal.param_tree_cover_threshold")
        field = next(f for f in rc.find(vw.TextField).widgets if f.label == label)
        assert field.v_model == "30"
    finally:
        rc.close()
