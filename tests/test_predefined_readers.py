"""Readers of param-dependent catalogue fields go through the resolvers.

Exercised on a synthetic entry whose every dynamic field depends on a
``mode`` choice, injected with ``monkeypatch.setitem`` so the tests do not
depend on which real layer uses the feature.
"""

from pathlib import Path

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.scripts.predefined import PREDEFINED_CATALOGUE  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402
from spatialrisk.variables.models import DataType, RasterType  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEY = "demo_dyn"
MODE = "vars.modal.param_aggregation"
YEAR = "vars.modal.predefined_year_label"


def _by_mode(mask, level):
    return lambda params: mask if params["mode"] == "mask" else level


DEMO = {
    "label_key": "vars.predefined.precipitation",
    "description_key": "vars.predefined_info.precipitation",
    "group": "climate",
    "var_type": "GEEVar",
    "temporal": True,
    "raster_type": _by_mode("categorical", "continuous"),
    "years": _by_mode([2001, 2002], [2010, 2011, 2012]),
    "default_scale": _by_mode(500, 1000),
    "vis_params": _by_mode(
        {"palette": ["ffffff", "000000"], "min": 0, "max": 1},
        {"palette": ["ffffff", "08306b"]},
    ),
    "unit_key": _by_mode("legend.unit.plain_value", "legend.unit.m_value"),
    "params": [
        {
            "key": "mode",
            "label_key": MODE,
            "hint_key": "vars.modal.param_aggregation_hint",
            "type": "choice",
            "default": "mask",
            "options": ["mask", "level"],
            "option_label_key_prefix": "vars.modal.agg_",
            "suffix_prefix": "",
            "omit_default": True,
        }
    ],
    "get_image": lambda aoi, year=None, mode="mask": "IMAGE",
}


@pytest.fixture(autouse=True)
def demo(monkeypatch):
    """Demo."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, KEY, DEMO)


def _local_raster(var_name, raster_type="continuous"):
    """A stand-in LocalRasterVar carrying just ``name`` + ``raster_type``."""
    rt = type("RT", (), {"value": raster_type})()
    return type("LocalRasterVar", (), {"name": var_name, "raster_type": rt})()


@pytest.mark.parametrize(
    "path",
    [
        "gui/tile/variables_tile.py",
        "gui/scripts/legend_data.py",
        "gui/scripts/variable_styles.py",
        "gui/solara_app.py",
    ],
)
def test_no_raw_catalogue_lookups(path):
    """No raw catalogue lookups."""
    src = (ROOT / path).read_text()
    assert "PREDEFINED_CATALOGUE[" not in src
    assert "PREDEFINED_CATALOGUE.get(" not in src


def test_local_style_follows_the_params_in_the_name():
    """Local style follows the params in the name."""
    from gui.scripts.variable_styles import resolve_variable_style

    mask = resolve_variable_style(_local_raster(KEY, "categorical"))
    assert (mask["vmin"], mask["vmax"]) == (0, 1)
    assert mask["render_kind"] == "catalogue_palette"
    level = resolve_variable_style(_local_raster(f"{KEY}_level"))
    assert level["vmin"] is None and level["vmax"] is None
    assert level["render_kind"] == "catalogue_palette"


def test_legend_entry_is_resolved():
    """Legend entry is resolved."""
    from gui.scripts.legend_data import _catalogue_entry

    assert _catalogue_entry(_local_raster(f"{KEY}_level"))["unit_key"] == (
        "legend.unit.m_value"
    )
    assert _catalogue_entry(_local_raster(KEY))["raster_type"] == "categorical"
    assert _catalogue_entry(_local_raster("my_custom_layer")) is None


def test_build_predefined_resolves_type_and_scale(monkeypatch):
    """Build predefined resolves type and scale."""
    import ee

    import gui.scripts.predefined_variables as pv
    from gui.tile.variables_tile import _build_predefined

    monkeypatch.setattr(
        pv, "resolve_aoi_ee", lambda _result: ee.Geometry.__new__(ee.Geometry)
    )

    class _Reactive:
        value = object()

    class _FakeState:
        aoi_result = _Reactive()

    monkeypatch.setattr("gui.store.state_manager.app_state", _FakeState)

    var = _build_predefined(
        {
            "source": "predefined",
            "type": "GEEVar",
            "name": f"{KEY}_level",
            "predefined_key": KEY,
            "params": {"mode": "level"},
            "year": 2010,
            "data_type": DataType.raster,
        },
        None,
    )
    assert var.raster_type == RasterType.continuous
    assert var.default_scale == 1000


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def test_modal_year_list_and_raster_type_follow_the_params():
    """Modal year list and raster type follow the params."""
    added = []
    _box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=added.append,
            initial_entry={
                "source": "predefined",
                "predefined_key": KEY,
                "params": {"mode": "level"},
                "year": 2010,
            },
        ),
        handle_error=False,
    )
    try:
        assert list(_select(rc, YEAR).items) == [2012, 2011, 2010]  # newest first (#35)
        _select(rc, MODE).v_model = "mask"
        assert list(_select(rc, YEAR).items) == [2002, 2001]
        _select(rc, YEAR).v_model = 2002
        submit = t("common.add")
        [b for b in rc.find(vw.Btn).widgets if submit in str(b.children)][0].click()
        assert added[0]["name"] == KEY
        assert added[0]["raster_type"] == RasterType.categorical
    finally:
        rc.close()
