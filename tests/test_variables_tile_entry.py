"""Regression tests for VariablesTile._variable_to_entry round-tripping."""

from gui.tile.variables_tile import _variable_to_entry
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.models import DataType, RasterType


class _FakeProject:
    base_raster = None


def test_predefined_gee_roundtrips_as_predefined():
    """A catalogue-backed GEEVar (gee_images set, no path) must round-trip.

    As a predefined entry, so editing rebuilds the image instead of producing
    path='None' and failing GEEVar validation.
    """
    var = GEEVar(
        name="altitude",  # a PREDEFINED_CATALOGUE key
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=["dummy"],
    )
    entry = _variable_to_entry("altitude", var, _FakeProject())

    assert entry["source"] == "predefined"
    assert entry["predefined_key"] == "altitude"
    # Must never stringify a missing path into the literal "None".
    assert entry.get("asset_id") != "None"


def test_gee_without_path_does_not_emit_literal_none():
    """Defensive: any GEEVar lacking a path must yield an empty asset_id, not 'None'."""
    var = GEEVar(
        name="custom_layer",  # not in the catalogue
        data_type=DataType.raster,
        gee_images=["dummy"],
    )
    entry = _variable_to_entry("custom_layer", var, _FakeProject())

    assert entry.get("asset_id", "") != "None"


def test_entry_key_matches_add_key_convention():
    """entry_key must predict the storage key on_add computes.

    After building the variable (name_year, or bare name when year is empty)
    so the duplicate-add confirmation fires exactly when the add would
    overwrite.
    """
    from gui.tile.variables_tile import entry_key

    assert entry_key({"name": "forest", "year": "2020"}) == "forest_2020"
    assert entry_key({"name": "altitude", "year": ""}) == "altitude"
    assert entry_key({"name": "altitude"}) == "altitude"


def test_parameterised_predefined_roundtrips_with_params():
    """Editing forest_gfc_tc30 must reopen the modal on the forest_gfc entry.

    With 30 in the threshold field — so predefined_key is the catalogue key,
    name keeps the suffix, and params carries the parsed value.
    """
    var = GEEVar(
        name="forest_gfc_tc30",
        year=2020,
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        gee_images=["dummy"],
    )
    entry = _variable_to_entry("forest_gfc_tc30_2020", var, _FakeProject())

    assert entry["source"] == "predefined"
    assert entry["predefined_key"] == "forest_gfc"
    assert entry["name"] == "forest_gfc_tc30"
    assert entry["params"] == {"tree_cover_threshold": 30}
    assert entry["year"] == "2020"


def test_unparameterised_predefined_roundtrips_with_empty_params():
    """Altitude has no params; the key must still be present and empty.

    So the modal can prefill unconditionally.
    """
    var = GEEVar(
        name="altitude",
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=["dummy"],
    )
    entry = _variable_to_entry("altitude", var, _FakeProject())

    assert entry["predefined_key"] == "altitude"
    assert entry["params"] == {}


def test_build_predefined_forwards_params_to_get_image(monkeypatch):
    """The chosen threshold must reach the catalogue's get_image call.

    The bug this feature fixes is that it was invoked positionally as
    (aoi, year).
    """
    import ee

    import gui.scripts.predefined_variables as pv
    from gui.tile.variables_tile import _build_predefined
    from spatialrisk.variables.models import DataType, RasterType

    seen = {}

    def _fake_get_image(aoi, year=None, **params):
        seen["aoi"] = aoi
        seen["year"] = year
        seen["params"] = params
        return "IMAGE"

    monkeypatch.setitem(
        pv.PREDEFINED_CATALOGUE,
        "forest_gfc",
        {**pv.PREDEFINED_CATALOGUE["forest_gfc"], "get_image": _fake_get_image},
    )
    # Variable.aoi is typed as an ee object and pydantic isinstance-checks it, so
    # a plain string would fail validation. An uninitialized instance satisfies
    # the check without needing an Earth Engine session.
    fake_aoi = ee.Geometry.__new__(ee.Geometry)
    monkeypatch.setattr(pv, "resolve_aoi_ee", lambda _result: fake_aoi)

    class _Reactive:
        value = object()

    class _FakeState:
        aoi_result = _Reactive()

    monkeypatch.setattr("gui.store.state_manager.app_state", _FakeState)

    var = _build_predefined(
        {
            "source": "predefined",
            "type": "GEEVar",
            "name": "forest_gfc_tc45",
            "predefined_key": "forest_gfc",
            "params": {"tree_cover_threshold": 45},
            "year": 2020,
            "data_type": DataType.raster,
            "raster_type": RasterType.categorical,
        },
        None,
    )

    assert seen["params"] == {"tree_cover_threshold": 45}
    assert seen["year"] == 2020
    assert var.name == "forest_gfc_tc45"  # suffix kept: it names the output file


def test_build_predefined_without_params_is_unchanged(monkeypatch):
    """Unparameterised layers must still be called as (aoi, year)."""
    import ee

    import gui.scripts.predefined_variables as pv
    from gui.tile.variables_tile import _build_predefined
    from spatialrisk.variables.models import DataType, RasterType

    seen = {}

    def _fake_get_image(aoi, year=None, **params):
        seen["params"] = params
        return "IMAGE"

    monkeypatch.setitem(
        pv.PREDEFINED_CATALOGUE,
        "altitude",
        {**pv.PREDEFINED_CATALOGUE["altitude"], "get_image": _fake_get_image},
    )
    monkeypatch.setattr(
        pv, "resolve_aoi_ee", lambda _result: ee.Geometry.__new__(ee.Geometry)
    )

    class _Reactive:
        value = object()

    class _FakeState:
        aoi_result = _Reactive()

    monkeypatch.setattr("gui.store.state_manager.app_state", _FakeState)

    _build_predefined(
        {
            "source": "predefined",
            "type": "GEEVar",
            "name": "altitude",
            "predefined_key": "altitude",
            "year": None,
            "data_type": DataType.raster,
            "raster_type": RasterType.continuous,
        },
        None,
    )

    assert seen["params"] == {}


def test_build_predefined_passes_catalogue_scale(monkeypatch):
    """ERA5's ~11 km native scale must reach the GEEVar.

    ``GEEVar.download`` exports at ``default_scale or 30``; without the
    pass-through the ~11 km pixels would be exported at 30 m (~370x
    oversampled files).
    """
    import ee

    import gui.scripts.predefined_variables as pv
    from gui.tile.variables_tile import _build_predefined
    from spatialrisk.variables.models import DataType, RasterType

    monkeypatch.setitem(
        pv.PREDEFINED_CATALOGUE,
        "precipitation",
        {
            **pv.PREDEFINED_CATALOGUE["precipitation"],
            "get_image": lambda aoi, year=None, **params: "IMAGE",
        },
    )
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
            "name": "precipitation",
            "predefined_key": "precipitation",
            "year": 2020,
            "data_type": DataType.raster,
            "raster_type": RasterType.continuous,
        },
        None,
    )

    assert var.default_scale == 11132


def test_build_predefined_without_scale_stays_none(monkeypatch):
    """Entries that declare no default_scale keep today's behaviour."""
    import ee

    import gui.scripts.predefined_variables as pv
    from gui.tile.variables_tile import _build_predefined
    from spatialrisk.variables.models import DataType, RasterType

    monkeypatch.setitem(
        pv.PREDEFINED_CATALOGUE,
        "altitude",
        {
            **pv.PREDEFINED_CATALOGUE["altitude"],
            "get_image": lambda aoi, year=None, **params: "IMAGE",
        },
    )
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
            "name": "altitude",
            "predefined_key": "altitude",
            "year": None,
            "data_type": DataType.raster,
            "raster_type": RasterType.continuous,
        },
        None,
    )

    assert var.default_scale is None


def test_custom_gee_asset_gets_current_aoi(monkeypatch):
    """A custom GEE asset variable must carry the selected AOI.

    ``_download`` clips the export to ``self.aoi.geometry()``; a custom asset
    built without an AOI fails at download time instead of at add time.
    """
    import ee

    import gui.scripts.predefined_variables as predefined
    from gui.store.state_manager import app_state
    from gui.tile.variables_tile import _build_variable
    from spatialrisk import Project

    # Pydantic validates ``aoi`` against ee types and ``project`` against
    # Project; an uninitialised-client-safe stand-in, as elsewhere in this file.
    aoi = ee.Geometry.__new__(ee.Geometry)
    monkeypatch.setattr(app_state.aoi_result, "value", object())
    monkeypatch.setattr(predefined, "resolve_aoi_ee", lambda _r: aoi)
    project = Project(project_name="p")

    var = _build_variable(
        {
            "source": "custom",
            "type": "GEEVar",
            "name": "custom_layer",
            "year": None,
            "path": "projects/p/assets/x",
            "default_scale": None,
            "data_type": DataType.raster,
        },
        project,
    )

    assert var.aoi is aoi
