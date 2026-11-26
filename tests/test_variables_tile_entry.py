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
    assert var.params == {"tree_cover_threshold": 45}  # kept for the Details dialog


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


def test_custom_gee_asset_carries_raster_type(monkeypatch):
    """The modal's raster-type pick for a GEE asset must reach the GEEVar.

    ``to_local_raster`` refuses to convert without one ("raster_type must be
    provided ..."), so dropping it turns every custom asset download into an
    error after the export has already succeeded.
    """
    import ee

    import gui.scripts.predefined_variables as predefined
    from gui.store.state_manager import app_state
    from gui.tile.variables_tile import _build_variable
    from spatialrisk import Project

    monkeypatch.setattr(app_state.aoi_result, "value", object())
    monkeypatch.setattr(
        predefined, "resolve_aoi_ee", lambda _r: ee.Geometry.__new__(ee.Geometry)
    )

    var = _build_variable(
        {
            "source": "custom",
            "type": "GEEVar",
            "name": "custom_layer",
            "year": None,
            "path": "projects/p/assets/x",
            "default_scale": None,
            "data_type": DataType.raster,
            "raster_type": RasterType.categorical,
        },
        Project(project_name="p"),
    )

    assert var.raster_type == RasterType.categorical


def test_custom_gee_asset_roundtrips_raster_type():
    """Editing a custom GEE asset must prefill the raster type it was saved with."""
    var = GEEVar(
        name="custom_layer",
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        path="projects/p/assets/x",
    )
    entry = _variable_to_entry("custom_layer", var, _FakeProject())

    assert entry["source"] == "custom"
    assert entry["raster_type"] == "categorical"


def test_local_raster_band_becomes_single_band_vrt(tmp_path, monkeypatch):
    """A band picked from a multi-band file is exposed as a one-band VRT.

    Every reader takes band 1, so the layer's path must be a file holding only
    the picked band; editing it shows the original file and band again.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from gui.tile.variables_tile import _build_variable
    from spatialrisk import Project

    src = tmp_path / "multi.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 3, 1, 1),
    ) as dst:
        for b in (1, 2, 3):
            dst.write(np.full((3, 4), b * 10, dtype="uint8"), b)
    monkeypatch.setenv("SPATIAL_RISK_DATA_DIR", str(tmp_path / "data"))
    project = Project(project_name="p")

    var = _build_variable(
        {
            "source": "custom",
            "type": "LocalRasterVar",
            "name": "multi",
            "year": None,
            "path": src,
            "band": 2,
            "raster_type": RasterType.continuous,
            "data_type": DataType.raster,
        },
        project,
    )

    with rasterio.open(var.path) as ds:
        assert ds.count == 1
        assert ds.read(1).max() == 20
    entry = _variable_to_entry("multi", var, _FakeProject())
    assert entry["path"] == str(src)
    assert entry["band"] == 2


def _project(tmp_path, monkeypatch):
    from spatialrisk import Project

    monkeypatch.setenv("SPATIAL_RISK_DATA_DIR", str(tmp_path / "data"))
    return Project(project_name="p")


def test_downloaded_catalogue_layer_reopens_as_predefined(tmp_path, monkeypatch):
    """A downloaded catalogue layer edits as predefined, not as a custom file."""
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    project = _project(tmp_path, monkeypatch)
    name = "deforestation_gfc_tc30_2015_2020"
    var = LocalRasterVar(
        name=name,
        # Where GEEVar._download writes it.
        path=project.folders.data_raw_folder / f"{name}.tif",
        raster_type=RasterType.categorical,
        project=project,
        # Copied from the GEEVar by to_local_raster: marks a catalogue layer.
        params={"tree_cover_threshold": 30, "start_year": 2015, "end_year": 2020},
    )
    entry = _variable_to_entry(var.name, var, project)

    assert entry["source"] == "predefined"
    assert entry["type"] == "GEEVar"
    assert entry["predefined_key"] == "deforestation_gfc"
    assert entry["params"] == {
        "tree_cover_threshold": 30,
        "start_year": 2015,
        "end_year": 2020,
    }
    assert entry["year"] == ""
    assert "vis_params" not in entry  # no override: the catalogue look applies


def test_catalogue_named_file_at_a_user_path_stays_custom(tmp_path, monkeypatch):
    """A user's own slope.tif is not a download: Save must keep its file."""
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    project = _project(tmp_path, monkeypatch)
    user_file = tmp_path / "mine" / "slope.tif"
    var = LocalRasterVar(
        name="slope",
        path=user_file,
        raster_type=RasterType.continuous,
        project=project,
    )
    entry = _variable_to_entry("slope", var, project)

    assert entry["source"] == "custom"
    assert entry["path"] == str(user_file)


def test_catalogue_named_file_without_project_stays_custom():
    """No project to locate the download folder: the safe side is custom."""
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    var = LocalRasterVar(
        name="slope", path="/tmp/slope.tif", raster_type=RasterType.continuous
    )
    assert _variable_to_entry("slope", var, _FakeProject())["source"] == "custom"


def test_override_round_trips(tmp_path, monkeypatch):
    """A saved display override survives build -> download -> reopen."""
    import ee

    import gui.scripts.predefined_variables as pv
    from gui.tile.variables_tile import _build_predefined

    monkeypatch.setitem(
        pv.PREDEFINED_CATALOGUE,
        "forest_gfc",
        {
            **pv.PREDEFINED_CATALOGUE["forest_gfc"],
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

    project = _project(tmp_path, monkeypatch)
    vis = {"palette": ["000000", "ff0000"], "min": 0, "max": 1}
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
            "vis_params": vis,
        },
        project,
    )
    assert var.vis_params == vis

    entry = _variable_to_entry("forest_gfc_tc45_2020", var, project)
    assert entry["source"] == "predefined"
    assert entry["params"] == {"tree_cover_threshold": 45}
    assert entry["vis_params"] == vis

    # The downloaded copy (GEEVar.to_local_raster) keeps both.
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    local = LocalRasterVar(
        name=var.name,
        path=var.expected_local_path,
        raster_type=RasterType.categorical,
        project=project,
        year=2020,
        params=var.params,
        vis_params=var.vis_params,
    )
    entry = _variable_to_entry("forest_gfc_tc45_2020", local, project)
    assert entry["source"] == "predefined"
    assert entry["params"] == {"tree_cover_threshold": 45}
    assert entry["vis_params"] == vis
    assert entry["year"] == "2020"


def test_custom_layer_named_like_nothing_stays_custom():
    """A local raster whose name is no catalogue layer keeps the custom branch."""
    from spatialrisk.variables.local_raster_var import LocalRasterVar

    var = LocalRasterVar(
        name="my_dem",
        path="/tmp/my_dem.tif",
        raster_type=RasterType.continuous,
    )
    entry = _variable_to_entry("my_dem", var, _FakeProject())

    assert entry["source"] == "custom"
    assert entry["type"] == "LocalRasterVar"
    assert entry["path"] == "/tmp/my_dem.tif"
