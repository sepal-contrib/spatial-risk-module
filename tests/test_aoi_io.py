"""AOI persistence helpers."""

import json
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from pysepal.solara.components.aoi import AoiResult, AoiSpec
from shapely.geometry import box

import spatialrisk.project as proj
from gui.scripts.aoi_io import (
    AOI_GEOMETRY_FILENAME,
    admin_code_chain,
    attach_aoi,
    load_aoi,
    persist_aoi,
    write_aoi,
)

# The legacy ``"asset"`` manifest dict (pre-AoiSpec manifests carry only this).
LEGACY_ASSET = {
    "asset_id": "users/me/aoi",
    "type": "TABLE",
    "column": "ALL",
    "value": None,
}


def _gdf(bounds=(12.40, 43.89, 12.52, 43.99)):
    return gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[box(*bounds)], crs="EPSG:4326")


def _aoi(method="DRAW", name="san_marino", gee=True, admin=None, gdf=None, spec=None):
    return SimpleNamespace(
        method=method, name=name, gee=gee, admin=admin, gdf=gdf, spec=spec
    )


def _draw_spec(name="san_marino", gdf=None):
    """A DRAW spec shaped like the one ``process_draw`` records (JSON lists)."""
    gdf = _gdf() if gdf is None else gdf
    return AoiSpec(method="DRAW", name=name, geo_json=json.loads(gdf.to_json()))


def _asset_spec():
    return AoiSpec(
        method="ASSET", asset_id="users/me/aoi", asset_type="TABLE", column="ALL"
    )


def _gaul_names(chains):
    """Stand in for ``pygaul.Names(admin=..., complete=True)`` offline.

    ``chains`` maps a leaf code to its level-0..n codes; an unknown code raises
    the ``ValueError`` pygaul raises for a code outside GAUL 2024.
    """

    def names(admin="", complete=False, **_):
        assert complete, "the chain lookup needs every level's columns"
        if admin not in chains:
            raise ValueError(f'The requested "{admin}" is not part of FAO GAUL 2024.')
        row = {f"gaul{lvl}_code": code for lvl, code in enumerate(chains[admin])}
        return pd.DataFrame([row])

    return names


@pytest.fixture
def draw_aoi_result():
    """A v4 DRAW selection: geometry plus the spec that produced it."""
    gdf = _gdf()
    return AoiResult(
        method="DRAW", name="san_marino", gdf=gdf, gee=False, spec=_draw_spec(gdf=gdf)
    )


@pytest.fixture
def legacy_admin_manifest(monkeypatch):
    """A pre-AoiSpec ADMIN2 manifest; pygaul resolves its chain offline."""
    import pygaul

    monkeypatch.setattr(
        pygaul, "Names", _gaul_names({"100001": ("101", "1001", "100001")})
    )
    return {"method": "ADMIN2", "name": "ABC_x_y", "gee": False, "admin": "100001"}


# --- write_aoi --------------------------------------------------------------


def test_vector_aoi_writes_sidecar_and_metadata(tmp_path):
    """Vector AOI with geometry writes both sidecar and metadata."""
    meta = write_aoi(tmp_path, _aoi(gdf=_gdf()))

    digest = meta.pop("geometry_digest", None)
    assert meta == {
        "method": "DRAW",
        "name": "san_marino",
        "gee": True,
        "admin": None,
        "geometry_file": AOI_GEOMETRY_FILENAME,
    }
    assert isinstance(digest, str) and digest
    assert (tmp_path / AOI_GEOMETRY_FILENAME).exists()


def test_geometryless_aoi_writes_metadata_only(tmp_path):
    """Geometry-less AOI writes metadata only, no sidecar."""
    meta = write_aoi(
        tmp_path, _aoi(method="ADMIN1", name="COL_x", admin="21758", gdf=None)
    )

    assert meta == {"method": "ADMIN1", "name": "COL_x", "gee": True, "admin": "21758"}
    assert "geometry_file" not in meta
    assert not (tmp_path / AOI_GEOMETRY_FILENAME).exists()


def test_none_aoi_returns_none_and_removes_stale_sidecar(tmp_path):
    """None AOI returns None and removes any stale sidecar."""
    write_aoi(tmp_path, _aoi(gdf=_gdf()))
    assert (tmp_path / AOI_GEOMETRY_FILENAME).exists()

    assert write_aoi(tmp_path, None) is None
    assert not (tmp_path / AOI_GEOMETRY_FILENAME).exists()


def test_geometryless_resave_removes_stale_sidecar(tmp_path):
    """Re-saving a geometry-less AOI removes any stale sidecar from before."""
    write_aoi(tmp_path, _aoi(gdf=_gdf()))
    assert (tmp_path / AOI_GEOMETRY_FILENAME).exists()

    # Re-saving with a geometry-less AOI must not leave a dangling pointer.
    write_aoi(tmp_path, _aoi(method="ASSET", gdf=None))
    assert not (tmp_path / AOI_GEOMETRY_FILENAME).exists()


def test_non_wgs84_geometry_is_reprojected(tmp_path):
    """Non-WGS84 geometry is reprojected to EPSG:4326 for the sidecar."""
    # A box in Web Mercator metres near San Marino.
    gdf = gpd.GeoDataFrame(
        {"name": ["aoi"]},
        geometry=[box(1_380_000, 5_440_000, 1_395_000, 5_460_000)],
        crs="EPSG:3857",
    )
    write_aoi(tmp_path, _aoi(gdf=gdf))

    written = gpd.read_file(tmp_path / AOI_GEOMETRY_FILENAME)
    assert written.crs.to_epsg() == 4326
    minx, miny, maxx, maxy = written.total_bounds
    assert 10 < minx < 14 and 43 < miny < 45  # plausible lon/lat, not metres


# --- load_aoi ---------------------------------------------------------------


def test_load_roundtrips_geometry_and_metadata(tmp_path):
    """Writing and loading an AOI preserves geometry and metadata."""
    meta = write_aoi(tmp_path, _aoi(gdf=_gdf()))
    restored = load_aoi(tmp_path, meta)

    assert restored.method == "DRAW"
    assert restored.name == "san_marino"
    assert restored.gee is True
    assert restored.gdf is not None
    assert restored.gdf.total_bounds == pytest.approx(
        [12.40, 43.89, 12.52, 43.99], abs=1e-6
    )


def test_load_metadata_only_aoi_has_no_geometry(tmp_path):
    """Metadata-only AOI loads without a geometry GeoDataFrame."""
    meta = write_aoi(tmp_path, _aoi(method="ADMIN1", admin="21758", gdf=None))
    restored = load_aoi(tmp_path, meta)

    assert restored is not None
    assert restored.method == "ADMIN1"
    assert restored.admin == "21758"
    assert restored.gdf is None


def test_load_gee_admin_rebuilds_feature_collection(tmp_path, monkeypatch):
    """A GEE admin AOI persists only its GAUL ``admin`` code (no sidecar).

    Loading must rebuild the lazy EE FeatureCollection that selection produced
    (``pygaul.Items(admin=...)``) so the restored AOI is usable downstream
    (``resolve_aoi_ee``) without the user re-selecting the area.
    """
    import pygaul

    sentinel = object()
    captured = {}

    def fake_items(admin):
        captured["admin"] = admin
        return sentinel

    monkeypatch.setattr(pygaul, "Items", fake_items)

    meta = write_aoi(tmp_path, _aoi(method="ADMIN0", name="GUY", admin="197", gdf=None))
    restored = load_aoi(tmp_path, meta)

    assert restored.gdf is None
    assert restored.feature_collection is sentinel
    assert captured["admin"] == "197"


def test_load_gee_admin_degrades_when_rebuild_fails(tmp_path, monkeypatch):
    """If the FeatureCollection rebuild fails, loading degrades gracefully.

    When the rebuild fails (EE not ready, offline), loading degrades to a
    metadata-only AOI instead of raising.
    """
    import pygaul

    def boom(admin):
        raise RuntimeError("Earth Engine not initialized")

    monkeypatch.setattr(pygaul, "Items", boom)

    meta = write_aoi(tmp_path, _aoi(method="ADMIN0", name="GUY", admin="197", gdf=None))
    restored = load_aoi(tmp_path, meta)

    assert restored is not None
    assert restored.admin == "197"
    assert restored.feature_collection is None


def test_load_none_metadata_returns_none(tmp_path):
    """Loading with None metadata returns None."""
    assert load_aoi(tmp_path, None) is None


def test_load_tolerates_missing_sidecar(tmp_path):
    """Loading tolerates a missing geometry sidecar referenced in metadata."""
    # Manifest references a sidecar that was deleted/moved — degrade, don't crash.
    meta = {
        "method": "DRAW",
        "name": "x",
        "gee": False,
        "geometry_file": AOI_GEOMETRY_FILENAME,
    }
    restored = load_aoi(tmp_path, meta)

    assert restored is not None
    assert restored.gdf is None


# --- Project.aoi field round-trip ------------------------------------------


def test_project_persists_aoi_metadata(tmp_path, monkeypatch):
    """Project.aoi metadata round-trips through save/load."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)

    p = proj.Project(project_name="proj_with_aoi")
    p.aoi = {
        "method": "DRAW",
        "name": "x",
        "gee": True,
        "admin": None,
        "geometry_file": AOI_GEOMETRY_FILENAME,
    }
    p.save()

    loaded = proj.Project.load("proj_with_aoi")
    assert loaded.aoi == p.aoi


def test_project_persists_asset_aoi_metadata(tmp_path, monkeypatch):
    """Project persists and restores ASSET AOI metadata."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)

    p = proj.Project(project_name="proj_asset_aoi")
    p.aoi = {
        "method": "ASSET",
        "name": "aoi",
        "gee": True,
        "admin": None,
        "asset": {
            "asset_id": "users/me/aoi",
            "type": "TABLE",
            "column": "ALL",
            "value": None,
        },
    }
    p.save()

    loaded = proj.Project.load("proj_asset_aoi")
    assert loaded.aoi == p.aoi


def test_project_without_aoi_loads_none(tmp_path, monkeypatch):
    """Project without saved AOI loads None."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)

    proj.Project(project_name="proj_no_aoi").save()

    loaded = proj.Project.load("proj_no_aoi")
    assert loaded.aoi is None


# --- ASSET AOI persist + rebuild --------------------------------------------


def test_write_includes_asset_for_asset_method(tmp_path):
    """An ASSET AOI still writes the legacy ``asset`` dict next to its spec.

    The picker inputs now live on ``AoiResult.spec``; the legacy key is kept
    for one release so an older app version still reads the manifest. It must
    be the full dict, not ``spec.asset_id``: the rebuild does
    ``(asset or {}).get(...)``, which a bare string would break silently.
    """
    meta = write_aoi(
        tmp_path, _aoi(method="ASSET", name="aoi", gdf=None, spec=_asset_spec())
    )

    assert meta["asset"] == LEGACY_ASSET
    assert meta["aoi_spec"]["asset_id"] == "users/me/aoi"


def test_write_omits_asset_for_non_asset_method(tmp_path):
    """Write omits asset dict for non-ASSET method AOIs."""
    spec = AoiSpec(method="ADMIN0", admin_codes=("197",))
    meta = write_aoi(
        tmp_path, _aoi(method="ADMIN0", name="GUY", admin="197", gdf=None, spec=spec)
    )
    assert "asset" not in meta


def test_load_restores_the_asset_spec(tmp_path):
    """The restored ASSET AOI carries its picker inputs on ``spec``."""
    meta = write_aoi(
        tmp_path, _aoi(method="ASSET", name="aoi", gdf=None, spec=_asset_spec())
    )
    restored = load_aoi(tmp_path, meta)

    assert restored.spec == _asset_spec()
    assert restored.spec.asset_data() == LEGACY_ASSET


def test_load_asset_rebuilds_from_the_spec_alone(tmp_path, monkeypatch):
    """A manifest carrying only ``aoi_spec`` (no legacy key) still rebuilds."""
    import ee

    sentinel = object()
    captured = {}
    monkeypatch.setattr(
        ee, "FeatureCollection", lambda aid: captured.update({"aid": aid}) or sentinel
    )
    meta = {
        "method": "ASSET",
        "name": "aoi",
        "gee": True,
        "aoi_spec": _asset_spec().to_dict(),
    }

    restored = load_aoi(tmp_path, meta)

    assert restored.feature_collection is sentinel
    assert captured["aid"] == "users/me/aoi"


def test_load_asset_rebuilds_feature_collection(tmp_path, monkeypatch):
    """Load rebuilds the feature collection from asset metadata."""
    import ee

    import gui.scripts.aoi_io as aoi_io

    sentinel = object()
    captured = {}
    monkeypatch.setattr(
        ee, "FeatureCollection", lambda aid: captured.update({"aid": aid}) or sentinel
    )

    meta = {
        "method": "ASSET",
        "name": "aoi",
        "gee": True,
        "asset": {
            "asset_id": "users/me/aoi",
            "type": "TABLE",
            "column": "ALL",
            "value": None,
        },
    }
    restored = aoi_io.load_aoi(tmp_path, meta)

    assert restored.gdf is None
    assert restored.feature_collection is sentinel
    assert captured["aid"] == "users/me/aoi"


def test_load_asset_degrades_when_rebuild_fails(tmp_path, monkeypatch):
    """Load degrades gracefully when asset feature collection rebuild fails."""
    import ee

    import gui.scripts.aoi_io as aoi_io

    monkeypatch.setattr(
        ee,
        "FeatureCollection",
        lambda aid: (_ for _ in ()).throw(RuntimeError("no ee")),
    )
    meta = {
        "method": "ASSET",
        "name": "aoi",
        "gee": True,
        "asset": {
            "asset_id": "users/me/aoi",
            "type": "TABLE",
            "column": "ALL",
            "value": None,
        },
    }
    restored = aoi_io.load_aoi(tmp_path, meta)
    assert restored is not None
    assert restored.feature_collection is None


# --- AoiSpec persistence: v4 restores the picker from ``AoiResult.spec`` ----


def test_write_persists_the_spec(tmp_path, draw_aoi_result):
    """The manifest carries ``aoi_spec`` next to the existing metadata."""
    meta = write_aoi(tmp_path, draw_aoi_result)

    assert meta["aoi_spec"]["method"] == "DRAW"
    assert meta["aoi_spec"]["schema_version"] >= 1
    assert meta["aoi_spec"] == draw_aoi_result.spec.to_dict()
    assert meta["geometry_file"] == AOI_GEOMETRY_FILENAME  # existing keys kept


def test_attach_persists_the_same_spec_as_write(tmp_path, monkeypatch, draw_aoi_result):
    """Selection-time attach and manual save must agree, spec included.

    Otherwise every attach after a save would look like a change and rewrite.
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_spec")

    attach_aoi(p, draw_aoi_result, data_dir=tmp_path)

    assert p.aoi["aoi_spec"] == draw_aoi_result.spec.to_dict()
    assert p.aoi == write_aoi(tmp_path / "attach_spec", draw_aoi_result)


def test_restore_spec_round_trip(tmp_path, draw_aoi_result):
    """Write, then JSON manifest, then load gives back an equal spec."""
    meta = json.loads(json.dumps(write_aoi(tmp_path, draw_aoi_result)))

    restored = load_aoi(tmp_path, meta)

    assert restored.spec == draw_aoi_result.spec
    assert restored.method == "DRAW"
    assert restored.gdf is not None


def test_restore_admin_spec_round_trip_keeps_the_code_chain(tmp_path):
    """Admin codes come back as the tuple the picker's cascade expects."""
    spec = AoiSpec(method="ADMIN2", admin_codes=("101", "1001", "100001"))
    aoi = _aoi(method="ADMIN2", name="ABC_x_y", gee=False, admin="100001", spec=spec)
    meta = json.loads(json.dumps(write_aoi(tmp_path, aoi)))

    assert load_aoi(tmp_path, meta).spec == spec


def test_restore_legacy_admin2_manifest_derives_the_code_chain(
    tmp_path, legacy_admin_manifest
):
    """A pre-spec ADMIN2 manifest restores its cascade from the leaf code."""
    result = load_aoi(tmp_path, legacy_admin_manifest)

    assert result.method == "ADMIN2"
    assert result.admin == "100001"
    assert result.spec == AoiSpec(
        method="ADMIN2", admin_codes=("101", "1001", "100001")
    )


def test_restore_legacy_admin_manifest_without_a_resolvable_chain(
    tmp_path, monkeypatch
):
    """When pygaul cannot place the code: no spec, but the AOI still loads."""
    import pygaul

    monkeypatch.setattr(pygaul, "Names", _gaul_names({}))
    meta = {"method": "ADMIN1", "name": "ABC_x", "gee": False, "admin": "1001"}

    result = load_aoi(tmp_path, meta)

    assert result is not None
    assert result.admin == "1001"
    assert result.spec is None


def test_restore_legacy_admin0_manifest_needs_no_lookup(tmp_path, monkeypatch):
    """At level 0 the leaf is the whole chain, so pygaul is never asked."""
    import pygaul

    def no_lookup(**_):
        raise AssertionError("ADMIN0 must not look its chain up")

    monkeypatch.setattr(pygaul, "Names", no_lookup)
    meta = {"method": "ADMIN0", "name": "GUY", "gee": False, "admin": "197"}

    assert load_aoi(tmp_path, meta).spec == AoiSpec(
        method="ADMIN0", admin_codes=("197",)
    )


def test_restore_legacy_draw_manifest_rebuilds_the_drawing(tmp_path):
    """A pre-spec DRAW manifest seeds the draw control from its sidecar."""
    meta = write_aoi(tmp_path, _aoi(gdf=_gdf()))
    assert "aoi_spec" not in meta  # what every saved project holds today

    spec = load_aoi(tmp_path, meta).spec

    assert spec.method == "DRAW"
    assert spec.name == "san_marino"
    drawn = gpd.GeoDataFrame.from_features(spec.geo_json["features"])
    assert drawn.total_bounds == pytest.approx([12.40, 43.89, 12.52, 43.99], abs=1e-6)


def test_restore_legacy_draw_manifest_without_its_sidecar_has_no_spec(tmp_path):
    """No geometry to seed the control from: nothing to restore the picker to."""
    meta = {"method": "DRAW", "name": "x", "gee": False, "geometry_file": "gone.json"}

    assert load_aoi(tmp_path, meta).spec is None


def _dated_legacy_draw_manifest(project_dir):
    """A pre-spec DRAW manifest whose sidecar carries an ISO-date attribute.

    Like one a QGIS edit adds. GDAL's GeoJSON driver reads such a string back
    as a datetime column, whose ``Timestamp`` values ``json`` cannot encode.
    """
    gdf = _gdf()
    gdf["edited"] = ["2024-05-06"]
    meta = write_aoi(project_dir, _aoi(gdf=gdf))
    sidecar = gpd.read_file(project_dir / AOI_GEOMETRY_FILENAME)
    assert pd.api.types.is_datetime64_any_dtype(sidecar["edited"])  # the trap
    return meta


def test_restore_legacy_draw_manifest_with_a_date_attribute_still_loads(tmp_path):
    """A date in the sidecar must not fail the load or drop the drawing."""
    meta = _dated_legacy_draw_manifest(tmp_path)

    result = load_aoi(tmp_path, meta)

    assert result is not None
    assert result.gdf is not None
    assert result.spec.method == "DRAW"
    drawn = gpd.GeoDataFrame.from_features(result.spec.geo_json["features"])
    assert drawn.total_bounds == pytest.approx([12.40, 43.89, 12.52, 43.99], abs=1e-6)


def test_restore_unreadable_spec_over_a_dated_sidecar_still_loads(tmp_path):
    """The newer-schema fallback reaches the same sidecar and must survive it."""
    meta = _dated_legacy_draw_manifest(tmp_path)
    meta["aoi_spec"] = {"schema_version": 99, "method": "DRAW"}

    result = load_aoi(tmp_path, meta)

    assert result is not None
    assert result.gdf is not None
    assert result.spec.method == "DRAW"


def test_restore_survives_a_legacy_spec_that_cannot_be_built(tmp_path, monkeypatch):
    """Whatever breaks the spec synthesis, the project opens with a blank picker."""
    import gui.scripts.aoi_io as aoi_io

    def broken(metadata, gdf):
        raise TypeError("cannot synthesize")

    monkeypatch.setattr(aoi_io, "_legacy_spec", broken)
    meta = write_aoi(tmp_path, _aoi(gdf=_gdf()))

    result = load_aoi(tmp_path, meta)

    assert result is not None
    assert result.gdf is not None
    assert result.spec is None


def test_restore_legacy_asset_manifest_rebuilds_the_picker_inputs(tmp_path):
    """The legacy ``asset`` dict maps back onto the spec field for field."""
    meta = {"method": "ASSET", "name": "aoi", "gee": False, "asset": LEGACY_ASSET}

    spec = load_aoi(tmp_path, meta).spec

    assert spec.asset_data() == LEGACY_ASSET


@pytest.mark.parametrize("method", ["SHAPE", "POINTS"])
def test_restore_legacy_file_manifest_has_no_spec(tmp_path, method):
    """Old manifests never stored the file path, so there is nothing to seed."""
    meta = write_aoi(tmp_path, _aoi(method=method, gdf=_gdf()))

    result = load_aoi(tmp_path, meta)

    assert result.spec is None
    assert result.gdf is not None  # the AOI itself is still restored


def test_restore_newer_schema_falls_back_to_the_legacy_fields(tmp_path):
    """A spec from a newer pysepal must not fail the project load."""
    meta = write_aoi(tmp_path, _aoi(gdf=_gdf()))
    meta["aoi_spec"] = {**_draw_spec().to_dict(), "schema_version": 99}

    result = load_aoi(tmp_path, meta)

    assert result.gdf is not None
    assert result.spec.method == "DRAW"  # rebuilt from the sidecar instead
    assert result.spec.schema_version == 1


def test_restore_malformed_spec_falls_back_to_the_legacy_fields(tmp_path):
    """A spec payload with no method is treated like a missing one."""
    meta = {"method": "ADMIN0", "name": "GUY", "gee": False, "admin": "197"}
    meta["aoi_spec"] = {"schema_version": 1}

    assert load_aoi(tmp_path, meta).spec == AoiSpec(
        method="ADMIN0", admin_codes=("197",)
    )


def test_admin_code_chain_walks_up_from_the_leaf(monkeypatch):
    """The shared helper: levels 0..n for a leaf, derived from pygaul."""
    import pygaul

    monkeypatch.setattr(
        pygaul, "Names", _gaul_names({"100001": ("101", "1001", "100001")})
    )

    assert admin_code_chain("100001", 2) == ("101", "1001", "100001")
    assert admin_code_chain(100001, 2) == ("101", "1001", "100001")  # int code
    assert admin_code_chain("197", 0) == ("197",)
    assert admin_code_chain("999", 1) is None  # not in GAUL 2024


# --- persist_aoi ------------------------------------------------------------


def test_persist_keeps_stored_aoi_when_session_state_is_empty(tmp_path):
    """Persist keeps stored AOI when session state is empty.

    A save must never turn a persisted AOI into nothing.

    Session state going empty while a project holds a saved AOI means something
    dropped it (a widget teardown, a failed restore) — not that the user asked
    for the AOI to be removed. Overwriting is silent, unrecoverable data loss,
    so the stored metadata and its sidecar are kept.
    """
    stored = write_aoi(tmp_path, _aoi(gdf=_gdf()))
    assert (tmp_path / AOI_GEOMETRY_FILENAME).exists()

    kept = persist_aoi(tmp_path, None, stored)

    assert kept == stored
    assert (tmp_path / AOI_GEOMETRY_FILENAME).exists(), "sidecar geometry was deleted"


def test_persist_writes_a_new_aoi_over_the_stored_one(tmp_path):
    """Persist allows writing a new AOI over the stored one.

    Guarding an empty AOI must not block a real change of area.
    """
    stored = write_aoi(tmp_path, _aoi(method="ADMIN0", name="GUY", admin="197"))

    meta = persist_aoi(tmp_path, _aoi(method="ADMIN0", name="BOL", admin="178"), stored)

    assert meta["name"] == "BOL"
    assert meta["admin"] == "178"


def test_persist_stores_nothing_when_there_is_nothing_stored(tmp_path):
    """No AOI and none saved is an ordinary empty project."""
    assert persist_aoi(tmp_path, None, None) is None


def test_persist_of_empty_state_leaves_no_stale_sidecar(tmp_path):
    """Persist guarding metadata doesn't resurrect geometry for metadata-only AOI."""
    stored = write_aoi(tmp_path, _aoi(method="ADMIN0", name="GUY", admin="197"))

    kept = persist_aoi(tmp_path, None, stored)

    assert kept == stored
    assert not (tmp_path / AOI_GEOMETRY_FILENAME).exists()


# --- build_asset_feature_collection (strict builder) --------------------------


def test_strict_builder_rejects_a_filter_with_no_value():
    """AssetSelectComponent publishes {column: X, value: None} mid-filter.

    Accepting it would silently allocate over the whole unfiltered collection.
    """
    from gui.scripts.aoi_io import build_asset_feature_collection

    asset = {
        "asset_id": "users/me/table",
        "type": "TABLE",
        "column": "adm1",
        "value": None,
    }

    with pytest.raises(ValueError, match="adm1"):
        build_asset_feature_collection(asset)


def test_strict_builder_rejects_an_unknown_asset_type():
    """An unrecognised type must not fall through to None."""
    from gui.scripts.aoi_io import build_asset_feature_collection

    with pytest.raises(ValueError, match="FOLDER"):
        build_asset_feature_collection({"asset_id": "users/me/x", "type": "FOLDER"})


def test_forgiving_builder_still_degrades_to_none():
    """load_aoi's contract is unchanged: a bad asset loads as metadata-only."""
    from gui.scripts.aoi_io import _rebuild_asset_feature_collection

    asset = {
        "asset_id": "users/me/table",
        "type": "TABLE",
        "column": "adm1",
        "value": None,
    }

    assert _rebuild_asset_feature_collection(asset) is None


# --- attach_aoi: selection-time persistence ---------------------------------
#
# Job-completion saves call ``project.save()`` directly, which serializes
# whatever ``project.aoi`` holds at that moment. Before attach_aoi existed the
# AOI was only attached inside the manual Save flow, so a project driven
# through the workflow but never manually saved reloaded with every artifact
# EXCEPT its AOI (silently: the manifest simply lacked the key).


def test_attach_writes_sidecar_and_sets_project_aoi(tmp_path, monkeypatch):
    """Selecting an AOI immediately persists geometry + attaches metadata."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_fresh")

    assert attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path) is True

    assert p.aoi["method"] == "DRAW"
    assert (tmp_path / "attach_fresh" / AOI_GEOMETRY_FILENAME).exists()


def test_attach_does_not_materialize_unsaved_project(tmp_path, monkeypatch):
    """A never-saved project must not gain a manifest from AOI selection."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_unsaved")

    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)

    assert not (tmp_path / "attach_unsaved" / "attach_unsaved_project.json").exists()


def test_attach_updates_existing_manifest_in_place(tmp_path, monkeypatch):
    """A previously saved project's manifest gains the AOI on selection."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_saved")
    p.save()

    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)

    loaded = proj.Project.load("attach_saved")
    assert loaded.aoi["method"] == "DRAW"
    assert load_aoi(tmp_path / "attach_saved", loaded.aoi).gdf is not None


def test_attach_survives_job_style_bare_save(tmp_path, monkeypatch):
    """A background job's bare save carries the AOI (the testag regression).

    Once an AOI is selected, project.save() must serialize it with no manual
    Save in between.
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_job")

    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)
    p.save()  # what every job-completion site does

    loaded = proj.Project.load("attach_job")
    restored = load_aoi(tmp_path / "attach_job", loaded.aoi)
    assert restored is not None and restored.method == "DRAW"
    assert restored.gdf is not None


def test_attach_skips_rewrite_when_metadata_unchanged(tmp_path, monkeypatch):
    """Re-running with the same AOI (e.g. right after a load) writes nothing.

    Otherwise every project load would bump the manifest mtime.
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_idem")
    p.save()
    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)

    manifest = tmp_path / "attach_idem" / "attach_idem_project.json"
    sidecar = tmp_path / "attach_idem" / AOI_GEOMETRY_FILENAME
    before = (manifest.stat().st_mtime_ns, sidecar.stat().st_mtime_ns)

    assert attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path) is False

    assert (manifest.stat().st_mtime_ns, sidecar.stat().st_mtime_ns) == before


def test_attach_noop_without_project_or_aoi(tmp_path):
    """No project or no AOI: nothing to attach, nothing written."""
    assert attach_aoi(None, _aoi(gdf=_gdf()), data_dir=tmp_path) is False
    assert (
        attach_aoi(proj.Project(project_name="attach_none"), None, data_dir=tmp_path)
        is False
    )
    assert not (tmp_path / "attach_none").exists()


# --- restore guard: no attach while a project load is mid-flight ------------
#
# do_load sets aoi_result and project as two separate reactive writes, and
# Solara can run a render (and effects) BETWEEN them — the attach effect then
# saw (new project's AOI, old project) and wrote one project's AOI into the
# other's folder (testag got test_Taka's ADMIN AOI, its drawn sidecar
# unlinked, 2026-08-28). The load flow must mark the window and the attach
# path must refuse to run inside it.


def _app_state():
    from gui.store.state_manager import AppState

    return AppState()


def test_attach_current_aoi_refuses_during_restore(tmp_path, monkeypatch):
    """Inside the restoring window, nothing is written — ever."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    state = _app_state()
    old = proj.Project(project_name="old_proj")
    old.save()
    state.project.set(old)

    with state.restoring_project():
        # the mid-load transient: the NEW project's AOI paired with the OLD one
        state.aoi_result.set(_aoi(name="other_projects_aoi", gdf=_gdf()))
        assert state.attach_current_aoi(data_dir=tmp_path) is False

    loaded = proj.Project.load("old_proj")
    assert loaded.aoi is None, "restore-window attach wrote another project's AOI"
    assert not (tmp_path / "old_proj" / AOI_GEOMETRY_FILENAME).exists()


def test_attach_current_aoi_runs_after_restore_window(tmp_path, monkeypatch):
    """Once the window closes, a genuine selection persists as usual."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    state = _app_state()
    p = proj.Project(project_name="cur_proj")
    p.save()
    state.project.set(p)

    with state.restoring_project():
        state.aoi_result.set(_aoi(gdf=_gdf()))

    assert state.attach_current_aoi(data_dir=tmp_path) is True
    assert proj.Project.load("cur_proj").aoi["method"] == "DRAW"


def test_restoring_project_nests_until_outermost_exit():
    """Two nested windows: the guard stays True until the OUTER one exits.

    ``_restoring_depth`` is a solara.reactive counter (session-scoped: reactive
    values are stored per-kernel, so two browser sessions never share one
    guard — no second kernel context is needed here to prove that, it's a
    property of how Solara stores reactive state). Nesting matters because an
    inner window closing must not prematurely re-enable attaches while an
    outer window covering it is still open.
    """
    state = _app_state()
    assert state.restoring is False

    with state.restoring_project():
        assert state.restoring is True
        with state.restoring_project():
            assert state.restoring is True
        # Inner window closed; outer is still open.
        assert state.restoring is True

    assert state.restoring is False


def test_restoring_project_nesting_clears_fully_on_inner_exception():
    """An exception inside a nested window still fully unwinds the counter."""
    state = _app_state()

    with pytest.raises(RuntimeError):
        with state.restoring_project():
            with state.restoring_project():
                raise RuntimeError("load failed")

    assert state.restoring is False


def test_new_project_state_mid_swap_does_not_attach_outgoing_aoi(tmp_path, monkeypatch):
    """A render between new_project_state's sets must not attach the old AOI.

    Subscribing directly to ``project`` simulates Solara's render-between-sets:
    the listener fires synchronously from inside ``project.set(...)``, the same
    point at which an interleaved render's attach-on-select effect would run.
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    state = _app_state()
    old = proj.Project(project_name="outgoing")
    old.save()
    state.project.set(old)
    state.aoi_result.set(_aoi(name="outgoing_aoi", gdf=_gdf()))

    captured = []
    unsubscribe = state.project.subscribe(
        lambda _new: captured.append(state.attach_current_aoi(data_dir=tmp_path))
    )
    try:
        new = proj.Project(project_name="incoming")
        new.save()
        state.new_project_state(new)
    finally:
        unsubscribe()

    assert captured == [False], "mid-swap attach must be refused"
    assert proj.Project.load("incoming").aoi is None
    assert not (tmp_path / "incoming" / AOI_GEOMETRY_FILENAME).exists()
    assert proj.Project.load("outgoing").aoi is None


def test_close_project_state_mid_teardown_does_not_attach(tmp_path, monkeypatch):
    """A render between close_project_state's sets must not attach either.

    Same simulated-render technique as the new_project_state test above,
    applied to teardown (project -> None).
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    state = _app_state()
    old = proj.Project(project_name="closing")
    old.save()
    state.project.set(old)
    state.aoi_result.set(_aoi(name="closing_aoi", gdf=_gdf()))

    captured = []
    unsubscribe = state.project.subscribe(
        lambda _new: captured.append(state.attach_current_aoi(data_dir=tmp_path))
    )
    try:
        state.close_project_state()
    finally:
        unsubscribe()

    assert captured == [False], "mid-teardown attach must be refused"
    assert proj.Project.load("closing").aoi is None


# --- attach_aoi: geometry digest + crash-safety consistency contract --------
#
# Two review-confirmed bugs: (1) attach_aoi compared only metadata, so a
# same-method/same-name geometry edit (e.g. reshaping a drawn rectangle) was
# treated as idempotent and silently dropped; (2) idempotency was judged from
# in-memory ``project.aoi``, so a failed ``project.save()`` could never be
# retried, and the write ordering could leave a manifest referencing a
# missing sidecar (or vice versa) across a crash.


def test_attach_rewrites_when_geometry_changes_under_same_metadata(
    tmp_path, monkeypatch
):
    """A geometry edit with unchanged method/name is NOT a no-op (Bug 1)."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_geom_edit")
    p.save()

    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)
    first_digest = p.aoi["geometry_digest"]

    shifted = _gdf(bounds=(20.0, 10.0, 20.2, 10.2))
    assert attach_aoi(p, _aoi(gdf=shifted), data_dir=tmp_path) is True
    assert p.aoi["geometry_digest"] != first_digest

    sidecar = tmp_path / "attach_geom_edit" / AOI_GEOMETRY_FILENAME
    written = gpd.read_file(sidecar)
    assert written.total_bounds == pytest.approx([20.0, 10.0, 20.2, 10.2], abs=1e-6)

    loaded = proj.Project.load("attach_geom_edit")
    assert loaded.aoi["geometry_digest"] == p.aoi["geometry_digest"]


def test_attach_write_load_attach_roundtrip_is_a_noop(tmp_path, monkeypatch):
    """write_aoi -> load_aoi -> attach_aoi must not look like a geometry edit.

    The digest is computed on a precision-snapped copy of the geometry so it
    survives a GeoJSON write/read round-trip: loading a project and handing
    the restored AOI back through attach_aoi (as the app does) must be a
    true no-op, or every load would bump the manifest mtime.
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_roundtrip")
    p.save()
    # With a spec, as every v4 selection has: the spec must survive the
    # manifest round trip unchanged too, or each load would rewrite it.
    attach_aoi(p, _aoi(gdf=_gdf(), spec=_draw_spec()), data_dir=tmp_path)

    manifest = tmp_path / "attach_roundtrip" / "attach_roundtrip_project.json"
    sidecar = tmp_path / "attach_roundtrip" / AOI_GEOMETRY_FILENAME
    manifest_mtime = manifest.stat().st_mtime_ns
    sidecar_mtime = sidecar.stat().st_mtime_ns

    restored = load_aoi(tmp_path / "attach_roundtrip", p.aoi)
    assert attach_aoi(p, restored, data_dir=tmp_path) is False

    assert manifest.stat().st_mtime_ns == manifest_mtime
    assert sidecar.stat().st_mtime_ns == sidecar_mtime


def test_attach_heals_legacy_manifest_without_digest_then_settles(
    tmp_path, monkeypatch
):
    """A pre-digest manifest triggers one healing rewrite, then is idempotent."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_legacy")
    aoi = _aoi(gdf=_gdf())
    legacy_meta = write_aoi(tmp_path / "attach_legacy", aoi)
    del legacy_meta["geometry_digest"]
    p.aoi = legacy_meta
    p.save()

    assert attach_aoi(p, aoi, data_dir=tmp_path) is True  # heals
    assert "geometry_digest" in p.aoi

    assert attach_aoi(p, aoi, data_dir=tmp_path) is False  # now idempotent


def test_attach_heals_legacy_manifest_without_spec_then_settles(tmp_path, monkeypatch):
    """A pre-spec manifest gains its synthesized spec once, then stays put.

    Loading a legacy project synthesizes a spec, so the first attach of the
    restored AOI writes it; the next load reads that spec back unchanged and
    the manifest is not touched again.
    """
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_legacy_spec")
    p.save()
    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)  # a pre-v4 selection
    assert "aoi_spec" not in p.aoi

    restored = load_aoi(tmp_path / "attach_legacy_spec", p.aoi)
    assert attach_aoi(p, restored, data_dir=tmp_path) is True  # heals
    assert p.aoi["aoi_spec"]["method"] == "DRAW"

    manifest = tmp_path / "attach_legacy_spec" / "attach_legacy_spec_project.json"
    reloaded = load_aoi(
        tmp_path / "attach_legacy_spec",
        json.loads(manifest.read_text(encoding="utf-8"))["aoi"],
    )
    assert attach_aoi(p, reloaded, data_dir=tmp_path) is False  # now idempotent


def test_attach_retries_manifest_save_after_a_previous_failure(tmp_path, monkeypatch):
    """A raised project.save() must be retried, not silently skipped (Bug 2a)."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_retry")
    p.save()

    real_save = proj.Project.save
    calls = {"n": 0}

    def flaky_save(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return real_save(self, *a, **kw)

    monkeypatch.setattr(proj.Project, "save", flaky_save)

    aoi = _aoi(gdf=_gdf())
    with pytest.raises(RuntimeError, match="disk full"):
        attach_aoi(p, aoi, data_dir=tmp_path)

    manifest = tmp_path / "attach_retry" / "attach_retry_project.json"
    assert json.loads(manifest.read_text(encoding="utf-8")).get("aoi") is None

    # Same AOI again: disk still disagrees with `expected`, so this must
    # retry the save rather than treat project.aoi as already matching.
    assert attach_aoi(p, aoi, data_dir=tmp_path) is True
    assert calls["n"] == 2
    assert json.loads(manifest.read_text(encoding="utf-8")).get("aoi") is not None


def test_attach_metadata_only_replace_unlinks_sidecar_only_after_save(
    tmp_path, monkeypatch
):
    """Metadata-only AOI replacing a geometry one: sidecar outlives the save call."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_replace")
    p.save()
    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)

    sidecar = tmp_path / "attach_replace" / AOI_GEOMETRY_FILENAME
    assert sidecar.exists()

    real_save = proj.Project.save
    seen = {}

    def spy_save(self, *a, **kw):
        seen["sidecar_existed_during_save"] = sidecar.exists()
        return real_save(self, *a, **kw)

    monkeypatch.setattr(proj.Project, "save", spy_save)

    admin_aoi = _aoi(method="ADMIN1", name="COL_x", admin="21758", gdf=None)
    assert attach_aoi(p, admin_aoi, data_dir=tmp_path) is True

    assert seen["sidecar_existed_during_save"] is True
    assert not sidecar.exists()


def test_attach_metadata_only_replace_keeps_sidecar_when_save_raises(
    tmp_path, monkeypatch
):
    """If the manifest save fails, the still-referenced sidecar must survive."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_replace_fail")
    p.save()
    attach_aoi(p, _aoi(gdf=_gdf()), data_dir=tmp_path)

    sidecar = tmp_path / "attach_replace_fail" / AOI_GEOMETRY_FILENAME
    assert sidecar.exists()

    def boom_save(self, *a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(proj.Project, "save", boom_save)

    admin_aoi = _aoi(method="ADMIN1", name="COL_x", admin="21758", gdf=None)
    with pytest.raises(RuntimeError, match="disk full"):
        attach_aoi(p, admin_aoi, data_dir=tmp_path)

    assert sidecar.exists(), "sidecar must survive a failed manifest save"


def test_attach_trusts_disk_over_memory_when_they_disagree(tmp_path, monkeypatch):
    """A manifest that fell out of sync with project.aoi is healed, not trusted."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="attach_disk_truth")
    p.save()

    aoi = _aoi(gdf=_gdf())
    attach_aoi(p, aoi, data_dir=tmp_path)

    manifest = tmp_path / "attach_disk_truth" / "attach_disk_truth_project.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["aoi"] = None
    manifest.write_text(json.dumps(data), encoding="utf-8")

    # project.aoi (in memory) still matches `expected`; disk does not.
    assert attach_aoi(p, aoi, data_dir=tmp_path) is True
    assert json.loads(manifest.read_text(encoding="utf-8"))["aoi"] is not None


def test_project_save_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
    """Project.save() writes via temp file + os.replace; nothing lingers."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    p = proj.Project(project_name="save_atomic")
    saved_path = p.save()

    assert saved_path.exists()
    leftovers = [
        f
        for f in saved_path.parent.iterdir()
        if f.name.startswith(".") and f.name.endswith(".tmp")
    ]
    assert leftovers == []


def test_new_and_close_forget_the_picker_spec():
    """New/Close leave no spec behind for the next project to inherit."""
    state = _app_state()

    state.aoi_spec.set(_draw_spec())
    state.new_project_state(proj.Project(project_name="fresh_spec"))
    assert state.aoi_spec.value is None

    state.aoi_spec.set(_draw_spec())
    state.close_project_state()
    assert state.aoi_spec.value is None


def test_restoring_flag_clears_even_when_load_raises(tmp_path):
    """A failed load must not leave the guard stuck (attach disabled forever)."""
    state = _app_state()
    try:
        with state.restoring_project():
            raise RuntimeError("load failed")
    except RuntimeError:
        pass
    assert state.restoring is False
