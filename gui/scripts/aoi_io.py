"""AOI persistence for the Spatial Risk GUI.

The AOI is session state (``app_state.aoi_result``), not part of the library
``Project`` model. We persist it as:

* a light, library-agnostic metadata dict stored on ``project.aoi`` (and thus
  inside ``<project>_project.json``); plus
* the geometry itself in a sidecar ``aoi.geojson`` (WGS84) in the project
  folder, so the manifest stays small and the geometry is QGIS-openable.

Only AOIs that already carry a resolved ``gdf`` (DRAW / SHAPE / POINTS, and
admin selections whose geometry has been fetched) get a geometry sidecar.
GEE-lazy AOIs (admin / asset under GEE expose ``gdf=None``) are persisted as
metadata only. For admin AOIs that metadata (the GAUL ``admin`` code) is enough
to rebuild the lazy EE ``feature_collection`` on load (see ``load_aoi``), so the
AOI stays usable downstream without re-selection; asset AOIs remain
geometry-less on load. See ``write_aoi`` for the boundary.

The picker's own record of the selection is persisted too, as ``aoi_spec``
(pysepal's ``AoiSpec.to_dict()``), so a load restores the picker through
``AoiView(spec=)``. Manifests written before it get a spec synthesized from
their legacy fields (see ``_legacy_spec``).

Kept free of Solara/ipyvuetify so it can be unit-tested without a render
harness; pysepal/geopandas are imported lazily inside the functions.
"""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("spatial_risk")

AOI_GEOMETRY_FILENAME = "aoi.geojson"
_ADMIN_METHODS = ("ADMIN0", "ADMIN1", "ADMIN2")

# Grid size (decimal degrees, WGS84) that geometry digests are snapped to
# before hashing. ~0.11m at the equator — far finer than any AOI selection
# workflow (draw tool, admin boundary, uploaded shapefile) is meaningfully
# precise to, and far coarser than the floating-point noise a GeoJSON
# write -> read round-trip can introduce. See ``_geometry_digest``.
_DIGEST_GRID_SIZE = 1e-6


def admin_code_chain(admin_code: Any, level: int) -> Optional[Tuple[str, ...]]:
    """Return every GAUL code from level 0 down to ``admin_code``, or None.

    pysepal's admin cascade is restored from the whole chain
    (``AoiSpec.admin_codes``, ``AdminLevelSelector(codes=)``), but a pre-v4
    manifest (and a stored borders selection) records only the leaf. This
    mirrors pysepal's own ``_derive_admin_chain``: a local pygaul parquet
    lookup, no Earth Engine. At level 0 the leaf already is the whole chain,
    so no lookup is made.

    Args:
        admin_code: The leaf GAUL 2024 code (str or int).
        level: The admin level of ``admin_code`` (0, 1 or 2).

    Returns:
        The codes for levels ``0..level`` as strings, or None when pygaul cannot
        place the code (not a GAUL 2024 code, lookup failure) — a caller then
        restores without a cascade rather than failing.
    """
    code = str(admin_code)
    if level == 0:
        return (code,)
    try:
        import pygaul

        row = pygaul.Names(admin=code, complete=True).iloc[0]
        return tuple(str(row[f"gaul{lvl}_code"]) for lvl in range(level + 1))
    except Exception:
        logger.warning(
            "Could not derive the GAUL code chain of ADMIN%s code %s; the "
            "admin picker cannot be restored to it.",
            level,
            code,
            exc_info=True,
        )
        return None


def _rebuild_admin_feature_collection(admin_code: str) -> Optional[Any]:
    """Rebuild the lazy GEE FeatureCollection for a persisted admin AOI.

    Admin selections persist only their GAUL ``admin`` code, not geometry.
    This mirrors what pysepal's ``process_admin`` does on selection — the
    FeatureCollection is ``pygaul.Items(admin=<code>)`` — so a loaded admin AOI
    is usable downstream (``resolve_aoi_ee``) and can frame the map without the
    user re-selecting it.

    Earth Engine must already be initialized (the GUI does this when the AOI
    panel mounts). Returns None if the rebuild fails (EE not ready, offline,
    bad code) so loading degrades to a metadata-only AOI instead of erroring.
    """
    try:
        import pygaul

        return pygaul.Items(admin=admin_code)
    except Exception:  # pragma: no cover - exercised via degrade test (patched)
        logger.debug(
            "Could not rebuild AOI FeatureCollection for admin=%s; "
            "loading as metadata-only.",
            admin_code,
            exc_info=True,
        )
        return None


def build_asset_feature_collection(asset: Dict[str, Any]) -> Any:
    """Rebuild the EE object for an ASSET selection, raising on any problem.

    Mirrors pysepal's ``process_asset`` validation. The strict counterpart of
    :func:`_rebuild_asset_feature_collection`: an allocation run that cannot
    build its borders must fail loudly, where loading persisted AOI metadata
    degrades to None.

    Raises:
        ValueError: no asset id, a column filter with no value, or an asset
            type the AOI workflow does not support.
    """
    import ee

    aid = (asset or {}).get("asset_id")
    if not aid:
        raise ValueError("No Earth Engine asset selected.")

    atype = asset.get("type", "TABLE")
    column, value = asset.get("column", "ALL"), asset.get("value")
    if column not in (None, "ALL") and value is None:
        raise ValueError(f"The filter on column '{column}' has no value selected.")

    if atype == "TABLE":
        obj = ee.FeatureCollection(aid)
        if column not in (None, "ALL"):
            obj = obj.filter(ee.Filter.eq(column, value))
        return obj
    if atype == "IMAGE":
        return ee.Image(aid)
    if atype == "IMAGE_COLLECTION":
        return ee.ImageCollection(aid)
    raise ValueError(f"Unsupported asset type {atype!r}.")


def _rebuild_asset_feature_collection(asset: Dict[str, Any]) -> Optional[Any]:
    """Best-effort rebuild for ``load_aoi``: None instead of an exception.

    Returns None (metadata-only fallback) if EE isn't ready or the asset is
    unavailable, so loading a project never crashes.
    """
    try:
        return build_asset_feature_collection(asset)
    except Exception:  # pragma: no cover - exercised via degrade test (patched)
        logger.debug(
            "Could not rebuild ASSET FeatureCollection; metadata-only.", exc_info=True
        )
    return None


def _to_wgs84(gdf: Any) -> Any:
    """Reproject ``gdf`` to EPSG:4326 if it isn't already."""
    if getattr(gdf, "crs", None) is not None and gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(epsg=4326)
    return gdf


def _geometry_digest(gdf: Any) -> str:
    """Stable content digest for a WGS84 GeoDataFrame's geometries.

    Each geometry is snapped to a ``_DIGEST_GRID_SIZE`` (1e-6 deg, ~11cm)
    precision grid with ``shapely.set_precision`` before hashing the
    concatenated WKB with sha256.

    This is what makes the digest STABLE across a ``write_aoi`` ->
    ``load_aoi`` round-trip: the sidecar is a GeoJSON file, and reading it
    back with geopandas can introduce sub-nanometre floating-point noise
    relative to the geometry that was written (text serialization and
    reparsing of doubles). That noise is many orders of magnitude smaller
    than the 1e-6 degree grid, so snapping produces bit-identical results
    before and after the round-trip, and a load-then-reattach never looks
    like a geometry change. A real edit (a shifted vertex, a redrawn
    rectangle) moves coordinates by amounts humans can see on a map — far
    larger than the grid — so it always changes the digest.
    """
    import shapely

    h = hashlib.sha256()
    for geom in gdf.geometry:
        snapped = shapely.set_precision(geom, _DIGEST_GRID_SIZE)
        h.update(shapely.to_wkb(snapped))
    return h.hexdigest()


def _aoi_metadata(
    aoi: Any, geometry_file: Optional[str], gdf: Any = None
) -> Dict[str, Any]:
    """Build the serializable metadata dict for an AoiResult-like object.

    ``gdf``, when given, must already be normalized to WGS84 (see
    ``_to_wgs84``) — callers that write and callers that merely compare must
    use the same normalized geometry so their digests agree.
    """
    meta: Dict[str, Any] = {
        "method": getattr(aoi, "method", None),
        "name": getattr(aoi, "name", None),
        "gee": bool(getattr(aoi, "gee", False)),
        "admin": getattr(aoi, "admin", None),
    }
    if geometry_file is not None:
        meta["geometry_file"] = geometry_file
    if gdf is not None:
        meta["geometry_digest"] = _geometry_digest(gdf)

    # The picker's own record of the selection: what load_aoi hands back to
    # AoiView(spec=) so the picker shows (and re-runs) it.
    spec = getattr(aoi, "spec", None)
    if spec is not None:
        meta["aoi_spec"] = spec.to_dict()
        if spec.method == "ASSET":
            # Legacy key, kept for one release so an older app still reads the
            # manifest. The full dict, never spec.asset_id: the rebuild does
            # ``(asset or {}).get(...)``, which a bare string silently breaks.
            meta["asset"] = spec.asset_data()
    return meta


def _write_sidecar_geojson(gdf: Any, sidecar: Path) -> None:
    """Write ``gdf`` to ``sidecar`` atomically.

    Writes to a temp file in the same directory (so ``os.replace`` is an
    atomic rename on the same filesystem) then replaces the sidecar in one
    step, so a crash mid-write never leaves a truncated/corrupt
    ``aoi.geojson``. GDAL's GeoJSON driver accepts any filename as long as
    ``driver="GeoJSON"`` is passed explicitly, so the ``.tmp`` suffix is fine.
    """
    tmp_path = sidecar.parent / f".{sidecar.name}.{uuid.uuid4().hex}.tmp"
    try:
        gdf.to_file(tmp_path, driver="GeoJSON")
        os.replace(tmp_path, sidecar)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_aoi(project_dir: Path, aoi: Any) -> Optional[Dict[str, Any]]:
    """Persist ``aoi`` for a project and return the metadata dict to store.

    Writes the geometry to ``<project_dir>/aoi.geojson`` (WGS84) when the AOI
    carries a ``gdf``; otherwise persists metadata only. Any stale sidecar from
    a previous save is removed when the current AOI has no geometry (or is None),
    so the manifest's ``geometry_file`` pointer never dangles.

    Args:
        project_dir: The project folder (created if missing).
        aoi: A pysepal ``AoiResult`` (or compatible) or None.

    Returns:
        The metadata dict to assign to ``project.aoi``, or None when there is no
        AOI to persist.
    """
    project_dir = Path(project_dir)
    sidecar = project_dir / AOI_GEOMETRY_FILENAME

    if aoi is None:
        sidecar.unlink(missing_ok=True)
        return None

    project_dir.mkdir(parents=True, exist_ok=True)

    gdf = getattr(aoi, "gdf", None)
    if gdf is None:
        # Metadata-only AOI (e.g. GEE admin/asset): drop any stale geometry.
        sidecar.unlink(missing_ok=True)
        return _aoi_metadata(aoi, geometry_file=None)

    # Normalize to WGS84 so the sidecar matches what zoom_bounds expects.
    gdf = _to_wgs84(gdf)
    _write_sidecar_geojson(gdf, sidecar)
    return _aoi_metadata(aoi, geometry_file=AOI_GEOMETRY_FILENAME, gdf=gdf)


def persist_aoi(
    project_dir: Path, aoi: Any, stored: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Decide what a save should store on ``project.aoi``.

    ``write_aoi`` faithfully persists whatever session state holds, including
    "nothing" — which is right when the project never had an AOI, and wrong
    when it had one a moment ago. An AOI vanishing from session state means
    something dropped it, not that the user asked for it to be removed: the
    picker is step 1 and gates the whole workflow, so there is no flow in which
    saving is how you clear it. Overwriting the stored metadata (and unlinking
    its sidecar) in that case is unrecoverable — the geometry only lives here.

    So an empty AOI over a stored one keeps the stored one, loudly. This is
    defence-in-depth: the wipe that motivated it (pysepal's AoiView nulling the
    caller's reactive when the widget unmounted) is fixed upstream, but the
    failure mode was silent and cost real data.

    Args:
        project_dir: The project folder.
        aoi: The AOI in session state (an ``AoiResult`` or None).
        stored: The metadata already on the project (``project.aoi``), if any.

    Returns:
        The metadata dict to assign to ``project.aoi``, or None.
    """
    if aoi is None and stored:
        logger.warning(
            "Refusing to overwrite the saved AOI (%s) with an empty one: the "
            "project has a stored AOI but session state is empty. Keeping the "
            "saved AOI; re-select the area if you meant to change it.",
            stored.get("name") or stored.get("method"),
        )
        return stored

    return write_aoi(project_dir, aoi)


def _read_manifest_aoi(manifest: Path) -> Any:
    """Return the ``"aoi"`` value committed in ``manifest``, or None.

    Used by :func:`attach_aoi` to judge idempotency from disk rather than
    from ``project.aoi``: if a previous ``project.save()`` raised, the
    in-memory dict was already updated but nothing landed on disk, and the
    only way to notice that (and retry) is to read the committed state back.
    """
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("aoi")


def attach_aoi(project: Any, aoi: Any, data_dir: Path) -> bool:
    """Persist ``aoi`` into ``project`` the moment it is selected.

    The manual Save flow is not the only writer of the project manifest: every
    job completion (variable download, processing, training, inference, …)
    calls ``project.save()`` directly, serializing whatever ``project.aoi``
    holds right then. When the AOI was only attached inside the Save button's
    flow, a project driven through the workflow without a manual save wrote
    manifest after manifest with ``aoi: null`` — and reloaded with all its
    artifacts but no AOI, silently. Attaching (and writing the geometry
    sidecar) at selection time makes every later save carry it.

    The manifest itself is rewritten only when one already exists on disk:
    a freshly created project must not materialize just because an AOI was
    picked — it appears, as before, on its first save or job completion
    (which now includes the AOI).

    Idempotent and cheap to re-run: when the AOI already matches and the
    sidecar is in place (e.g. right after a project load hands the restored
    AOI back through the same code path), nothing is touched — so loading a
    project does not bump its manifest mtime. The "already matches" check is
    read from the manifest on disk when one exists, not from ``project.aoi``:
    that makes a failed ``project.save()`` retryable on the next call (the
    in-memory ``project.aoi`` was already updated, but disk wasn't, so disk
    disagreeing with ``expected`` is exactly what should trigger a retry) and
    means a manifest that fell out of sync with memory some other way is
    healed rather than trusted. Without a manifest yet, the fallback is the
    same in-memory + sidecar check as before.

    Consistency contract for what gets written and in what order:

    * A geometry AOI (DRAW/SHAPE/POINTS/…): the sidecar is written first,
      then ``project.aoi`` is updated, then the manifest is saved (if it
      exists). The manifest can therefore never end up referencing a sidecar
      that doesn't exist yet.
    * A metadata-only AOI (GEE admin/asset) that may be replacing a geometry
      one: ``project.aoi`` is updated and the manifest saved *first*; the
      stale sidecar is unlinked only after that save succeeds (or
      immediately, unconditionally, when there is no manifest to save yet).
      A crash between the save and the unlink leaves an orphaned sidecar,
      which is harmless (nothing references it) and gets cleaned up
      opportunistically the next time this AOI is found to already match.
    * ``project.save()`` is never caught here: raising is what makes the
      disk-based idempotency check see the old committed state and retry on
      the next call.

    Args:
        project: The open ``spatialrisk`` Project (or None).
        aoi: The freshly selected ``AoiResult`` (or None). None is a no-op:
            dropping a stored AOI is reserved for explicit user flows, per
            :func:`persist_aoi`.
        data_dir: The projects root (``DATA_DIR``).

    Returns:
        True when something was written, False on a no-op.
    """
    if project is None or aoi is None:
        return False

    project_dir = Path(data_dir) / project.project_name
    sidecar = project_dir / AOI_GEOMETRY_FILENAME
    manifest = project_dir / f"{project.project_name}_project.json"

    has_geometry = getattr(aoi, "gdf", None) is not None
    gdf = _to_wgs84(aoi.gdf) if has_geometry else None
    expected = _aoi_metadata(
        aoi, geometry_file=AOI_GEOMETRY_FILENAME if has_geometry else None, gdf=gdf
    )

    manifest_exists = manifest.exists()
    committed = _read_manifest_aoi(manifest) if manifest_exists else project.aoi

    sidecar_ok = (not has_geometry) or sidecar.exists()
    if committed == expected and sidecar_ok:
        if not has_geometry and sidecar.exists():
            # Opportunistic cleanup: an orphaned sidecar from a crash between
            # a metadata-only save and its post-save unlink. The manifest
            # already agrees with `expected`, so there's nothing to persist —
            # just tidy the leftover file.
            sidecar.unlink(missing_ok=True)
        return False

    if has_geometry:
        project_dir.mkdir(parents=True, exist_ok=True)
        _write_sidecar_geojson(gdf, sidecar)
        project.aoi = expected
        if manifest_exists:
            project.save()
    else:
        project.aoi = expected
        if manifest_exists:
            project.save()
        sidecar.unlink(missing_ok=True)

    return True


def _legacy_spec(metadata: Dict[str, Any], gdf: Any) -> Optional[Any]:
    """Synthesize an ``AoiSpec`` for a manifest written before specs existed.

    Every project saved by the pysepal 3.x app is such a manifest, and that
    app restored every method into the picker — so dropping the spec would
    load them all with a blank picker. Rebuilt from what the old manifest has:

    * DRAW: the drawing, from the sidecar geometry (``to_json``, not
      ``__geo_interface__``: its tuple coordinates would never compare equal
      to the JSON lists a round-tripped spec holds). ``default=str`` because
      ``read_file`` parses an ISO-date-like attribute (a QGIS edit adds one)
      into a ``Timestamp``, which ``json`` cannot encode.
    * ASSET: the legacy ``asset`` dict, field for field.
    * ADMIN0/1/2: the GAUL code chain (:func:`admin_code_chain`).
    * SHAPE/POINTS: None — the file path was never stored.

    Returns None whenever the inputs are not there; the AOI still loads.
    """
    from pysepal.solara.components.aoi import AoiSpec

    method = metadata.get("method")
    if method == "DRAW" and gdf is not None:
        return AoiSpec(
            method="DRAW",
            name=metadata.get("name") or None,
            geo_json=json.loads(gdf.to_json(default=str)),
        )
    if method == "ASSET":
        asset = metadata.get("asset")
        if isinstance(asset, dict) and asset.get("asset_id"):
            return AoiSpec(
                method="ASSET",
                asset_id=asset["asset_id"],
                asset_type=asset.get("type"),
                column=asset.get("column"),
                value=asset.get("value"),
            )
        return None
    admin = metadata.get("admin")
    if method in _ADMIN_METHODS and admin:
        codes = admin_code_chain(admin, int(method[-1]))
        return AoiSpec(method=method, admin_codes=codes) if codes else None
    return None


def _restore_spec(metadata: Dict[str, Any], gdf: Any) -> Optional[Any]:
    """Return the manifest's ``AoiSpec``, or a synthesized one for old manifests.

    A spec this pysepal cannot read (a newer ``schema_version``, a payload with
    no method) falls back to the legacy fields instead of failing the load, and
    a legacy synthesis that fails returns None: a project must always open,
    even if its picker comes back blank.
    """
    payload = metadata.get("aoi_spec")
    if payload:
        from pysepal.solara.components.aoi import AoiSpec

        try:
            return AoiSpec.from_dict(payload)
        except Exception:
            logger.warning(
                "Could not read the saved AOI selection (%r); rebuilding it from "
                "the legacy AOI fields instead.",
                payload.get("schema_version") if isinstance(payload, dict) else None,
                exc_info=True,
            )
    try:
        return _legacy_spec(metadata, gdf)
    except Exception:
        logger.warning(
            "Could not rebuild the AOI selection from the legacy %r fields; the "
            "AOI loads with a blank picker.",
            metadata.get("method"),
            exc_info=True,
        )
        return None


def load_aoi(project_dir: Path, metadata: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Reconstruct an ``AoiResult`` from persisted metadata + sidecar geometry.

    Args:
        project_dir: The project folder containing ``aoi.geojson``.
        metadata: The ``project.aoi`` dict (or None).

    Returns:
        A reconstructed ``AoiResult``, or None when there is nothing to restore.
        Vector AOIs carry their ``gdf`` (from the sidecar). GEE admin AOIs carry
        no sidecar — their lazy EE ``feature_collection`` is rebuilt from the
        persisted ``admin`` code (``gdf`` stays None). ``feature_collection`` is
        None only when neither applies or the rebuild fails. ``spec`` holds the
        picker's selection for ``AoiView(spec=)`` — read from the manifest, or
        synthesized for a pre-spec one (see :func:`_legacy_spec`) — and is None
        when there is nothing to restore the picker to.
    """
    if not metadata:
        return None

    import geopandas as gpd
    from pysepal.solara.components.aoi import AoiResult

    method = metadata.get("method")
    admin = metadata.get("admin")
    gee = bool(metadata.get("gee", False))

    gdf = None
    geometry_file = metadata.get("geometry_file")
    if geometry_file:
        path = Path(project_dir) / geometry_file
        if path.exists():
            gdf = gpd.read_file(path)

    spec = _restore_spec(metadata, gdf)

    # GEE admin selections (ADMIN0/1/2) carry no geometry sidecar — only the
    # GAUL ``admin`` code is persisted. Rebuild the lazy EE FeatureCollection so
    # the restored AOI has usable geometry; otherwise it loads "present" but
    # empty and the Variables step fails with "no usable geometry — re-select
    # the area", forcing the user to reselect just to continue.
    feature_collection = None
    asset = spec.asset_data() if spec is not None else metadata.get("asset")
    if gdf is None and gee and admin and method in _ADMIN_METHODS:
        feature_collection = _rebuild_admin_feature_collection(admin)
    elif gee and method == "ASSET" and asset:
        feature_collection = _rebuild_asset_feature_collection(asset)

    return AoiResult(
        method=method,
        name=metadata.get("name"),
        gdf=gdf,
        feature_collection=feature_collection,
        admin=admin,
        gee=gee,
        spec=spec,
    )
