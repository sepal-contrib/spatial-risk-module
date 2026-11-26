"""AOI helpers shared by every predefined layer builder."""

import ee


def get_aoi_ee_feature(gdf):
    """Convert a GeoDataFrame AOI to an ee.Feature for GEE operations."""
    import json

    geojson = json.loads(gdf.dissolve().to_json())
    geometry = ee.Geometry(geojson["features"][0]["geometry"])
    return ee.Feature(geometry)


def resolve_aoi_ee(aoi_result):
    """Resolve an AoiResult to an Earth Engine object for variable extraction.

    AOI selections expose their geometry one of two ways:
      * DRAW / local selections -> a GeoDataFrame in ``aoi_result.gdf``.
      * Admin-boundary / GEE-asset selections -> an ee object in
        ``aoi_result.feature_collection`` (``gdf`` is None — fetched lazily on GEE).

    Prefer the GEE object when present (it covers admin/asset selections, which
    have no local gdf); otherwise convert the local gdf. The returned object
    supports ``.geometry()``/``.clip()``/``.filterBounds()`` either way.
    """
    fc = getattr(aoi_result, "feature_collection", None)
    if fc is not None:
        return fc
    if getattr(aoi_result, "gdf", None) is not None:
        return get_aoi_ee_feature(aoi_result.gdf)
    raise ValueError(
        "Selected AOI has no usable geometry. Re-select the area "
        "(admin boundaries need GEE enabled to fetch their geometry)."
    )


def _aoi_bounds(aoi):
    """Bounding rectangle of an AOI (Geometry, Feature or FeatureCollection).

    Use this — never the AOI itself or ``aoi.geometry()`` — as the
    ``filterBounds`` argument of a spatial pre-filter. Passing the AOI makes
    Earth Engine materialise its geometry inside the download request, and a
    large table asset (a single feature of ~1.1M coordinates, reproduced
    2026-09-11) then fails every tile with "Description length exceeds
    maximum". Pre-filters are an optimisation only: each layer is clipped to
    the AOI afterwards (``clip`` accepts the AOI object directly), so the
    rectangle is sufficient.
    """
    geom = aoi if isinstance(aoi, ee.Geometry) else aoi.geometry()
    return geom.bounds()
