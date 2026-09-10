"""Map helpers for the Spatial Risk GUI.

Kept free of Solara/ipyvuetify so the map-interaction logic can be unit-tested
without a render harness.
"""


GOOGLE_SATELLITE = "SATELLITE"
"""pysepal basemap key for the Google Satellite XYZ tiles (imagery, no labels)."""


def add_satellite_basemap(map_, basemap: str = GOOGLE_SATELLITE) -> bool:
    """Add a satellite basemap alongside the theme-driven CartoDB one.

    ``SepalMap`` starts with a single base layer (``CartoDB.Positron`` /
    ``DarkMatter``, swapped on theme change), so the layers control offers no
    basemap choice. Adding Google Satellite gives the user two entries in the
    basemap radio group. The new layer starts hidden, keeping CartoDB the one
    actually drawn until satellite is picked.

    Idempotent, so it stays safe on the session-memoized map.

    Note: pysepal replaces the CartoDB layer on every theme change and its
    layers control re-selects the first visible base layer, which sends the
    selection back to CartoDB after a light/dark toggle.

    Args:
        map_: SepalMap instance (or None).
        basemap: key from pysepal's ``basemap_tiles``.

    Returns:
        True if the basemap was added, False if there was no map or it was
        already present.
    """
    if map_ is None:
        return False

    # Lazy so the rest of this module stays free of the pysepal import chain.
    from pysepal.mapping.basemaps import basemap_tiles

    name = basemap_tiles[basemap].name
    if any(getattr(lyr, "name", None) == name for lyr in map_.layers):
        return False

    map_.add_basemap(basemap)

    for lyr in map_.layers:
        if getattr(lyr, "name", None) == name:
            lyr.visible = False

    return True


def is_mappable(var) -> bool:
    """True if a variable can be drawn on the map.

    A variable is mappable when it carries a fetched GEE image (``gee_images``),
    is a GEE asset the map toggle can still resolve by id (``GEEVar`` whose
    ``path`` is a ``users/`` or ``projects/`` asset id), or is a local
    file-backed layer (``LocalRasterVar`` / ``LocalVectorVar``).
    Kept type-name based so this module stays free of the variable import chain.
    """
    if getattr(var, "gee_images", None):
        return True
    type_name = type(var).__name__
    if type_name == "GEEVar":
        path = getattr(var, "path", None)
        return isinstance(path, str) and path.startswith(("users/", "projects/"))
    return type_name in ("LocalRasterVar", "LocalVectorVar")


def add_vector_on_map(map_, path, name: str, key: str, style: dict = None):
    """Draw a local vector file on ``map_`` as a GeoJSON outline overlay.

    Reprojects to WGS84 (required by ipyleaflet) and replaces any existing layer
    registered under ``key`` so re-toggling doesn't stack duplicates.

    Args:
        map_: SepalMap instance.
        path: Path to a vector file readable by geopandas (shp/geojson/gpkg/…).
        name: Display name for the layer.
        key: Unequivocal map-layer key (used for later removal).
        style: ipyleaflet GeoJSON style; defaults to a black, unfilled outline.
    """
    import geopandas as gpd
    from pysepal.mapping import get_ipygeojson

    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    style = style or {"color": "#000000", "weight": 2, "fillOpacity": 0.0}

    map_.remove_layer(key, none_ok=True)
    layer = get_ipygeojson(gdf, name, style)
    map_.add_layer(layer, key=key)
    return layer


def clear_project_overlays(map_) -> None:
    """Remove every app-added overlay layer from the shared map, keeping basemaps.

    The ``SepalMap`` instance is memoized for the session and reused across
    project switches, so a previously open project's variable / sample-point /
    prediction / AOI layers linger when another project is loaded or created.
    ``remove_all(base=False)`` drops all overlays while preserving the base
    layers (basemaps). No-op when there is no map.
    """
    if map_ is None:
        return
    map_.remove_all(base=False)


def sync_draw_control_visibility(
    map_, aoi_active: bool, was_hidden: bool, project_switched: bool = False
) -> bool:
    """Keep the AOI draw control on the map only while the AOI tab is active.

    ``rv.TabsItems`` hides inactive tabs client-side without unmounting them,
    so ``AoiView`` (which owns ``map_.dc`` while the DRAW method is selected)
    never runs its unmount cleanup on a tab switch — the Geoman toolbar and the
    editable drawn shape would stay on the shared map on every step. This
    helper removes the control when the user leaves the AOI tab and re-adds it
    on return, going through ``add_control``/``remove_control`` directly:
    ``dc.show()``/``dc.hide()`` also wipe ``dc.data``, which would lose the
    drawn shape.

    Args:
        map_: SepalMap instance (or None).
        aoi_active: whether the AOI tab is the active workflow step.
        was_hidden: the value returned by the previous call — True when this
            helper removed the control on leaving the AOI tab.
        project_switched: whether a project load/close happened since the
            previous call. Only then may an absent control drop the flag.

    Returns:
        The new ``was_hidden`` state for the caller to carry to the next call.
        While away, the flag is PRESERVED when the control is absent: the
        effect re-runs for reasons other than leaving the AOI tab (moving
        between two non-AOI tabs, a loading flip), and recomputing the flag
        from the map on those runs forgot the earlier hide — the toolbar then
        never came back ("appears once" bug). A project switch is the one
        event that invalidates the memory: its restore decides the control's
        fate itself (a DRAW restore re-seeds it, a non-DRAW one keeps it off),
        so on ``project_switched`` an absent control resets the flag and a
        pending re-add is skipped rather than second-guessing the restore.
    """
    dc = getattr(map_, "dc", None)
    if map_ is None or dc is None:
        return was_hidden

    if aoi_active:
        if was_hidden and not project_switched and dc not in map_.controls:
            map_.add_control(dc)
        return False

    if dc in map_.controls:
        map_.remove_control(dc)
        return True
    return False if project_switched else was_hidden


def zoom_map_to_aoi(map_, aoi) -> bool:
    """Zoom ``map_`` to ``aoi`` using pysepal's built-in SepalMap zoom methods.

    Mirrors ``AoiView``'s own zoom logic so a loaded project frames its AOI the
    same way a freshly selected one does:

    * vector AOIs (DRAW / admin-with-geometry) → ``zoom_bounds`` on the
      GeoDataFrame's WGS84 ``total_bounds``;
    * GEE-only AOIs → ``zoom_ee_object`` on the feature collection.

    Args:
        map_: SepalMap instance (or None).
        aoi: pysepal ``AoiResult`` (or None).

    Returns:
        True if a zoom was performed, False if there was nothing to zoom to
        (no map, no AOI, or an AOI carrying neither geometry nor a feature
        collection — e.g. a non-GEE admin selection whose geometry was not
        fetched).
    """
    if map_ is None or aoi is None:
        return False

    gdf = getattr(aoi, "gdf", None)
    if gdf is not None:
        map_.zoom_bounds(gdf.total_bounds)
        return True

    feature_collection = getattr(aoi, "feature_collection", None)
    if feature_collection is not None:
        map_.zoom_ee_object(feature_collection)
        return True

    return False


def draw_aoi_on_map(map_, aoi, key: str = "aoi", style: dict = None) -> bool:
    """Draw a vector ``aoi``'s geometry on ``map_`` as a GeoJSON layer.

    Mirrors ``AoiView``'s own vector rendering so a restored AOI looks the same
    as a freshly selected one. Replaces any existing layer under ``key`` so
    re-running on reload doesn't stack duplicates. No-op (returns False) when
    there is no map or no geometry to draw (e.g. a GEE-lazy admin/asset AOI).

    Returns True if a layer was drawn, False otherwise.
    """
    if map_ is None or aoi is None:
        return False

    gdf = getattr(aoi, "gdf", None)
    if gdf is None:
        return False

    # Imported lazily so this module (and zoom_map_to_aoi's tests) stay free of
    # the pysepal/ipyleaflet import chain.
    from pysepal.mapping import get_ipygeojson

    try:
        map_.remove_layer(key, none_ok=True)
    except Exception:
        # Layer may not exist yet, or the map API may differ — drawing proceeds.
        pass

    name = getattr(aoi, "name", None) or key
    layer = get_ipygeojson(gdf, name, style)
    map_.add_layer(layer, key=key)
    return True


def show_aoi_on_map(map_, aoi) -> bool:
    """Draw ``aoi`` and zoom the map to it. Returns True if anything happened."""
    drew = draw_aoi_on_map(map_, aoi)
    zoomed = zoom_map_to_aoi(map_, aoi)
    return drew or zoomed


def sample_layer_keys(key: str) -> tuple:
    """Return the (event, forest) layer keys for a sample-set base ``key``."""
    return (f"{key}__event", f"{key}__forest")


def _split_by_target(gdf, target_col: str = "target"):
    """Split a points GeoDataFrame into (event, forest) by target class.

    Pure (no map/pysepal). ``event`` is target == 1 (deforestation),
    ``forest`` is everything else.
    """
    event = gdf[gdf[target_col] == 1]
    forest = gdf[gdf[target_col] != 1]
    return event, forest


def add_sample_points_on_map(map_, points_path, name: str, key: str):
    """Draw a sample set's points as two colored GeoJSON layers.

    Red points = event (strata == 1), green = forest (strata == 0). Reprojects
    to WGS84 (ipyleaflet requirement) and replaces any existing layers under the
    derived keys so re-toggling doesn't stack duplicates.
    """
    import geopandas as gpd
    from pysepal.mapping import get_ipygeojson

    gdf = gpd.read_file(points_path)
    if gdf.crs is None:
        import warnings

        warnings.warn(
            f"Sample points '{key}' have no CRS; drawing as-is without "
            "reprojection to WGS84.",
            stacklevel=2,
        )
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    event, forest = _split_by_target(gdf, "strata")
    event_key, forest_key = sample_layer_keys(key)

    def _point_style(color):
        return {
            "radius": 4,
            "color": color,
            "fillColor": color,
            "fillOpacity": 0.7,
            "weight": 1,
        }

    for subset, layer_key, color, suffix in (
        (event, event_key, "#d62728", "event"),
        (forest, forest_key, "#2ca02c", "forest"),
    ):
        map_.remove_layer(layer_key, none_ok=True)
        if len(subset) == 0:
            continue
        layer = get_ipygeojson(subset, f"{name} ({suffix})", _point_style(color))
        map_.add_layer(layer, key=layer_key)


def remove_sample_points_from_map(map_, key: str):
    """Remove both layers a sample set placed on the map."""
    for layer_key in sample_layer_keys(key):
        map_.remove_layer(layer_key, none_ok=True)
