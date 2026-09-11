"""FAO GAUL data utilities.

Functions for selecting FAO GAUL (Global Administrative Unit Layers) features
from Google Earth Engine.
"""

import ee


def get_fao_gaul_features(
    level: int, code: str, filter_attribute: str = "iso3_code"
) -> ee.FeatureCollection:
    """
    Selects features from FAO GAUL 2024 dataset based on the level and code provided.

    This function allows users to select administrative features at any level
    (0, 1, or 2) using ISO codes. It returns a FeatureCollection containing the
    selected features.

    Parameters
    ----------
    level : int
        The administrative level to select (0, 1, or 2).
        Level 0: Country level
        Level 1: First administrative level (e.g., states/provinces)
        Level 2: Second administrative level (e.g., districts)
    code : str
        ISO code for level 0, or identifier for levels 1 and 2.
        For level 0: ISO-3 country code (e.g., "BRA" for Brazil)
        For level 1: ADM1_CODE value (e.g., "DZA01" for Adrar, Algeria)
        For level 2: ADM2_CODE value (e.g., "DZA0101" for Adrar district in
        Adrar province, Algeria)
    filter_attribute : str, optional
        The attribute name to use for filtering. Default is "iso3_code".

    Returns:
    -------
    ee.FeatureCollection
        The selected FeatureCollection from the FAO GAUL 2024 dataset.

    Raises:
    ------
    ValueError
        If level is not 0, 1, or 2, or if code is not provided.

    Notes:
    -----
    The function assumes Earth Engine has already been initialized (`ee.Initialize()`)
    in your script or notebook.
    """
    # Validate inputs
    if level not in (0, 1, 2):
        raise ValueError("`level` must be 0, 1, or 2.")

    if not code:
        raise ValueError("`code` must be provided.")

    # Define FAO GAUL 2024 paths
    gaul_paths = {
        0: "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L0",
        1: "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L1",
        2: "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L2",
    }

    # Get the FeatureCollection for the specified level
    fao_gaul_fc = ee.FeatureCollection(gaul_paths[level])

    # Select features based on the code using the filter attribute
    selected_features = fao_gaul_fc.filter(ee.Filter.eq(filter_attribute, code))

    return selected_features


def _as_feature_collection(aoi):
    """Wrap a Geometry / Feature / FeatureCollection AOI as a FeatureCollection.

    The spatial join in :func:`get_fao_gaul_subj` needs a collection on the
    secondary side. Only asset-backed collections keep their geometry out of
    the request; an inline Geometry/Feature (drawn or uploaded AOI) is small
    enough to serialise, and is wrapped as is.
    """
    if isinstance(aoi, ee.FeatureCollection):
        return aoi
    if isinstance(aoi, ee.Feature):
        return ee.FeatureCollection([aoi])
    return ee.FeatureCollection([ee.Feature(aoi)])


def get_fao_gaul_subj(level: int, feature_collection: ee.FeatureCollection):
    """
    Selects features from FAO GAUL 2024 dataset based on the level provided.

    This function allows users to select administrative features at any level (1 or 2).
    It returns a FeatureCollection containing the selected features.

    Parameters
    ----------
    level : int
        The administrative level to select.
        Level 1: First administrative level (e.g., states/provinces)
        Level 2: Second administrative level (e.g., districts)
    feature_collection : ee.FeatureCollection
        A FeatureCollection used to filter the FAO GAUL dataset.

    Returns:
    -------
    ee.FeatureCollection
        The selected FeatureCollection from the FAO GAUL 2024 dataset

    Raises:
    ------
    ValueError
        If `level` is not 1 or 2.
    """
    # Validate inputs
    if level not in (1, 2):
        raise ValueError("`level` must be 1 or 2.")

    # Define FAO GAUL 2024 paths
    gaul_paths = {
        1: "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L1",
        2: "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L2",
    }

    gaul_names_attribute = {
        1: "gaul1_name",
        2: "gaul2_name",
    }

    # Get the FeatureCollection for the specified level
    fao_gaul_fc = ee.FeatureCollection(gaul_paths[level])

    # Get the attribute name for the specified level
    fao_gaul_attribute = gaul_names_attribute[level]

    # Select the GAUL features intersecting the AOI *without* putting the
    # AOI's computed geometry in the expression: ``filterBounds(aoi)`` (or
    # ``aoi.geometry()``) inlines it into any download request built on the
    # result, and a large table asset then fails every tile with
    # "Description length exceeds maximum". A bounding-box pre-filter is not
    # enough on its own — it can select far more districts than actually
    # intersect (345 vs 106 for a Peruvian Amazon AOI), and the consecutive
    # codes are exported as a byte. So: cheap bbox pre-filter, then an exact
    # spatial join against the AOI *collection*, whose geometry EE loads
    # server-side instead of serialising it into the request. Verified
    # 2026-09-11 to select exactly the same features as ``filterBounds`` for
    # admin, drawn and asset AOIs, and to stay lazy (no getInfo at build time).
    aoi_fc = _as_feature_collection(feature_collection)
    candidates = fao_gaul_fc.filterBounds(aoi_fc.geometry().bounds())
    selected_features = ee.Join.simple().apply(
        candidates,
        aoi_fc,
        ee.Filter.intersects(leftField=".geo", rightField=".geo", maxError=10),
    )

    # You might want to use fao_gaul_attribute here for further operations
    # For example, you could select features based on this attribute

    return selected_features, fao_gaul_attribute
