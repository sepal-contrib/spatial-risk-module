"""Generate sample point locations from a raster + mask (Solara-free)."""
from pathlib import Path
from typing import Optional

from spatialrisk.sampling.random import RandomSampling
from spatialrisk.sampling.stratified import StratifiedSampling
from spatialrisk.sampling.systematic import SystematicSampling
from spatialrisk.sampling.types import SamplingStrategy

_STRATEGIES = {
    SamplingStrategy.random: RandomSampling,
    SamplingStrategy.stratified: StratifiedSampling,
    SamplingStrategy.systematic: SystematicSampling,
}


def generate_points(
    raster_path,
    mask_path: Optional[Path] = None,
    *,
    strategy: str,
    n_samples: Optional[int],
    allocation: Optional[str] = None,
    seed: Optional[int] = None,
    adapt: bool = False,
    spacing_m: Optional[float] = None,
    rows_per_stripe: Optional[int] = None,
):
    """Draw sample locations and return a GeoDataFrame of point centres.

    The raster is never held in memory: it is walked as full-width row stripes
    (see :mod:`spatialrisk.sampling.blocked`) so peak memory scales with one
    stripe plus the number of points, not with the raster. The whole-raster
    version cost 21.6 bytes of working RAM per raster pixel — about 45 GiB on a
    2.22 Gpx country raster, which is where sample generation ran out of memory.

    Output is unchanged for a given seed: the strategies draw the same ranks
    from the same RNG calls and a second stripe pass converts those ranks back
    into pixel coordinates.

    ``rows_per_stripe`` overrides the stripe height, for tests and tuning; by
    default the scan picks a whole number of tile rows (512 for the usual 256 px
    tiles) so a stripe boundary never makes GDAL decode a tile twice.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio

    from spatialrisk.sampling.blocked import RasterScan

    strategy_enum = SamplingStrategy(strategy)

    # Reject an unbounded request BEFORE opening anything. n_samples=None means
    # "return every valid pixel", which is ~373 GiB of point construction on a
    # 2.22 Gpx raster. Only systematic-with-spacing means it intentionally (the
    # grid spacing bounds the result there). The dialog guards this too, but
    # `Sample` can be built directly in code and bypass it, so the service
    # boundary has to be the one that cannot be skipped.
    if n_samples is None and not (
        strategy_enum is SamplingStrategy.systematic and spacing_m is not None
    ):
        raise ValueError(
            "n_samples is required for "
            f"'{strategy_enum.value}' sampling: n_samples=None means 'every valid "
            "pixel', which is unbounded on a large raster. Only systematic "
            "sampling with an explicit spacing_m may omit it."
        )

    impl = _STRATEGIES[strategy_enum]()
    with RasterScan(raster_path, mask_path, rows_per_stripe=rows_per_stripe) as scan:
        transform = scan.transform
        crs = scan.crs
        # pixel area in hectares from a projected transform (m^2 -> ha)
        pixel_area_ha = abs(transform.a * transform.e) / 10_000.0
        # pixel size (row, col) in metres for distance-based spacing
        res_m = (abs(transform.e), abs(transform.a))

        rows, cols, values = impl.select_blocked(
            scan,
            n_samples=n_samples,
            seed=seed,
            allocation=allocation,
            adapt=adapt,
            pixel_area_ha=pixel_area_ha,
            spacing_m=spacing_m,
            res_m=res_m,
        )

    # Vectorised throughout: list(rows)/list(cols) plus a [Point(x, y) ...]
    # comprehension cost 240 bytes per point; ndarrays straight into
    # transform.xy plus points_from_xy measure 23 bytes per point and ~9x
    # faster (1 M points: 229 MiB/2.8 s -> 23 MiB/0.3 s).
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    gdf = gpd.GeoDataFrame(
        {
            "strata": values.astype(int),
            "row": np.asarray(rows, dtype=int),
            "col": np.asarray(cols, dtype=int),
        },
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )
    return gdf
