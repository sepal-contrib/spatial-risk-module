"""Raster/vector geo helpers: CRS estimation, reprojection, rasterization."""

import warnings
from pathlib import Path
from typing import Literal

import dask
import fiona
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
from odc.geo import xr
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import shape

from spatialrisk.parallel import worker_threads

# Chunk spec for every lazy raster read in the package. rioxarray turns
# ``chunks="auto"`` (and ``True``) into a dimension-order tuple internally,
# which its own ``.chunk()`` call then deprecates -- one FutureWarning per
# process, landing on whatever variable happens to be processed first. Naming
# the dimensions avoids that path.
RASTER_CHUNKS = {"band": 1, "x": "auto", "y": "auto"}


def calculate_utm_rioxarray(
    input_tif_file_path: str | Path,
    output_mode: Literal["str", "int"] = "str",
) -> str | int | None:
    """
    Estimate the UTM CRS of a GeoTIFF file using rioxarray.

    Parameters
    ----------
    input_tif_file_path : str | Path
        The path to the raster file.
    output_mode : Literal["str", "int"], default="str"
        How to return the CRS.  ``"str"`` returns an EPSG string (e.g.,
        ``EPSG:32633``); ``"int"`` returns the numeric EPSG code.

    Returns:
    -------
    str | int | None
        The CRS representation or ``None`` if it could not be estimated.

    Raises:
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If an unsupported `output_mode` is supplied.
    RuntimeError
        If rioxarray fails to open the raster.

    Examples:
    --------
    >>> calculate_utm_rioxarray("sample.tif")
    'EPSG:32633'
    >>> calculate_utm_rioxarray("sample.tif", output_mode="int")
    32633
    """
    # Validate path
    path = Path(input_tif_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster file {input_tif_file_path!r} does not exist")

    try:
        raster = rioxarray.open_rasterio(
            path,
            chunks={"band": 1, "x": "auto", "y": "auto"},
            cache=False,
            lock=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to open raster {path!s}") from exc

    try:
        utm_crs = raster.rio.estimate_utm_crs()
        if utm_crs is None:
            return None

        if output_mode == "str":
            return utm_crs.to_string()
        elif output_mode == "int":
            epsg_code = utm_crs.to_epsg()  # may still be None
            return epsg_code
        else:  # pragma: no cover
            raise ValueError(f"output_mode must be 'str' or 'int', got {output_mode!r}")
    finally:
        # Ensure the dataset is closed to free resources
        raster.close()


def get_centroid(shapefile_path):
    """
    Get the centroid of the first feature in a shapefile using Shapely and Fiona.

    Parameters:
        shapefile_path (str): Path to the input shapefile.

    Returns:
        tuple: A tuple containing the latitude and longitude of the centroid.
    """
    # Open the shapefile
    with fiona.open(shapefile_path, "r") as shapefile:
        # Get the first feature
        first_feature = next(iter(shapefile))

        # Convert the feature geometry to a Shapely geometry object
        geom = shape(first_feature["geometry"])

        # Calculate the centroid
        centroid = geom.centroid

        # Get the coordinates of the centroid
        longitude, latitude = centroid.x, centroid.y

    return (longitude, latitude)


def get_utm_proj_str_from_lat_lon(lon, lat):
    """Return the UTM/UPS EPSG code string for a WGS84 longitude/latitude.

    - UTM: EPSG:326xx (Northern) or EPSG:327xx (Southern)
    - UPS: EPSG:5041 (North, >84°N), EPSG:5042 (South, <-80°S)

    Handles special cases for Norway and Svalbard.
    """
    # UPS zones for polar regions
    if lat >= 84:
        return "EPSG:5041"  # UPS North
    elif lat <= -80:
        return "EPSG:5042"  # UPS South

    # Special cases for Norway and Svalbard
    if lat > 55 and lat < 64 and lon > 2 and lon < 6:
        zone_number = 32
    elif lat > 71 and lon >= 6 and lon < 9:
        zone_number = 31
    elif lat > 71 and ((lon >= 9 and lon < 12) or (lon >= 18 and lon < 21)):
        zone_number = 33
    elif lat > 71 and ((lon >= 21 and lon < 24) or (lon >= 30 and lon < 33)):
        zone_number = 35
    else:
        zone_number = int((lon + 180) / 6) + 1

    if lat >= 0:
        epsg_code = 32600 + zone_number  # Northern Hemisphere
    else:
        epsg_code = 32700 + zone_number  # Southern Hemisphere

    return f"EPSG:{epsg_code}"


def process_forest_loss(input1_path, input2_path, output_path):
    """Combine two forest rasters into a loss raster (1 = loss between them)."""
    # Open the input rasters
    with rasterio.open(input1_path) as src1:
        input1 = src1.read(1)
        bounds1 = src1.bounds
        profile = src1.profile
        nodata1 = src1.nodata

    with rasterio.open(input2_path) as src2:
        input2 = src2.read(1)
        bounds2 = src2.bounds
        nodata2 = src2.nodata

    # Check if the bounds of input1 are equal to or larger than those of input2
    if not (
        bounds1.left <= bounds2.left
        and bounds1.right >= bounds2.right
        and bounds1.top >= bounds2.top
        and bounds1.bottom <= bounds2.bottom
    ):
        raise ValueError(
            "The bounds of input1 must be equal to or larger than those of input2."
        )

    # Create masks for valid data
    valid_mask = (input1 != nodata1) & (input2 != nodata2)

    # Initialize output with nodata (255)
    output = np.full(input1.shape, 255, dtype=np.uint8)

    # Set values based on conditions:
    # 1 where input1 == 1 and input2 == 0
    # 0 where input1 == 1 and input2 == 1
    # nodata (255) for all other cases

    # Create condition for 1s: input1 == 1 AND input2 == 0
    condition_1 = (input1 == 1) & (input2 == 0)

    # Create condition for 0s: input1 == 1 AND input2 == 1
    condition_0 = (input1 == 1) & (input2 == 1)

    # Apply conditions only where both inputs are valid
    output[valid_mask & condition_1] = 0
    output[valid_mask & condition_0] = 1

    # Update the profile for the output raster
    profile.update(dtype=rasterio.uint8, compress="deflate", nodata=255)

    # Write the output raster
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)


def raster_is_all_nodata(raster_path: str | Path, band: int = 1) -> bool:
    """True when every pixel of ``band`` equals the file's declared nodata.

    Cheap first pass: a decimated (max 1024x1024) read — any valid pixel there
    settles it without loading the raster. Only when the decimated view is
    entirely nodata does it fall back to a full windowed scan, so sparse valid
    pixels can't slip through the decimation. Files with no declared nodata
    are never "all nodata".
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        if nodata is None:
            return False

        def _all_nodata(arr):
            if np.isnan(nodata):
                return bool(np.isnan(arr).all())
            return bool((arr == nodata).all())

        out_shape = (min(src.height, 1024), min(src.width, 1024))
        if not _all_nodata(src.read(band, out_shape=out_shape)):
            return False
        if out_shape == (src.height, src.width):
            return True
        for _, window in src.block_windows(band):
            if not _all_nodata(src.read(band, window=window)):
                return False
        return True


def reproject_shapefile(
    input_path: str,
    output_path: str,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Reprojects a shapefile to a target CRS.

    Args:
        input_path (str): Path to input shapefile
        target_crs (str): Target CRS (e.g., "EPSG:4326")
        output_path (str, optional): Path to save reprojected shapefile.
            If None, returns GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Reprojected GeoDataFrame
    """
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(target_crs)
    if output_path:
        gdf.to_file(output_path)
    return None


# Destination tile size (pixels) for the warp. odc.geo defaults the *output*
# chunk shape to the *input* array's chunk shape, so a coarse source (ERA5-Land
# precipitation is one ~55x54-pixel chunk) would tile a 20166x19960 30 m output
# into 135790 blocks — one GDAL warp setup each, ~100x slower than necessary
# and a stream of NotGeoreferencedWarnings from the temporary block datasets.
DST_CHUNK = 2048

# Source tile size (pixels) for the warp, matched to DST_CHUNK. Left at
# rioxarray's "auto" a uint8 source is read in 128 MiB (~11.6k x 11.6k) tiles,
# and every destination tile copies the concatenation of every source tile it
# overlaps -- up to four of them per worker thread. Measured 2026-09-18 on the
# Bolivia project (45k x 49k source, 12k x 12k destination crop, 16 threads):
# 6.8 GB peak with "auto", 1.0 GB with 2048.
SRC_CHUNK = DST_CHUNK

# GDAL block cache during a warp, in bytes. The default is 5 % of physical RAM
# and, on a single-pass read-then-write, buys nothing but that much extra RSS
# (measured: a 24k x 24k warp's peak dropped 1.9 GB -> 1.1 GB at the same wall
# clock). Same reasoning as ``parallel.SCAN_CACHEMAX_BYTES``.
WARP_CACHEMAX_BYTES = 64 * 1024 * 1024


def xr_reproject(
    raster_path: str = None,
    geobox=None,
    resampling_method="nearest",
    output_path: str = None,
    cast_dtype: str = None,
    **rasterio_kwargs,
):
    """Reproject a raster onto a target geobox and write it as a GeoTIFF.

    Parameters
    ----------
    raster_path : str
        Path to the input raster.
    geobox : odc.geo.geobox.GeoBox
        The spatial template defining the shape, coordinates, dimensions,
        and transform of the output raster.
    resampling_method : str, optional
        Resampling algorithm accepted by ``odc.geo.xr.xr_reproject``
        (e.g. 'nearest', 'bilinear').
    output_path : str, optional
        File path for the reprojected GeoTIFF.
    cast_dtype : str, optional
        Cast the source to this dtype *before* warping. Left as ``None`` the
        source dtype is preserved, which is what every caller wants unless the
        warp's own fill value matters to it: ``odc`` fills outside the source
        footprint with ``resolve_fill_value``, which is 0 for an integer array
        but NaN for a float one. Casting to a float dtype therefore makes the
        fill distinguishable from a genuine 0. The cast rides the existing dask
        graph, so it costs no extra pass or temporary file.
    **rasterio_kwargs :
        Unused; kept for signature compatibility.

    Returns:
    -------
    None
        The result is written to ``output_path``.
    """
    # Read the raster lazily, tiled to match the destination tiling (see
    # SRC_CHUNK). Nothing is decoded until the store below pulls a tile.
    raster_array = rioxarray.open_rasterio(
        raster_path,
        chunks={"band": 1, "x": SRC_CHUNK, "y": SRC_CHUNK},
        cache=False,
        lock=False,
    )
    if cast_dtype is not None:
        raster_array = raster_array.astype(cast_dtype)

    # Convert numpy array to a full xarray.DataArray
    # and set array name if supplied
    # Pin the destination tiling to the output grid (see DST_CHUNK) instead of
    # letting it inherit the source's chunk shape.
    dst_ny, dst_nx = geobox.shape
    da_reprojected = xr.xr_reproject(
        src=raster_array,
        how=geobox,
        resampling=resampling_method,
        chunks=(min(DST_CHUNK, dst_ny), min(DST_CHUNK, dst_nx)),
    )

    # DEFLATE predictor is dtype-specific: 2 (horizontal differencing) is only
    # defined for integer samples, 3 is the floating-point variant.
    predictor = 3 if np.issubdtype(da_reprojected.dtype, np.floating) else 2

    # odc.geo warps one in-memory numpy block at a time, passing the
    # georeferencing beside the array as src_transform/dst_transform. GDAL still
    # warns that the block dataset it wraps them in "has no geotransform" --
    # noise about a temporary, never about the input (which the geobox above
    # guarantees is georeferenced). Suppressed only around the warp, and only
    # for that one category, so a genuinely non-georeferenced input still fails
    # loudly in the read above.
    #
    # Three things keep this warp's memory independent of the output size, all
    # measured 2026-09-18 on the Bolivia project (harmonize-all at country
    # scale killed the kernel on a 16 GB machine):
    #
    # * ``lock=True``. rioxarray's default ``lock=None`` takes the *non-dask*
    #   branch of its writer, which evaluates the whole dask graph into one
    #   numpy array before the first ``write``: the entire output layer in
    #   RAM, with the concatenation copies on top (a float32 layer at Peru
    #   scale is >10 GB before those). With a lock the writer goes through
    #   ``dask.array.store`` and each destination tile is written as soon as
    #   it is warped, so the resident set is a handful of tiles.
    # * The worker pool. dask's threaded scheduler defaults to one thread per
    #   core, and every worker holds its own source window and destination
    #   tile. ``worker_threads`` is the shared half-the-cores policy the
    #   sampling and evaluation scans already follow.
    # * The GDAL block cache, capped as on the scan paths (WARP_CACHEMAX_BYTES).
    #
    # Full-size uint8 layer, 8 workers, this machine: 10.2 GB peak before,
    # see tests/test_xr_reproject_streaming.py for the guard.
    with (
        warnings.catch_warnings(),
        rasterio.Env(GDAL_CACHEMAX=WARP_CACHEMAX_BYTES),
        dask.config.set(scheduler="threads", num_workers=worker_threads()),
    ):
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        da_reprojected.rio.to_raster(
            output_path,
            driver="GTiff",
            compress="DEFLATE",
            predictor=predictor,
            bigtiff="YES",
            tiled=True,
            lock=True,
        )

    # Explicitly close references - not strictly required but tidy.
    del raster_array
    del da_reprojected
