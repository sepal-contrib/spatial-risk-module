"""Download Earth Engine images to local GeoTIFFs with geedim."""

import threading
from typing import Any, Optional, Union

import ee
import geedim as gd  # noqa

from spatialrisk.gee.progress import notify_download_wait

# geedim runs every download through ``geedim.utils.AsyncRunner``: a
# process-wide singleton wrapping ONE ``asyncio.Runner`` (one event loop).
# ``AsyncRunner.run`` only checks whether the *calling* thread already has a
# running loop, so two plain worker threads (the GUI starts one per download
# action) both call ``run_until_complete`` on that shared loop and the second
# dies with "This event loop is already running". Downloads therefore take
# turns here; tiles within one download still fetch concurrently. A download
# that has to wait says so once (``notify_download_wait``) before blocking.
_geedim_lock = threading.Lock()


def _export_bounds(region: Any) -> Any:
    """The export region for geedim: the *bounds* of ``region``, never its geometry.

    Putting the computed ``.geometry()`` of a table asset in a download request
    makes Earth Engine materialise it into the request description; for a
    large AOI (a single-feature asset of ~1.1M coordinates reproduced this,
    2026-09-11) the server answers "Description length exceeds maximum" on
    every tile. The export grid only depends on the bounding rectangle anyway
    — ``bounds()`` yields the identical transform and shape — while the
    footprint itself is applied by ``clip``, which accepts the collection.
    """
    if region is None:
        return None
    geom = region.geometry() if hasattr(region, "geometry") else region
    return geom.bounds()


# ------------------------------------------------------------------
#  Download gee raster data using geedim
# ------------------------------------------------------------------
def download_ee_image(
    image: ee.Image,
    filename: str,
    region: Optional[Union[ee.Geometry, ee.Feature, ee.FeatureCollection]] = None,
    crs: Optional[str] = None,
    crs_transform: Optional[list] = None,
    scale: Optional[float] = None,
    resampling: str = "near",
    dtype: Optional[str] = None,
    overwrite: bool = True,
    num_threads: Optional[int] = None,
    max_tile_size: Optional[int] = None,
    max_tile_dim: Optional[int] = None,
    shape: Optional[tuple[int, int]] = None,
    scale_offset: bool = False,
    unmask_value: Optional[int] = None,
    nodata_value: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Download an Earth Engine Image as a GeoTIFF.

    Parameters
    ----------
    image : ee.Image
        The image to be downloaded.
    filename : str
        Name of the destination file.
    region : ee.Geometry | ee.Feature | ee.FeatureCollection | None, optional
        Area of interest. Used as the ``clip`` footprint (when ``unmask_value``
        is set) and, through its bounds, as the export extent. Pass the AOI
        object itself, not ``aoi.geometry()``. Defaults to the entire image
        granule.
    crs : str | None, optional
        Reproject image(s) to this EPSG or WKT CRS. Where image bands have
        different CRS, all are reprojected to this CRS. Defaults to the CRS of
        the minimum scale band.
    crs_transform : list[float] | None, optional
        List of 6 numbers specifying an affine transform in the specified CRS.
    scale : float | None, optional
        Resample image(s) to this pixel scale (meters).  Where image bands have
        different scales, all are resampled to this scale. Defaults to the
        minimum scale of image bands.
    resampling : str, optional
        Resampling method 'near', 'bilinear', 'bicubic', or 'average'.
    dtype : str | None, optional
        Convert to this data type ('uint8', 'int8', 'uint16', 'int16',
        'uint32', 'int32', 'float32' or 'float64').  Defaults to auto select.
    overwrite : bool, optional
        Overwrite the destination file if it exists. Defaults to True.
    num_threads : int | None, optional
        Number of tiles to download concurrently.
    max_tile_size : int | None, optional
        Maximum tile size (MB).  If None, defaults to the Earth Engine download size
        limit (32 MB).
    max_tile_dim : int | None, optional
        Maximum tile width/height (pixels).  Defaults to Earth Engine download limit
        (10000).
    shape : tuple[int, int] | None, optional
        Desired output dimensions (height, width) in pixels.
    scale_offset : bool, optional
        Whether to apply any EE band scales and offsets to the image.
    unmask_value : int | None, optional
        Value used for masked pixels. Set to a non zero value if you want zeros to be
        treated as data.
    nodata_value : int | None, optional
        Value used for no data raster value.

    Returns:
    -------
    None
        The function writes the GeoTIFF in place; it returns ``None``.
    """
    try:
        import geedim as gd  # noqa
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Please install geedim using 'pip install geedim' or "
            "'conda install -c conda-forge geedim'"
        ) from exc

    if not isinstance(image, ee.Image):  # pragma: no cover
        raise ValueError("image must be an ee.Image.")

    # Apply unmasking/clip logic before export. ``clip`` takes a Geometry,
    # Feature or FeatureCollection as given — never ``region.geometry()``, see
    # ``_export_bounds``.
    if unmask_value is not None:
        if region is not None:
            image = image.clip(region)
        image = image.unmask(unmask_value, sameFootprint=False)

    if not _geedim_lock.acquire(blocking=False):
        notify_download_wait()
        _geedim_lock.acquire()
    try:
        img = image.gd.prepareForExport(
            crs=crs,
            region=_export_bounds(region),
            scale=scale,
            resampling=resampling,
            dtype=dtype,
        )
        nodata = True if nodata_value is None else nodata_value
        img.gd.toGeoTIFF(file=filename, overwrite=overwrite, nodata=nodata, **kwargs)
    finally:
        _geedim_lock.release()

    print(f"File {filename}, downloaded")
