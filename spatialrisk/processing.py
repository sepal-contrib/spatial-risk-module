"""Raster processing helpers: warps, rasterization, distances and change layers."""

from typing import TYPE_CHECKING, List

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from osgeo import gdal

from spatialrisk.gdal_env import configure_gdal_tmpdir
from spatialrisk.variables.models import RasterType

if TYPE_CHECKING:
    from spatialrisk.project import Project
    from spatialrisk.variables import LocalRasterVar


def reproject_raster_gdal_warp(
    input_file: str,
    output_file: str,
    target_epsg: str,
    resolution: int | float = 30,
    resampling_method: str = "near",
) -> None:
    """
    Reproject a raster to a given EPSG code with GDAL, saved with DEFLATE compression.

    Parameters:
    input_file (str): The path to the input raster file.
    output_file (str): The path where the reprojected raster file will be saved.
    target_epsg (str): The EPSG code of the target coordinate reference system
        (e.g., 'EPSG:4326').
    resolution (int | float): Target resolution in the units of the target CRS.
        Default is 30.
    resampling_method (str): Resampling algorithm ('near', 'bilinear', 'cubic',
        etc.). Default is 'near'.

    Returns:
    None
    """
    import os

    # Enable GDAL exceptions to catch errors
    gdal.UseExceptions()

    # Open the input dataset
    dataset = gdal.Open(input_file)
    if not dataset:
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Get projection from the original raster
    src_proj = dataset.GetProjection()

    # For small files, don't use BIGTIFF as it can cause issues
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    creation_options = ["COMPRESS=DEFLATE", "PREDICTOR=2"]
    if file_size_mb > 2000:  # Only use BIGTIFF for files larger than 2GB
        creation_options.append("BIGTIFF=YES")

    param = gdal.WarpOptions(
        warpOptions=["overwrite"],
        srcSRS=src_proj,
        dstSRS=target_epsg,
        targetAlignedPixels=True,
        resampleAlg=resampling_method,
        xRes=resolution,
        yRes=resolution,
        multithread=True,
        creationOptions=creation_options,
    )

    # Perform reprojection
    result = gdal.Warp(output_file, input_file, format="GTiff", options=param)

    if result is None:
        raise RuntimeError("gdal.Warp() failed - check input file and parameters")

    # Close datasets
    result = None
    dataset = None


def xr_rasterize(
    shapefile_path: str = None,
    geobox=None,
    crs=None,
    output_path: str = None,
    mode: str = "binary",
    **rasterio_kwargs,
):
    """
    Rasterizes a vector shapefile into a raster array.

    This function provides unified functionality for both binary and unique ID
    rasterization.

    Parameters
    ----------
    shapefile_path : str
        Path to the input shapefile containing vector data.
    geobox : odc.geo.geobox.GeoBox
        The spatial template defining the shape, coordinates, dimensions, and transform
        of the output raster.
    crs : str or CRS object, optional
        If ``geobox``'s coordinate reference system (CRS) cannot be
        determined, provide a CRS using this parameter.
        (e.g. 'EPSG:3577').
    output_path : string, optional
        Provide an optional string file path to export the rasterized
        data as a GeoTIFF file.
    mode : str, optional
        Rasterization mode: 'binary' or 'unique'.
        - 'binary': Creates a boolean raster with 1s and 0s (default)
        - 'unique': Creates a raster with unique integer IDs for each feature
    **rasterio_kwargs :
        A set of keyword arguments to ``rasterio.features.rasterize``.
        Can include: 'all_touched', 'merge_alg', 'dtype'.

    Returns:
    -------
    da_rasterized : xarray.DataArray
        The rasterized vector data.
    """
    import geopandas as gpd
    import rasterio
    from odc.geo import xr

    # Read the shapefile
    gdf = gpd.read_file(filename=shapefile_path, engine="fiona")

    # Reproject vector data to raster's CRS
    gdf_reproj = gdf.to_crs(crs=geobox.crs)

    # Handle different modes
    if mode == "binary":
        # Binary mode: rasterize into a boolean array with 1s and 0s
        shapes = gdf_reproj.geometry
        values = [1] * len(gdf_reproj)  # All features set to 1
        shapes_and_values = list(zip(shapes, values))

    elif mode == "unique":
        # Unique ID mode: rasterize using unique integer IDs for each feature
        shapes = gdf_reproj.geometry
        # Create unique integer IDs starting from 1
        values = list(range(1, len(gdf_reproj) + 1))
        shapes_and_values = list(zip(shapes, values))

    else:
        raise ValueError("Mode must be either 'binary' or 'unique'")

    # Determine appropriate dtype
    if mode == "unique":
        # For unique IDs, check if there are too many features
        num_features = len(values) if values else 0
        if num_features > 255:
            raise ValueError(
                f"Cannot rasterize with mode='unique': {num_features} features found, "
                f"but unique mode only supports up to 255 unique values. "
                f"Consider using mode='binary' instead to create a simple "
                f"presence/absence raster."
            )
        dtype = "uint8"
    else:
        # Binary mode only needs uint8
        dtype = "uint8"

    # Allow user to override dtype via kwargs
    dtype = rasterio_kwargs.pop("dtype", dtype)

    # Rasterize shapes into a numpy array
    im = rasterio.features.rasterize(
        shapes=shapes_and_values if mode == "unique" else shapes,
        out_shape=geobox.shape,
        transform=geobox.transform,
        dtype=dtype,
        **rasterio_kwargs,
    )

    # Convert numpy array to a full xarray.DataArray
    # and set array name if supplied
    da_rasterized = xr.wrap_xr(im=im, gbox=geobox)

    da_rasterized.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        bigtiff="YES",
        tiled=True,
    )

    # Explicitly close references - not strictly required but tidy.
    del im
    del da_rasterized


def distance_to_edge_gdal_no_mask(
    input_file,
    dist_file,
    values=0,
    nodata=0,
    max_distance_value=4294967295,
    input_nodata=True,
    verbose=False,
):
    """Compute the shortest distance to given pixel values in a raster.

    The original nodata mask is preserved in the output.
    """
    # ComputeProximity() needs a writable scratch dir for its Float32 working
    # band (the destination is UInt32). Cheap and idempotent, so re-assert it
    # here in case this helper is used without the app's import-time setup.
    configure_gdal_tmpdir()

    # Read input file
    src_ds = gdal.Open(input_file)
    srcband = src_ds.GetRasterBand(1)

    # Create raster of distance
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(
        dist_file,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_UInt32,
        ["COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=YES"],
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    dstband = dst_ds.GetRasterBand(1)

    # Use_input_nodata
    ui_nodata = "YES" if input_nodata else "NO"

    # Compute distance
    val = "VALUES=" + str(values)
    use_input_nodata = "USE_INPUT_NODATA=" + ui_nodata
    max_distance = "MAXDIST=" + str(max_distance_value)
    distance_nodata = "NODATA=" + str(nodata)
    cb = gdal.TermProgress_nocb if verbose else 0
    gdal.ComputeProximity(
        srcband,
        dstband,
        [val, use_input_nodata, max_distance, distance_nodata, "DISTUNITS=GEO"],
        callback=cb,
    )

    # Set nodata value
    dstband.SetNoDataValue(max_distance_value)

    # Flush to disk
    dstband.FlushCache()
    dst_ds.FlushCache()

    # Clean up
    srcband = None
    dstband = None
    del src_ds, dst_ds


def process_change_xarray(input1_path, input2_path, output_path, op="loss"):
    """Pixel-wise change detection between two presence masks (1=present, 0=absent).

    Output follows the far-target convention (1 = event of interest):
    - op="loss": 1 where present(t1) -> absent(t2), 0 where present stayed
      present, 255 otherwise (incl. nodata and absent-at-t1).
    - op="gain": 1 where absent(t1) -> present(t2), 0 where absent stayed
      absent, 255 otherwise.

    Inputs must share the same grid (aligned processed layers).
    """
    if op not in ("loss", "gain"):
        raise ValueError(f"op must be 'loss' or 'gain', got {op!r}")

    # Open the input rasters
    input1 = rioxarray.open_rasterio(
        input1_path,
        chunks="auto",
        cache=False,
        lock=False,
    ).squeeze()
    input2 = rioxarray.open_rasterio(
        input2_path,
        chunks="auto",
        cache=False,
        lock=False,
    ).squeeze()

    # Check bounds properly - extract bounds tuple values
    bounds1 = input1.rio.bounds()
    bounds2 = input2.rio.bounds()

    if not (
        bounds1[0] <= bounds2[0]  # left
        and bounds1[2] >= bounds2[2]  # right
        and bounds1[3] >= bounds2[3]  # top
        and bounds1[1] <= bounds2[1]  # bottom
    ):
        raise ValueError(
            "The bounds of input1 must be equal to or larger than those of input2."
        )

    # Create masks for valid data
    nodata1 = input1.rio.nodata
    nodata2 = input2.rio.nodata
    valid_mask = (input1 != nodata1) & (input2 != nodata2)

    if op == "loss":
        event = valid_mask & (input1 == 1) & (input2 == 0)
        stable = valid_mask & (input1 == 1) & (input2 == 1)
    else:  # gain
        event = valid_mask & (input1 == 0) & (input2 == 1)
        stable = valid_mask & (input1 == 0) & (input2 == 0)

    output = xr.where(event, 1, xr.where(stable, 0, 255)).astype("uint8")

    # Set proper metadata
    output.rio.write_nodata(255, inplace=True)
    output.rio.write_crs(input1.rio.crs, inplace=True)
    output.rio.write_transform(input1.rio.transform(), inplace=True)

    output.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        bigtiff="YES",
        tiled=True,
    )


def process_forest_loss_xarray(input1_path, input2_path, output_path):
    """Back-compat wrapper: forest loss = change detection with op="loss".

    Kept for the legacy notebook path (make_forest_loss_var /
    get_forest_loss_calculated). New code should call process_change_xarray.
    """
    process_change_xarray(input1_path, input2_path, output_path, op="loss")


def generate_deforestation_raster(
    raster_1: "LocalRasterVar",
    raster_2: "LocalRasterVar",
    raster_3: "LocalRasterVar",
    project: "Project",
):
    """
    Generate a deforestation raster from three input rasters.

    Parameters:
    - raster1_path: Path to the first raster file (period 1).
    - raster2_path: Path to the second raster file (period 2).
    - raster3_path: Path to the third raster file (period 3).
    - output_path: Path to save the output raster file.
    """
    # Import here to avoid circular dependency
    from spatialrisk.variables import LocalRasterVar

    # Open the input rasters
    with (
        rasterio.open(raster_1.path) as src1,
        rasterio.open(raster_2.path) as src2,
        rasterio.open(raster_3.path) as src3,
    ):
        # Read the data into numpy arrays
        raster1 = src1.read(1)
        raster2 = src2.read(1)
        raster3 = src3.read(1)

        # Create an output array initialized with NoData value (0)
        output_raster = np.zeros_like(raster1, dtype=np.uint8)

        # Set the values based on deforestation periods
        output_raster[
            (raster1 == 1) & (raster2 == 0)
        ] = 1  # Deforestation in period 1-2
        output_raster[
            (raster2 == 1) & (raster3 == 0)
        ] = 2  # Deforestation in period 2-3
        # Set the remaining forest value only where no deforestation has been marked
        output_raster[
            (output_raster == 0) & (raster3 == 1)
        ] = 3  # Remaining forest in period 3

    # Define the metadata for the output raster
    meta = src1.meta
    meta.update({"count": 1, "dtype": np.uint8, "nodata": 0, "compress": "deflate"})

    # extract years data from each of the rasters
    years = [raster_1.year, raster_2.year, raster_3.year]

    output_name = "defostack" + "_".join(map(str, years)) + ".tif"
    output_path = str(project.folders.data_raw_folder / output_name)

    # Write the output raster to a file
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(output_raster, 1)

    return LocalRasterVar(
        name="defostack",
        path=output_path,
        raster_type="categorical",
        project=project,
        tags=["deforestation"],
    )


def make_forest_loss_var(
    project: "Project",
    start_layer: "LocalRasterVar",
    end_layer: "LocalRasterVar",
) -> "LocalRasterVar":
    """Create one forest-loss target from two forest layers (start -> end year).

    Idempotent: if the output raster already exists, it is reused rather than
    regenerated. Returns an (unregistered) LocalRasterVar; the caller decides
    whether to add_as_raw().
    """
    from pathlib import Path

    from spatialrisk.variables import LocalRasterVar

    start_year = start_layer.year
    end_year = end_layer.year
    var_name = f"forest_loss_{start_year}_{end_year}"
    output_path = project.folders.data_raw_folder / f"{var_name}.tif"

    if not output_path.exists():
        print(f"Calculating forest loss between {start_year} and {end_year}...")
        process_forest_loss_xarray(
            str(start_layer.path),
            str(end_layer.path),
            str(output_path),
        )

    return LocalRasterVar(
        name=var_name,
        path=Path(output_path),
        raster_type=RasterType.categorical,
        project=project,
        tags=["deforestation", "forest_loss", f"{start_year}_{end_year}"],
        default_crs=end_layer.default_crs or start_layer.default_crs,
        default_resolution=end_layer.default_resolution
        or start_layer.default_resolution,
    )


def get_forest_loss_calculated(
    project: "Project", forest_layers: List["LocalRasterVar"]
) -> List["LocalRasterVar"]:
    """
    Calculate forest loss between different time periods.

    Creates three forest loss rasters:
    1. Loss between year[0] and year[1]
    2. Loss between year[0] and year[2]
    3. Loss between year[1] and year[2]

    Parameters
    ----------
    project : Project
        The project containing configuration
    forest_layers : List[LocalRasterVar]
        List of exactly 3 forest raster layers, each with a year attribute.
        Years are automatically extracted from the layers.

    Returns:
    -------
    List[LocalRasterVar]
        Three LocalRasterVar objects for the generated forest loss rasters
    """
    # Import here to avoid circular dependency
    from spatialrisk.variables import LocalRasterVar

    # Validate input - must have exactly 3 layers
    if len(forest_layers) != 3:
        raise ValueError(
            f"Exactly 3 forest layers are required, got {len(forest_layers)}"
        )

    # Build layer_by_year mapping and validate
    layer_by_year = {}
    for layer in forest_layers:
        if layer.year is None:
            raise ValueError(
                f"Forest layer '{layer.name}' is missing a 'year' attribute"
            )
        if layer.year in layer_by_year:
            raise ValueError(f"Duplicate forest layer detected for year {layer.year}")
        layer_by_year[layer.year] = layer

    # Get years from the layers themselves (sorted)
    years = sorted(layer_by_year.keys())

    ordered_layers = []
    for year in years:
        try:
            ordered_layers.append(layer_by_year[year])
        except KeyError as exc:
            raise ValueError(
                f"No forest layer provided for project year {year}"
            ) from exc

    pairings = [
        (ordered_layers[0], ordered_layers[1]),
        (ordered_layers[0], ordered_layers[2]),
        (ordered_layers[1], ordered_layers[2]),
    ]

    forest_loss_vars: List[LocalRasterVar] = []
    for start_layer, end_layer in pairings:
        forest_loss_vars.append(make_forest_loss_var(project, start_layer, end_layer))

    print("✓ Forest loss calculation complete!")
    return forest_loss_vars


def display_raster(
    name, path, raster_type=None, ax=None, return_fig=False, max_size=1024
):
    """
    Display a raster with appropriate visualization based on its type.

    Uses intelligent downsampling for fast visualization of large rasters.

    Parameters
    ----------
    name : str
        Name of the raster to display in the title
    path : str or Path
        Path to the raster file
    raster_type : RasterType or str, optional
        Type of raster ('categorical' or 'continuous'). If None, treats as continuous.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    return_fig : bool, optional
        If True, returns (fig, ax) tuple instead of showing the plot.
    max_size : int, optional
        Maximum dimension for display (default: 1024). Rasters larger than this
        will be downsampled for faster visualization.

    Returns:
    -------
    tuple or None
        If return_fig=True, returns (fig, ax) tuple. Otherwise returns None.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    # Open the raster file using rasterio
    with rasterio.open(path) as src:
        # Calculate optimal downsampling factor for display
        height, width = src.height, src.width
        max_dim = max(height, width)

        if max_dim > max_size:
            # Calculate the output shape for downsampling
            scale_factor = max_size / max_dim
            out_height = int(height * scale_factor)
            out_width = int(width * scale_factor)

            # Choose resampling method based on raster type
            is_categorical = False
            if raster_type is not None:
                raster_type_str = str(raster_type).lower()
                is_categorical = "categorical" in raster_type_str

            # Use nearest neighbor for categorical, average for continuous
            resampling_method = (
                Resampling.nearest if is_categorical else Resampling.average
            )

            # Read with downsampling - much faster!
            raster_data = src.read(
                1, out_shape=(out_height, out_width), resampling=resampling_method
            )
        else:
            # Small enough, read full resolution
            raster_data = src.read(1)

        nodata = src.nodata

        # Mask nodata values
        if nodata is not None:
            raster_data = np.ma.masked_equal(raster_data, nodata)

        # Create a plot for the raster data (or use provided axes)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            created_fig = True
        else:
            fig = ax.get_figure()
            created_fig = False

        # Determine if raster is categorical or continuous
        is_categorical = False
        if raster_type is not None:
            # Handle both RasterType enum and string
            raster_type_str = str(raster_type).lower()
            is_categorical = "categorical" in raster_type_str

        if is_categorical:
            # Categorical raster: use discrete colormap
            # Get unique values (excluding masked/nodata)
            unique_vals = (
                np.unique(raster_data.compressed())
                if np.ma.is_masked(raster_data)
                else np.unique(raster_data)
            )
            n_categories = len(unique_vals)

            # Use appropriate categorical colormap
            if n_categories <= 10:
                cmap = plt.cm.get_cmap("tab10", n_categories)
            elif n_categories <= 20:
                cmap = plt.cm.get_cmap("tab20", n_categories)
            else:
                cmap = plt.cm.get_cmap("nipy_spectral", n_categories)

            ax.imshow(raster_data, cmap=cmap, interpolation="nearest")
            ax.set_title(
                f"{name}\n(Categorical - {n_categories} classes)", fontsize=10, pad=5
            )
        else:
            # Continuous raster: use continuous colormap
            ax.imshow(raster_data, cmap="viridis")

            # Add statistics to title
            if np.ma.is_masked(raster_data):
                vmin, vmax = raster_data.min(), raster_data.max()
                ax.set_title(
                    f"{name}\n(Continuous: [{vmin:.2f}, {vmax:.2f}])",
                    fontsize=10,
                    pad=5,
                )
            else:
                ax.set_title(f"{name}\n(Continuous)", fontsize=10, pad=5)

        # Turn off axes for cleaner look in grid layouts
        ax.axis("off")

        # Only apply tight_layout and show if we created the figure
        if created_fig and not return_fig:
            plt.tight_layout()
            plt.show()

        if return_fig:
            return fig, ax

        return None

        # Show the plot
        plt.show()
