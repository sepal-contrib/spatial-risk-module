"""
Dataset module for deforestation risk modeling.

Handles variable selection, validation, and data transformation.
Provides data in different formats for different model types.
Works exclusively with LocalRasterVar instances.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pydantic import BaseModel, Field, ConfigDict


_EXTRACT_NUM_THREADS_ENV = "SPATIAL_RISK_EXTRACT_NUM_THREADS"

EXTRACT_GDAL_CACHEMAX_BYTES = 64 * 1024 * 1024
"""``GDAL_CACHEMAX`` while extracting point values.

GDAL's block cache defaults to ~5% of physical RAM and keeps every decoded
tile the extraction touches, so on a 60 GB host a windowed read of a 1.6 Gpx
stack still peaked at 1.5 GiB (measured 2026-09-15). Each tile is read
exactly once here, so the cache buys nothing; 64 MiB caps it while leaving
room for the handful of tiles in flight across the reader threads.
"""


def _extract_num_threads() -> int:
    """Reader threads for :func:`_read_values_at`: half the cores, min 1.

    Tile decoding is CPU-bound and GDAL releases the GIL, so a pool of
    per-thread dataset handles scales almost linearly (2.5 s -> 0.55 s per
    layer with 8 threads on a 400 Mpx DEFLATE layer, 2026-09-15).
    ``GDAL_NUM_THREADS`` does not help: it only parallelises *within* one
    read call, and each call here decodes a single tile.

    Uses the affinity mask rather than ``os.cpu_count()`` because SEPAL runs
    the app in a container whose cgroup quota is smaller than the host. Half
    the cores leaves room for the Solara server and concurrent jobs.
    ``SPATIAL_RISK_EXTRACT_NUM_THREADS`` overrides the default.
    """
    env_val = os.environ.get(_EXTRACT_NUM_THREADS_ENV)
    if env_val:
        return max(1, int(env_val))
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    return max(1, cores // 2)


def _read_values_at(src, rows, cols, num_threads: int = 1):
    """Read ``src`` band 1 at ``(rows, cols)`` without loading the whole band.

    Points are grouped by the raster's native block (tile or strip) and each
    touched block is decoded exactly once, so memory scales with the number
    of touched blocks rather than the raster size. A country-scale 30 m layer
    is several GiB as a full band; a few thousand points touch a few MiB.

    With ``num_threads > 1`` the touched blocks are dealt round-robin to a
    thread pool where every worker opens its own handle on ``src.name``
    (rasterio dataset handles are not thread-safe). Workers write disjoint
    slices of the output, so no locking is needed. Small jobs stay serial:
    the pool is only worth its start-up when there are blocks to share.

    ``rows``/``cols`` must already be clipped in-bounds. Returns an array in
    the band dtype with one value per point, in input order.
    """
    from rasterio.windows import Window

    n = len(rows)
    out = np.empty(n, dtype=src.dtypes[0])
    if n == 0:
        return out
    bh, bw = src.block_shapes[0]
    n_bcols = -(-src.width // bw)
    block_key = (rows // bh) * n_bcols + (cols // bw)
    order = np.argsort(block_key, kind="stable")
    keys, starts = np.unique(block_key[order], return_index=True)
    ends = np.append(starts[1:], n)
    blocks = list(zip(keys.tolist(), starts.tolist(), ends.tolist()))

    def _read(handle, items):
        for key, s0, s1 in items:
            idx = order[s0:s1]
            br, bc = divmod(key, n_bcols)
            r0, c0 = br * bh, bc * bw
            h = min(bh, src.height - r0)
            w = min(bw, src.width - c0)
            block = handle.read(1, window=Window(c0, r0, w, h))
            out[idx] = block[rows[idx] - r0, cols[idx] - c0]

    if num_threads <= 1 or len(blocks) < 2 * num_threads:
        _read(src, blocks)
        return out

    from concurrent.futures import ThreadPoolExecutor

    def _worker(items):
        with rasterio.open(src.name) as handle:
            _read(handle, items)

    chunks = [blocks[i::num_threads] for i in range(num_threads)]
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        # list() re-raises any worker exception instead of swallowing it.
        list(pool.map(_worker, chunks))
    return out


class Dataset(BaseModel):
    """Dataset configuration for model data preparation.

    Handles variable selection, temporal logic, and data transformation.
    Provides data in different formats for different model types.
    Works exclusively with LocalRasterVar instances (raster variables only).

    Attributes
    ----------
    project : Project
        Project instance with all variables
    target : LocalRasterVar, optional
        Target variable instance
    features : List[LocalRasterVar]
        List of feature variable instances
    year : int, optional
        Year for temporal variables
    name : str, optional
        Dataset identifier (e.g., "calibration_2020")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: Any = Field(..., repr=False)  # Project type to avoid circular import
    target: Optional[Any] = None  # LocalRasterVar instance
    features: List[Any] = Field(
        default_factory=list
    )  # List of LocalRasterVar instances
    year: Optional[int] = None
    name: Optional[str] = None

    def __init__(self, project: Any, name: Optional[str] = None, **data):
        """Initialize dataset configuration.

        Parameters
        ----------
        project : Project
            Project instance with all variables
        name : str, optional
            Dataset name/identifier
        """
        super().__init__(project=project, name=name, **data)

    def set_target(
        self, name: Optional[str] = None, year: Optional[int] = None
    ) -> Optional[List[str]]:
        """Set or discover target variable.

        Parameters
        ----------
        name : str, optional
            Target variable name. If None, returns available targets.
        year : int, optional
            Year for temporal variables. Required if target is multitemporal.

        Returns
        -------
        List[str] or None
            If name is None, returns list of available target variables.
            If name is provided, sets target and returns None.
        """
        if name is None:
            # Discovery mode - return available targets
            available = self.project.list_unique_variable_names()
            print("\n📊 Available target variables:")
            for var in available:
                instances = self.project.get_all_instances(var)
                is_temporal = self.project.is_temporal(var)
                years = self.project.get_variable_years(var)

                if is_temporal:
                    print(f"  • {var} (temporal: {', '.join(map(str, years))})")
                else:
                    print(f"  • {var} (static)")
            return available
        else:
            # Configuration mode - set target
            instances = self.project.get_all_instances(name)
            if not instances:
                raise ValueError(
                    f"Target variable '{name}' not found in processed variables"
                )

            is_temporal = self.project.is_temporal(name)
            years = self.project.get_variable_years(name)

            # Require year for multitemporal variables
            if is_temporal and year is None:
                raise ValueError(
                    f"Target variable '{name}' is multitemporal. "
                    f"You must specify a year parameter.\n"
                    f"Available years: {', '.join(map(str, years))}\n"
                    f"Example: dataset.set_target('{name}', year={years[0]})"
                )

            # Validate year if provided
            if year is not None:
                if not is_temporal:
                    raise ValueError(
                        f"Target variable '{name}' is static (not temporal). "
                        f"Do not specify a year parameter."
                    )
                if year not in years:
                    raise ValueError(
                        f"Year {year} not available for target variable '{name}'.\n"
                        f"Available years: {', '.join(map(str, years))}"
                    )
                # Set the year if provided
                self.year = year

            # Get the variable instance (with year if temporal)
            year_param = self.year if is_temporal else None
            self.target = self.project.get_variable(name, year=year_param)

            if is_temporal:
                print(f"✓ Target set: {name} (year: {year})")
            else:
                print(f"✓ Target set: {name} (static)")
            return None

    def set_features(self, names: Optional[List[str]] = None) -> Optional[List[str]]:
        """Set or discover feature variables.

        Parameters
        ----------
        names : List[str], optional
            Feature variable names. If None, returns available features.

        Returns
        -------
        List[str] or None
            If names is None, returns list of available features.
            If names is provided, sets features and returns None.

        Notes
        -----
        If any feature is temporal, a year must be set (either via set_target with year parameter,
        or by calling set_year() before or after set_features).
        All temporal features will use the same year as specified in self.year.
        """
        if names is None:
            # Discovery mode - return available features
            available = self.project.list_unique_variable_names()
            print("\n📊 Available feature variables:")
            for var in available:
                instances = self.project.get_all_instances(var)
                is_temporal = self.project.is_temporal(var)
                years = self.project.get_variable_years(var)

                if is_temporal:
                    print(f"  • {var} (temporal: {', '.join(map(str, years))})")
                else:
                    print(f"  • {var} (static)")
            return available
        else:
            # Configuration mode - set features
            static_features = []
            temporal_features = []

            for name in names:
                instances = self.project.get_all_instances(name)
                if not instances:
                    raise ValueError(
                        f"Feature variable '{name}' not found in processed variables"
                    )

                if self.project.is_temporal(name):
                    temporal_features.append(name)
                else:
                    static_features.append(name)

            # Check if temporal features require a year to be set
            if temporal_features and self.year is None:
                raise ValueError(
                    f"Temporal features detected: {', '.join(temporal_features)}\n"
                    f"You must set a year before or when setting features with temporal variables.\n"
                    f"Either:\n"
                    f"  1. Call set_target() with year parameter first (e.g., set_target('target', year=2020)), OR\n"
                    f"  2. Call set_year() before set_features() (e.g., dataset.set_year(2020))\n"
                    f"All temporal features will use the same year."
                )

            # Validate that all temporal features have data for the specified year
            if temporal_features and self.year is not None:
                missing_vars = []
                for name in temporal_features:
                    available_years = self.project.get_variable_years(name)
                    if self.year not in available_years:
                        missing_vars.append(f"{name} (available: {available_years})")

                if missing_vars:
                    raise ValueError(
                        f"Year {self.year} not available for temporal features:\n  "
                        + "\n  ".join(missing_vars)
                    )

            # Store variable instances (with year if temporal)
            feature_instances = []
            for name in names:
                is_temporal = self.project.is_temporal(name)
                year_param = self.year if is_temporal else None
                var = self.project.get_variable(name, year=year_param)
                if var is None:
                    raise ValueError(
                        f"Could not resolve feature variable '{name}'"
                        + (f" for year {year_param}" if year_param else "")
                        + ". It is not available in the processed variables."
                    )
                feature_instances.append(var)

            self.features = feature_instances
            print(f"✓ Features set: {len(names)} variables")
            if static_features:
                print(f"  Static: {', '.join(static_features)}")
            if temporal_features:
                year_info = f" (year: {self.year})" if self.year else ""
                print(f"  Temporal{year_info}: {', '.join(temporal_features)}")
            return None

    def set_year(self, year: int) -> None:
        """Set year for temporal variables.

        Parameters
        ----------
        year : int
            Year to use for temporal variables
        """
        # Check if year is available for all temporal variables
        all_vars = [self.target] + self.features if self.target else self.features
        temporal_vars = [v for v in all_vars if v and self.project.is_temporal(v.name)]

        missing_vars = []
        for var in temporal_vars:
            available_years = self.project.get_variable_years(var.name)
            if year not in available_years:
                missing_vars.append(f"{var.name} (available: {available_years})")

        if missing_vars:
            raise ValueError(
                f"Year {year} not available for temporal variables:\n  "
                + "\n  ".join(missing_vars)
            )

        self.year = year
        print(f"✓ Year set: {year}")
        print(f"✓ All temporal variables available for this year")

    def get_available_years(self) -> List[int]:
        """Get years available for all configured variables.

        Returns
        -------
        List[int]
            Sorted list of years available for all temporal variables
        """
        all_vars = [self.target] + self.features if self.target else self.features
        temporal_vars = [v for v in all_vars if v and self.project.is_temporal(v.name)]

        if not temporal_vars:
            return []

        # Get intersection of years across all temporal variables
        years_sets = [
            set(self.project.get_variable_years(var.name)) for var in temporal_vars
        ]

        common_years = sorted(set.intersection(*years_sets) if years_sets else set())
        return common_years

    def validate(self) -> bool:
        """Validate the dataset configuration.

        Checks:
        - Target is set
        - Features are set
        - Year is set (if temporal variables present)
        - All variables exist and are processed
        - All variables have matching spatial properties

        Returns
        -------
        bool
            True if validation passes

        Raises
        ------
        ValueError
            If validation fails
        """
        print("\n🔍 Validating dataset configuration...")

        # Check target is set
        if not self.target:
            raise ValueError("Target variable not set. Use set_target() first.")

        # Check features are set
        if not self.features:
            raise ValueError("Features not set. Use set_features() first.")

        # Check year for temporal variables
        all_vars = [self.target] + self.features
        temporal_vars = [v for v in all_vars if self.project.is_temporal(v.name)]

        if temporal_vars and self.year is None:
            raise ValueError(
                f"Year must be set for temporal variables: {', '.join([v.name for v in temporal_vars])}\n"
                f"Use set_year() or check available years with get_available_years()"
            )

        # Check all variables exist and are processed
        print("✓ Checking variable existence...")
        for var in all_vars:
            if var is None:
                raise ValueError(f"Variable instance is None")
            if not var.path or not var.path.exists():
                raise ValueError(
                    f"Variable '{var.name}' has no valid file path: {var.path}"
                )

        print("✓ All variables exist")
        print("✓ All variables processed and available")

        # Check spatial properties match
        print("✓ Checking spatial compatibility...")

        print("\n✅ Dataset validation passed!")
        return True

    def show(
        self,
        include_target: bool = True,
        include_features: bool = True,
        ncols: int = 3,
        max_size: int = 1024,
    ) -> None:
        """Display all variables in the dataset in a grid layout.

        Parameters
        ----------
        include_target : bool, optional
            Whether to display the target variable (default: True)
        include_features : bool, optional
            Whether to display the feature variables (default: True)
        ncols : int, optional
            Number of columns in the grid layout (default: 3)
        max_size : int, optional
            Maximum dimension for display (default: 1024). Larger rasters will be
            downsampled for faster visualization.

        Examples
        --------
        >>> dataset.show()  # Show all variables in 3-column grid
        >>> dataset.show(ncols=2)  # Show in 2-column grid
        >>> dataset.show(include_target=False)  # Show only features
        >>> dataset.show(include_features=False)  # Show only target
        >>> dataset.show(max_size=512)  # Use more aggressive downsampling for speed
        """
        import matplotlib.pyplot as plt

        if not self.target and not self.features:
            print("⚠️  No variables configured in dataset")
            return

        # Collect variables to display
        vars_to_show = []
        if include_target and self.target:
            vars_to_show.append(("TARGET", self.target))
        if include_features and self.features:
            vars_to_show.extend([(var.name, var) for var in self.features])

        if not vars_to_show:
            print("⚠️  No variables to display")
            return

        # Calculate grid dimensions
        n_vars = len(vars_to_show)
        nrows = (n_vars + ncols - 1) // ncols  # Ceiling division

        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

        # Flatten axes array for easier indexing
        if n_vars == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

        # Display each variable using its show() method
        for idx, (var_label, var) in enumerate(vars_to_show):
            ax = axes[idx]
            # Use the variable's show method with the subplot axes and downsampling
            var.show(ax=ax, return_fig=True, max_size=max_size)

            # Add distinctive border/background for target variable
            if var_label == "TARGET":
                # Add a bold red border around the target
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d62728")  # Red color
                    spine.set_linewidth(4)
                    spine.set_visible(True)

                # Add "TARGET" label badge in top-left corner
                ax.text(
                    0.02,
                    0.98,
                    "TARGET",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#d62728",
                        edgecolor="none",
                        alpha=0.9,
                    ),
                    verticalalignment="top",
                    horizontalalignment="left",
                    zorder=1000,
                )

        # Hide unused subplots
        for idx in range(n_vars, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"\n✅ Displayed {n_vars} variable(s) in {nrows}x{ncols} grid")

    def register(
        self, project: Any, key: Optional[str] = None, auto_save: bool = True
    ) -> None:
        """Register this dataset with a project. Mirrors model.register()."""
        project.add_dataset(self, key=key, auto_save=auto_save)

    def extract_at_points(self, points, *, drop_nodata: bool = True):
        """Extract target + feature values at the given sample points.

        Points are mapped onto EACH raster's own grid (reprojected to that
        raster's CRS as needed), so extraction stays correct when dataset
        layers do not share a grid. cell_id is keyed on the TARGET grid (the
        canonical location id) and is not meaningful for feature layers on a
        different grid.

        Parameters
        ----------
        points : geopandas.GeoDataFrame
            Point geometries (any CRS).
        drop_nodata : bool
            Drop points that hit nodata/NaN or fall outside ANY layer (logged).

        Returns
        -------
        pandas.DataFrame
            Columns: [target_name, feature1…featureN, cell_id, trial].
        """
        import logging
        import numpy as np
        import pandas as pd
        import rasterio

        logger = logging.getLogger("spatial_risk")
        if self.target is None or not self.features:
            raise ValueError("Dataset target and features must be set.")

        all_vars = [self.target] + self.features
        n_pts = len(points)
        df_data = {}
        valid = np.ones(n_pts, dtype=bool)
        target_ncols = None
        target_cell = None

        num_threads = _extract_num_threads()
        # Reproject once per distinct CRS, not once per layer.
        reprojected = {}

        def _pts_in(crs):
            if points.crs == crs:
                return points
            key = crs.to_wkt()
            if key not in reprojected:
                reprojected[key] = points.to_crs(crs)
            return reprojected[key]

        with rasterio.Env(GDAL_CACHEMAX=EXTRACT_GDAL_CACHEMAX_BYTES):
            for i, var in enumerate(all_vars):
                with rasterio.open(var.path) as src:
                    nodata = src.nodata
                    vheight, vwidth = src.height, src.width

                    vpts = _pts_in(src.crs)
                    r, c = rasterio.transform.rowcol(
                        src.transform,
                        vpts.geometry.x.to_numpy(),
                        vpts.geometry.y.to_numpy(),
                    )
                    r = np.asarray(r, dtype=int)
                    c = np.asarray(c, dtype=int)
                    in_bounds = (r >= 0) & (r < vheight) & (c >= 0) & (c < vwidth)

                    # Out-of-bounds points are masked out below; clipping keeps
                    # the block lookup in range (same as the old clipped index).
                    rc = np.clip(r, 0, vheight - 1)
                    cc = np.clip(c, 0, vwidth - 1)
                    vals = _read_values_at(src, rc, cc, num_threads=num_threads)

                layer_valid = in_bounds.copy()
                if np.issubdtype(vals.dtype, np.floating):
                    layer_valid &= ~np.isnan(vals)
                if nodata is not None:
                    layer_valid &= vals != nodata
                valid &= layer_valid
                df_data[var.name] = vals

                if i == 0:  # target raster defines cell_id
                    target_ncols = vwidth
                    target_cell = r * vwidth + c

        df_data["cell_id"] = target_cell
        df_data["trial"] = 1
        df = pd.DataFrame(df_data)

        if drop_nodata:
            dropped = int((~valid).sum())
            if dropped:
                logger.info(
                    "extract_at_points: dropped %d/%d points on nodata/out-of-bounds.",
                    dropped,
                    n_pts,
                )
            df = df[valid].reset_index(drop=True)
        return df

    def get_file_paths(self) -> Dict[str, Path]:
        """Get file paths for all configured variables.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping variable names to file paths
        """
        all_vars = [self.target] + self.features if self.target else self.features
        paths = {}

        for var in all_vars:
            if var and var.path:
                paths[var.name] = var.path

        return paths

    def to_raster_stack(
        self, output_path: Optional[Path] = None, format: str = "VRT"
    ) -> Path:
        """Create stacked raster file with all variables as bands.

        Parameters
        ----------
        output_path : Path, optional
            Output path. If None, creates in project data folder.
        format : str, optional
            Output format: 'VRT' (default) or 'GTiff'

        Returns
        -------
        Path
            Path to stacked raster file
        """
        if output_path is None:
            output_path = (
                self.project.folders.processed_data_folder
                / f"{self.name or 'dataset'}_stack.{format.lower()}"
            )

        paths = self.get_file_paths()

        if format == "VRT":
            # Create VRT file
            from rasterio.vrt import WarpedVRT

            # TODO: Implement VRT stacking
            raise NotImplementedError("VRT stacking not yet implemented")
        else:
            # Create GTiff stack
            # TODO: Implement GTiff stacking
            raise NotImplementedError("GTiff stacking not yet implemented")

    def export_geotiffs(
        self, output_folder: Path, prefix: str = "", suffix: str = ""
    ) -> Dict[str, Path]:
        """Export each variable as separate GeoTIFF file.

        Parameters
        ----------
        output_folder : Path
            Output folder path
        prefix : str, optional
            Prefix for output filenames
        suffix : str, optional
            Suffix for output filenames

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping variable names to exported file paths
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        paths = self.get_file_paths()
        exported = {}

        for var_name, var_path in paths.items():
            output_name = f"{prefix}{var_name}{suffix}.tif"
            output_path = output_folder / output_name

            # Copy file
            import shutil

            shutil.copy2(var_path, output_path)
            exported[var_name] = output_path
            print(f"✓ Exported: {var_name} → {output_path}")

        return exported

    def get_spatial_info(self) -> Dict[str, Any]:
        """Get spatial properties of the dataset.

        Returns
        -------
        Dict[str, Any]
            Dictionary with spatial metadata
        """
        if not self.target:
            raise ValueError("Target not set")

        if not self.target.path:
            raise ValueError("Target variable has no valid path")

        with rasterio.open(self.target.path) as src:
            return {
                "crs": str(src.crs),
                "bounds": src.bounds,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
                "nodata": src.nodata,
            }

    def describe(self) -> Dict[str, Any]:
        """Get summary statistics for all variables.

        Returns
        -------
        Dict[str, Any]
            Dictionary with variable statistics
        """
        all_vars = [self.target] + self.features if self.target else self.features
        stats = {}

        for var in all_vars:
            if not var or not var.path:
                continue

            with rasterio.open(var.path) as src:
                arr = src.read(1)
                valid_arr = arr[
                    ~np.isnan(arr) & (arr != src.nodata if src.nodata else True)
                ]

                stats[var.name] = {
                    "min": float(np.min(valid_arr)) if len(valid_arr) > 0 else None,
                    "max": float(np.max(valid_arr)) if len(valid_arr) > 0 else None,
                    "mean": float(np.mean(valid_arr)) if len(valid_arr) > 0 else None,
                    "std": float(np.std(valid_arr)) if len(valid_arr) > 0 else None,
                    "count": len(valid_arr),
                }

        return stats
