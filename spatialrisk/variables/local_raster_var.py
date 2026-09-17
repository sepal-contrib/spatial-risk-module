"""Local-filesystem raster Variable: reprojection and post-processing helpers."""

import warnings
from pathlib import Path
from typing import List, Optional

import ee
import odc.geo.xr  # noqa: F401  # do not delete this - registers the .odc accessor
import rioxarray
from pydantic import Field, field_validator

from spatialrisk.geo_utils import RASTER_CHUNKS, raster_is_all_nodata, xr_reproject
from spatialrisk.harmonization import geobox_signature
from spatialrisk.processing import (
    display_raster,
    distance_to_edge_gdal_no_mask,
    reproject_raster_gdal_warp,
)
from spatialrisk.utilities.file_helpers import copy_and_rename_file
from spatialrisk.variables.models import DataType, PostProcessing, RasterType
from spatialrisk.variables.variable import Variable


class LocalRasterVar(Variable):
    """
    Local filesystem-based raster variable.

    - Handles raster data (.tif, etc.)
    - Can be reprojected and post-processed
    - Use add_as_raw() or add_as_processed() to register to project.
    """

    path: Path
    data_type: DataType = Field(default=DataType.raster, frozen=True)
    raster_type: RasterType
    post_processing: List[PostProcessing] = []
    processing_history: List[str] = Field(
        default_factory=list
    )  # Track processing steps
    default_crs: Optional[str] = None
    default_resolution: Optional[float] = None

    @field_validator("path")
    @classmethod
    def _warn_if_path_missing(cls, v: Path) -> Path:
        """Warn when constructing a raster variable pointing to a non-existent file."""
        if not Path(v).exists():
            warnings.warn(
                f"LocalRasterVar path does not exist: {v}",
                UserWarning,
                stacklevel=2,
            )
        return v

    def show(self, ax=None, return_fig=False, max_size=1024):
        """
        Display the raster with appropriate visualization based on its type.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and shows it.
        return_fig : bool, optional
            If True, returns (fig, ax) tuple. Otherwise displays and returns None.
        max_size : int, optional
            Maximum dimension for display (default: 1024). Larger rasters will be
            downsampled for faster visualization.

        Returns:
        -------
        tuple or None
            If return_fig=True, returns (fig, ax). Otherwise returns None.
        """
        return display_raster(
            self.name,
            self.path,
            self.raster_type,
            ax=ax,
            return_fig=return_fig,
            max_size=max_size,
        )

    def add_as_raw(self, auto_save: bool = True) -> "LocalRasterVar":
        """
        Add this variable to the project's raw_variables collection.

        Raw variables are typically unprocessed data downloaded from sources
        like Google Earth Engine or other data providers.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after adding
            the variable.

        Returns:
        -------
        LocalRasterVar
            Returns self for method chaining.

        Raises:
        ------
        ValueError
            If the variable is not associated with a project.

        Examples:
        --------
        >>> var = LocalRasterVar(name="dem", ..., project=project)
        >>> var.add_as_raw()  # Auto-saves by default
        >>> var.add_as_raw(auto_save=False)  # Skip auto-save
        """
        if self.project is None:
            raise ValueError(
                "Cannot add to project: this variable is not associated with "
                "a project. Please set the 'project' parameter when creating "
                "the variable."
            )

        # Use name + year for storage key
        storage_key = f"{self.name}_{self.year}" if self.year else self.name
        self.project.raw_variables[storage_key] = self
        print(f"✓ Added '{self.name}' to raw variables (key: {storage_key})")

        if auto_save:
            self.project.save()

        return self

    def add_as_processed(self, auto_save: bool = True) -> "LocalRasterVar":
        """
        Add this variable to the project's processed variables collection.

        Processed variables are typically derived from raw variables through
        operations like reprojection, rasterization, or post-processing.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after adding
            the variable.

        Returns:
        -------
        LocalRasterVar
            Returns self for method chaining.

        Raises:
        ------
        ValueError
            If the variable is not associated with a project.

        Examples:
        --------
        >>> reprojected = raw_var.reproject(...).add_as_processed()  # Auto-saves
        >>> reprojected = raw_var.reproject(...).add_as_processed(auto_save=False)
        """
        if self.project is None:
            raise ValueError(
                "Cannot add to project: this variable is not associated with "
                "a project. Please set the 'project' parameter when creating "
                "the variable."
            )

        # Use name + year for storage key
        storage_key = f"{self.name}_{self.year}" if self.year else self.name
        self.project.processed_vars[storage_key] = self
        print(f"✓ Added '{self.name}' to processed variables (key: {storage_key})")

        if auto_save:
            self.project.save()

        return self

    def download(self):
        """Copy from the default path to the project."""
        copy_and_rename_file(self.path)

    def get_base_geobox(self):
        """
        Get the geobox from this raster file to use as a spatial reference.

        Returns:
        -------
        odc.geo.geobox.GeoBox
            The geobox defining spatial properties of this raster.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.path}")

        raster_array = rioxarray.open_rasterio(
            str(self.path),
            chunks=RASTER_CHUNKS,
            cache=False,
            lock=False,
        )

        # Return GeoBox using the .odc accessor (registered by odc.geo.xr import)
        return raster_array.odc.geobox

    def reproject(
        self,
        target_epsg: str,
        resolution: Optional[float] = None,
        resampling: Optional[str] = None,
        **kwargs,
    ) -> "LocalRasterVar":
        """
        Reproject this raster layer to a target EPSG coordinate system.

        Returns a new LocalRasterVar with the reprojected data.

        Parameters
        ----------
        target_epsg : str
            Target EPSG code (e.g., 'EPSG:4326' or '4326').
        resolution : float, optional
            Target resolution. If None, uses default_resolution or 30.
        resampling : str, optional
            Resampling method. If None, automatically selects based on raster_type:
            - categorical: 'near' (nearest neighbor)
            - continuous: 'bilinear'
        **kwargs
            Additional keyword arguments to pass to reproject_raster_gdal_warp.

        Returns:
        -------
        LocalRasterVar
            A new LocalRasterVar instance with the reprojected data.
        """
        # Normalize EPSG format
        if not target_epsg.startswith("EPSG:"):
            target_epsg = f"EPSG:{target_epsg}"

        # Determine output path using project folders
        output_folder = self.project.folders.processed_data_folder
        # Build filename suffix from processing history + current step
        # Handle legacy variables without processing_history
        history = getattr(self, "processing_history", [])
        filename_suffix = (
            "_".join([*history, "reprojected"]) if history else "reprojected"
        )
        year_suffix = f"_{self.year}" if self.year else ""
        output_path = output_folder / f"{self.name}_{filename_suffix}{year_suffix}.tif"

        # Determine resolution
        _resolution = resolution or self.default_resolution or 30

        # Auto-select resampling method based on raster_type if not specified
        if resampling is None:
            if self.raster_type == RasterType.categorical:
                resampling = "near"
            elif self.raster_type == RasterType.continuous:
                resampling = "bilinear"
            else:
                resampling = "near"  # default fallback

        # Reproject
        reproject_raster_gdal_warp(
            input_file=str(self.path),
            output_file=str(output_path),
            target_epsg=target_epsg,
            resolution=_resolution,
            resampling_method=resampling,
            **kwargs,
        )

        # Update processing history
        history = getattr(self, "processing_history", [])
        new_history = [*history, "reprojected"]

        # Create and return new LocalRasterVar using model_construct to bypass
        # validation
        return LocalRasterVar.model_construct(
            name=self.name,  # Keep original name
            raster_type=self.raster_type,
            path=output_path,
            default_crs=target_epsg,
            default_resolution=_resolution,
            post_processing=self.post_processing,
            processing_history=new_history,
            project=self.project,
            data_type=DataType.raster,
            active=True,
            year=self.year,
            tags=self.tags.copy() if self.tags else [],
        )

    def reproject_and_match(
        self,
        geobox,
        resampling: Optional[str] = None,
        output_suffix: str = "reprojected_matched",
        **kwargs,
    ) -> "LocalRasterVar":
        """
        Reproject this raster using xarray and odc-geo to match a target geobox.

        Returns a new LocalRasterVar with the reprojected data.

        Parameters
        ----------
        geobox : odc.geo.geobox.GeoBox
            The spatial template defining the shape, coordinates, dimensions,
            and transform of the output raster.
        resampling : str, optional
            Resampling method. If None, automatically selects based on raster_type:
            - categorical: 'nearest'
            - continuous: 'bilinear'
            Valid options: 'nearest', 'bilinear', 'cubic', 'cubic_spline',
            'lanczos', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3'
        output_suffix : str, optional
            Suffix to add to the output filename. Default is 'reprojected_matched'.
        **kwargs
            Additional keyword arguments (currently unused, for future expansion).

        Returns:
        -------
        LocalRasterVar
            A new LocalRasterVar instance with the reprojected data.

        Raises:
        ------
        FileNotFoundError
            If the input raster file doesn't exist.
        ValueError
            If the geobox is invalid or reprojection fails.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.path}")

        # Determine output path using project folders
        output_folder = self.project.folders.processed_data_folder
        # Build filename suffix from processing history + current step
        # Handle legacy variables without processing_history
        history = getattr(self, "processing_history", [])
        filename_suffix = (
            "_".join([*history, output_suffix]) if history else output_suffix
        )
        year_suffix = f"_{self.year}" if self.year else ""
        output_path = output_folder / f"{self.name}_{filename_suffix}{year_suffix}.tif"

        # Auto-select resampling method based on raster_type if not specified
        if resampling is None:
            if self.raster_type == RasterType.categorical:
                resampling = "nearest"
            elif self.raster_type == RasterType.continuous:
                resampling = "bilinear"
            else:
                resampling = "nearest"  # default fallback

        # Perform xarray-based reprojection
        xr_reproject(
            raster_path=str(self.path),
            geobox=geobox,
            resampling_method=resampling,
            output_path=str(output_path),
        )

        # An all-nodata output from a valid input is silent corruption (seen
        # 2026-08-03: a full 27.5M-px constant-nodata write). Fail loudly and
        # keep the bad file on disk for diagnosis; a re-run overwrites it.
        if raster_is_all_nodata(output_path) and not raster_is_all_nodata(self.path):
            raise ValueError(
                f"Reprojection of '{self.name}' produced an all-nodata "
                f"raster ({output_path}) from a valid input ({self.path}). "
                "The output was not registered; re-run the processing step."
            )

        # Extract CRS and resolution from geobox
        target_crs = f"EPSG:{geobox.crs.to_epsg()}"
        target_resolution = abs(geobox.resolution.x)  # Use x resolution

        # Update processing history
        history = getattr(self, "processing_history", [])
        new_history = [*history, output_suffix]

        # Create and return new LocalRasterVar (name unchanged)
        return LocalRasterVar.model_construct(
            name=self.name,  # Keep original name
            raster_type=self.raster_type,
            path=output_path,
            default_crs=target_crs,
            default_resolution=target_resolution,
            post_processing=self.post_processing,
            processing_history=new_history,
            project=self.project,
            data_type=DataType.raster,
            active=self.active,
            year=self.year,
            tags=self.tags.copy() if self.tags else [],
            # Stamped from the target geobox, not from the file just written:
            # xr_reproject writes *onto* this geobox, so the two are the same
            # grid by construction, and taking it from here keeps the
            # already-harmonized check a string compare instead of a reopen.
            grid_signature=geobox_signature(geobox),
        )

    def apply_post_processing(self, post_process: PostProcessing) -> "LocalRasterVar":
        """
        Apply a post-processing step to this raster and create a new LocalRasterVar.

        Parameters
        ----------
        post_process : PostProcessing
            Post-processing step to apply ('edge' or 'dist').

        Returns:
        -------
        LocalRasterVar
            A new LocalRasterVar with the post-processing applied.
        """
        if post_process == PostProcessing.edge:
            return self._create_post_var(
                self.path,
                suffix="edge",
                raster_type=RasterType.continuous,
            )
        elif post_process == PostProcessing.dist:
            return self._create_post_var(
                self.path,
                suffix="dist",
                raster_type=RasterType.continuous,
            )
        else:
            raise ValueError(f"Unknown post-processing step: {post_process}")

    def _apply_post(self, raster_path: Path) -> List["LocalRasterVar"]:
        """
        Create LocalRasterVars, one per post-processing step in self.post_processing.

        These are registered automatically via model_post_init.

        Returns:
        -------
        List[LocalRasterVar]
            List of new LocalRasterVar instances for post-processing outputs.
        """
        post_vars = []
        for step in self.post_processing:
            if step == PostProcessing.edge:
                post_var = self._create_post_var(
                    raster_path, suffix="edge", raster_type=RasterType.categorical
                )
                post_vars.append(post_var)
            elif step == PostProcessing.dist:
                post_var = self._create_post_var(
                    raster_path, suffix="dist", raster_type=RasterType.continuous
                )
                post_vars.append(post_var)
        return post_vars

    def _create_post_var(
        self,
        base_path: Path,
        suffix: str,
        raster_type: Optional[RasterType] = None,
    ) -> "LocalRasterVar":
        """
        Create a LocalRasterVar for post-processing output and execute the processing.

        Parameters
        ----------
        base_path : Path
            The base raster path to use as input.
        suffix : str
            Suffix to add to the variable name and file (e.g., 'edge', 'dist').
        raster_type : RasterType, optional
            Desired output raster type. If not provided, it will be derived from
            the post-processing step (edge → continuous, dist → continuous).

        Returns:
        -------
        LocalRasterVar
            A new LocalRasterVar instance.
        """
        # Make the output raster type explicit based on the post-processing step
        # to avoid inheriting the parent's type incorrectly.
        if suffix == "edge":
            # Edge outputs are treated as continuous (e.g., gradients / edge strength)
            output_raster_type = RasterType.continuous
        elif suffix == "dist":
            output_raster_type = RasterType.continuous
        else:
            output_raster_type = raster_type or self.raster_type

        # Create a new variable name by appending the suffix
        new_var_name = f"{self.name}_{suffix}"

        # Determine output path
        output_folder = self.project.folders.processed_data_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        # Build filename suffix from processing history + current step
        # Handle legacy variables without processing_history
        history = getattr(self, "processing_history", [])
        filename_suffix = "_".join([*history, suffix]) if history else suffix
        year_suffix = f"_{self.year}" if self.year else ""
        output_path = output_folder / f"{self.name}_{filename_suffix}{year_suffix}.tif"

        # Perform the actual post-processing
        if suffix == "edge":
            # Distance from the nearest feature (value==1) - i.e., outward
            # distance to patch
            distance_to_edge_gdal_no_mask(
                input_file=str(base_path),
                dist_file=str(output_path),
                values=0,  # Distance to feature pixels
                nodata=0,
                max_distance_value=4294967295,
                input_nodata=True,
                verbose=False,
            )

        elif suffix == "dist":
            # Distance calculation: distance from background (value==0) into the feature
            distance_to_edge_gdal_no_mask(
                input_file=str(base_path),
                dist_file=str(output_path),
                values=1,  # Distance to background pixels (non-feature)
                nodata=0,
                max_distance_value=4294967295,
                input_nodata=True,
                verbose=False,
            )

        # Update processing history
        history = getattr(self, "processing_history", [])
        new_history = [*history, suffix]

        # Create and return the new LocalRasterVar with a new name
        post_var = LocalRasterVar.model_construct(
            name=new_var_name,  # NEW variable name with suffix
            raster_type=output_raster_type,
            path=output_path,
            project=self.project,
            default_crs=self.default_crs,
            default_resolution=self.default_resolution,
            processing_history=new_history,
            data_type=DataType.raster,
            active=True,
            post_processing=self.post_processing + [PostProcessing(suffix)],
            year=self.year,
            tags=self.tags.copy() if self.tags else [],
        )

        return post_var

    def to_gee_var(self) -> ee.Image:
        """
        Convert a local raster to a Google Earth Engine Image.

        Note: This requires uploading to GEE assets first.

        Raises:
        ------
        NotImplementedError
            Raster upload to GEE requires manual asset upload.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Local file not found: {self.path}")

        raise NotImplementedError(
            "Converting local raster to GEE Image requires uploading to GEE assets. "
            "Please use GEE's asset upload tools manually."
        )

    def use_as_base_raster(self, auto_save: bool = True) -> "LocalRasterVar":
        """
        Set this raster as the base raster for the project.

        The base raster is used as a spatial reference for reprojecting and warping
        other layers to match its extent, resolution, and CRS.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after setting
            the base raster.

        Returns:
        -------
        LocalRasterVar
            Returns self for method chaining.

        Raises:
        ------
        ValueError
            If the variable is not associated with a project.
        FileNotFoundError
            If the raster file doesn't exist.

        Examples:
        --------
        >>> dem.use_as_base_raster()  # Auto-saves by default
        >>> dem.use_as_base_raster(auto_save=False)  # Skip auto-save
        """
        if self.project is None:
            raise ValueError(
                "Cannot set as base raster: this variable is not associated "
                "with a project. Please ensure the variable has a project "
                "reference."
            )

        if not self.path.exists():
            raise FileNotFoundError(
                f"Cannot set as base raster: raster file not found at {self.path}"
            )

        # The one file open, at the only moment the project's reference grid
        # can change. Unlike a harmonized output — written *onto* a geobox we
        # already hold — the base's grid only exists on disk, so it is read
        # once here and every later status check compares strings instead of
        # reopening this file.
        self.grid_signature = geobox_signature(self.get_base_geobox())

        # Set this raster as the project's base raster
        self.project.base_raster = self

        print(
            f"✓ Set '{self.name}' as base raster for project "
            f"'{self.project.project_name}'"
        )
        print(f"  Path: {self.path}")
        print(f"  CRS: {self.default_crs or 'Not specified'}")
        print(f"  Resolution: {self.default_resolution or 'Not specified'}")

        if auto_save:
            self.project.save()

        return self


# Rebuild models after Project is imported to resolve forward references
try:
    from spatialrisk.project import Project

    # Rebuild Variable classes first
    Variable.model_rebuild()
    LocalRasterVar.model_rebuild()

    # Then rebuild Project to ensure it sees the updated Variable classes
    Project.model_rebuild()
except ImportError:
    pass  # Project not yet available
