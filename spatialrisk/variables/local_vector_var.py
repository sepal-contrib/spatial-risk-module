import warnings
from pathlib import Path
from typing import Optional

import ee
from pydantic import Field, field_validator
from spatialrisk.processing import xr_rasterize
from spatialrisk.utilities.file_helpers import copy_and_rename_file
from spatialrisk.variables.models import DataType, RasterType, RasterizationMethod
from spatialrisk.variables.variable import Variable
from spatialrisk.variables.local_raster_var import LocalRasterVar


class LocalVectorVar(Variable):
    """
    Local filesystem-based vector variable.
    - Handles vector data (.shp, .geojson, etc.)
    - Can be rasterized to create LocalRasterVar
    - Use add_as_raw() or add_as_processed() to register to project
    """

    path: Path
    data_type: DataType = Field(default=DataType.vector, frozen=True)
    rasterization_method: Optional[RasterizationMethod] = None
    default_crs: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _warn_if_path_missing(cls, v: Path) -> Path:
        """Warn when constructing a vector variable pointing to a non-existent file."""
        if not Path(v).exists():
            warnings.warn(
                f"LocalVectorVar path does not exist: {v}",
                UserWarning,
                stacklevel=2,
            )
        return v

    def add_as_raw(self, auto_save: bool = True) -> "LocalVectorVar":
        """
        Add this variable to the project's raw_variables collection.

        Raw variables are typically unprocessed data downloaded from sources
        like Google Earth Engine or other data providers.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after adding the variable.

        Returns
        -------
        LocalVectorVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.
        """
        if self.project is None:
            raise ValueError(
                "Cannot add to project: this variable is not associated with a project. "
                "Please set the 'project' parameter when creating the variable."
            )

        # Use name + year for storage key
        storage_key = f"{self.name}_{self.year}" if self.year else self.name
        self.project.raw_variables[storage_key] = self
        print(f"✓ Added '{self.name}' to raw variables (key: {storage_key})")

        if auto_save:
            self.project.save()

        return self

    def add_as_processed(self, auto_save: bool = True) -> "LocalVectorVar":
        """
        Add this variable to the project's processed variables collection.

        Processed variables are typically derived from raw variables through
        operations like reprojection, rasterization, or post-processing.

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after adding the variable.

        Returns
        -------
        LocalVectorVar
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variable is not associated with a project.
        """
        if self.project is None:
            raise ValueError(
                "Cannot add to project: this variable is not associated with a project. "
                "Please set the 'project' parameter when creating the variable."
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

    def rasterize(
        self,
        base: "LocalRasterVar",
        rasterization_method: Optional[RasterizationMethod] = None,
        **kwargs,
    ) -> "LocalRasterVar":
        """
        Rasterize this vector layer using a base raster as spatial reference.

        Parameters
        ----------
        base : LocalRasterVar
            A LocalRasterVar to use as spatial reference.
        rasterization_method : RasterizationMethod, optional
            How to rasterize the vector data. If not provided, uses self.rasterization_method.
            Must be provided either here or when creating the LocalVectorVar.
        **kwargs
            Additional keyword arguments to pass to xr_rasterize.

        Returns
        -------
        LocalRasterVar
            A new LocalRasterVar instance.
        """
        # Check if base is a raster using data_type instead of isinstance to handle module reloads
        if not hasattr(base, "data_type") or base.data_type != DataType.raster:
            raise ValueError(
                "base must be a LocalRasterVar instance (raster data type)"
            )

        # Use provided rasterization_method or fall back to self's value
        _rasterization_method = rasterization_method or self.rasterization_method
        if _rasterization_method is None:
            raise ValueError(
                "rasterization_method must be provided either as parameter or set in LocalVectorVar"
            )

        # Get geobox from base raster
        geobox = base.get_base_geobox()

        # Determine output path using project folders
        output_folder = self.project.folders.data_raw_folder
        # Year in the filename, mirroring LocalRasterVar.reproject_and_match:
        # add_as_processed registers under `{name}_{year}`, so without it two
        # years of one vector share a file and the later run overwrites the
        # earlier. Static vectors keep the unsuffixed path they already have.
        year_suffix = f"_{self.year}" if self.year else ""
        output_path = output_folder / f"{self.name}{year_suffix}.tif"

        # Map RasterizationMethod to mode parameter
        mode_mapping = {
            RasterizationMethod.binary: "binary",
            RasterizationMethod.unique: "unique",
        }
        mode = mode_mapping.get(_rasterization_method, "binary")

        # Rasterize
        xr_rasterize(
            shapefile_path=str(self.path),
            geobox=geobox,
            output_path=str(output_path),
            mode=mode,
            **kwargs,
        )

        # Create and return LocalRasterVar
        raster_type = (
            RasterType.categorical if mode == "unique" else RasterType.continuous
        )

        return LocalRasterVar.model_construct(
            name=self.name,
            raster_type=raster_type,
            path=output_path,
            default_crs=self.default_crs,
            project=self.project,
            data_type=DataType.raster,
            active=True,
            year=self.year,
            processing_history=[
                "rasterized"
            ],  # Track that this came from vector rasterization
            tags=self.tags.copy() if self.tags else [],
        )

    def to_gee_var(self) -> ee.FeatureCollection:
        """
        Convert this local vector to a Google Earth Engine FeatureCollection.

        Returns
        -------
        ee.FeatureCollection
            An Earth Engine FeatureCollection.

        Raises
        ------
        FileNotFoundError
            If the local file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Local file not found: {self.path}")

        import json
        import geopandas as gpd
        gdf = gpd.read_file(str(self.path))
        return ee.FeatureCollection(json.loads(gdf.to_json()))


# Rebuild models after Project is imported to resolve forward references
try:
    from spatialrisk.project import Project

    # Rebuild Variable classes first
    Variable.model_rebuild()
    LocalVectorVar.model_rebuild()
    LocalRasterVar.model_rebuild()

    # Then rebuild Project to ensure it sees the updated Variable classes
    Project.model_rebuild()
except ImportError:
    pass  # Project not yet available
