"""GEE-backed Variable that downloads assets to local raster/vector variables."""

from pathlib import Path
from typing import Any, List, Optional, Union

import ee
from pydantic import Field, model_validator

from spatialrisk.gee.ee_raster_export import download_ee_image
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.local_vector_var import LocalVectorVar
from spatialrisk.variables.models import (
    DataType,
    PostProcessing,
    RasterizationMethod,
    RasterType,
)
from spatialrisk.variables.variable import Variable


class GEEVar(Variable):
    """
    Google Earth Engine-backed variable.

    - Accepts either an asset id (gee_image as str or path as asset id) or
      an ee.Image object.
    - Optional local output path; if missing → defaults to CWD/<name>[_year].tif.
    """

    path: Optional[Union[Path, str]] = None  # local target OR asset id
    gee_images: Optional[List[Union[str, Any]]] = Field(default=None, repr=False)
    default_scale: Optional[float] = None
    default_crs: Optional[str] = None

    # Optional metadata that will be used when converting to LocalVar
    raster_type: Optional[RasterType] = None  # for raster data
    rasterization_method: Optional[RasterizationMethod] = None  # for vector data
    post_processing: List[PostProcessing] = []  # for raster data
    # Explicit export fill/nodata. When unset, derived from raster_type:
    # continuous -> -32768, categorical/unset -> 255. A layer whose values can
    # collide with the default sentinel must set this (e.g. a categorical
    # export whose class ids can reach 255).
    export_nodata: Optional[float] = None

    @model_validator(mode="after")
    def _chk_source(self):
        if self.gee_images is not None:
            return self
        if isinstance(self.path, str) and (
            self.path.startswith("users/") or self.path.startswith("projects/")
        ):
            return self
        raise ValueError(
            "GEEVar needs `gee_images` (ee.Image or asset id str list) or "
            "`path` set to an asset id."
        )

    def _resolve_export_nodata(self, raster_type: Optional[RasterType] = None) -> float:
        """Fill/nodata for the GeoTIFF export: a value the layer cannot contain.

        ``export_nodata`` wins when set. Otherwise continuous layers get
        -32768 (int16-safe; impossible for altitude in m, precipitation in mm,
        slope in degrees, temperature in degC), and categorical/unset layers
        keep the byte convention 255. 255 on a continuous layer punched holes
        through real data — every genuine 255 m elevation pixel became fill.
        """
        if self.export_nodata is not None:
            return self.export_nodata
        _raster_type = raster_type or self.raster_type
        if _raster_type == RasterType.continuous:
            return -32768
        return 255

    def resolve_images(self) -> List[Any]:
        """The layer's ``ee.Image`` list, building it from an asset id if needed.

        An asset-id GEEVar (``path`` set, no ``gee_images``) is valid per
        ``_chk_source``. The image is built lazily here — ``ee.Image`` needs an
        initialised client, which construction must not require — and kept, so
        both the map toggle and the download see the same variable become
        mappable after the first call.
        """
        if not self.gee_images:
            if isinstance(self.path, str) and (
                self.path.startswith("users/") or self.path.startswith("projects/")
            ):
                self.gee_images = [ee.Image(self.path)]
            else:
                raise ValueError("gee_images must be provided for download.")
        return self.gee_images

    def _download(
        self,
        overwrite: bool = False,
        raster_type: Optional[RasterType] = None,
    ) -> List[Path]:
        """
        Internal method to download GEE data to local file(s).

        Parameters
        ----------
        raster_type : RasterType, optional
            Conversion-time override from ``to_local_raster`` — must be
            resolved here, before the export, so it also drives the
            nodata sentinel (not only the LocalRasterVar metadata).
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).

        Returns:
        -------
        List[Path]
            List of paths to the downloaded files.
        """
        output_path: Path = None
        extensions = {"vector": ".shp", "raster": ".tif"}

        images = self.resolve_images()

        # Get the output folder
        output_folder = self.project.folders.data_raw_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        local_paths = []

        # Process images

        filename = f"{self.name}_{self.year}" if self.year is not None else self.name
        output_path = output_folder / filename

        output_path = output_path.with_suffix(extensions[self.data_type])
        local_paths.append(output_path)

        if overwrite or not output_path.exists():

            if self.data_type == DataType.vector:
                from spatialrisk.gee.vector_export import ee_export_vector

                ee_export_vector(
                    images[0],
                    output_path,
                    selectors=["gaul0_name", "iso3_code"],
                    keep_zip=False,
                    timeout=600,
                    verbose=False,
                )

            elif self.data_type == DataType.raster:
                nodata = self._resolve_export_nodata(raster_type)
                download_ee_image(
                    images[0],
                    output_path,
                    scale=self.default_scale or 30,
                    crs=self.default_crs or "EPSG:4326",
                    # The AOI object itself: the helper clips with it and
                    # exports over its bounds. ``self.aoi.geometry()`` inlines
                    # a table asset's computed geometry into every tile
                    # request and large AOIs fail with "Description length
                    # exceeds maximum".
                    region=self.aoi,
                    overwrite=True,
                    unmask_value=nodata,
                    nodata_value=nodata,
                )
        elif output_path.exists():
            print(f"{output_path} already exists. Skipping download.")

        # Verify all files were downloaded
        for local_path in local_paths:
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Download failed: file does not exist at {local_path}."
                )

        return local_paths

    def to_local_vector(
        self,
        overwrite: bool = False,
    ) -> Union["LocalVectorVar", List["LocalVectorVar"]]:
        """
        Download from GEE and convert to LocalVectorVar(s).

        Only works for vector data types.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).

        Returns:
        -------
        LocalVectorVar or List[LocalVectorVar]
            A LocalVectorVar instance if single image, or a list of
            LocalVectorVar instances if multiple images were downloaded.

        Raises:
        ------
        ValueError
            If data_type is not 'vector'.
        """
        if self.data_type != DataType.vector:
            raise ValueError(
                f"to_local_vector() can only be used with vector data types, "
                f"got '{self.data_type}'"
            )

        # Download the vector data
        local_paths = self._download(overwrite=overwrite)

        # Create LocalVectorVar instances for each downloaded file
        local_vars = []
        for i, local_path in enumerate(local_paths):

            local_var = LocalVectorVar.model_construct(
                name=self.name,
                rasterization_method=self.rasterization_method,
                path=local_path,
                default_crs=self.default_crs,
                project=self.project,
                data_type=DataType.vector,
                active=True,
                tags=self.tags.copy(),  # Copy tags from GEEVar
                year=self.year,
            )
            local_vars.append(local_var)

        # Return single instance if only one, otherwise return list
        return local_vars[0] if len(local_vars) == 1 else local_vars

    def to_local_raster(
        self,
        raster_type: Optional[RasterType] = None,
        rasterization_method: Optional[RasterizationMethod] = None,
        base: Optional["LocalRasterVar"] = None,
        post_processing: Optional[List[PostProcessing]] = None,
        overwrite: bool = False,
        **rasterize_kwargs,
    ) -> Union["LocalRasterVar", List["LocalRasterVar"]]:
        """
        Download from GEE and convert to LocalRasterVar(s).

        - For raster data: downloads directly and returns LocalRasterVar(s).
        - For vector data: downloads as LocalVectorVar(s), then rasterizes to
          LocalRasterVar(s).

        Parameters
        ----------
        raster_type : RasterType, optional
            Required if data_type is 'raster' and not set in GEEVar.
            If not provided, uses self.raster_type.
        rasterization_method : RasterizationMethod, optional
            Required if data_type is 'vector' and not set in GEEVar.
            Used when rasterizing vector data.
        base : LocalRasterVar, optional
            Required if data_type is 'vector'. The base raster to use as
            spatial reference for rasterization.
        post_processing : List[PostProcessing], optional
            Post-processing steps to apply (only for rasters).
            If not provided, uses self.post_processing.
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).
        **rasterize_kwargs
            Additional keyword arguments passed to LocalVectorVar.rasterize().

        Returns:
        -------
        LocalRasterVar or List[LocalRasterVar]
            A LocalRasterVar instance if single image, or a list of
            LocalRasterVar instances if multiple images were downloaded.

        Raises:
        ------
        ValueError
            If required parameters are missing based on data_type.
        """
        if self.data_type == DataType.vector:
            # For vector data: download as LocalVectorVar, then rasterize
            if base is None:
                raise ValueError(
                    "base (LocalRasterVar) must be provided when converting "
                    "a vector GEEVar to raster"
                )

            # Use provided rasterization_method or fall back to self's value
            _rasterization_method = rasterization_method or self.rasterization_method
            if _rasterization_method is None:
                raise ValueError(
                    "rasterization_method must be provided either as parameter "
                    "or set in GEEVar when converting vector to raster"
                )

            # Download as LocalVectorVar(s)
            local_vectors = self.to_local_vector(overwrite=overwrite)

            # Ensure we have a list to iterate over
            if not isinstance(local_vectors, list):
                local_vectors = [local_vectors]

            # Rasterize each LocalVectorVar to create LocalRasterVar(s)
            local_rasters = []
            for local_vector in local_vectors:
                local_raster = local_vector.rasterize(
                    base=base,
                    rasterization_method=_rasterization_method,
                    **rasterize_kwargs,
                )
                local_rasters.append(local_raster)

            # Return single instance if only one, otherwise return list
            return local_rasters[0] if len(local_rasters) == 1 else local_rasters

        else:  # DataType.raster
            # Resolve raster_type BEFORE downloading: it drives the export's
            # nodata sentinel, not just the LocalRasterVar metadata.
            _raster_type = raster_type or self.raster_type
            local_paths = self._download(overwrite=overwrite, raster_type=_raster_type)

            if _raster_type is None:
                raise ValueError(
                    "raster_type must be provided either as parameter or set "
                    "in GEEVar when converting a raster GEEVar to LocalRasterVar"
                )

            # Use provided post_processing or fall back to self's value
            _post_processing = (
                post_processing if post_processing is not None else self.post_processing
            )

            # Create LocalRasterVar instances for each downloaded file
            local_vars = []
            for i, local_path in enumerate(local_paths):

                local_var = LocalRasterVar.model_construct(
                    name=self.name,
                    raster_type=_raster_type,
                    post_processing=_post_processing,
                    path=local_path,
                    default_crs=self.default_crs,
                    default_resolution=self.default_scale,
                    project=self.project,
                    data_type=DataType.raster,
                    active=True,
                    tags=self.tags.copy(),  # Copy tags from GEEVar
                    vis_params=self.vis_params,
                    year=self.year,
                )
                local_vars.append(local_var)

            # Return single instance if only one, otherwise return list
            return local_vars[0] if len(local_vars) == 1 else local_vars


# Rebuild GEEVar after Variable and Project are available
try:
    from spatialrisk.project import Project

    types_namespace = {"Project": Project}
    GEEVar.model_rebuild(_types_namespace=types_namespace)
except ImportError:
    pass  # Project not yet available
