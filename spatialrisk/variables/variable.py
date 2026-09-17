"""Base model shared by every variable (local rasters, vectors, GEE-backed)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ee
from pydantic import BaseModel, ConfigDict, Field

from spatialrisk.variables.models import DataType


class Variable(BaseModel):
    """Base class for all variables (local and GEE-based)."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        # Exclude non-serializable fields by default
        json_encoders={
            Path: str,
        },
    )

    name: str  # Clean variable identifier: "towns", "forest_gfc", "altitude"
    data_type: DataType
    year: Optional[int] = None  # Optional year for temporal variables
    active: bool = True  # Variables are active by default
    tags: List[str] = Field(default_factory=list)  # Tags for categorizing variables
    # Optional map visualization chosen by the user (GEE ``vis_params`` shape:
    # ``{"palette": [hex, ...], "min": .., "max": ..}``; hex without ``#``).
    # Catalogue layers leave it None and take their look from the catalogue.
    vis_params: Optional[Dict[str, Any]] = None
    grid_signature: Optional[str] = None
    """Signature of the geobox this file sits on (see harmonization.geobox_signature).

    On the base raster it records the base's own grid; on a harmonized output it
    records the grid that output was matched to. A layer is already harmonized
    exactly when the two strings are equal. ``None`` means "not recorded" — a
    project written before this field existed — and sends the entry to the disk
    check instead.
    """
    project: Optional["Project"] = Field(
        default=None, repr=False, exclude=True, validate_default=False
    )  # Excluded from JSON serialization and __repr__
    aoi: Union[ee.Feature, ee.Geometry, ee.FeatureCollection] = Field(
        default=None, repr=False, exclude=True  # Excluded from JSON serialization
    )

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Override model_dump to ensure project and aoi are always excluded.

        This prevents issues with forward references during serialization.
        """
        # Ensure project and aoi are in the exclude set
        if exclude is None:
            exclude = {"project", "aoi"}
        elif isinstance(exclude, set):
            exclude = exclude | {"project", "aoi"}
        elif isinstance(exclude, dict):
            exclude = {**exclude, "project": True, "aoi": True}
        else:
            # If exclude is some other type, create a dict
            exclude = {"project": True, "aoi": True}

        return super().model_dump(mode=mode, include=include, exclude=exclude, **kwargs)

    def activate(self, auto_save: bool = True) -> "Variable":
        """
        Activate this variable for processing.

        Active variables will be included in batch operations like
        project.reproject_all() and project.rasterize_all().

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after activation.

        Returns:
        -------
        Variable
            Returns self for method chaining.

        Examples:
        --------
        >>> var.activate()
        >>> var.deactivate().activate()  # Toggle state
        """
        self.active = True
        print(f"✓ Activated '{self.name}'")

        if auto_save and self.project is not None:
            self.project.save()

        return self

    def deactivate(self, auto_save: bool = True) -> "Variable":
        """
        Deactivate this variable to exclude it from processing.

        Inactive variables will be skipped in batch operations like
        project.reproject_all() and project.rasterize_all().

        Parameters
        ----------
        auto_save : bool, optional
            If True (default), automatically saves the project after deactivation.

        Returns:
        -------
        Variable
            Returns self for method chaining.

        Examples:
        --------
        >>> var.deactivate()
        >>> var.deactivate(auto_save=False)  # Skip auto-save
        """
        self.active = False
        print(f"✓ Deactivated '{self.name}'")

        if auto_save and self.project is not None:
            self.project.save()

        return self


try:
    from spatialrisk.project import Project

    # Build type namespace with Project for all variable classes
    types_namespace = {"Project": Project}

    # Rebuild Variable base class first
    Variable.model_rebuild(_types_namespace=types_namespace)

    # Then rebuild Project to ensure it sees the updated Variable classes
    Project.model_rebuild()
except ImportError:
    pass  # Project not yet available
