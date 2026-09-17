"""The Project model: the manifest every other module reads and writes."""

import json
import logging
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from box import Box
from pydantic import BaseModel, ConfigDict, Field

from spatialrisk.gdal_env import configure_gdal_tmpdir
from spatialrisk.log_utils import log_progress

# Imported for their runtime presence in this module's namespace, not for direct
# use: Project's fields annotate them as forward references (``Union["LocalVectorVar",
# "LocalRasterVar"]``) that pydantic resolves against this namespace on
# model_rebuild(). Removing them breaks Project construction at import time.
from spatialrisk.variables import LocalRasterVar, LocalVectorVar  # noqa: F401
from spatialrisk.variables.models import DataType


def _resolve_data_dir() -> Path:
    """Resolve the canonical project data directory.

    Order: ``SPATIAL_RISK_DATA_DIR`` env var, else the SEPAL-wide convention
    ``~/module_results/spatial_risk_module`` (every SEPAL module must write its
    outputs under ``~/module_results/<module>``).
    """
    env = os.environ.get("SPATIAL_RISK_DATA_DIR")
    if env:
        return Path(env).resolve()
    return (Path.home() / "module_results" / "spatial_risk_module").resolve()


DATA_DIR: Path = _resolve_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)
# Must run before any GDAL algorithm call: GDAL otherwise writes its scratch
# files to the CWD, which is read-only on SEPAL. Importing any ``spatialrisk.*``
# submodule runs ``spatialrisk/__init__.py``, which imports this module, so this
# single call covers voila, solara, notebooks and tests alike.
configure_gdal_tmpdir()
# Backward-compatible alias: save()/load()/initialize_folders() read this name.
downloads_folder = DATA_DIR
# Deprecated: kept only because ``initialize_folders()`` publishes it as
# ``project.folders["root_folder"]``, which notebooks may still reach for. It used
# to be ``Path.cwd().parent`` -- on SEPAL the CWD is the read-only shared module
# mount, so anything that took this key as a place to write landed there. Nothing
# in this repo reads it and it is never persisted (``save()`` serialises an
# explicit whitelist of fields and ``folders`` is a property, not a model field),
# so retargeting it at the module's real, writable output root changes no saved
# manifest and breaks no load.
root_folder: Path = DATA_DIR

logger = logging.getLogger("spatial_risk")


def _stringify_paths(obj: Any) -> Any:
    """Recursively convert pathlib.Path objects to str for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        t = type(obj)
        return t(_stringify_paths(v) for v in obj)
    return obj


class Project(BaseModel):
    """
    A Pydantic model representing a deforestation risk analysis project.

    Stores project metadata, variables, and manages folder structure.
    Can be serialized to/from JSON for persistence.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_name: str
    years: Optional[List[int]] = Field(
        default=None,
        description="Deprecated: Years are now auto-discovered from variables",
    )
    raw_variables: Dict[str, Union["LocalVectorVar", "LocalRasterVar"]] = Field(
        default_factory=dict
    )
    processed_variables: Dict[str, Union["LocalVectorVar", "LocalRasterVar"]] = Field(
        default_factory=dict
    )
    base_raster: Optional["LocalRasterVar"] = None
    models: Dict[str, Any] = Field(default_factory=dict)
    datasets: Dict[str, Any] = Field(default_factory=dict)
    samples: Dict[str, Any] = Field(default_factory=dict)
    predictions: Dict[str, Any] = Field(default_factory=dict)
    evaluations: Dict[str, Any] = Field(default_factory=dict)
    allocations: Dict[str, Any] = Field(default_factory=dict)
    # AOI descriptor (GUI-populated, library-agnostic): light metadata only
    # (method, name, gee, admin, geometry_file). The geometry itself lives in a
    # sidecar ``aoi.geojson`` in the project folder, written/read by the GUI —
    # the project model stays free of geopandas/pysepal types. None when no AOI.
    aoi: Optional[Dict[str, Any]] = None

    def _relink_backrefs(self) -> None:
        """Point every contained variable/model/prediction's ``.project`` at self.

        pydantic's shallow ``model_copy()`` shares child objects with the original
        and does not run ``model_post_init``, so a copy's children keep their
        ``.project`` pointing at the *original* project. The GUI replaces
        ``project.value`` via ``project.set(p.model_copy())`` on every action, so
        without re-linking, operations that mutate via a variable's ``.project``
        back-reference (e.g. ``use_as_base_raster`` -> ``self.project.base_raster
        = self``) hit the discarded original instead of the live project.
        """
        for var in self.raw_variables.values():
            var.project = self
        for var in self.processed_variables.values():
            var.project = self
        if self.base_raster is not None:
            self.base_raster.project = self
        for model in self.models.values():
            if hasattr(model, "project"):
                model.project = self
        for prediction in self.predictions.values():
            if hasattr(prediction, "project"):
                prediction.project = self
        for dataset in self.datasets.values():
            if hasattr(dataset, "project"):
                dataset.project = self
        for sample in self.samples.values():
            if hasattr(sample, "project"):
                sample.project = self

    def model_copy(self, *, update=None, deep=False) -> "Project":
        """Copy the project and re-link all child ``.project`` back-references.

        See ``_relink_backrefs`` for why this is required.
        """
        copied = super().model_copy(update=update, deep=deep)
        copied._relink_backrefs()
        return copied

    @staticmethod
    def _ensure_model_schemas() -> None:
        """Resolve the forward references between Project and the variable models."""
        from spatialrisk.variables import (
            GEEVar,
            LocalRasterVar,
            LocalVectorVar,
        )
        from spatialrisk.variables.variable import Variable

        # Rebuild variable models first so they know about Project
        types_namespace = {"Project": Project}

        Variable.model_rebuild(_types_namespace=types_namespace)
        LocalVectorVar.model_rebuild(_types_namespace=types_namespace)
        LocalRasterVar.model_rebuild(_types_namespace=types_namespace)
        GEEVar.model_rebuild(_types_namespace=types_namespace)

        # Finally rebuild Project to include the updated variable schemas
        project_namespace = {
            "LocalVectorVar": LocalVectorVar,
            "LocalRasterVar": LocalRasterVar,
            "Variable": Variable,
        }

        Project.model_rebuild(_types_namespace=project_namespace)

    @property
    def folders(self) -> Box:
        """Initialize and return project folder structure."""
        return self.initialize_folders()

    @property
    def raw_vars(self) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """Alias for raw variables."""
        return self.raw_variables

    @property
    def processed_vars(self) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """Alias for processed variables."""
        return self.processed_variables

    @property
    def variables(
        self,
    ) -> Dict[str, Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]]:
        """Return both raw and processed variables as two separate collections.

        The resulting dictionary is structured as::

            {
                "raw": {
                    "var1": <LocalVar>,
                    "var2": <LocalVar>,
                    ...
                },
                "processed": {
                    "var3": <LocalVar>,
                    "var4": <LocalVar>,
                    ...
                }
            }

        This provides clear separation between raw and processed datasets.
        """
        print("project.variables → {'raw': {...}, 'processed': {...}} view")
        return {
            "raw": dict(self.raw_variables),
            "processed": dict(self.processed_variables),
        }

    def add_variable(self, variable: Union["LocalVectorVar", "LocalRasterVar"]) -> None:
        """
        Add a variable to the project's processed variables collection.

        Note: This method is kept for backward compatibility.
        Prefer using variable.add_as_raw() or variable.add_as_processed() instead.

        Parameters
        ----------
        variable : LocalVectorVar | LocalRasterVar
            The variable to add to the project
        """
        print(f"Adding variable: {variable.name}")
        self.processed_variables[variable.name] = variable

    def get_variable(
        self, name: str, year: Optional[int] = None, source: str = "processed"
    ) -> Optional[Union["LocalVectorVar", "LocalRasterVar"]]:
        """
        Get a variable by name and optional year.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'towns', 'forest_gfc')
        year : int, optional
            Year for temporal variables. If None, returns static variable.
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns:
        -------
        LocalVectorVar | LocalRasterVar | None
            The variable if found, None otherwise
        """
        variables = (
            self.processed_variables if source == "processed" else self.raw_variables
        )

        # Construct storage key
        storage_key = f"{name}_{year}" if year else name

        # Try storage key lookup (fast path)
        if storage_key in variables:
            return variables[storage_key]

        # Slow path: resolve by the variable's .name. A single-year variable is
        # keyed "name_year" yet reported non-temporal (is_temporal needs 2+
        # years), so callers look it up with year=None and miss the fast path
        # above. Match on .name to recover it.
        matches = [var for var in variables.values() if var.name == name]
        if year is not None:
            for var in matches:
                if var.year == year:
                    return var
            return None
        # No year requested: return the sole instance if unambiguous; a
        # genuinely temporal variable (multiple instances) stays None so the
        # caller is forced to specify a year.
        if len(matches) == 1:
            return matches[0]
        return None

    def get_all_instances(
        self, name: str, source: str = "processed"
    ) -> List[Union["LocalVectorVar", "LocalRasterVar"]]:
        """
        Get all year instances of a variable by name.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'towns', 'forest_gfc')
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns:
        -------
        List[LocalVectorVar | LocalRasterVar]
            List of all variable instances with matching name
        """
        variables = (
            self.processed_variables if source == "processed" else self.raw_variables
        )
        return [var for var in variables.values() if var.name == name]

    def is_temporal(self, name: str, source: str = "processed") -> bool:
        """
        Check if a variable has multiple year instances (is temporal).

        Parameters
        ----------
        name : str
            Variable name
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns:
        -------
        bool
            True if variable has multiple years, False otherwise
        """
        instances = self.get_all_instances(name, source)
        unique_years = set(var.year for var in instances if var.year is not None)
        return len(unique_years) > 1

    def get_variable_years(self, name: str, source: str = "processed") -> List[int]:
        """
        Get list of available years for a variable.

        Parameters
        ----------
        name : str
            Variable name
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns:
        -------
        List[int]
            Sorted list of years for the variable
        """
        instances = self.get_all_instances(name, source)
        years = sorted(set(var.year for var in instances if var.year is not None))
        return years

    def get_available_years(self, source: str = "all") -> List[int]:
        """
        Get all unique years across all variables in the project.

        This automatically discovers years from the variables that have been added,
        rather than relying on a manually specified years list.

        Parameters
        ----------
        source : str, optional
            Variable source: 'processed' (default), 'raw', or 'all'

        Returns:
        -------
        List[int]
            Sorted list of all unique years found in variables

        Examples:
        --------
        >>> project.get_available_years()  # All years from all variables
        [2015, 2020, 2024]
        >>> project.get_available_years(source='raw')  # Only from raw variables
        [2015, 2020, 2024]
        """
        years_set = set()

        if source in ("processed", "all"):
            for var in self.processed_variables.values():
                if var.year is not None:
                    years_set.add(var.year)

        if source in ("raw", "all"):
            for var in self.raw_variables.values():
                if var.year is not None:
                    years_set.add(var.year)

        return sorted(years_set)

    def list_unique_variable_names(self, source: str = "processed") -> List[str]:
        """
        Get list of unique variable names.

        Parameters
        ----------
        source : str, optional
            Variable source: 'processed' (default) or 'raw'

        Returns:
        -------
        List[str]
            Sorted list of unique variable names
            (e.g., ['altitude', 'towns', 'forest_gfc'])
        """
        variables = (
            self.processed_variables if source == "processed" else self.raw_variables
        )
        unique_names = sorted(set(var.name for var in variables.values()))
        return unique_names

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def add_model(
        self,
        model: Any,
        key: Optional[str] = None,
        auto_save: bool = True,
    ) -> None:
        """Add a trained ML model to the project's model registry.

        Parameters
        ----------
        model : BaseRiskModel
            Trained model instance (GLMModel, RFModel, ICARModel, …).
        key : str, optional
            Storage key. Defaults to "{model_type}_{name}" or "{model_type}".
        auto_save : bool
            If True, saves the project JSON after registering.
        """
        model.project = self
        model.project_name = self.project_name
        storage_key = key or (
            f"{model.model_type}_{model.name}" if model.name else model.model_type
        )
        self.models[storage_key] = model
        print(f"  Model registered as project.models['{storage_key}']")
        if auto_save:
            self.save()

    def delete_model(
        self, key: str, *, delete_files: bool = True, auto_save: bool = False
    ) -> bool:
        """Remove a registered model and (optionally) its on-disk artifacts.

        Predictions produced by the model are left in place — delete them
        separately via :meth:`delete_prediction` if desired.

        Returns True if a model was found and removed.
        """
        model = self.models.pop(key, None)
        if model is None:
            return False
        if delete_files:
            for path in model.output_files():
                self._safe_unlink(path)
        logger.info("Model deleted: project.models['%s']", key)
        if auto_save:
            self.save()
        return True

    def get_model(self, key: str) -> Optional[Any]:
        """Return the model stored under *key*, or None if not found.

        Parameters
        ----------
        key : str
            Storage key used when the model was registered.
        """
        return self.models.get(key)

    def list_models(self) -> List[str]:
        """Return sorted list of registered model keys."""
        return sorted(self.models.keys())

    def add_dataset(
        self, dataset: Any, key: Optional[str] = None, auto_save: bool = True
    ) -> None:
        """Add a dataset to the project's dataset registry.

        Parameters
        ----------
        dataset : Dataset
            Configured dataset instance.
        key : str, optional
            Storage key. Defaults to dataset.name.
        auto_save : bool
            If True, saves the project JSON after registering.
        """
        storage_key = key or dataset.name
        if not storage_key:
            raise ValueError("Dataset must have a name or provide a key parameter.")
        dataset.project = self
        self.datasets[storage_key] = dataset
        print(f"  Dataset registered as project.datasets['{storage_key}']")
        if auto_save:
            self.save()

    def get_dataset(self, key: str) -> Optional[Any]:
        """Return the dataset stored under *key*, or None if not found."""
        return self.datasets.get(key)

    def list_datasets(self) -> List[str]:
        """Return sorted list of registered dataset keys."""
        return sorted(self.datasets.keys())

    def add_sample(
        self, sample: Any, key: Optional[str] = None, auto_save: bool = True
    ) -> None:
        """Register a Sample under ``key`` (defaults to sample.name)."""
        storage_key = key or sample.name
        if not storage_key:
            raise ValueError("Sample must have a name or provide a key.")
        sample.project = self
        self.samples[storage_key] = sample
        print(f"  Sample registered as project.samples['{storage_key}']")
        if auto_save:
            self.save()

    def get_sample(self, key: str) -> Optional[Any]:
        """Return the sample stored under *key*, or None."""
        return self.samples.get(key)

    def list_samples(self) -> List[str]:
        """Return sorted list of registered sample keys."""
        return sorted(self.samples.keys())

    def delete_sample(self, key: str, auto_save: bool = True) -> None:
        """Remove a sample from the registry and delete its points file."""
        sample = self.samples.pop(key, None)
        if sample is None:
            return
        path = getattr(sample, "points_path", None)
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                print(f"  ⚠ Could not delete sample file: {path}")
        pm = getattr(sample, "pmtiles_path", None)
        if pm is not None:
            try:
                Path(pm).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete sample pmtiles: %s", pm)
        if auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Prediction registry
    # ------------------------------------------------------------------

    def add_prediction(
        self,
        prediction: Any,
        key: Optional[str] = None,
        auto_save: bool = True,
    ) -> None:
        """Add a Prediction to the project's prediction registry.

        Parameters
        ----------
        prediction : Prediction
            Prediction instance (one output raster).
        key : str, optional
            Storage key. Defaults to ``prediction.storage_key()``.
        auto_save : bool
            If True, saves the project JSON after registering.
        """
        prediction.project = self
        storage_key = key or prediction.storage_key()
        self.predictions[storage_key] = prediction
        print(f"  Prediction registered as project.predictions['{storage_key}']")
        if auto_save:
            self.save()

    def delete_prediction(
        self, key: str, *, delete_file: bool = True, auto_save: bool = False
    ) -> bool:
        """Remove a registered prediction and (optionally) its output raster.

        Returns True if a prediction was found and removed.
        """
        prediction = self.predictions.pop(key, None)
        if prediction is None:
            return False
        if delete_file and getattr(prediction, "path", None):
            self._safe_unlink(prediction.path)
            # The viewer's overview sidecar: GDAL would serve a leftover .ovr
            # to the next raster written under this name.
            self._safe_unlink(str(prediction.path) + ".ovr")
        logger.info("Prediction deleted: project.predictions['%s']", key)
        if auto_save:
            self.save()
        return True

    def add_allocation(self, run, auto_save: bool = True) -> str:
        """Register an AllocationRun under its history-safe storage key."""
        key = run.storage_key()
        self.allocations[key] = run
        if auto_save:
            self.save()
        return key

    def delete_allocation(self, key: str) -> bool:
        """Remove an allocation record (artifacts are removed by the caller).

        Deletion is transactional in the GUI helper: the record is dropped here
        without autosaving, the manifest is committed, and only then are the run's
        files removed — see gui/scripts/allocation_runner.py::delete_allocation_run.
        """
        if key not in self.allocations:
            return False
        del self.allocations[key]
        return True

    def _project_dir(self) -> Path:
        """Folder holding this project's files (manifest, rasters, model artifacts)."""
        return downloads_folder / self.project_name

    def _safe_unlink(self, path: Union[str, Path]) -> bool:
        """Delete *path*, but only if it exists and lives inside the project folder.

        The within-project guard prevents a malformed or unexpected absolute path
        from removing files elsewhere on disk. Returns True if a file was removed;
        missing/out-of-scope/locked files are skipped with a warning, never raised.
        """
        try:
            target = Path(path).resolve()
        except (OSError, RuntimeError):
            return False
        project_dir = self._project_dir().resolve()
        if project_dir not in target.parents:
            logger.warning(
                "Refusing to delete %s — outside project folder %s", target, project_dir
            )
            return False
        if not target.exists():
            return False
        try:
            target.unlink()
            return True
        except OSError as exc:
            logger.warning("Could not delete %s: %s", target, exc)
            return False

    def get_prediction(self, key: str) -> Optional[Any]:
        """Return the prediction stored under *key*, or None if not found."""
        return self.predictions.get(key)

    def list_predictions(self) -> List[str]:
        """Return registered prediction keys in insertion order."""
        return list(self.predictions.keys())

    # ------------------------------------------------------------------
    # Evaluation registry
    # ------------------------------------------------------------------

    def add_evaluation(
        self,
        record: Any,
        key: Optional[str] = None,
        auto_save: bool = True,
    ) -> None:
        """Register a saved evaluation run. Defaults the key to record.storage_key()."""
        storage_key = key or record.storage_key()
        self.evaluations[storage_key] = record
        print(f"  Evaluation registered as project.evaluations['{storage_key}']")
        if auto_save:
            self.save()

    def get_evaluation(self, key: str) -> Optional[Any]:
        """Return the evaluation record under *key*, or None."""
        return self.evaluations.get(key)

    def list_evaluations(self) -> List[str]:
        """Return registered evaluation keys in insertion order."""
        return list(self.evaluations.keys())

    def delete_evaluation(self, key: str, auto_save: bool = False) -> bool:
        """Remove an evaluation record (registry only; on-disk artifacts stay)."""
        removed = self.evaluations.pop(key, None)
        if removed is None:
            return False
        logger.info("Evaluation deleted: project.evaluations['%s']", key)
        if auto_save:
            self.save()
        return True

    def filter_predictions(
        self,
        model_key: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **attrs: Any,
    ) -> Dict[str, Any]:
        """Return the subset of predictions matching the given criteria.

        Parameters
        ----------
        model_key : str, optional
            Keep only predictions from this model key.
        dataset_name : str, optional
            Keep only predictions from this dataset.
        **attrs
            Additional exact-match filters on any Prediction attribute
            (e.g. ``year=2020``, ``window=5``, ``active=True``).
        """
        result: Dict[str, Any] = {}
        for key, pred in self.predictions.items():
            if model_key is not None and pred.model_key != model_key:
                continue
            if dataset_name is not None and pred.dataset_name != dataset_name:
                continue
            if any(getattr(pred, attr, None) != value for attr, value in attrs.items()):
                continue
            result[key] = pred
        return result

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save the project to a JSON file in the project folder.

        Parameters
        ----------
        filename : str, optional
            Custom filename for the project file.
            If None, uses '{project_name}_project.json'

        Returns:
        -------
        Path
            Path to the saved JSON file
        """
        # Ensure schemas are up-to-date before serializing any variables
        self._ensure_model_schemas()

        if filename is None:
            filename = f"{self.project_name}_project.json"

        project_folder = self.folders.project_folder
        project_folder.mkdir(parents=True, exist_ok=True)

        save_path = project_folder / filename

        # Prepare data for serialization
        data = {
            "project_name": self.project_name,
            "raw_variables": {},
            "processed_variables": {},
        }

        # Only include years if explicitly set (for backward compatibility)
        if self.years is not None:
            data["years"] = self.years

        # Serialize raw variables. GEEVars are session-only: they hold live
        # ee.Image objects (not JSON-serializable) and load() only ever
        # reconstructs Local*Var, so a persisted GEEVar could never be read
        # back. Skip them — they are materialized to local vars before save.
        # Registries are snapshotted with list(): background workers insert
        # into these shared dicts while a save may be walking them.
        for var_name, var in list(self.raw_variables.items()):
            if type(var).__name__ == "GEEVar":
                continue
            data["raw_variables"][var_name] = var.model_dump(mode="json")

        # Serialize processed variables
        for var_name, var in list(self.processed_variables.items()):
            data["processed_variables"][var_name] = var.model_dump(mode="json")

        # Serialize base_raster if it exists
        if self.base_raster is not None:
            data["base_raster"] = self.base_raster.model_dump(mode="json")

        # Serialize registered ML models
        if self.models:
            data["models"] = {}
            for key, model in list(self.models.items()):
                data["models"][key] = model.model_dump(mode="json")

        # Serialize registered datasets
        if self.datasets:
            data["datasets"] = {}
            for key, dataset in list(self.datasets.items()):
                data["datasets"][key] = {
                    "name": dataset.name,
                    "year": dataset.year,
                    "target_name": dataset.target.name if dataset.target else None,
                    "target_year": dataset.target.year if dataset.target else None,
                    "feature_names": [f.name for f in dataset.features],
                }

        # Serialize registered samples (location-only; the GPKG is the truth).
        if self.samples:
            data["samples"] = {}
            for key, s in list(self.samples.items()):
                data["samples"][key] = {
                    "name": s.name,
                    "raster_var_name": s.raster_var_name,
                    "mask_var_name": s.mask_var_name,
                    "strategy": s.strategy,
                    "n_samples": s.n_samples,
                    "spacing_m": s.spacing_m,
                    "allocation": s.allocation,
                    "adapt": s.adapt,
                    "seed": s.seed,
                    "points_path": str(s.points_path) if s.points_path else None,
                    "pmtiles_path": str(s.pmtiles_path) if s.pmtiles_path else None,
                    "crs": s.crs,
                    "n_total": s.n_total,
                    "class_counts": s.class_counts,
                    "created_at": s.created_at,
                }

        # Serialize registered predictions
        if self.predictions:
            data["predictions"] = {}
            for key, prediction in list(self.predictions.items()):
                data["predictions"][key] = prediction.model_dump(mode="json")

        # Serialize saved evaluation runs
        if self.evaluations:
            data["evaluations"] = {}
            for key, record in list(self.evaluations.items()):
                data["evaluations"][key] = record.model_dump(mode="json")

        # Serialize saved allocation runs
        if self.allocations:
            data["allocations"] = {}
            for key, run in list(self.allocations.items()):
                data["allocations"][key] = run.model_dump(mode="json")

        # Serialize the AOI descriptor (geometry lives in the sidecar file)
        if self.aoi:
            data["aoi"] = self.aoi

        # Write to file atomically: serialize to a temp file in the same
        # folder, then os.replace() onto the final path. os.replace is an
        # atomic rename on a given filesystem, so a crash or interrupt
        # mid-write can never leave a truncated/corrupt manifest — readers
        # always see either the previous complete file or the new one.
        tmp_path = project_folder / f".{save_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            tmp_path.write_text(
                json.dumps(data, indent=4, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            os.replace(tmp_path, save_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        print(f"Project saved to: {save_path}")

        return save_path

    @classmethod
    def load(cls, project_name: str, filename: Optional[str] = None) -> "Project":
        """
        Load a project from a JSON file.

        Parameters
        ----------
        project_name : str
            Name of the project (used to locate the project folder)
        filename : str, optional
            Custom filename for the project file.
            If None, uses '{project_name}_project.json'

        Returns:
        -------
        Project
            Loaded project instance with all variables
        """
        # Ensure schemas are up-to-date before instantiating variables
        cls._ensure_model_schemas()

        from spatialrisk.variables import LocalRasterVar, LocalVectorVar

        if filename is None:
            filename = f"{project_name}_project.json"

        project_folder = downloads_folder / project_name
        load_path = project_folder / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Project file not found: {load_path}")

        # Load JSON data
        data = json.loads(load_path.read_text(encoding="utf-8"))

        # Create project instance without variables first
        project = cls(
            project_name=data["project_name"],
            years=data.get("years"),
            aoi=data.get("aoi"),
        )

        # Reconstruct raw variables
        for var_name, var_data in data.get("raw_variables", {}).items():
            # Convert Path strings back to Path objects
            if "path" in var_data and var_data["path"]:
                var_data["path"] = Path(var_data["path"])

            # Determine which class to use based on data_type
            if var_data.get("data_type") == "vector":
                var = LocalVectorVar(**var_data)
            elif var_data.get("data_type") == "raster":
                var = LocalRasterVar(**var_data)
            else:
                raise ValueError(
                    f"Unknown data_type for variable {var_name}: "
                    f"{var_data.get('data_type')}"
                )

            # Set project reference and add to raw_variables
            var.project = project
            project.raw_variables[var_name] = var

        # Reconstruct processed variables
        for var_name, var_data in data.get("processed_variables", {}).items():
            # Convert Path strings back to Path objects
            if "path" in var_data and var_data["path"]:
                var_data["path"] = Path(var_data["path"])

            # Determine which class to use based on data_type
            if var_data.get("data_type") == "vector":
                var = LocalVectorVar(**var_data)
            elif var_data.get("data_type") == "raster":
                var = LocalRasterVar(**var_data)
            else:
                raise ValueError(
                    f"Unknown data_type for variable {var_name}: "
                    f"{var_data.get('data_type')}"
                )

            # Set project reference and add to processed_variables
            var.project = project
            project.processed_variables[var_name] = var

        # Reconstruct base_raster if it exists
        if "base_raster" in data and data["base_raster"]:
            base_data = data["base_raster"]

            # Convert Path strings back to Path objects
            if "path" in base_data and base_data["path"]:
                base_data["path"] = Path(base_data["path"])

            # Create the base raster variable
            project.base_raster = LocalRasterVar(**base_data)
            # Set project reference
            project.base_raster.project = project

        # Reconstruct registered ML models
        if "models" in data and data["models"]:
            from spatialrisk.mlmodels import (
                GLMModel,
                ICARModel,
                JNRBenchmarkModel,
                MWModel,
                RFModel,
            )

            _MODEL_REGISTRY = {
                "glm": GLMModel,
                "rf": RFModel,
                "icar": ICARModel,
                "jnr": JNRBenchmarkModel,
                "mw": MWModel,
            }
            for key, model_data in data["models"].items():
                model_type = model_data.get("model_type", "")
                # Legacy GUI keys: the JNR benchmark was stored under a
                # "benchmark[_name]" key before the family token was unified
                # on model_type ("jnr[_name]").
                if model_type == "jnr" and key.split("_")[0] == "benchmark":
                    new_key = "jnr" + key[len("benchmark") :]
                    print(f"  Migrating legacy model key '{key}' → '{new_key}'")
                    key = new_key
                model_cls = _MODEL_REGISTRY.get(model_type)
                if model_cls is None:
                    print(
                        f"  Warning: unknown model_type '{model_type}' "
                        f"for key '{key}' — skipped"
                    )
                    continue
                # Convert Path strings back to Path objects
                for path_field in ("model_path", "samples_path", "rho_path"):
                    if path_field in model_data and model_data[path_field]:
                        model_data[path_field] = Path(model_data[path_field])
                model = model_cls(**model_data)
                model.project = project
                project.models[key] = model
            print(f"Loaded {len(project.models)} model(s)")

        # Reconstruct registered datasets
        if "datasets" in data and data["datasets"]:
            from spatialrisk.dataset import Dataset

            for key, ds_data in data["datasets"].items():
                ds = Dataset(
                    project=project,
                    name=ds_data.get("name"),
                    year=ds_data.get("year"),
                )
                target_name = ds_data.get("target_name")
                feature_names = ds_data.get("feature_names", [])
                if target_name:
                    # The dataset's stored year applies to temporal features and is
                    # already restored via the constructor above. Only pass it to
                    # set_target when the target itself is temporal, since set_target
                    # rejects a year argument for static targets.
                    target_is_temporal = project.is_temporal(target_name)
                    ds.set_target(
                        target_name,
                        year=ds_data.get("year") if target_is_temporal else None,
                    )
                if feature_names:
                    missing = [
                        n for n in feature_names if not project.get_all_instances(n)
                    ]
                    valid_names = [
                        n for n in feature_names if project.get_all_instances(n)
                    ]
                    if missing:
                        print(
                            f"  ⚠ Dataset '{key}': feature(s) not found in "
                            f"processed variables, skipped: {missing}"
                        )
                    if valid_names:
                        ds.set_features(valid_names)
                project.datasets[key] = ds
            print(f"Loaded {len(project.datasets)} dataset(s)")

        # Reconstruct registered samples (location-only; no regeneration).
        if "samples" in data and data["samples"]:
            from pathlib import Path as _Path

            from spatialrisk.sample import Sample

            loaded = 0
            for key, s_data in data["samples"].items():
                if "raster_var_name" not in s_data:
                    print(f"  ⚠ Sample '{key}' uses the old schema — skipped.")
                    continue
                s = Sample(
                    name=s_data.get("name", key),
                    raster_var_name=s_data["raster_var_name"],
                    mask_var_name=s_data.get("mask_var_name"),
                    strategy=s_data.get("strategy", "random"),
                    n_samples=s_data.get("n_samples"),
                    spacing_m=s_data.get("spacing_m"),
                    allocation=s_data.get("allocation"),
                    adapt=s_data.get("adapt", False),
                    seed=s_data.get("seed"),
                    points_path=(
                        _Path(s_data["points_path"])
                        if s_data.get("points_path")
                        else None
                    ),
                    pmtiles_path=(
                        _Path(s_data["pmtiles_path"])
                        if s_data.get("pmtiles_path")
                        else None
                    ),
                    crs=s_data.get("crs"),
                    n_total=s_data.get("n_total", 0),
                    class_counts=s_data.get("class_counts", {}),
                    created_at=s_data.get("created_at"),
                )
                s.project = project
                project.samples[key] = s
                loaded += 1
            print(f"Loaded {loaded} sample(s)")

        # Reconstruct registered predictions
        if "predictions" in data and data["predictions"]:
            from spatialrisk.predictions.prediction import Prediction

            for key, pred_data in data["predictions"].items():
                if pred_data.get("path"):
                    pred_data["path"] = Path(pred_data["path"])
                if pred_data.get("defrate_path"):
                    pred_data["defrate_path"] = Path(pred_data["defrate_path"])
                prediction = Prediction(**pred_data)
                prediction.project = project
                project.predictions[key] = prediction
            print(f"Loaded {len(project.predictions)} prediction(s)")

        # Reconstruct saved evaluation runs
        if "evaluations" in data and data["evaluations"]:
            from spatialrisk.evaluations import EvaluationRecord

            for key, ev_data in data["evaluations"].items():
                project.evaluations[key] = EvaluationRecord(**ev_data)
            print(f"Loaded {len(project.evaluations)} evaluation(s)")

        # Reconstruct saved allocation runs
        if "allocations" in data and data["allocations"]:
            from spatialrisk.allocations import AllocationRun

            for key, run_data in data["allocations"].items():
                project.allocations[key] = AllocationRun(**run_data)
            print(f"Loaded {len(project.allocations)} allocation run(s)")

        print(f"Project loaded from: {load_path}")
        print(f"Loaded {len(project.processed_variables)} processed variables")
        return project

    def reproject_and_match_all(
        self,
        target_epsg: Optional[str] = None,
        resolution: Optional[float] = None,
        source: str = "raw",
        keys: Optional[Iterable[str]] = None,
        add_to_processed: bool = True,
        auto_save: bool = True,
        **reproject_kwargs,
    ) -> Dict[str, "LocalRasterVar"]:
        """
        Reproject all raster variables in the project.

        Parameters
        ----------
        target_epsg : str, optional
            Target EPSG code. If None, uses base_raster's CRS (base_raster must be set).
        resolution : float, optional
            Target resolution. If None, uses base_raster's resolution
            (base_raster must be set).
        source : str, optional
            Which variables to reproject: 'raw' or 'processed' (default: 'raw').
        keys : iterable of str, optional
            Restrict the run to these source-collection keys. ``None`` (default)
            reprojects every raster, which is what the notebooks expect;
            ``process_actions.run_processing`` passes the keys that
            ``harmonization_status_from_disk`` reports as pending, so an
            already-aligned layer is not re-derived. An empty iterable means
            "nothing to do".
        add_to_processed : bool, optional
            Whether to add reprojected variables to the processed collection
            (default: True).
        auto_save : bool, optional
            Whether to auto-save after each reprojection (default: True).
        **reproject_kwargs
            Additional arguments passed to LocalRasterVar.reproject().

        Returns:
        -------
        Dict[str, LocalRasterVar]
            Dictionary of reprojected variables {name: LocalRasterVar}.

        Raises:
        ------
        ValueError
            If base_raster is not set when target_epsg or resolution is None.
            If source is not 'raw' or 'processed'.

        Examples:
        --------
        >>> # Reproject all raw variables to base raster's CRS
        >>> project.reproject_all()

        >>> # Reproject to specific CRS
        >>> project.reproject_all(target_epsg="EPSG:32618", resolution=30)
        """
        # Determine source collection
        if source == "raw":
            source_vars = self.raw_variables
        elif source == "processed":
            source_vars = self.processed_variables
        else:
            raise ValueError(f"source must be 'raw' or 'processed', got '{source}'")

        # Determine target CRS and resolution
        if target_epsg is None or resolution is None:
            if self.base_raster is None:
                raise ValueError(
                    "base_raster must be set when target_epsg or resolution is "
                    "not provided. "
                    "Use variable.use_as_base_raster() to set a base raster first."
                )

        # Reproject all active raster variables
        reprojected_vars = {}
        skipped_count = 0

        # `keys is None` means "everything"; an empty iterable means "nothing".
        selected = None if keys is None else set(keys)
        # Filter on data_type (not isinstance) so module reloads don't break it.
        raster_pairs = [
            (k, v)
            for k, v in source_vars.items()
            if v.data_type == DataType.raster and (selected is None or k in selected)
        ]
        for var_key, var in log_progress(
            raster_pairs, "Reprojecting", label=lambda kv: kv[0]
        ):
            reprojected = var.reproject_and_match(
                geobox=self.base_raster.get_base_geobox(),
            )

            if add_to_processed:
                reprojected.add_as_processed(auto_save=auto_save)

            storage_key = (
                f"{reprojected.name}_{reprojected.year}"
                if reprojected.year
                else reprojected.name
            )
            reprojected_vars[storage_key] = reprojected

        print(f"\n✅ Reprojected {len(reprojected_vars)} raster variables")
        if skipped_count > 0:
            print(f"   ({skipped_count} inactive variables skipped)")
        return reprojected_vars

    def rasterize_all(
        self,
        source: str = "raw",
        keys: Optional[Iterable[str]] = None,
        add_to_processed: bool = True,
        auto_save: bool = True,
        **rasterize_kwargs,
    ) -> Dict[str, "LocalRasterVar"]:
        """
        Rasterize all vector variables in the project using the base raster.

        Parameters
        ----------
        source : str, optional
            Which variables to rasterize: 'raw' or 'processed' (default: 'raw').
        keys : iterable of str, optional
            Restrict the run to these source-collection keys. ``None`` (default)
            rasterizes every active vector. An empty iterable means "nothing
            to do". See ``reproject_and_match_all``.
        add_to_processed : bool, optional
            Whether to add rasterized variables to processed collection (default: True).
        auto_save : bool, optional
            Whether to auto-save after each rasterization (default: True).
        **rasterize_kwargs
            Additional arguments passed to LocalVectorVar.rasterize().

        Returns:
        -------
        Dict[str, LocalRasterVar]
            Dictionary of rasterized variables {name: LocalRasterVar}.

        Raises:
        ------
        ValueError
            If base_raster is not set.
            If source is not 'raw' or 'processed'.

        Examples:
        --------
        >>> # Set base raster first
        >>> dem.use_as_base_raster()

        >>> # Rasterize all raw vector variables
        >>> project.rasterize_all()
        """
        # Check base raster is set
        if self.base_raster is None:
            raise ValueError(
                "base_raster must be set before rasterizing. "
                "Use variable.use_as_base_raster() to set a base raster first."
            )

        # Determine source collection
        if source == "raw":
            source_vars = self.raw_variables
        elif source == "processed":
            source_vars = self.processed_variables
        else:
            raise ValueError(f"source must be 'raw' or 'processed', got '{source}'")

        # Rasterize all active vector variables
        rasterized_vars = {}
        skipped_count = 0

        # `keys is None` means "everything"; an empty iterable means "nothing".
        selected = None if keys is None else set(keys)

        for var_key, var in source_vars.items():
            if selected is not None and var_key not in selected:
                continue
            if not var.active:
                print(f"⏭️  Skipping '{var_key}' (inactive)")
                skipped_count += 1

        # Filter on data_type (not isinstance) so module reloads don't break it.
        vector_pairs = [
            (k, v)
            for k, v in source_vars.items()
            if v.active
            and v.data_type == DataType.vector
            and (selected is None or k in selected)
        ]
        for var_key, var in log_progress(
            vector_pairs, "Rasterizing", label=lambda kv: kv[0]
        ):
            rasterized = var.rasterize(base=self.base_raster, **rasterize_kwargs)

            if add_to_processed:
                rasterized.add_as_processed(auto_save=auto_save)

            storage_key = (
                f"{rasterized.name}_{rasterized.year}"
                if rasterized.year
                else rasterized.name
            )
            rasterized_vars[storage_key] = rasterized

        print(f"\n✅ Rasterized {len(rasterized_vars)} vector variables")
        if skipped_count > 0:
            print(f"   ({skipped_count} inactive variables skipped)")
        return rasterized_vars

    def list_variables(
        self,
        source: str = "processed",
        **filters: Any,
    ) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """Return variables from the requested collection applying simple filters.

        Parameters
        ----------
        source : str, optional
            Collection to inspect: 'processed' (default), 'raw', or 'both'.
        **filters : dict
            Attribute filters evaluated as equality checks. Iterables (except
            strings/bytes) are treated as lists of acceptable values. Callables
            are invoked with the attribute value and must return True to keep it.
        """
        if source == "processed":
            candidates = self.processed_vars
        elif source == "raw":
            candidates = self.raw_vars
        elif source == "both":
            # Combine both, with processed overriding raw for duplicate names
            candidates = {}
            candidates.update(self.raw_vars)
            candidates.update(self.processed_vars)
        else:
            raise ValueError("source must be 'processed', 'raw', or 'both'")

        def matches(var: Union["LocalVectorVar", "LocalRasterVar"]) -> bool:
            for attr, expected in filters.items():
                if not hasattr(var, attr):
                    raise AttributeError(
                        f"Variable '{var.name}' has no attribute '{attr}'"
                    )

                value = getattr(var, attr)

                if callable(expected):
                    if not expected(value):
                        return False
                elif isinstance(expected, Iterable) and not isinstance(
                    expected, (str, bytes, bytearray)
                ):
                    if value not in expected:
                        return False
                else:
                    if value != expected:
                        return False

            return True

        if not filters:
            return dict(candidates)

        return {name: var for name, var in candidates.items() if matches(var)}

    def filter_by_tags(
        self,
        tags: Union[str, List[str]],
        match_all: bool = False,
        look_up_in: Optional[str] = None,
        **filters,
    ) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """
            Filter variables by tags.

            Parameters
            ----------
            tags : str or List[str]
                Tag(s) to filter by. Can be a single tag string or a list of tags.
            match_all : bool, optional
                If True, variables must have ALL specified tags (AND logic).
                If False (default), variables must have AT LEAST ONE tag (OR logic).
            look_up_in : str, optional
                Collection to search: 'processed' (default), 'raw', or 'both'.
            **filters : keyword arguments
                Additional filter criteria (same as list_variables).

        Returns:
            -------
            Dict[str, Variable]
                Dictionary of variables matching the tag criteria and any
                additional filters.

        Examples:
            --------
            >>> # Get all variables with 'climate' tag
        >>> project.filter_by_tags('climate')

        >>> # Get variables with either 'roads' OR 'infrastructure' tag
        >>> project.filter_by_tags(['roads', 'infrastructure'])

            >>> # Get active variables with BOTH 'climate' AND 'temperature' tags
            >>> project.filter_by_tags(
            ...     ['climate', 'temperature'], match_all=True, active=True
            ... )

            >>> # Get raw raster variables with 'elevation' tag
            >>> project.filter_by_tags(
            ...     'elevation', look_up_in='raw', data_type='raster'
            ... )
        """
        # Normalize tags to a list
        if isinstance(tags, str):
            tags = [tags]

        if look_up_in is None:
            look_up_in = filters.pop("source", "processed")

        # Get all variables matching the basic filters
        variables = self.list_variables(source=look_up_in, **filters)

        # Filter by tags
        result = {}
        for var_name, var in variables.items():
            if match_all:
                # Variable must have ALL specified tags
                if all(tag in var.tags for tag in tags):
                    result[var_name] = var
            else:
                # Variable must have AT LEAST ONE of the specified tags
                if any(tag in var.tags for tag in tags):
                    result[var_name] = var

        return result

    def filter_by_attrs(
        self,
        source: str = "processed",
        **attrs,
    ) -> Dict[str, Union["LocalVectorVar", "LocalRasterVar"]]:
        """
        Filter variables by any attribute(s).

        This is an alias for list_variables() with a more explicit name for filtering.
        You can filter by any variable attribute such as year, name, active status,
        data_type, tags, or any custom attributes.

        Parameters
        ----------
        source : str, optional
            Collection to search: 'processed' (default), 'raw', or 'both'.
        **attrs : keyword arguments
            Attribute filters evaluated as equality checks. You can use:
            - Simple values: year=2020, active=True, data_type='raster'
            - Lists of acceptable values: year=[2019, 2020, 2021]
            - Callable functions: year=lambda y: y >= 2015
            - Tags (special): tags=["tag1"] or tags="tag1" checks if ANY tag
              matches (OR logic)

        Returns:
        -------
        Dict[str, Variable]
            Dictionary of variables that match all specified attribute criteria.

        Examples:
        --------
        >>> # Filter by year
        >>> project.filter_by_attrs(year=2020)

        >>> # Filter by multiple years
        >>> project.filter_by_attrs(year=[2019, 2020, 2021])

        >>> # Filter by tags (checks if variable has ANY of these tags)
        >>> project.filter_by_attrs(source="raw", tags=["town"])
        >>> # Has either "town" OR "city"
        >>> project.filter_by_attrs(tags=["town", "city"])

        >>> # Filter by name pattern (using callable)
        >>> project.filter_by_attrs(name=lambda n: 'forest' in n.lower())

        >>> # Filter active raster variables from 2020
        >>> project.filter_by_attrs(year=2020, active=True, data_type='raster')

        >>> # Filter by year range (using callable)
        >>> project.filter_by_attrs(year=lambda y: y >= 2015 and y <= 2020)

        >>> # Search in raw variables
        >>> project.filter_by_attrs(source='raw', year=2019)

        >>> # Search in both raw and processed
        >>> project.filter_by_attrs(source='both', active=True)

        Notes:
        -----
        - String and bytes values are compared for exact equality.
        - Iterable values (lists, tuples, sets) are treated as "value in list" checks.
        - Tags are special: tags=["tag1", "tag2"] checks whether the variable
          has ANY of these tags.
        - Callable values are invoked with the attribute value and must return True.
        - All filters must match for a variable to be included (AND logic).
        """
        # Special handling for tags attribute
        if "tags" in attrs:
            tags_filter = attrs.pop("tags")

            # Normalize tags to a list
            if isinstance(tags_filter, str):
                tags_filter = [tags_filter]

            # Use filter_by_tags for tag filtering, then apply other filters
            if tags_filter and isinstance(tags_filter, (list, tuple)):
                # Get variables matching the tags
                result = self.filter_by_tags(tags_filter, look_up_in=source)

                # If there are additional filters, apply them
                if attrs:
                    filtered_result = {}
                    for var_name, var in result.items():
                        # Check if variable matches all other filters
                        matches = True
                        for attr, expected in attrs.items():
                            if not hasattr(var, attr):
                                matches = False
                                break

                            value = getattr(var, attr)

                            if callable(expected):
                                if not expected(value):
                                    matches = False
                                    break
                            elif isinstance(expected, Iterable) and not isinstance(
                                expected, (str, bytes, bytearray)
                            ):
                                if value not in expected:
                                    matches = False
                                    break
                            else:
                                if value != expected:
                                    matches = False
                                    break

                        if matches:
                            filtered_result[var_name] = var

                    return filtered_result
                else:
                    return result

        # No tag filtering, use standard list_variables
        return self.list_variables(source=source, **attrs)

    def reset(
        self,
        source: str = "processed",
        auto_save: bool = True,
        confirm: bool = True,
    ) -> int:
        """
        Remove all variables from the specified collection.

        This clears out all variables from either the raw or processed collection,
        useful for cleaning up or starting fresh without deleting the entire project.

        Parameters
        ----------
        source : str, optional
            Which collection to reset: 'processed' (default), 'raw', or 'both'.
        auto_save : bool, optional
            If True (default), automatically saves the project after reset.
        confirm : bool, optional
            If True (default), shows a warning message before clearing.
            Set to False to skip confirmation (useful in scripts).

        Returns:
        -------
        int
            Number of variables removed.

        Raises:
        ------
        ValueError
            If source is not 'processed', 'raw', or 'both'.

        Examples:
        --------
        >>> # Reset processed variables (default)
        >>> project.reset()

        >>> # Reset raw variables
        >>> project.reset(source='raw')

        >>> # Reset both collections
        >>> project.reset(source='both')

        >>> # Reset without confirmation (in automated scripts)
        >>> project.reset(confirm=False, auto_save=False)

        Notes:
        -----
        This does NOT delete the actual files on disk, only removes the
        variable references from the project. Use with caution as this
        operation cannot be undone unless you reload the project from disk.
        """
        if source not in ["processed", "raw", "both"]:
            raise ValueError("source must be 'processed', 'raw', or 'both'")

        removed_count = 0

        if source == "processed" or source == "both":
            count = len(self.processed_variables)
            if confirm and count > 0:
                print(
                    f"⚠️  WARNING: About to remove {count} processed "
                    f"variable(s) from project"
                )

            self.processed_variables.clear()
            removed_count += count

            if count > 0:
                print(f"✓ Removed {count} processed variable(s)")

        if source == "raw" or source == "both":
            count = len(self.raw_variables)
            if confirm and count > 0:
                print(
                    f"⚠️  WARNING: About to remove {count} raw variable(s) from project"
                )

            self.raw_variables.clear()
            removed_count += count

            if count > 0:
                print(f"✓ Removed {count} raw variable(s)")

        if removed_count == 0:
            print(f"i️  No variables to remove from '{source}' collection")
        else:
            print(f"\n✅ Total removed: {removed_count} variable(s)")

        if auto_save and removed_count > 0:
            self.save()

        return removed_count

    def create_model_folder(self, model: str, test_name: Optional[str] = None) -> Path:
        """Path of *model*'s folder inside this project (not created on disk)."""
        # Create the folder path
        project_folder = downloads_folder / self.project_name
        model_folder = project_folder / model

        # Add test_name as subfolder if provided
        if test_name:
            model_folder = model_folder / test_name

        model_folder.mkdir(parents=True, exist_ok=True)

        # Create a meaningful key for project.folders
        if test_name:
            folder_key = f"{model}_{test_name}"
        else:
            folder_key = model

        # Save to project.folders (reinitialize to update the Box object)
        folders = self.initialize_folders()
        folders[folder_key] = model_folder

        print(f"✅ Created model folder: {model_folder}")
        print(f"📁 Saved as: project.folders.{folder_key}")
        return model_folder

    def initialize_folders(self, step=None, it_name=""):
        """Create this project's folder tree and return it as a dotted Box."""
        if step and not it_name:
            raise ValueError(
                "A suffix must be provided when a specific step is specified."
            )

        it_name = f"{it_name}_" if it_name else it_name

        project_folder = downloads_folder / self.project_name
        project_folder.mkdir(parents=True, exist_ok=True)

        folders = {
            "data_raw_folder": project_folder / "data_raw",
            "processed_data_folder": project_folder / "data",
            "sampling_folder": project_folder / "far_samples",
            "samples_folder": project_folder / "samples",
            "rmj_mw": project_folder / "rmj_mw",
            "plots_folder": project_folder / "plots",
            "rmj_bm": project_folder / f"{it_name}rmj_bm",
            "glm_model": project_folder / f"{it_name}far_glm",
            "icar_model": project_folder / f"{it_name}far_icar",
            "rf_model": project_folder / f"{it_name}far_rf",
        }

        if step:
            folder = folders.get(step)
            folder.mkdir(parents=True, exist_ok=True)

        else:
            for folder in folders.values():
                folder.mkdir(parents=True, exist_ok=True)

        folders.update(
            {
                "root_folder": root_folder,
                "downloads_folder": downloads_folder,
                "project_folder": project_folder,
            }
        )

        # Return a Box object for dot notation access
        return Box(folders)


# Rebuild the Project model now that the Variable classes can be imported, so the
# forward references in its field annotations resolve. The names shadow the
# module-level imports on purpose: this block runs after those classes are fully
# defined, which is what makes model_rebuild() succeed.
try:
    from spatialrisk.variables import (
        GEEVar,
        LocalRasterVar,  # noqa: F811
        LocalVectorVar,  # noqa: F811
        Variable,
    )

    # Rebuild Variable classes first to ensure they're fully defined
    Variable.model_rebuild()
    LocalVectorVar.model_rebuild()
    LocalRasterVar.model_rebuild()
    GEEVar.model_rebuild()

    # Then rebuild Project
    Project.model_rebuild()
except ImportError:
    pass  # Variables not yet available
