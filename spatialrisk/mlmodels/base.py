"""Base class for risk probability models.

Provides a generic Pydantic-based foundation for ML models that:
- Own a Dataset and Sample object; extract the training table at fit time
- Store dataset metadata, formula, parameters, and training date
- Generate raster predictions from a Dataset object
- Serialize to/from JSON for project persistence
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger("spatial_risk")


def _mask_values(mask_value) -> tuple:
    """Normalise apply()'s ``mask_value`` (scalar or list) to a tuple."""
    if isinstance(mask_value, (list, tuple)):
        return tuple(mask_value)
    return (mask_value,)


class BaseRiskModel(BaseModel):
    """Generic base class for risk probability ML models.

    Stores all model metadata in serializable fields. The actual trained
    ML object is held in a private attribute and persisted as a pickle file
    whose name includes a timestamp to prevent overwriting.

    Attributes:
    ----------
    name : str, optional
        Base name for the model, e.g. "calibration". Used in pickle filename.
    model_type : str
        Model type identifier: "glm", "rf", or "icar".
    project_name : str, optional
        Name of the parent project. Used for path reconstruction after load.
    dataset_name : str, optional
        Name of the dataset used for training, e.g. "calibration_2020".
    sample_name : str, optional
        Name of the Sample used for training.
    target_name : str, optional
        Name of the target variable, e.g. "forest_loss_2015_2020".
    feature_names : list of str
        Names of the feature variables used in training.
    year : int, optional
        Year associated with the training dataset.
    formula : str, optional
        Patsy formula string used for training.
    parameters : dict
        Model-specific hyperparameters (solver, n_trees, mcmc iterations, …).
    model_path : Path, optional
        Path to the saved pickle file.
    samples_path : Path, optional
        Path to the CSV file used for training (if samples came from a file).
    trained : bool
        Whether the model has been fitted.
    trained_at : str, optional
        ISO-format datetime string set when fit() completes.
    n_samples : int, optional
        Number of samples used during training.
    deviance : float, optional
        Model deviance (2 x log-loss x n_samples) from training.
    project : any
        Live Project reference. Excluded from serialization.
    dataset : any
        Live Dataset reference. Excluded from serialization.
    sample : any
        Live Sample reference. Excluded from serialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    name: Optional[str] = None
    model_type: str = ""

    # Dataset metadata (serializable)
    project_name: Optional[str] = None
    dataset_name: Optional[str] = None
    sample_name: Optional[str] = None
    target_name: Optional[str] = None
    feature_names: List[str] = Field(default_factory=list)
    year: Optional[int] = None

    # Formula and parameters
    formula: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # File paths
    model_path: Optional[Path] = None
    samples_path: Optional[Path] = None

    # Training results
    trained: bool = False
    trained_at: Optional[str] = None
    n_samples: Optional[int] = None
    deviance: Optional[float] = None

    # Live references — excluded from serialization
    project: Optional[Any] = Field(default=None, exclude=True, repr=False)
    dataset: Optional[Any] = Field(default=None, exclude=True, repr=False)
    sample: Optional[Any] = Field(default=None, exclude=True, repr=False)

    # In-memory ML objects — not serialized
    _ml_model: Any = PrivateAttr(default=None)
    _x_design_info: Any = PrivateAttr(default=None)
    # Small training sample used to reconstruct DesignInfo after loading
    _design_sample: Any = PrivateAttr(default=None)
    # Transient: a user-chosen name for the NEXT apply()'s prediction(s). Set by
    # the inference runner just before apply(); consumed by _register_prediction
    # to key/name the output(s). None → fall back to the provenance-derived key.
    _pending_pred_name: Optional[str] = PrivateAttr(default=None)
    # Transient: run-time choices for the NEXT apply() that are arguments to it
    # rather than model config (the ML mask layer). Same lifecycle as
    # _pending_pred_name; frozen onto the Prediction as ``run_params``.
    _pending_run_params: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_samples(
        self,
        formula: Optional[str] = None,
        output_csv: Optional[Union[str, Path]] = None,
    ):
        """Extract the training table from (dataset, sample) and resolve formula."""
        from spatialrisk.far_helpers import (
            generate_patsy_formula,
            inject_categorical_levels,
        )

        if self.dataset is None:
            raise ValueError("dataset must be set before calling fit().")
        if self.sample is None:
            raise ValueError("sample must be set before calling fit().")

        df = self.dataset.extract_at_points(self.sample.load_points())

        if output_csv is not None:
            from pathlib import Path

            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            self.samples_path = output_csv

        self.target_name = self.dataset.target.name
        self.dataset_name = getattr(self.dataset, "name", None) or self.dataset_name
        self.feature_names = [v.name for v in self.dataset.features]
        if self.dataset.year is not None:
            self.year = self.dataset.year

        resolved = formula or self.formula or generate_patsy_formula(self.dataset)
        # The GUI shows/edits bare C(x) terms; prediction re-parses the stored
        # string, so the categorical level domains must be re-armed here.
        resolved = inject_categorical_levels(resolved, self.dataset)
        self.formula = resolved
        return df, resolved

    def _resolve_dataset(self, dataset: Optional[Any]) -> Any:
        """Return dataset to use for apply(), validating feature compatibility."""
        active = dataset if dataset is not None else self.dataset
        if active is None:
            raise ValueError(
                "No dataset available. Pass dataset= or set model.dataset"
                " before calling apply()."
            )
        # Validate features when a different dataset is provided
        if dataset is not None and dataset is not self.dataset and self.feature_names:
            available = {v.name for v in dataset.features}
            missing = [f for f in self.feature_names if f not in available]
            if missing:
                raise ValueError(
                    f"Provided dataset is missing required feature(s): {missing}"
                )
        return active

    def _ensure_design_info(self) -> None:
        """Rebuild patsy design info from the training CSV when the pickle lost it.

        patsy's DesignInfo is not picklable, so a loaded model re-derives it
        from the formula and the saved samples. Raises when neither is there.
        """
        if self._x_design_info is not None:
            return
        if self.samples_path is not None and Path(self.samples_path).exists():
            import pandas as pd
            from patsy import dmatrices

            df = pd.read_csv(self.samples_path).dropna()
            _, x_ref = dmatrices(self.formula, df, NA_action="drop")
            self._x_design_info = x_ref.design_info
            return
        raise RuntimeError(
            "Cannot reconstruct design info: samples_path not set or "
            "file missing. Re-run fit() to regenerate samples."
        )

    def _stamp_now(self) -> str:
        """Return current datetime as ISO string and set trained_at."""
        self.trained_at = datetime.now().isoformat()
        return self.trained_at

    def _pickle_filename(self) -> str:
        """Build a date-stamped pickle filename to avoid overwriting."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.name or "model"
        return f"{self.model_type}_{base}_{ts}.pickle"

    def _default_folder(self) -> Optional[Path]:
        """Return the project model folder for this model type, if available."""
        if self.project is None:
            return None
        folders = self.project.folders
        key_map = {
            "glm": "glm_model",
            "rf": "rf_model",
            "icar": "icar_model",
        }
        folder_key = key_map.get(self.model_type)
        if folder_key and hasattr(folders, folder_key):
            return getattr(folders, folder_key)
        return None

    def _resolve_output_folder(
        self,
        folder: Optional[Union[str, Path]] = None,
        param: str = "folder",
    ) -> Path:
        """Resolve where this model writes: explicit folder, else project folder.

        This is the rule :meth:`save` has always applied, factored out so every
        site that writes a model artifact — training CSVs, pickles, rho rasters,
        rmj outputs — obeys it. With no folder and no project attached it raises
        rather than guessing.

        The guess it replaces was ``Path.cwd()``. Nothing in the GUI reached it
        (the tiles always set ``model.project`` and pass a folder), but a script
        or notebook did, and then artifacts landed in whatever directory the
        process happened to start in. On SEPAL that directory is the read-only
        shared module mount, so the guess does not merely misplace the file — it
        fails with ``Read-only file system``.

        ``param`` names the keyword in the error message, because not every
        caller spells it ``folder`` (``MWModel.apply`` takes ``output_folder``).
        """
        if folder is not None:
            return Path(folder)
        default = self._default_folder()
        if default is None:
            raise RuntimeError(
                "Cannot determine output folder: no project is attached. "
                f"Set model.project first or pass {param}= explicitly."
            )
        return Path(default)

    # ------------------------------------------------------------------
    # Fit / Apply (implemented in subclasses)
    # ------------------------------------------------------------------

    def fit(
        self,
        formula: Optional[str] = None,
        folder: Optional[Union[str, Path]] = None,
    ) -> "BaseRiskModel":
        """Train the model using the attached dataset and sampling.

        Parameters
        ----------
        formula : str, optional
            Patsy formula. If omitted, falls back to self.formula or
            auto-generates via generate_patsy_formula(self.dataset).
        folder : str or Path, optional
            Folder for saving the model pickle.

        Returns:
        --------
        self
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def apply(
        self,
        output_file: Union[str, Path],
        dataset: Optional[Any] = None,
        mask: Optional[Union[str, Path]] = None,
        mask_value: Union[int, float, list] = 0,
        *,
        workers: Optional[int] = None,
    ) -> Path:
        """Generate a probability raster from a Dataset object.

        Parameters
        ----------
        output_file : str or Path
            Path for the output GeoTIFF.
        dataset : Dataset, optional
            Dataset with target and features configured. If omitted, uses
            self.dataset. When provided, must contain all features in
            self.feature_names.
        mask : str or Path, optional
            Path to a mask raster. Pixels matching ``mask_value`` (or the
            raster's nodata) are set to nodata (0) in the output.
            If omitted, prediction runs over the full raster stack.
        mask_value : int, float, or list of int/float, optional
            Value(s) in the mask raster that identify pixels to suppress.
            Defaults to 0. Ignored when ``mask`` is None.
        workers : int, optional
            Stripe worker threads for
            :func:`spatialrisk.mlmodels.windowed_predict.predict_windowed`.
            None lets each predictor apply its own default (the resource
            policy, except RF's serial one); 1 runs on the calling thread.

        Returns:
        --------
        Path
            Path to the written GeoTIFF.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, folder: Optional[Union[str, Path]] = None) -> Path:
        """Save the trained ML object as a date-stamped pickle file.

        Parameters
        ----------
        folder : str or Path, optional
            Target folder. Falls back to the project model folder; raises when
            neither is available.

        Returns:
        --------
        Path
            Path to the written pickle file.
        """
        if self._ml_model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        out_dir = self._resolve_output_folder(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = self._pickle_filename()
        out_path = out_dir / filename

        payload = {
            "ml_model": self._ml_model,
            "design_sample": self._design_sample,
            "formula": self.formula,
            "samples_path": self.samples_path,
        }
        with open(out_path, "wb") as fh:
            pickle.dump(payload, fh)

        self.model_path = out_path
        print(f"  Model saved to: {out_path}")
        return out_path

    def load_model(self) -> None:
        """Load the ML object from self.model_path into memory."""
        if self.model_path is None:
            raise RuntimeError("model_path is not set. Train and save first.")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Pickle not found: {self.model_path}")
        with open(self.model_path, "rb") as fh:
            payload = pickle.load(fh)
        self._ml_model = payload["ml_model"]
        self._design_sample = payload.get("design_sample")
        if payload.get("formula") is not None:
            self.formula = payload["formula"]
        if payload.get("samples_path") is not None:
            self.samples_path = payload["samples_path"]

    # ------------------------------------------------------------------
    # Project registration
    # ------------------------------------------------------------------

    def register(
        self,
        project: Any,
        key: Optional[str] = None,
        auto_save: bool = True,
    ) -> None:
        """Register this model in the project's models dict.

        Parameters
        ----------
        project : Project
            Parent project instance.
        key : str, optional
            Storage key. Defaults to "{model_type}_{name}" or "{model_type}".
        auto_save : bool
            If True, calls project.save() after registering.
        """
        self.project = project
        self.project_name = project.project_name

        storage_key = key or (
            f"{self.model_type}_{self.name}" if self.name else self.model_type
        )
        project.models[storage_key] = self
        print(f"  Model registered as project.models['{storage_key}']")

        if auto_save:
            project.save()

    def _model_key(self) -> str:
        """Return this model's key in ``project.models``.

        Prefers an identity reverse-lookup (honors custom keys passed to
        ``register``/``add_model``); falls back to the default key formula.
        """
        if self.project is not None:
            for key, model in self.project.models.items():
                if model is self:
                    return key
        return f"{self.model_type}_{self.name}" if self.name else self.model_type

    def output_files(self) -> List[Path]:
        """On-disk artifacts this model owns (for cleanup when it is deleted).

        Base models persist a pickle (``model_path``) and an optional training
        sample CSV (``samples_path``). Model types that write extra rasters
        (iCAR's rho raster, MW's deforestation-rate maps) extend this list.
        """
        return [Path(p) for p in (self.model_path, self.samples_path) if p]

    @staticmethod
    def _ensure_display_overviews(path: Union[str, Path]) -> None:
        """Build the viewer's overview pyramid for a written output raster.

        Best-effort: a raster that cannot be optimised (read-only folder, a
        stubbed writer that produced nothing) still registers as a prediction,
        it just draws slower.
        """
        import spatialrisk.overviews as overviews

        if not Path(path).exists():
            return
        try:
            overviews.ensure_overviews(path, min_pixels=overviews.OVERVIEW_MIN_PIXELS)
        except Exception:
            logger.exception("Could not build overviews for %s", path)

    def _register_prediction(
        self,
        path: Union[str, Path],
        dataset: Optional[Any] = None,
        year: Optional[int] = None,
        window: Optional[int] = None,
        auto_save: bool = True,
        defrate_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Any]:
        """Build and register a Prediction for an output raster.

        No-ops (returns None) when the model has no project reference, so direct
        ``apply()`` calls outside a project context keep working unchanged.

        Parameters
        ----------
        path : str or Path
            The written output raster.
        dataset : Dataset, optional
            Dataset used for this prediction. Falls back to ``self.dataset``.
        year : int, optional
            Period of the prediction. Falls back to ``self.year``.
        window : int, optional
            Moving-window size discriminator (MW only).
        auto_save : bool
            Passed through to project registration.
        defrate_path : str or Path, optional
            Per-category deforestation-rate table written alongside this output
            (MW/JNR). Consumed by the allocation tool.
        """
        # A written prediction is above all a display artifact: the viewer
        # draws it from 256 px tiles, and without a pyramid every zoomed-out
        # tile decodes the whole raster (see spatialrisk.overviews). Done
        # before the project guard so a direct apply() gets it too.
        self._ensure_display_overviews(path)

        if self.project is None:
            return None

        from spatialrisk.predictions.prediction import (
            Prediction,
            build_dataset_snapshot,
        )

        ds = dataset if dataset is not None else self.dataset
        # A pending name (set by the inference runner) makes the prediction's key
        # and label user-chosen instead of provenance-derived, so distinct runs no
        # longer overwrite each other. Multi-output runs (MW windows) stay distinct
        # via the window suffix; model_key/dataset_name fields are kept intact so
        # evaluation labelling (which reads those fields) is unaffected.
        pending_name = self._pending_pred_name
        prediction = Prediction(
            name=pending_name,
            path=Path(path),
            model_key=self._model_key(),
            dataset_name=(getattr(ds, "name", None) or self.dataset_name or "unknown"),
            year=year if year is not None else self.year,
            window=window,
            model_snapshot=self.model_dump(mode="json"),
            dataset_snapshot=build_dataset_snapshot(ds),
            run_params=dict(self._pending_run_params or {}),
            defrate_path=Path(defrate_path) if defrate_path else None,
        )
        key = None
        if pending_name:
            key = pending_name + (f"_w{window}" if window is not None else "")
        prediction.add_to_project(self.project, key=key, auto_save=auto_save)
        return prediction

    # ------------------------------------------------------------------
    # Serialization override
    # ------------------------------------------------------------------

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Exclude live references (project, dataset, sample) from serialization."""
        kwargs.setdefault("exclude", set())
        if isinstance(kwargs["exclude"], set):
            kwargs["exclude"] = kwargs["exclude"] | {"project", "dataset", "sample"}
        return super().model_dump(**kwargs)
