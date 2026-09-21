"""Sample — a persisted, location-only set of sample points.

Generated from a raster variable (grid/CRS/domain; values are strata for
stratified sampling) and a mask variable (restricts valid pixels). Stores only
point locations — feature extraction happens at training time against a chosen
Dataset (see Dataset.extract_at_points). Solara-free; geo deps imported lazily.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("spatial_risk")


class Sample(BaseModel):
    """A persisted, location-only set of sample points drawn from a raster variable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    project: Any = Field(default=None, repr=False, exclude=True)

    raster_var_name: str
    mask_var_name: Optional[str] = None
    strategy: str = "random"
    n_samples: Optional[int] = 10000
    allocation: Optional[str] = None
    adapt: bool = False
    seed: Optional[int] = None
    spacing_m: Optional[float] = None

    points_path: Optional[Path] = None
    pmtiles_path: Optional[Path] = None
    crs: Optional[str] = None

    n_total: int = 0
    class_counts: Dict[str, int] = Field(default_factory=dict)
    created_at: Optional[str] = None

    def model_dump(self, **kwargs):
        """Serialize the sample, always excluding the back-reference to the project."""
        exclude = kwargs.get("exclude")
        if exclude is None:
            kwargs["exclude"] = {"project"}
        elif isinstance(exclude, set):
            kwargs["exclude"] = exclude | {"project"}
        return super().model_dump(**kwargs)

    def _resolve_path(self, var_name: Optional[str]):
        if var_name is None:
            return None
        if self.project is None:
            raise ValueError("Sample has no project reference.")
        var = self.project.get_variable(var_name)
        if var is None or getattr(var, "path", None) is None:
            raise ValueError(f"Variable '{var_name}' not found or has no path.")
        return var.path

    def generate(self) -> "Sample":
        """Draw the points, write them to ``points_path``, and build the PMTiles.

        :meth:`draw_points` then :meth:`build_tiles`; callers that want to
        report the two phases separately (the sampling tile does, since tiling
        can take longer than sampling) call them one after the other.
        """
        self.draw_points()
        self.build_tiles()
        return self

    def draw_points(self) -> "Sample":
        """Draw the sample and write it to ``points_path``; no tiles yet.

        After this the sample is complete as data (``n_total``,
        ``class_counts``, ``created_at`` and the GPKG on disk); only the map
        archive is missing, so ``pmtiles_path`` stays as it was.
        """
        from spatialrisk.sampling import generate_points

        raster_path = self._resolve_path(self.raster_var_name)
        mask_path = self._resolve_path(self.mask_var_name)

        gdf = generate_points(
            raster_path,
            mask_path,
            strategy=self.strategy,
            n_samples=self.n_samples,
            allocation=self.allocation,
            seed=self.seed,
            adapt=self.adapt,
            spacing_m=self.spacing_m,
        )

        if self.points_path is not None:
            Path(self.points_path).parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(self.points_path, driver="GPKG")

        self.crs = str(gdf.crs) if gdf.crs is not None else None
        self.n_total = int(len(gdf))
        self.class_counts = {
            str(int(k)): int(v) for k, v in gdf["strata"].value_counts().items()
        }
        self.created_at = datetime.now().isoformat(timespec="seconds")
        return self

    def build_tiles(self) -> "Sample":
        """Build the PMTiles archive for the points on disk (non-fatal on error).

        A failed or unavailable conversion leaves ``pmtiles_path`` None and
        the sample renders through the GeoJSON fallback.
        """
        if self.points_path is not None:
            try:
                # force: a re-run over the same paths must rebuild the archive,
                # never keep tiles from the previous point set.
                self.ensure_pmtiles(force=True)
            except Exception:
                logger.exception(
                    "PMTiles conversion failed for sample '%s'; GeoJSON fallback",
                    self.name,
                )
                self.pmtiles_path = None
        return self

    def ensure_pmtiles(self, *, force: bool = False) -> bool:
        """Make ``pmtiles_path`` point at an existing archive, converting if needed.

        Backfills samples generated without tippecanoe (old projects, or a
        deploy environment that lacked the binary) and heals a manifest path
        whose file is gone. Returns True when a usable archive exists after the
        call. Without tippecanoe or points it returns False and leaves the
        sample on the GeoJSON fallback; conversion errors propagate to the
        caller. ``force`` rebuilds even when the archive already exists.
        """
        if (
            not force
            and self.pmtiles_path is not None
            and Path(self.pmtiles_path).exists()
        ):
            return True
        if self.points_path is None or not Path(self.points_path).exists():
            return False

        from spatialrisk import pmtiles_convert

        if not pmtiles_convert.tippecanoe_available():
            logger.warning(
                "tippecanoe unavailable; sample '%s' will render via "
                "GeoJSON fallback",
                self.name,
            )
            return False
        pm = Path(self.points_path).with_suffix(".pmtiles")
        pmtiles_convert.gpkg_to_pmtiles(self.points_path, pm)
        self.pmtiles_path = pm
        return True

    def load_points(self):
        """Read the generated points back as a GeoDataFrame."""
        import geopandas as gpd

        if self.points_path is None or not Path(self.points_path).exists():
            raise FileNotFoundError(
                f"Sample points not found for '{self.name}': {self.points_path}"
            )
        return gpd.read_file(self.points_path)

    def register(
        self, project: Any, key: Optional[str] = None, auto_save: bool = True
    ) -> None:
        """Attach this sample to ``project`` under ``key`` (defaults to its name)."""
        project.add_sample(self, key=key, auto_save=auto_save)
