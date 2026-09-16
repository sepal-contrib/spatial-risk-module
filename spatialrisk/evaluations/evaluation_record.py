"""Persisted record of one Step 8 evaluation run (one truth vs N maps).

A plain pydantic model with NO live ``project`` back-reference, so it serializes
without recursion and needs no ``_relink_backrefs`` handling. The full output
table is stored inline in ``indices`` so the GUI popup reads it straight from the
record — robust across save/load even if a later same-truth run overwrites the
on-disk ``indices_all.csv``.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationPlotArtifact(BaseModel):
    """One run's saved predicted-vs-observed pair for a single map + cell size.

    Records WHERE a run's per-cell data landed, so an interactive chart can be
    rebuilt from the exact files that run produced rather than from whatever
    currently sits at the shared ``evaluation/<truth_tag>/`` path (which a later
    run against the same truth overwrites). Paths are absolute strings, one
    artifact per prediction per cell size.
    """

    prediction_key: str
    model: str
    period: str
    csize_px: int
    points_csv: str
    png_path: str
    # Run-scoped per-category rate table this map was scored with. None for
    # records saved before the field existed; consumers derive the legacy path.
    defrate_csv: Optional[str] = None


class EvaluationRecord(BaseModel):
    """Persisted evaluation run record with truth, predictions, and artifacts."""

    name: Optional[str] = None
    truth_tag: str
    truth_defor: str
    truth_forest: str
    time_interval: int
    prediction_keys: List[str] = Field(default_factory=list)
    csizes: List[int] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)  # shown index columns; [] = all
    created_at: str
    indices: List[Dict[str, Any]] = Field(default_factory=list)
    csv_path: Optional[str] = None
    run_id: str
    # Run-scoped artifact paths. Empty is the legacy-compatible default: records
    # saved before run-scoping have no such files and fall back to deriving the
    # PNG path from ``Path(csv_path).parent``.
    artifacts: List[EvaluationPlotArtifact] = Field(default_factory=list)

    def storage_key(self) -> str:
        """Deterministic, history-safe registry key: truth + timestamp + run id."""
        compact = self.created_at.replace("-", "").replace(":", "").replace("T", "")
        return f"{self.truth_tag}__{compact}_{self.run_id}"
