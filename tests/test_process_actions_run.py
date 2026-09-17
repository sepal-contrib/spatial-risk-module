"""Process-step actions: base raster, harmonization run, post-processing."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gui.scripts import process_actions


class _Proj:
    def __init__(self):
        self.raw_variables = {}
        self.processed_variables = {}
        self.base_raster = None
        self.saved = False
        self.reproject_keys = None
        self.rasterize_keys = None

    def reproject_and_match_all(self, source="raw", keys=None):
        self.reprojected = source
        self.reproject_keys = list(keys) if keys is not None else None

    def rasterize_all(self, source="raw", keys=None):
        self.rasterized = source
        self.rasterize_keys = list(keys) if keys is not None else None

    def save(self):
        self.saved = True


def test_set_base_raster_reprojects_and_sets():
    """Reprojects the chosen raw raster and registers it as the base."""
    p = _Proj()
    raw = MagicMock(name="rawbase")
    reprojected = MagicMock(name="reprojected")
    raw.reproject.return_value = reprojected
    p.raw_variables["subj"] = raw

    out = process_actions.set_base_raster(p, "subj", "EPSG:5490", 30.0)

    raw.reproject.assert_called_once_with(target_epsg="EPSG:5490", resolution=30.0)
    reprojected.use_as_base_raster.assert_called_once()
    assert out is reprojected


def test_run_processing_sequences_steps():
    """Materialize -> reproject -> rasterize -> save, when a layer is pending."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=["subj"], current=[])
    with patch("gui.scripts.process_actions.materialize_raw_layers") as mat, patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        process_actions.run_processing(p)

    mat.assert_called_once_with(p)
    assert p.reprojected == "raw"
    assert p.rasterized == "raw"
    assert p.saved is True


def test_apply_post_processing_adds_processed():
    """The edge/dist output is registered as a processed variable."""
    p = _Proj()
    var = MagicMock(name="processed")
    derived = MagicMock(name="derived")
    var.apply_post_processing.return_value = derived
    p.processed_variables["rivers"] = var

    out = process_actions.apply_post_processing(p, "rivers", "dist")

    var.apply_post_processing.assert_called_once_with("dist")
    derived.add_as_processed.assert_called_once()
    assert out is derived


def test_run_processing_raises_without_base_raster():
    """No base raster means no grid to harmonize onto."""
    import pytest

    p = _Proj()  # base_raster is None by default
    with pytest.raises(ValueError, match="base raster"):
        process_actions.run_processing(p)


def test_run_processing_logs_reproject_and_rasterize(caplog):
    """Reproject/rasterize/complete are logged when a layer is pending."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=["subj"], current=[])
    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        with patch("gui.scripts.process_actions.materialize_raw_layers"), patch(
            "gui.scripts.process_actions.harmonization_status_from_disk",
            return_value=status,
        ):
            process_actions.run_processing(p)
    text = caplog.text.lower()
    assert "reproject" in text
    assert "rasteriz" in text
    assert "complete" in text


def test_apply_post_processing_logs_step(caplog):
    """The step name and variable key land in the log for the task tracker."""
    p = _Proj()
    var = MagicMock(name="processed")
    derived = MagicMock(name="derived")
    var.apply_post_processing.return_value = derived
    p.processed_variables["rivers"] = var
    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        process_actions.apply_post_processing(p, "rivers", "dist")
    assert "dist" in caplog.text.lower()
    assert "rivers" in caplog.text.lower()


def test_run_processing_only_passes_pending_keys():
    """Already-harmonized layers are not re-derived."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=["new_layer"], current=["old_a", "old_b"])
    with patch("gui.scripts.process_actions.materialize_raw_layers"), patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        out = process_actions.run_processing(p)

    assert p.reproject_keys == ["new_layer"]
    assert p.rasterize_keys == ["new_layer"]
    assert out == {"processed": ["new_layer"], "skipped": ["old_a", "old_b"]}


def test_run_processing_skips_the_bulk_calls_when_nothing_is_pending():
    """Nothing pending means no reproject/rasterize work at all."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=[], current=["old_a"])
    with patch(
        "gui.scripts.process_actions.materialize_raw_layers", return_value=[]
    ), patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        out = process_actions.run_processing(p)

    assert p.reproject_keys is None
    assert p.rasterize_keys is None
    assert out == {"processed": [], "skipped": ["old_a"]}


def test_run_processing_saves_even_when_nothing_was_pending_or_downloaded():
    """The early return must still persist in-memory-only edits.

    The Variables tile mutates ``raw_variables`` in memory and never saves, so
    before Step 3 became incremental the unconditional save at the end of every
    run was what persisted an add / edit / remove. Skipping the save here loses
    a source-variable removal on the next project load.
    """
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=[], current=["old_a"])
    with patch(
        "gui.scripts.process_actions.materialize_raw_layers", return_value=[]
    ), patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        process_actions.run_processing(p)

    assert p.saved is True


def test_run_processing_saves_materialization_even_with_nothing_pending():
    """A skipped download still replaced a GEEVar in raw_variables.

    GEEVar.to_local_raster skips the download when the file already exists, so
    the replacement can read as current. Returning without saving would lose
    the GEEVar -> LocalRasterVar swap on the next project load.
    """
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=[], current=["altitude"])
    with patch(
        "gui.scripts.process_actions.materialize_raw_layers",
        return_value=["altitude"],
    ), patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        process_actions.run_processing(p)

    assert p.saved is True
    assert p.reproject_keys is None


def test_run_processing_status_is_computed_after_downloading():
    """A GEEVar has no local file until it is materialized."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    calls = []
    with patch(
        "gui.scripts.process_actions.materialize_raw_layers",
        side_effect=lambda proj, **kw: calls.append("materialize"),
    ), patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        side_effect=lambda proj: (
            calls.append("status"),
            SimpleNamespace(pending=["x"], current=[]),
        )[1],
    ):
        process_actions.run_processing(p)

    assert calls == ["materialize", "status"]


def test_run_processing_logs_both_counts(caplog):
    """The user is told how much work was skipped."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=["new_layer"], current=["old_a", "old_b"])
    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        with patch("gui.scripts.process_actions.materialize_raw_layers"), patch(
            "gui.scripts.process_actions.harmonization_status_from_disk",
            return_value=status,
        ):
            process_actions.run_processing(p)
    # The exact rendered message, not "a 1 and a 2 appear somewhere": caplog.text
    # embeds module paths and line numbers, so a loose substring check drifts
    # towards vacuous as the file grows.
    assert "Harmonizing 1 layer(s); 2 already aligned." in caplog.text


def test_run_processing_keys_restricts_download_and_harmonization():
    """``keys=`` narrows the run to those raw keys (per-row harmonize button)."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=["a", "b"], current=["c"])
    with patch("gui.scripts.process_actions.materialize_raw_layers") as mat, patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        out = process_actions.run_processing(p, keys=["b", "c"])

    mat.assert_called_once_with(p, ["b", "c"])
    assert p.reproject_keys == ["b"]
    assert p.rasterize_keys == ["b"]
    assert out["processed"] == ["b"]
    assert sorted(out["skipped"]) == ["a", "c"]
    assert p.saved is True


def test_run_processing_keys_with_nothing_pending_saves_and_skips():
    """A key that is already current is a no-op run, but still saves."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=["a"], current=["c"])
    with patch("gui.scripts.process_actions.materialize_raw_layers"), patch(
        "gui.scripts.process_actions.harmonization_status_from_disk",
        return_value=status,
    ):
        out = process_actions.run_processing(p, keys=["c"])

    assert p.reproject_keys is None  # never called
    assert out["processed"] == []
    assert sorted(out["skipped"]) == ["a", "c"]
    assert p.saved is True
