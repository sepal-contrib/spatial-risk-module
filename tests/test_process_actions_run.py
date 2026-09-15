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
        "gui.scripts.process_actions.harmonization_status", return_value=status
    ):
        process_actions.run_processing(p)

    mat.assert_called_once_with(p)
    assert p.reprojected == "raw"
    assert p.rasterized == "raw"
    assert p.saved is True


def test_apply_post_processing_adds_processed():
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
            "gui.scripts.process_actions.harmonization_status", return_value=status
        ):
            process_actions.run_processing(p)
    text = caplog.text.lower()
    assert "reproject" in text
    assert "rasteriz" in text
    assert "complete" in text


def test_apply_post_processing_logs_step(caplog):
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
        "gui.scripts.process_actions.harmonization_status", return_value=status
    ):
        out = process_actions.run_processing(p)

    assert p.reproject_keys == ["new_layer"]
    assert p.rasterize_keys == ["new_layer"]
    assert out == {"processed": ["new_layer"], "skipped": ["old_a", "old_b"]}


def test_run_processing_skips_everything_when_nothing_is_pending():
    """Nothing pending and nothing downloaded means no bulk call and no save."""
    p = _Proj()
    p.base_raster = MagicMock(name="base")
    status = SimpleNamespace(pending=[], current=["old_a"])
    with patch(
        "gui.scripts.process_actions.materialize_raw_layers", return_value=[]
    ), patch("gui.scripts.process_actions.harmonization_status", return_value=status):
        out = process_actions.run_processing(p)

    assert p.reproject_keys is None
    assert p.rasterize_keys is None
    assert p.saved is False
    assert out == {"processed": [], "skipped": ["old_a"]}


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
    ), patch("gui.scripts.process_actions.harmonization_status", return_value=status):
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
        "gui.scripts.process_actions.harmonization_status",
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
            "gui.scripts.process_actions.harmonization_status", return_value=status
        ):
            process_actions.run_processing(p)
    assert "1" in caplog.text and "2" in caplog.text
    assert "already" in caplog.text.lower()
