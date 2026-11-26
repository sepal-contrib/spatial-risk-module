"""materialize_raw_layers: GEE layers become local rasters in raw_variables."""

from unittest.mock import MagicMock

from gui.scripts import process_actions

# Needed for GEEVar.model_rebuild() (forward reference), not used directly.
from spatialrisk.project import Project  # noqa: F401
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.models import DataType, RasterType

# GEEVar has a forward reference to Project; rebuild so instantiation works.
GEEVar.model_rebuild()


class _Proj:
    def __init__(self):
        self.raw_variables = {}


def test_materialize_replaces_raster_geevar(monkeypatch):
    """A raster GEEVar is replaced by the local var it downloads to."""
    p = _Proj()
    gv = GEEVar(
        name="altitude",
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=["img"],
    )
    p.raw_variables["altitude"] = gv

    local = MagicMock(name="LocalRasterVar")

    def fake_to_local_raster(*a, **k):
        # mimic add_as_raw registering the local var under the same key
        p.raw_variables["altitude"] = local
        return local

    monkeypatch.setattr(GEEVar, "to_local_raster", fake_to_local_raster, raising=True)

    done = process_actions.materialize_raw_layers(p)

    assert "altitude" in done
    assert p.raw_variables["altitude"] is local
    local.add_as_raw.assert_called_once_with(auto_save=False)


def test_materialize_skips_non_geevar():
    """Already-local variables are left alone."""
    p = _Proj()
    p.raw_variables["x"] = MagicMock(name="LocalRasterVar")  # already local
    assert process_actions.materialize_raw_layers(p) == []


def test_materialize_keys_filter_downloads_only_selected(monkeypatch):
    """``keys`` limits the download to those layers."""
    p = _Proj()
    for name in ("altitude", "slope"):
        p.raw_variables[name] = GEEVar(
            name=name,
            data_type=DataType.raster,
            raster_type=RasterType.continuous,
            gee_images=["img"],
        )

    downloaded = []

    def fake_to_local_raster(self, *a, **k):
        local = MagicMock(name=f"Local_{self.name}")
        p.raw_variables[self.name] = local
        downloaded.append(self.name)
        return local

    monkeypatch.setattr(GEEVar, "to_local_raster", fake_to_local_raster, raising=True)

    done = process_actions.materialize_raw_layers(p, keys=["slope"])

    assert done == ["slope"]
    assert downloaded == ["slope"]
    # the unselected variable is untouched
    assert isinstance(p.raw_variables["altitude"], GEEVar)


def test_materialize_keys_filter_ignores_local_and_unknown_keys():
    """Local or unknown keys in ``keys`` download nothing."""
    p = _Proj()
    p.raw_variables["x"] = MagicMock(name="LocalRasterVar")
    assert process_actions.materialize_raw_layers(p, keys=["x", "missing"]) == []


def test_materialize_logs_download_milestones(monkeypatch, caplog):
    """Start and end of a download are logged."""
    import logging

    p = _Proj()
    gv = GEEVar(
        name="altitude",
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=["img"],
    )
    p.raw_variables["altitude"] = gv

    local = MagicMock(name="LocalRasterVar")

    def fake_to_local_raster(*a, **k):
        p.raw_variables["altitude"] = local
        return local

    monkeypatch.setattr(GEEVar, "to_local_raster", fake_to_local_raster, raising=True)

    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        process_actions.materialize_raw_layers(p)

    text = caplog.text.lower()
    assert "downloading" in text
    assert "downloaded" in text


def test_one_failing_layer_does_not_skip_the_rest(monkeypatch):
    """Bulk download: the layer after a failure still downloads; the error is raised."""
    import pytest

    p = _Proj()
    for name in ("a_bad", "b_good"):
        p.raw_variables[name] = GEEVar(
            name=name,
            data_type=DataType.raster,
            raster_type=RasterType.continuous,
            gee_images=["img"],
        )
    local = MagicMock(name="LocalRasterVar")

    def fake_to_local_raster(self, *a, **k):
        if self.name == "a_bad":
            raise ValueError("outside the region")
        p.raw_variables[self.name] = local
        return local

    monkeypatch.setattr(GEEVar, "to_local_raster", fake_to_local_raster, raising=True)

    with pytest.raises(ValueError, match="outside the region"):
        process_actions.materialize_raw_layers(p)
    assert p.raw_variables["b_good"] is local


def test_several_failures_are_reported_together(monkeypatch):
    """Two failing layers: one error naming both."""
    import pytest

    p = _Proj()
    for name in ("a_bad", "b_bad"):
        p.raw_variables[name] = GEEVar(
            name=name,
            data_type=DataType.raster,
            raster_type=RasterType.continuous,
            gee_images=["img"],
        )

    def fake_to_local_raster(self, *a, **k):
        raise ValueError(f"{self.name} failed")

    monkeypatch.setattr(GEEVar, "to_local_raster", fake_to_local_raster, raising=True)

    with pytest.raises(RuntimeError) as err:
        process_actions.materialize_raw_layers(p)
    assert "a_bad" in str(err.value) and "b_bad" in str(err.value)
