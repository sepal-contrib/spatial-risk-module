"""MapBiomas footprint check: coverage maths, the check itself, validator helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mapbiomas_fake_ee import FakeEE, Node

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.predefined import PREDEFINED_CATALOGUE, mapbiomas
from gui.scripts.predefined.download_hooks import prepare_predefined_download
from gui.scripts.predefined.mapbiomas_registry import ROOT, region_param
from spatialrisk.variables.models import DataType

LULC_BRAZIL = (
    ROOT + "brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3"
)


@pytest.fixture
def fake_ee(monkeypatch):
    """Fake ee."""
    fake = FakeEE()
    monkeypatch.setattr(mapbiomas, "ee", fake)
    return fake


def test_region_mask_is_the_lulc_band0_footprint(fake_ee):
    """Region mask is the lulc band0 footprint."""
    mapbiomas._region_mask("brazil")
    assert fake_ee.calls == [
        ("Image", (LULC_BRAZIL,), {}),
        ("select", (0,), {}),
        ("mask", (), {}),
        ("gt", (0,), {}),
    ]


def test_region_coverage_reduces_the_mask_mean_over_the_aoi_geometry(fake_ee):
    """Region coverage reduces the mask mean over the aoi geometry."""
    fake_ee.info = 0.9956
    aoi = fake_ee.Geometry()
    assert mapbiomas.region_coverage(aoi, "brazil") == pytest.approx(0.9956)
    [kwargs] = fake_ee.kwargs_of("reduceRegion")
    assert kwargs["geometry"] is aoi
    assert kwargs["scale"] == 100
    assert kwargs["bestEffort"] is True
    assert kwargs["maxPixels"] == 1e7
    assert kwargs["tileScale"] == 2
    assert "Reducer.mean" in fake_ee.names()
    assert fake_ee.names().count("getInfo") == 1


def test_region_coverage_counts_masked_pixels_as_outside(fake_ee):
    """Review focus 1: pixels off the asset must lower the mean, not vanish from it."""
    mapbiomas.region_coverage(fake_ee.Geometry(), "brazil")
    names = fake_ee.names()
    assert ("unmask", (0,), {}) in fake_ee.calls
    assert names.index("unmask") < names.index("reduceRegion")


def test_region_coverage_uses_the_geometry_of_a_feature_collection(fake_ee):
    """Region coverage uses the geometry of a feature collection."""
    fake_ee.info = 1.0
    mapbiomas.region_coverage(Node(fake_ee, "FeatureCollection"), "peru")
    [kwargs] = fake_ee.kwargs_of("reduceRegion")
    assert kwargs["geometry"].origin == "geometry"


def test_region_coverage_none_is_zero(fake_ee):
    """An AOI the region never touches reduces to None: that is 0 % inside."""
    fake_ee.info = None
    assert mapbiomas.region_coverage(fake_ee.Geometry(), "brazil") == 0.0


@pytest.mark.parametrize(
    "coverage, pct", [(0.9, 90), (0.29, 29), (0.9499, 94), (0.0, 0)]
)
def test_footprint_check_fails_below_the_threshold(monkeypatch, coverage, pct):
    """Footprint check fails below the threshold."""
    seen = []
    monkeypatch.setattr(
        mapbiomas,
        "region_coverage",
        lambda aoi, region: seen.append((aoi, region)) or coverage,
    )
    assert mapbiomas._footprint_check("AOI", {"region": "peru"}, 2020) == (
        "vars.mapbiomas.error_outside_region",
        {"region": t("vars.mapbiomas.region_peru"), "pct": pct, "min": 95},
    )
    assert seen == [("AOI", "peru")]


@pytest.mark.parametrize("coverage", [0.95, 0.9956, 1.0])
def test_footprint_check_passes_at_or_above_the_threshold(monkeypatch, coverage):
    """Footprint check passes at or above the threshold."""
    monkeypatch.setattr(mapbiomas, "region_coverage", lambda aoi, region: coverage)
    assert mapbiomas._footprint_check("AOI", {"region": "brazil"}, 2020) is None


def test_footprint_check_without_aoi_makes_no_ee_call(monkeypatch):
    """Footprint check without aoi makes no ee call."""
    monkeypatch.setattr(
        mapbiomas,
        "region_coverage",
        lambda *a: pytest.fail("no EE call without an AOI"),
    )
    assert mapbiomas._footprint_check(None, {"region": "brazil"}, 2020) is None


def test_check_year():
    """Check year."""
    years = list(range(1985, 2024))
    assert mapbiomas._check_year("amazon", 2023, years) is None
    assert mapbiomas._check_year("amazon", None, years) is None
    assert mapbiomas._check_year("amazon", 2025, years) == (
        "vars.mapbiomas.error_year",
        {"region": t("vars.mapbiomas.region_amazon"), "first": 1985, "last": 2023},
    )


def test_check_period_against_the_monitor_months():
    """Check period against the monitor months."""
    brazil = t("vars.mapbiomas.region_brazil")
    period_error = (
        "vars.mapbiomas.error_period",
        {"region": brazil, "first": "2019-01", "last": "2026-10"},
    )
    assert mapbiomas._check_period("brazil", "fire_monitor", 2026, "oct") is None
    assert (
        mapbiomas._check_period("brazil", "fire_monitor", 2026, "nov") == period_error
    )
    assert (
        mapbiomas._check_period("brazil", "fire_monitor", 2026, "year") == period_error
    )  # partial year
    assert mapbiomas._check_period("brazil", "fire_monitor", 2025, "year") is None
    assert (
        mapbiomas._check_period("brazil", "fire_monitor", 2018, "dec") == period_error
    )
    assert mapbiomas._check_period("brazil", "fire_monitor", None, "aug") is None
    assert mapbiomas._check_period("peru", "fire_monitor", 2025, "aug") == (
        "vars.mapbiomas.error_not_available",
        {"region": t("vars.mapbiomas.region_peru")},
    )


# --- prepare_download hook on the download worker (Task 5) --------------------


def _outside(monkeypatch, coverage):
    calls = []
    monkeypatch.setattr(
        mapbiomas,
        "region_coverage",
        lambda aoi, region: calls.append(region) or coverage,
    )
    return calls


def test_hook_raises_the_localized_footprint_error(monkeypatch):
    """Hook raises the localized footprint error."""
    _outside(monkeypatch, 0.12)
    var = SimpleNamespace(name="mapbiomas_lulc_peru", aoi="AOI", year=2020)
    with pytest.raises(ValueError) as err:
        mapbiomas._prepare_mapbiomas_download(var, {"region": "peru"})
    assert str(err.value) == t(
        "vars.mapbiomas.error_outside_region",
        region=t("vars.mapbiomas.region_peru"),
        pct=12,
        min=95,
    )


def test_hook_returns_no_notice_inside_the_region(monkeypatch):
    """Hook returns no notice inside the region."""
    _outside(monkeypatch, 1.0)
    var = SimpleNamespace(name="mapbiomas_lulc_peru", aoi="AOI", year=2020)
    assert mapbiomas._prepare_mapbiomas_download(var, {"region": "peru"}) is None


def _probe_entry():
    return {
        "label_key": "vars.predefined.mapbiomas_lulc",
        "description_key": "vars.predefined_info.mapbiomas_lulc",
        "group": "mapbiomas",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        "years": [2020],
        "get_image": lambda aoi, year, **kw: None,
        "params": [region_param("lulc")],
        "prepare_download": mapbiomas._prepare_mapbiomas_download,
    }


def test_b2_dispatcher_runs_the_hook_with_the_region_from_the_name(
    monkeypatch, tmp_path
):
    """B2 dispatcher runs the hook with the region from the name."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, "mbprobe", _probe_entry())
    calls = _outside(monkeypatch, 0.5)
    var = SimpleNamespace(
        name="mbprobe_peru",
        aoi="AOI",
        year=2020,
        gee_images=["img"],
        expected_local_path=tmp_path / "mbprobe_peru_2020.tif",
    )
    with pytest.raises(ValueError):
        prepare_predefined_download(var)
    assert calls == ["peru"]


def test_b2_dispatcher_skips_the_check_for_a_reused_file(monkeypatch, tmp_path):
    """B2 dispatcher skips the check for a reused file."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, "mbprobe", _probe_entry())
    monkeypatch.setattr(
        mapbiomas,
        "region_coverage",
        lambda *a: pytest.fail("a reused file is not re-checked"),
    )
    existing = tmp_path / "mbprobe_peru_2020.tif"
    existing.write_bytes(b"")
    var = SimpleNamespace(
        name="mbprobe_peru",
        aoi="AOI",
        year=2020,
        gee_images=["img"],
        expected_local_path=existing,
    )
    assert prepare_predefined_download(var) is None


class GEEVar:
    """Stand-in recognised by ``materialize_raw_layers`` (it matches the class name).

    The real GEEVar validates ``aoi`` as an ee object; this one accepts "AOI".
    """

    def __init__(self, name, tmp_path):
        """Build a raster GEEVar stand-in whose file is not on disk yet."""
        self.name = name
        self.aoi = "AOI"
        self.year = 2020
        self.gee_images = ["img"]
        self.data_type = DataType.raster
        self.expected_local_path = tmp_path / f"{name}_2020.tif"
        self.exported = False

    def to_local_raster(self, overwrite=False):
        """Record the export instead of downloading."""
        self.exported = True
        return MagicMock(name="LocalRasterVar")


class _Proj:
    def __init__(self, var):
        self.raw_variables = {var.name: var}


def test_materialize_stops_before_the_export_when_the_aoi_is_outside(
    monkeypatch, tmp_path
):
    """Materialize stops before the export when the aoi is outside."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, "mbprobe", _probe_entry())
    _outside(monkeypatch, 0.3)
    var = GEEVar("mbprobe_peru", tmp_path)
    with pytest.raises(ValueError) as err:
        process_actions.materialize_raw_layers(_Proj(var))
    assert "30" in str(err.value)
    assert var.exported is False


def test_materialize_exports_when_the_aoi_is_inside(monkeypatch, tmp_path):
    """Materialize exports when the aoi is inside."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, "mbprobe", _probe_entry())
    _outside(monkeypatch, 0.999)
    var = GEEVar("mbprobe_peru", tmp_path)
    assert process_actions.materialize_raw_layers(_Proj(var)) == ["mbprobe_peru"]
    assert var.exported is True
