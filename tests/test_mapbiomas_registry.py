"""MapBiomas registry: verified asset facts, the LULC class table and lookups.

The registry is pure data (no ``ee``), so these tests run without Earth Engine.
The VERIFIED table pins the facts probed read-only on 2026-10-09 and re-verified
on 2026-11-24; when MapBiomas publishes a new collection, update the registry
record and this table together.
"""

import ast
import re
from pathlib import Path

import pytest

from gui.scripts.predefined import mapbiomas_registry as reg

R = reg.ROOT

VERIFIED = {
    # (region, product): (asset path under ROOT, band, (first, last) years)
    ("brazil", "lulc"): (
        "brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3",
        "classification_{year}",
        (1985, 2025),
    ),
    ("amazon", "lulc"): (
        "amazon/lulc/collection6/mapbiomas_collection60_integration_v1",
        "classification_{year}",
        (1985, 2023),
    ),
    ("bolivia", "lulc"): (
        "bolivia/lulc/collection3/mapbiomas_bolivia_collection3_integration_v1",
        "classification_{year}",
        (1985, 2024),
    ),
    ("colombia", "lulc"): (
        "colombia/collection2/mapbiomas_colombia_collection2_integration_v1",
        "classification_{year}",
        (1985, 2023),
    ),
    ("ecuador", "lulc"): (
        "ecuador/lulc/collection4/mapbiomas_ecuador_collection4_coverage_v1",
        "classification_{year}",
        (1985, 2025),
    ),
    ("paraguay", "lulc"): (
        "paraguay/lulc/collection3/mapbiomas_paraguay_collection3_coverage_v1",
        "classification_{year}",
        (1985, 2025),
    ),
    ("peru", "lulc"): (
        "peru/collection3/mapbiomas_peru_collection3_integration_v1",
        "classification_{year}",
        (1985, 2024),
    ),
    ("venezuela", "lulc"): (
        "venezuela/collection2/mapbiomas_venezuela_collection2_integration_v1",
        "classification_{year}",
        (1985, 2023),
    ),
    ("brazil", "deforestation"): (
        "brazil/lulc/collection11/mapbiomas_brazil_collection11_deforestation_secondary_vegetation_v5",
        "classification_{year}",
        (1987, 2025),
    ),
    ("bolivia", "deforestation"): (
        "bolivia/lulc/collection3/mapbiomas_bolivia_collection3_deforestation_secondary_vegetation_v1",
        "classification_{year}",
        (1985, 2024),
    ),
    ("paraguay", "deforestation"): (
        "paraguay/lulc/collection3/mapbiomas_paraguay_collection3_deforestation_secondary_vegetation_v1",
        "classification_{year}",
        (1985, 2025),
    ),
    ("peru", "deforestation"): (
        "peru/collection3/mapbiomas_peru_collection3_deforestation_secondary_vegetation_v4",
        "classification_{year}",
        (1999, 2024),
    ),
    ("brazil", "fire"): (
        "brazil/fire/collection5_1/mapbiomas_fire_collection51_monthly_burned_v1",
        "burned_monthly_{year}",
        (1985, 2025),
    ),
    ("colombia", "fire"): (
        "colombia/fire/collection1/mapbiomas_colombia_fire_collection1_monthly_burned_v1",
        "burned_monthly_{year}",
        (2000, 2025),
    ),
    ("paraguay", "fire"): (
        "paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_monthly_burned_v1",
        "burned_monthly_{year}",
        (1999, 2024),
    ),
    ("peru", "fire"): (
        "peru/fire/collection2/mapbiomas_peru_fire_collection2_monthly_burned_v1",
        "burned_monthly_{year}",
        (1990, 2025),
    ),
    ("brazil", "fire_severity"): (
        "brazil/fire/collection5_1/mapbiomas_fire_collection51_severity_class_v1",
        "severity_class_{year}",
        (1985, 2025),
    ),
    ("brazil", "fire_monitor"): (
        "brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1",
        "FireMonth",
        (2019, 2026),
    ),
    ("brazil", "logging"): (
        "brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_logging_v2",
        "logging_{year}",
        (1988, 2024),
    ),
    ("brazil", "water_annual"): (
        "brazil/water/collection5/mapbiomas_brazil_collection5_water_annual_v4",
        "classification_{year}",
        (1985, 2025),
    ),
    ("bolivia", "water_annual"): (
        "bolivia/water/collection4/mapbiomas_bolivia_collection4_water_v1",
        "classification_{year}",
        (1985, 2025),
    ),
    ("peru", "water_annual"): (
        "peru/water/collection3/mapbiomas_peru_collection3_water_v1",
        "classification_{year}",
        (1985, 2024),
    ),
    ("brazil", "water_monthly"): (
        "brazil/water/collection5/mapbiomas_brazil_collection5_water_monthly_v4",
        "water_monthly",
        (1985, 2025),
    ),
    ("bolivia", "water_monthly"): (
        "bolivia/water/collection4/mapbiomas_bolivia_collection4_water_monthly_v1",
        "water_monthly",
        (1985, 2025),
    ),
}

VERIFIED_MONTHS = {
    ("brazil", "fire_monitor"): (("2019-01", "2026-10"), "int"),
    ("brazil", "water_monthly"): (("1985-01", "2025-12"), "zero_padded"),
    ("bolivia", "water_monthly"): (("1985-01", "2025-12"), "zero_padded"),
}


def test_regions_are_letters_only_in_declared_order():
    """Regions are letters only in declared order."""
    assert reg.MAPBIOMAS_REGIONS == (
        "brazil",
        "amazon",
        "bolivia",
        "colombia",
        "ecuador",
        "paraguay",
        "peru",
        "venezuela",
    )
    assert all(re.fullmatch(r"[a-z]+", r) for r in reg.MAPBIOMAS_REGIONS)


@pytest.mark.parametrize("region_product", sorted(VERIFIED))
def test_verified_records(region_product):
    """Verified records."""
    path, band, years = VERIFIED[region_product]
    rec = reg.mapbiomas_asset(*region_product)
    assert rec["asset"] == R + path
    assert rec["band"] == band
    assert rec["years"] == years


def test_registry_has_exactly_the_verified_records():
    """Registry has exactly the verified records."""
    assert {
        (r, p)
        for r in reg.MAPBIOMAS_REGIONS
        for p in _all_products()
        if reg.has_product(r, p)
    } == set(VERIFIED)


@pytest.mark.parametrize("region_product", sorted(VERIFIED_MONTHS))
def test_verified_monthly_layouts(region_product):
    """Verified monthly layouts."""
    months, month_value = VERIFIED_MONTHS[region_product]
    rec = reg.mapbiomas_asset(*region_product)
    assert rec["months"] == months
    assert rec["month_value"] == month_value
    assert rec["month_property"] == "month"


def _all_products():
    return {p for family in reg.FAMILIES.values() for p in family} | {
        "lulc",
        "deforestation",
        "logging",
    }


@pytest.mark.parametrize("region_product", sorted(VERIFIED))
def test_every_record_is_well_formed(region_product):
    """Every record is well formed."""
    rec = reg.mapbiomas_asset(*region_product)
    assert rec["asset"].startswith(R)
    assert rec["layout"] in {"bands", "year_month_collection", "month_collection"}
    first, last = rec["years"]
    assert 1985 <= first <= last <= 2026
    if rec["layout"] in {"bands", "month_collection"}:
        assert "{year}" in rec["band"]
    else:
        assert "{year}" not in rec["band"]
        assert rec["year_property"] == "year"
    if rec["layout"] != "bands":
        lo, hi = rec["months"]
        assert re.fullmatch(r"\d{4}-\d{2}", lo) and re.fullmatch(r"\d{4}-\d{2}", hi)
        assert (int(lo[:4]), int(hi[:4])) == rec["years"]
    assert "MapBiomas" in rec["attribution"] and reg.LICENSE in rec["attribution"]


def test_code_groups_partition_the_known_codes():
    """Code groups partition the known codes."""
    assert tuple(reg.LULC_CODE_GROUPS) == reg.LULC_GROUPS
    seen = {}
    for group, codes in reg.LULC_CODE_GROUPS.items():
        for code in codes:
            assert code not in seen, f"code {code} in {seen.get(code)} and {group}"
            seen[code] = group
    assert not set(seen) & set(reg.LULC_NODATA_CODES)
    assert reg.LULC_CODE_TO_GROUP == seen
    assert reg.LULC_KNOWN_CODES == tuple(sorted(seen))


def test_group_codes_are_unique_single_letters_and_colours_are_hex():
    """Group codes are unique single letters and colours are hex."""
    codes = [reg.LULC_GROUP_CODES[g] for g in reg.LULC_GROUPS]
    assert all(re.fullmatch(r"[a-z]", c) for c in codes)
    assert len(set(codes)) == len(codes)
    assert all(
        re.fullmatch(r"[0-9a-f]{6}", reg.LULC_GROUP_COLORS[g]) for g in reg.LULC_GROUPS
    )


@pytest.mark.parametrize("region", reg.MAPBIOMAS_REGIONS)
def test_every_observed_code_is_classified(region):
    """Review focus 5: an unmapped code would be masked silently in every layer."""
    rec = reg.mapbiomas_asset(region, "lulc")
    unknown = (
        set(rec["observed_codes"])
        - set(reg.LULC_CODE_TO_GROUP)
        - set(reg.LULC_NODATA_CODES)
    )
    assert not unknown, f"{region}: add {sorted(unknown)} to LULC_CODE_GROUPS"


def test_inferred_codes_are_classified():
    """Inferred codes are classified."""
    assert set(reg.LULC_INFERRED_CODES) <= set(reg.LULC_CODE_TO_GROUP)


def test_mapbiomas_asset_returns_a_copy_with_class_tables():
    """Mapbiomas asset returns a copy with class tables."""
    lulc = reg.mapbiomas_asset("peru", "lulc")
    lulc["asset"] = "mutated"
    lulc["class_table"][3] = "mutated"
    again = reg.mapbiomas_asset("peru", "lulc")
    assert again["asset"].endswith("integration_v1")
    assert again["class_table"][3] == "forest"
    assert again["region"] == "peru" and again["product"] == "lulc"
    assert again["nodata_codes"] == (0, 27)
    assert (
        reg.mapbiomas_asset("brazil", "deforestation")["class_table"][4]
        == "primary_suppression"
    )
    assert reg.mapbiomas_asset("brazil", "lulc")["attribution"] == (
        "MapBiomas Brazil, Collection 11 (CC-BY-SA 4.0)"
    )


@pytest.mark.parametrize(
    "region, product",
    [
        ("ecuador", "deforestation"),
        ("colombia", "water_annual"),
        ("peru", "water_monthly"),
        ("bolivia", "fire"),
        ("brazil", "nope"),
        ("chile", "lulc"),
    ],
)
def test_mapbiomas_asset_unknown_raises_key_error(region, product):
    """Mapbiomas asset unknown raises key error."""
    with pytest.raises(KeyError):
        reg.mapbiomas_asset(region, product)
    assert not reg.has_product(region, product)


@pytest.mark.parametrize(
    "name, regions",
    [
        ("lulc", reg.MAPBIOMAS_REGIONS),
        ("deforestation", ("brazil", "bolivia", "paraguay", "peru")),
        ("fire", ("brazil", "colombia", "paraguay", "peru")),
        ("fire_severity", ("brazil",)),
        ("logging", ("brazil",)),
        ("water", ("brazil", "bolivia", "peru")),
        ("water_monthly", ("brazil", "bolivia")),
    ],
)
def test_regions_for(name, regions):
    """Regions for."""
    assert reg.regions_for(name) == regions


def test_year_list_and_month_available():
    """Year list and month available."""
    assert reg.year_list("amazon", "lulc")[-1] == 2023
    assert reg.year_list("brazil", "fire_monitor") == list(range(2019, 2027))
    assert reg.month_available("brazil", "fire_monitor", 2026, 10)
    assert not reg.month_available("brazil", "fire_monitor", 2026, 11)
    assert not reg.month_available("brazil", "fire_monitor", 2018, 12)
    assert not reg.month_available("peru", "water_monthly", 2020, 3)


def test_registry_never_imports_ee():
    """Registry never imports ee."""
    src = Path(reg.__file__).read_text()
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(src))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(node.module or "")]
        )
    }
    assert "ee" not in imported


# --- region param and AOI pre-selection (Task 2) ------------------------------

import types  # noqa: E402


def _aoi(method, name):
    """Stand-in for pysepal's AoiResult (only ``method`` and ``name`` are read)."""
    return types.SimpleNamespace(method=method, name=name)


def test_region_param_shape():
    """Region param shape."""
    spec = reg.region_param("fire")
    assert spec["key"] == "region"
    assert spec["type"] == "choice"
    assert spec["options"] == ["brazil", "colombia", "paraguay", "peru"]
    assert spec["default"] == "brazil"
    assert spec["suffix_prefix"] == ""
    assert spec["advanced"] is False and spec["omit_default"] is False
    assert spec["label_key"] == "vars.modal.param_mapbiomas_region"
    assert spec["hint_key"] == "vars.modal.param_mapbiomas_region_hint"
    assert spec["option_label_key_prefix"] == "vars.mapbiomas.region_"
    assert spec["aoi_default"](_aoi("ADMIN1", "COL_Meta")) == "colombia"


@pytest.mark.parametrize(
    "product", ["lulc", "deforestation", "fire", "logging", "water"]
)
def test_region_param_default_is_first_option_and_brazil(product):
    """Region param default is first option and brazil."""
    spec = reg.region_param(product)
    assert spec["default"] == spec["options"][0] == "brazil"


@pytest.mark.parametrize(
    "method, name, iso3",
    [
        ("ADMIN0", "BRA", "BRA"),
        ("ADMIN1", "COL_Cundinamarca", "COL"),
        ("ADMIN2", "PER_Cusco_Cusco", "PER"),
        ("ADMIN1", "UNK_Somewhere", "UNK"),
        ("DRAW", "BRA_drawn", None),
        ("ASSET", "BRA_Rondonia", None),
        ("ADMIN1", "bra_lower", None),
        ("ADMIN1", "", None),
        ("ADMIN1", None, None),
    ],
)
def test_aoi_iso3(method, name, iso3):
    """Aoi iso3."""
    assert reg.aoi_iso3(_aoi(method, name)) == iso3


def test_aoi_iso3_none():
    """Aoi iso3 none."""
    assert reg.aoi_iso3(None) is None


@pytest.mark.parametrize(
    "product, method, name, region",
    [
        ("lulc", "ADMIN1", "COL_Meta", "colombia"),
        ("fire", "ADMIN1", "COL_Meta", "colombia"),
        (
            "water",
            "ADMIN1",
            "COL_Meta",
            None,
        ),  # Colombia water is denied; amazon has no water
        ("lulc", "ADMIN0", "GUY", "amazon"),  # Pan-Amazon fallback
        ("fire", "ADMIN0", "GUY", None),  # no amazon fire
        ("deforestation", "ADMIN1", "BOL_Beni", "bolivia"),
        ("deforestation", "ADMIN1", "ECU_Napo", None),
        ("lulc", "ADMIN1", "ECU_Napo", "ecuador"),  # own collection wins over amazon
        ("water", "ADMIN1", "ECU_Napo", None),
        ("lulc", "ADMIN1", "ARG_Salta", None),
        ("lulc", "ADMIN1", "UNK_x", None),
        ("lulc", "DRAW", "drawn", None),
        ("lulc", "ADMIN1", "VEN_Bolivar", "venezuela"),
        ("fire", "ADMIN1", "VEN_Bolivar", None),
    ],
)
def test_region_for_aoi(product, method, name, region):
    """Region for aoi."""
    assert reg.region_for_aoi(product, _aoi(method, name)) == region


def test_region_for_aoi_without_aoi():
    """Region for aoi without aoi."""
    assert reg.region_for_aoi("lulc", None) is None


def test_aoi_param_defaults_reads_the_aoi_default_hook(monkeypatch):
    """Aoi param defaults reads the aoi default hook."""
    from gui.scripts.predefined import PREDEFINED_CATALOGUE, aoi_param_defaults

    def boom(_aoi_result):
        raise RuntimeError("hook failure must not break the modal")

    monkeypatch.setitem(
        PREDEFINED_CATALOGUE,
        "mbprobe",
        {
            "label_key": "vars.predefined.altitude",
            "description_key": "vars.predefined_info.altitude",
            "group": "terrain",
            "var_type": "GEEVar",
            "raster_type": "categorical",
            "temporal": True,
            "years": [2020],
            "get_image": lambda aoi, year, **kw: None,
            "params": [
                reg.region_param("fire"),
                {**reg.region_param("lulc"), "key": "other", "aoi_default": boom},
                {
                    **reg.region_param("lulc"),
                    "key": "bogus",
                    "aoi_default": lambda a: "chile",
                },
            ],
        },
    )
    assert aoi_param_defaults("mbprobe", _aoi("ADMIN1", "PER_Cusco")) == {
        "region": "peru"
    }
    assert aoi_param_defaults("mbprobe", _aoi("DRAW", "x")) == {}
    assert aoi_param_defaults("mbprobe", None) == {}
    assert aoi_param_defaults("altitude", _aoi("ADMIN1", "PER_Cusco")) == {}
    assert aoi_param_defaults("", _aoi("ADMIN1", "PER_Cusco")) == {}
