"""MapBiomas catalogue entries: names, builders (fake ee), years, validate, display."""

import itertools
import re
from unittest.mock import ANY

import pytest
from mapbiomas_fake_ee import FakeEE

from gui.i18n import t
from gui.scripts.predefined import (
    GROUP_ORDER,
    PREDEFINED_CATALOGUE,
    build_predefined_name,
    default_param_values,
    mapbiomas,
    resolve_predefined,
    resolved_entry,
    validate_entry,
)
from gui.scripts.predefined.mapbiomas_registry import (
    LULC_CODE_TO_GROUP,
    LULC_GROUPS,
    LULC_KNOWN_CODES,
    ROOT,
)
from gui.scripts.predefined.params import inactive_to_defaults, param_specs

MAPBIOMAS_KEYS = [
    k for k, e in PREDEFINED_CATALOGUE.items() if e.get("group") == "mapbiomas"
]
LULC_PERU = ROOT + "peru/collection3/mapbiomas_peru_collection3_integration_v1"


@pytest.fixture
def fake_ee(monkeypatch):
    """Fake ee."""
    fake = FakeEE()
    monkeypatch.setattr(mapbiomas, "ee", fake)
    return fake


def _full(key, **values):
    """Every param of ``key``: the defaults overlaid with ``values``."""
    return {**default_param_values(key), **values}


def _space(spec):
    kind = spec.get("type", "int")
    if kind == "choice":
        return list(spec["options"])
    if kind == "year":
        years = list(spec["options"])  # C's test_predefined_interval is exhaustive
        return [years[0], years[len(years) // 2], years[-1]]
    if kind == "multichoice":
        values = [o["value"] for o in spec["options"]]
        low = spec.get("min_selected", 1)
        return [
            list(combo)
            for size in range(low, len(values) + 1)
            for combo in itertools.combinations(values, size)
        ]
    return sorted({spec["min"], spec["default"], spec["max"]})


def _combos(key):
    """Distinct canonical value combos: inactive params at default, start < end."""
    specs = param_specs(key)
    seen = set()
    for combo in itertools.product(*[_space(s) for s in specs]):
        values = inactive_to_defaults(key, {s["key"]: v for s, v in zip(specs, combo)})
        if "start_year" in values and values["start_year"] >= values["end_year"]:
            continue
        marker = repr(sorted(values.items()))
        if marker not in seen:
            seen.add(marker)
            yield values


# --- every MapBiomas entry -----------------------------------------------------


@pytest.mark.parametrize("key", MAPBIOMAS_KEYS)
def test_every_mapbiomas_entry_is_wired(key):
    """Every mapbiomas entry is wired."""
    entry = PREDEFINED_CATALOGUE[key]
    assert entry["group"] == "mapbiomas"
    assert entry["prepare_download"] is mapbiomas._prepare_mapbiomas_download
    assert callable(entry["validate"])
    assert entry["get_image"].__module__ == mapbiomas.__name__
    region = param_specs(key)[0]
    assert region["key"] == "region" and region["omit_default"] is False


def test_mapbiomas_group_is_last_and_holds_the_module_entries():
    """Mapbiomas group is last and holds the module entries."""
    assert GROUP_ORDER[-1] == "mapbiomas"
    assert MAPBIOMAS_KEYS == list(mapbiomas.ENTRIES)


@pytest.mark.parametrize("key", MAPBIOMAS_KEYS)
def test_every_combination_round_trips(key):
    """Names are [a-z0-9_], start with the region, and carry no stray four-digit run."""
    count = 0
    for values in _combos(key):
        name = build_predefined_name(key, values)
        assert re.fullmatch(r"[a-z0-9_]+", name), name
        assert name.startswith(f"{key}_{values['region']}"), name
        digits = re.findall(r"[0-9]{4,}", name)
        if "start_year" in values:
            assert name.endswith(f"_{values['start_year']}_{values['end_year']}"), name
            assert digits == [str(values["start_year"]), str(values["end_year"])], name
        else:
            assert digits == [], name
        assert resolve_predefined(name) == (key, values), name
        count += 1
    assert count > 0


@pytest.mark.parametrize("key", MAPBIOMAS_KEYS)
def test_offered_years_validate(key):
    """Review focus 2: every offered year passes validate; the next year fails.

    A combination the region does not offer at all (severity outside Brazil,
    monthly water in Peru) fails for every year with error_not_available.
    """
    if PREDEFINED_CATALOGUE[key].get("interval_years"):
        pytest.skip("interval entry: its validate test covers the year range")
    for values in _combos(key):
        years = resolved_entry(key, values)["years"]
        assert years, values
        errors = [validate_entry(key, values, year) for year in years]
        assert all(e is None for e in errors) or all(
            e is not None and e[0] == "vars.mapbiomas.error_not_available"
            for e in errors
        ), (values, errors)
        assert validate_entry(key, values, years[-1] + 1) is not None, values


@pytest.mark.parametrize("key", MAPBIOMAS_KEYS)
def test_every_combination_builds_or_refuses(monkeypatch, key):
    """Builders refuse exactly what validate refuses; otherwise they end clip -> B1."""
    entry = PREDEFINED_CATALOGUE[key]
    for values in _combos(key):
        fake = FakeEE()
        monkeypatch.setattr(mapbiomas, "ee", fake)
        year = (
            None
            if entry.get("interval_years")
            else resolved_entry(key, values)["years"][-1]
        )
        if validate_entry(key, values, year) is not None:
            with pytest.raises(ValueError):
                entry["get_image"]("AOI", year, **values)
            continue
        entry["get_image"]("AOI", year, **values)
        assert fake.calls[-2:] == [
            ("clip", ("AOI",), {}),
            ("rename", ("B1",), {}),
        ], values


@pytest.mark.parametrize("key", MAPBIOMAS_KEYS)
def test_categorical_outputs_stay_below_the_255_export_nodata(key):
    """Categorical outputs stay below the 255 export nodata."""
    for values in _combos(key):
        entry = resolved_entry(key, values)
        if entry["raster_type"] != "categorical":
            continue
        vis = entry.get("vis_params") or {}
        top = max(
            [vis.get("max", 1)] + [c["value"] for c in entry.get("class_labels") or []]
        )
        assert top < 255, (key, values)


# --- mapbiomas_lulc (Task 6) ---------------------------------------------------


@pytest.mark.parametrize(
    "values, name",
    [
        ({"region": "brazil"}, "mapbiomas_lulc_brazil"),
        (
            {"region": "brazil", "groups": ["forest", "natural"]},
            "mapbiomas_lulc_brazil_gfn",
        ),
        (
            {"region": "amazon", "groups": ["farming", "water"]},
            "mapbiomas_lulc_amazon_gaw",
        ),
        ({"region": "peru", "mode": "groups"}, "mapbiomas_lulc_peru_groups"),
    ],
)
def test_lulc_names(values, name):
    """Lulc names."""
    full = _full("mapbiomas_lulc", **values)
    assert build_predefined_name("mapbiomas_lulc", full) == name
    assert resolve_predefined(name) == ("mapbiomas_lulc", full)


@pytest.mark.parametrize(
    "name",
    [
        "mapbiomas_lulc_brazil_groups_gf",  # groups is inactive in groups mode
        "mapbiomas_lulc_brazil_gf",  # default written out
        "mapbiomas_lulc_brazil_mask",  # default written out
        "mapbiomas_lulc_brazil_gnf",  # codes out of declared order
        "mapbiomas_lulc_chile",  # not a region
        "mapbiomas_lulc_gf",  # region missing
    ],
)
def test_lulc_rejected_names(name):
    """Lulc rejected names."""
    assert resolve_predefined(name) == (None, {})


def test_lulc_mask_builder(fake_ee):
    """Lulc mask builder."""
    mapbiomas._get_mapbiomas_lulc(
        "AOI", 2020, region="peru", mode="mask", groups=["forest", "water"]
    )
    codes = list(LULC_KNOWN_CODES)
    assert fake_ee.calls == [
        ("Image", (LULC_PERU,), {}),
        ("select", ("classification_2020",), {}),
        (
            "remap",
            (
                codes,
                [
                    1 if LULC_CODE_TO_GROUP[c] in ("forest", "water") else 0
                    for c in codes
                ],
            ),
            {},
        ),
        ("toByte", (), {}),
        ("clip", ("AOI",), {}),
        ("rename", ("B1",), {}),
    ]


def test_lulc_groups_builder_numbers_the_groups_from_one(fake_ee):
    """Lulc groups builder numbers the groups from one."""
    mapbiomas._get_mapbiomas_lulc("AOI", 2020, region="peru", mode="groups")
    [(codes, values)] = fake_ee.args_of("remap")
    assert values == [LULC_GROUPS.index(LULC_CODE_TO_GROUP[c]) + 1 for c in codes]
    assert set(values) == {1, 2, 3, 4, 5}


def test_lulc_remap_has_no_default(fake_ee):
    """Review focus 5: unknown codes and 0/27 stay masked (nodata), never 0."""
    mapbiomas._get_mapbiomas_lulc("AOI", 2020)
    [args] = fake_ee.args_of("remap")
    assert len(args) == 2 and fake_ee.kwargs_of("remap") == [{}]
    assert 0 not in args[0] and 27 not in args[0]


def test_lulc_years_follow_the_region():
    """Lulc years follow the region."""
    assert resolved_entry("mapbiomas_lulc", {})["years"] == list(range(1985, 2026))
    assert resolved_entry("mapbiomas_lulc", {"region": "amazon"})["years"] == list(
        range(1985, 2024)
    )


def test_lulc_validate_rejects_stale_year_after_region_switch():
    """Review focus 3: 2025 picked under brazil, then region switched to venezuela."""
    assert validate_entry("mapbiomas_lulc", {"region": "brazil"}, 2025) is None
    assert validate_entry("mapbiomas_lulc", {"region": "venezuela"}, 2025) == (
        "vars.mapbiomas.error_year",
        {"region": t("vars.mapbiomas.region_venezuela"), "first": 1985, "last": 2023},
    )


def test_lulc_builder_refuses_a_stale_year(fake_ee):
    """Lulc builder refuses a stale year."""
    with pytest.raises(ValueError, match="1985"):
        mapbiomas._get_mapbiomas_lulc("AOI", 2025, region="venezuela")
    assert fake_ee.calls == []


def test_lulc_display_follows_the_mode():
    """Lulc display follows the mode."""
    mask = resolved_entry("mapbiomas_lulc", {"groups": ["natural", "water"]})
    assert mask["raster_type"] == "categorical"
    assert mask["vis_params"] == {"palette": ["ffffff", "d6bc74"], "min": 0, "max": 1}
    assert mask["class_labels"] is None
    assert mask["legend_class_keys"] == ("legend.class.absent", "legend.class.present")
    groups = resolved_entry("mapbiomas_lulc", {"mode": "groups"})
    assert groups["vis_params"]["min"] == 1 and groups["vis_params"]["max"] == 5
    assert groups["class_labels"] == [
        {"value": i + 1, "label_key": f"vars.mapbiomas.group_{g}", "color": ANY}
        for i, g in enumerate(LULC_GROUPS)
    ]


# --- deforestation_mapbiomas (Task 8) -------------------------------------------

DEFO_BRAZIL = (
    ROOT
    + "brazil/lulc/collection11/"
    + "mapbiomas_brazil_collection11_deforestation_secondary_vegetation_v5"
)


def test_defo_is_an_interval_entry_over_the_union_of_region_years():
    """Defo is an interval entry over the union of region years."""
    entry = PREDEFINED_CATALOGUE["deforestation_mapbiomas"]
    assert entry["temporal"] is False and "years" not in entry
    assert entry["interval_years"] == list(range(1986, 2027))
    assert mapbiomas.defo_interval_years("brazil") == list(range(1988, 2027))
    assert mapbiomas.defo_interval_years("peru") == list(range(2000, 2026))
    assert [s["key"] for s in param_specs("deforestation_mapbiomas")] == [
        "region",
        "vegetation",
        "start_year",
        "end_year",
    ]
    assert param_specs("deforestation_mapbiomas")[0]["options"] == [
        "brazil",
        "bolivia",
        "paraguay",
        "peru",
    ]


@pytest.mark.parametrize(
    "values, name",
    [
        (
            {"region": "brazil", "start_year": 2015, "end_year": 2020},
            "deforestation_mapbiomas_brazil_2015_2020",
        ),
        (
            {
                "region": "peru",
                "vegetation": "all",
                "start_year": 2010,
                "end_year": 2020,
            },
            "deforestation_mapbiomas_peru_all_2010_2020",
        ),
    ],
)
def test_defo_names(values, name):
    """Defo names."""
    from spatialrisk.evaluation import interval_from_target

    full = _full("deforestation_mapbiomas", **values)
    assert build_predefined_name("deforestation_mapbiomas", full) == name
    assert resolve_predefined(name) == ("deforestation_mapbiomas", full)
    assert interval_from_target(name) == values["end_year"] - values["start_year"]


@pytest.mark.parametrize(
    "name",
    [
        "deforestation_mapbiomas_brazil_primary_2015_2020",  # default written out
        "deforestation_mapbiomas_colombia_2015_2020",  # no deforestation layer
        "deforestation_mapbiomas_brazil_2020_2015",  # reversed
        "deforestation_mapbiomas_brazil_1985_1990",  # start outside interval_years
    ],
)
def test_defo_rejected_names(name):
    """Defo rejected names."""
    assert resolve_predefined(name) == (None, {})


def test_defo_builder_masks_events_with_the_start_baseline(fake_ee):
    """Defo builder masks events with the start baseline."""
    mapbiomas._get_deforestation_mapbiomas(
        "AOI", region="brazil", start_year=2015, end_year=2018
    )
    assert fake_ee.args_of("Image")[0] == (DEFO_BRAZIL,)
    assert ("Image", (0,), {}) in fake_ee.calls
    assert [a[0] for a in fake_ee.args_of("select")] == [
        "classification_2014",
        "classification_2015",
        "classification_2016",
        "classification_2017",
    ]
    remaps = fake_ee.args_of("remap")
    assert remaps[0] == ([2], [1], 0)
    assert remaps[1:] == [([4], [1], 0)] * 3
    assert fake_ee.names().count("Or") == 3
    assert fake_ee.names()[-4:] == ["updateMask", "toByte", "clip", "rename"]


def test_defo_all_vegetation_adds_secondary_codes(fake_ee):
    """Defo all vegetation adds secondary codes."""
    mapbiomas._get_deforestation_mapbiomas(
        "AOI", region="paraguay", vegetation="all", start_year=2015, end_year=2016
    )
    assert fake_ee.args_of("remap") == [([2, 3], [1, 1], 0), ([4, 6], [1, 1], 0)]


def test_defo_builder_refuses_a_start_before_the_region_data(fake_ee):
    """Defo builder refuses a start before the region data."""
    with pytest.raises(ValueError):
        mapbiomas._get_deforestation_mapbiomas(
            "AOI", region="peru", start_year=1999, end_year=2005
        )
    assert fake_ee.calls == []


@pytest.mark.parametrize(
    "region, start, end, ok",
    [
        ("brazil", 1988, 2026, True),
        ("brazil", 1987, 2000, False),
        ("bolivia", 1986, 2025, True),
        ("bolivia", 2010, 2026, False),
        ("paraguay", 1986, 2026, True),
        ("peru", 2000, 2025, True),
        ("peru", 1999, 2005, False),  # review focus 2
        ("peru", 2000, 2026, False),
    ],
)
def test_defo_validate_checks_the_region_interval(region, start, end, ok):
    """Defo validate checks the region interval."""
    years = mapbiomas.defo_interval_years(region)
    err = validate_entry(
        "deforestation_mapbiomas",
        {"region": region, "start_year": start, "end_year": end},
        None,
    )
    assert (
        err is None
        if ok
        else err
        == (
            "vars.mapbiomas.error_interval_range",
            {
                "region": t(f"vars.mapbiomas.region_{region}"),
                "first": years[0],
                "last": years[-1],
            },
        )
    )


def test_defo_display():
    """Defo display."""
    entry = resolved_entry("deforestation_mapbiomas", {})
    assert entry["raster_type"] == "categorical"
    assert entry["vis_params"] == {"palette": ["d9d9d9", "e31a1c"], "min": 0, "max": 1}
    assert entry["legend_class_keys"] == ("legend.class.stable", "legend.class.loss")


@pytest.mark.parametrize("region", ["brazil", "bolivia", "paraguay", "peru"])
def test_defo_pairs_with_native_vegetation_lulc_at_start(region):
    """Defo pairs with native vegetation lulc at start."""
    from gui.tile.evaluation_helpers import expected_forest_for_truth

    start = mapbiomas.defo_interval_years(region)[0]
    name = f"deforestation_mapbiomas_{region}_{start}_{start + 1}"
    lulc_name, year = expected_forest_for_truth(name)
    assert (lulc_name, year) == (f"mapbiomas_lulc_{region}_gfn", start)
    assert validate_entry("mapbiomas_lulc", {"region": region}, year) is None


# --- mapbiomas_fire: burned area and severity (Task 9) ---------------------------

FIRE_PERU = (
    ROOT + "peru/fire/collection2/mapbiomas_peru_fire_collection2_monthly_burned_v1"
)
FIRE_MONITOR = ROOT + "brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1"


@pytest.mark.parametrize(
    "values, name",
    [
        ({"region": "brazil", "product": "burned"}, "mapbiomas_fire_brazil_burned"),
        (
            {"region": "peru", "product": "burned", "period": "sep"},
            "mapbiomas_fire_peru_burned_sep",
        ),
        (
            {
                "region": "brazil",
                "product": "burned",
                "period": "sep",
                "source": "monitor",
            },
            "mapbiomas_fire_brazil_burned_sep_monitor",
        ),
        (
            {"region": "brazil", "product": "burned", "source": "monitor"},
            "mapbiomas_fire_brazil_burned_monitor",
        ),
        ({"region": "brazil", "product": "severity"}, "mapbiomas_fire_brazil_severity"),
    ],
)
def test_fire_names(values, name):
    """Fire names."""
    full = _full("mapbiomas_fire", **values)
    assert build_predefined_name("mapbiomas_fire", full) == name
    assert resolve_predefined(name) == ("mapbiomas_fire", full)


@pytest.mark.parametrize(
    "name",
    [
        "mapbiomas_fire_brazil_severity_sep",  # period is inactive for severity
        "mapbiomas_fire_brazil_severity_monitor",  # source is inactive for severity
        "mapbiomas_fire_brazil",  # product is always written
        "mapbiomas_fire_brazil_burned_year",  # default written out
        "mapbiomas_fire_brazil_burned_collection",  # default written out
        "mapbiomas_fire_brazil_burned_monitor_sep",  # segments out of order
        "mapbiomas_fire_amazon_burned",  # no Pan-Amazon fire
    ],
)
def test_fire_rejected_names(name):
    """Fire rejected names."""
    assert resolve_predefined(name) == (None, {})


def _region_mask_calls(lulc_asset):
    return [
        ("Image", (lulc_asset,), {}),
        ("select", (0,), {}),
        ("mask", (), {}),
        ("gt", (0,), {}),
    ]


def test_fire_burned_collection_year(fake_ee):
    """Fire burned collection year."""
    mapbiomas._get_mapbiomas_fire("AOI", 2020, region="peru")
    assert fake_ee.calls == [
        ("Image", (FIRE_PERU,), {}),
        ("select", ("burned_monthly_2020",), {}),
        ("gt", (0,), {}),
        ("unmask", (0,), {}),
        *_region_mask_calls(LULC_PERU),
        ("updateMask", (ANY,), {}),
        ("toByte", (), {}),
        ("clip", ("AOI",), {}),
        ("rename", ("B1",), {}),
    ]


def test_fire_burned_collection_month_matches_the_burn_month(fake_ee):
    """Fire burned collection month matches the burn month."""
    mapbiomas._get_mapbiomas_fire("AOI", 2020, region="peru", period="sep")
    assert fake_ee.calls[2] == ("eq", (9,), {})
    assert "gt" not in fake_ee.names()[:3]


def test_fire_monitor_month(fake_ee):
    """Fire monitor month."""
    mapbiomas._get_mapbiomas_fire(
        "AOI", 2026, region="brazil", period="sep", source="monitor"
    )
    assert fake_ee.args_of("ImageCollection") == [(FIRE_MONITOR,)]
    assert fake_ee.args_of("Filter.eq") == [("year", 2026), ("month", 9)]
    names = fake_ee.names()
    assert ("select", ("FireMonth",), {}) in fake_ee.calls
    assert names.index("max") < names.index("gt") < names.index("unmask")


def test_fire_monitor_year_takes_every_month_of_the_year(fake_ee):
    """Fire monitor year takes every month of the year."""
    mapbiomas._get_mapbiomas_fire("AOI", 2025, region="brazil", source="monitor")
    assert fake_ee.args_of("Filter.eq") == [("year", 2025)]
    assert "max" in fake_ee.names()


def test_fire_severity(fake_ee):
    """Fire severity."""
    mapbiomas._get_mapbiomas_fire("AOI", 2020, region="brazil", product="severity")
    assert fake_ee.calls[1] == ("select", ("severity_class_2020",), {})
    assert fake_ee.calls[2] == ("unmask", (0,), {})


def test_fire_years():
    """Fire years."""
    assert resolved_entry("mapbiomas_fire", {"region": "paraguay"})["years"] == list(
        range(1999, 2025)
    )
    monitor = {"source": "monitor"}
    assert resolved_entry("mapbiomas_fire", monitor)["years"] == list(range(2019, 2026))
    assert resolved_entry("mapbiomas_fire", {**monitor, "period": "oct"})[
        "years"
    ] == list(range(2019, 2027))
    assert resolved_entry("mapbiomas_fire", {**monitor, "period": "nov"})[
        "years"
    ] == list(range(2019, 2026))


def test_fire_validate_monitor_months():
    """Review focus 2: 2026 annual is partial; Nov 2026 is not published yet."""
    params = {"region": "brazil", "product": "burned", "source": "monitor"}
    assert validate_entry("mapbiomas_fire", {**params, "period": "oct"}, 2026) is None
    for period in ("nov", "year"):
        err = validate_entry("mapbiomas_fire", {**params, "period": period}, 2026)
        assert err == (
            "vars.mapbiomas.error_period",
            {
                "region": t("vars.mapbiomas.region_brazil"),
                "first": "2019-01",
                "last": "2026-10",
            },
        )


def test_fire_validate_severity_and_monitor_outside_brazil():
    """Fire validate severity and monitor outside brazil."""
    peru = {"region": t("vars.mapbiomas.region_peru")}
    assert validate_entry(
        "mapbiomas_fire", {"region": "peru", "product": "severity"}, 2020
    ) == (
        "vars.mapbiomas.error_not_available",
        peru,
    )
    assert validate_entry(
        "mapbiomas_fire",
        {"region": "peru", "product": "burned", "source": "monitor"},
        2020,
    ) == ("vars.mapbiomas.error_not_available", peru)


def test_fire_validate_rejects_stale_year_after_region_switch():
    """Review focus 3: 1990 is fine for Brazil, not for Colombia."""
    assert validate_entry("mapbiomas_fire", {"region": "brazil"}, 1990) is None
    assert validate_entry("mapbiomas_fire", {"region": "colombia"}, 1990) == (
        "vars.mapbiomas.error_year",
        {"region": t("vars.mapbiomas.region_colombia"), "first": 2000, "last": 2025},
    )


def test_fire_display():
    """Fire display."""
    burned = resolved_entry("mapbiomas_fire", {})
    assert burned["raster_type"] == "categorical"
    assert burned["vis_params"] == {"palette": ["ffffff", "d84315"], "min": 0, "max": 1}
    assert burned["legend_class_keys"] == (
        "legend.class.unburned",
        "legend.class.burned",
    )
    assert burned["class_labels"] is None
    severity = resolved_entry("mapbiomas_fire", {"product": "severity"})
    assert [c["value"] for c in severity["class_labels"]] == [0, 1, 2, 3, 4, 5]
    assert severity["vis_params"]["max"] == 5


# --- mapbiomas_fire: since / frequency (Task 10) ----------------------------------


@pytest.mark.parametrize(
    "values, name",
    [
        ({"region": "brazil", "product": "since"}, "mapbiomas_fire_brazil_since"),
        (
            {"region": "peru", "product": "frequency", "window": 5},
            "mapbiomas_fire_peru_frequency_w5",
        ),
    ],
)
def test_fire_history_names(values, name):
    """Fire history names."""
    full = _full("mapbiomas_fire", **values)
    assert build_predefined_name("mapbiomas_fire", full) == name
    assert resolve_predefined(name) == ("mapbiomas_fire", full)


@pytest.mark.parametrize(
    "name",
    [
        "mapbiomas_fire_brazil_burned_w5",  # window is inactive for burned
        "mapbiomas_fire_brazil_since_sep",  # period is inactive for since
        "mapbiomas_fire_brazil_since_monitor",  # source is inactive for since
        "mapbiomas_fire_brazil_since_w10",  # default written out
        "mapbiomas_fire_brazil_since_w26",  # above max
    ],
)
def test_fire_history_rejected_names(name):
    """Fire history rejected names."""
    assert resolve_predefined(name) == (None, {})


def test_fire_since_builder(fake_ee):
    """Fire since builder."""
    mapbiomas._get_mapbiomas_fire(
        "AOI", 2020, region="brazil", product="since", window=10
    )
    bands = [a[0] for a in fake_ee.args_of("select") if a[0] != 0]
    assert bands == [f"burned_monthly_{y}" for y in range(2010, 2020)]
    [(images,)] = fake_ee.args_of("ImageCollection")
    assert len(images) == 10
    assert [a[0] for a in fake_ee.args_of("multiply")] == list(range(2010, 2020))
    assert ("Image", (2020,), {}) in fake_ee.calls
    assert ("where", (ANY, 11), {}) in fake_ee.calls
    assert fake_ee.names()[-4:] == ["updateMask", "toInt16", "clip", "rename"]


def test_fire_frequency_builder(fake_ee):
    """Fire frequency builder."""
    mapbiomas._get_mapbiomas_fire(
        "AOI", 2020, region="peru", product="frequency", window=5
    )
    bands = [a[0] for a in fake_ee.args_of("select") if a[0] != 0]
    assert bands == [f"burned_monthly_{y}" for y in range(2015, 2020)]
    assert "sum" in fake_ee.names() and "where" not in fake_ee.names()


def test_fire_history_years_and_validate():
    """Fire history years and validate."""
    since = {"region": "brazil", "product": "since"}
    assert resolved_entry("mapbiomas_fire", since)["years"] == list(range(1995, 2027))
    freq = {"region": "paraguay", "product": "frequency", "window": 25}
    assert resolved_entry("mapbiomas_fire", freq)["years"] == [2024, 2025]
    assert validate_entry("mapbiomas_fire", since, 1994) == (
        "vars.mapbiomas.error_year",
        {"region": t("vars.mapbiomas.region_brazil"), "first": 1995, "last": 2026},
    )


def test_fire_history_display():
    """Fire history display."""
    since = resolved_entry("mapbiomas_fire", {"product": "since", "window": 5})
    assert since["raster_type"] == "continuous"
    assert since["unit_key"] == "legend.unit.years_value"
    assert since["vis_params"]["min"] == 1 and since["vis_params"]["max"] == 6
    freq = resolved_entry("mapbiomas_fire", {"product": "frequency"})
    assert freq["raster_type"] == "continuous"
    assert freq["unit_key"] == "legend.unit.plain_value"
    assert freq["vis_params"]["min"] == 0 and freq["vis_params"]["max"] == 10
    assert resolved_entry("mapbiomas_fire", {})["unit_key"] is None


# --- mapbiomas_degradation (Task 14) ------------------------------------------------

LULC_BRAZIL = (
    ROOT + "brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3"
)
DEG = ROOT + "brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_"


@pytest.mark.parametrize("variable", ["edge", "secondary", "logging", "canopy"])
def test_degradation_names(variable):
    """Degradation names."""
    full = _full("mapbiomas_degradation", region="brazil", variable=variable)
    name = f"mapbiomas_degradation_brazil_{variable}"
    assert build_predefined_name("mapbiomas_degradation", full) == name
    assert resolve_predefined(name) == ("mapbiomas_degradation", full)


@pytest.mark.parametrize(
    "name",
    [
        "mapbiomas_degradation_brazil",  # variable is always written
        "mapbiomas_degradation_peru_edge",  # Brazil only
        "mapbiomas_degradation_brazil_edge_sep",  # no period param
    ],
)
def test_degradation_rejected_names(name):
    """Degradation rejected names."""
    assert resolve_predefined(name) == (None, {})


def test_degradation_edge_keeps_the_native_mask(fake_ee):
    """Degradation edge keeps the native mask."""
    mapbiomas._get_mapbiomas_degradation("AOI", 2020, variable="edge")
    assert fake_ee.calls == [
        ("Image", (DEG + "edge_age_v1",), {}),
        ("select", ("age_2020",), {}),
        ("toInt16", (), {}),
        ("clip", ("AOI",), {}),
        ("rename", ("B1",), {}),
    ]


def test_degradation_secondary_decodes_the_age(fake_ee):
    """Degradation secondary decodes the age."""
    mapbiomas._get_mapbiomas_degradation("AOI", 2020, variable="secondary")
    assert fake_ee.calls == [
        ("Image", (DEG + "secondary_vegetation_age_v1",), {}),
        ("select", ("age_2020",), {}),
        ("divide", (100,), {}),
        ("floor", (), {}),
        ("unmask", (0,), {}),
        *_region_mask_calls(LULC_BRAZIL),
        ("updateMask", (ANY,), {}),
        ("toInt16", (), {}),
        ("clip", ("AOI",), {}),
        ("rename", ("B1",), {}),
    ]


def test_degradation_logging_is_a_byte_mask(fake_ee):
    """Degradation logging is a byte mask."""
    mapbiomas._get_mapbiomas_degradation("AOI", 2020, variable="logging")
    assert fake_ee.calls[:4] == [
        ("Image", (DEG + "logging_v2",), {}),
        ("select", ("logging_2020",), {}),
        ("gt", (0,), {}),
        ("unmask", (0,), {}),
    ]
    assert fake_ee.names()[-3:] == ["toByte", "clip", "rename"]


def test_degradation_canopy_counts(fake_ee):
    """Degradation canopy counts."""
    mapbiomas._get_mapbiomas_degradation("AOI", 2020, variable="canopy")
    assert fake_ee.calls[1] == ("select", ("canopy_disturbance_frequency_2020",), {})
    assert fake_ee.calls[2] == ("unmask", (0,), {})
    assert fake_ee.names()[-3:] == ["toInt16", "clip", "rename"]


def test_degradation_years_and_validate():
    """Degradation years and validate."""
    assert resolved_entry("mapbiomas_degradation", {"variable": "edge"})[
        "years"
    ] == list(range(1985, 2025))
    logging = {"region": "brazil", "variable": "logging"}
    assert resolved_entry("mapbiomas_degradation", logging)["years"] == list(
        range(1988, 2025)
    )
    assert validate_entry("mapbiomas_degradation", logging, 1986) == (
        "vars.mapbiomas.error_year",
        {"region": t("vars.mapbiomas.region_brazil"), "first": 1988, "last": 2024},
    )


def test_degradation_display():
    """Degradation display."""
    edge = resolved_entry("mapbiomas_degradation", {"variable": "edge"})
    assert edge["raster_type"] == "continuous"
    assert edge["unit_key"] == "legend.unit.years_value"
    logging = resolved_entry("mapbiomas_degradation", {"variable": "logging"})
    assert logging["raster_type"] == "categorical"
    assert logging["unit_key"] is None
    assert logging["vis_params"] == {
        "palette": ["ffffff", "8e24aa"],
        "min": 0,
        "max": 1,
    }
    canopy = resolved_entry("mapbiomas_degradation", {"variable": "canopy"})
    assert canopy["unit_key"] == "legend.unit.plain_value"


# --- mapbiomas_water (Task 15) ------------------------------------------------------

WATER_ANNUAL = (
    ROOT + "brazil/water/collection5/mapbiomas_brazil_collection5_water_annual_v4"
)
WATER_MONTHLY = (
    ROOT + "brazil/water/collection5/mapbiomas_brazil_collection5_water_monthly_v4"
)


@pytest.mark.parametrize(
    "values, name",
    [
        ({"region": "brazil"}, "mapbiomas_water_brazil"),
        ({"region": "bolivia", "period": "sep"}, "mapbiomas_water_bolivia_sep"),
        ({"region": "peru"}, "mapbiomas_water_peru"),
    ],
)
def test_water_names(values, name):
    """Water names."""
    full = _full("mapbiomas_water", **values)
    assert build_predefined_name("mapbiomas_water", full) == name
    assert resolve_predefined(name) == ("mapbiomas_water", full)


@pytest.mark.parametrize(
    "name",
    [
        "mapbiomas_water_brazil_year",  # default written out
        "mapbiomas_water_colombia",  # no MapBiomas water for Colombia
        "mapbiomas_water_peru_sep_year",  # leftover segment
    ],
)
def test_water_rejected_names(name):
    """Water rejected names."""
    assert resolve_predefined(name) == (None, {})


def test_water_annual_builder(fake_ee):
    """Water annual builder."""
    mapbiomas._get_mapbiomas_water("AOI", 2020, region="brazil")
    assert fake_ee.calls == [
        ("Image", (WATER_ANNUAL,), {}),
        ("select", ("classification_2020",), {}),
        ("gt", (0,), {}),
        ("unmask", (0,), {}),
        *_region_mask_calls(LULC_BRAZIL),
        ("updateMask", (ANY,), {}),
        ("toByte", (), {}),
        ("clip", ("AOI",), {}),
        ("rename", ("B1",), {}),
    ]


def test_water_monthly_builder_filters_the_padded_month(fake_ee):
    """Water monthly builder filters the padded month."""
    mapbiomas._get_mapbiomas_water("AOI", 2020, region="brazil", period="sep")
    assert fake_ee.args_of("ImageCollection") == [(WATER_MONTHLY,)]
    assert fake_ee.args_of("Filter.eq") == [("year", 2020), ("month", "09")]
    assert ("select", ("water_monthly",), {}) in fake_ee.calls
    names = fake_ee.names()
    assert names.index("max") < names.index("gt") < names.index("unmask")


def test_water_years():
    """Water years."""
    assert resolved_entry("mapbiomas_water", {"region": "peru"})["years"] == list(
        range(1985, 2025)
    )
    assert resolved_entry("mapbiomas_water", {"region": "brazil", "period": "sep"})[
        "years"
    ] == list(range(1985, 2026))


def test_water_validate():
    """Review focus 3: Peru has no monthly water; stale years per region."""
    assert validate_entry(
        "mapbiomas_water", {"region": "peru", "period": "sep"}, 2020
    ) == (
        "vars.mapbiomas.error_not_available",
        {"region": t("vars.mapbiomas.region_peru")},
    )
    assert validate_entry("mapbiomas_water", {"region": "peru"}, 2025) == (
        "vars.mapbiomas.error_year",
        {"region": t("vars.mapbiomas.region_peru"), "first": 1985, "last": 2024},
    )
    assert validate_entry(
        "mapbiomas_water", {"region": "brazil", "period": "sep"}, 2026
    ) == (
        "vars.mapbiomas.error_period",
        {
            "region": t("vars.mapbiomas.region_brazil"),
            "first": "1985-01",
            "last": "2025-12",
        },
    )


def test_water_display():
    """Water display."""
    entry = resolved_entry("mapbiomas_water", {})
    assert entry["raster_type"] == "categorical"
    assert entry["vis_params"] == {"palette": ["ffffff", "2532e4"], "min": 0, "max": 1}
    assert entry["legend_class_keys"] == ("legend.class.absent", "legend.class.present")
