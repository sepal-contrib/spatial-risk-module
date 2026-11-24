"""Predefined ``roads`` layer: OSM (default) or GRIP4 with road-type selection.

Issue #47. The default must stay the bare ``roads`` name with the same Earth
Engine expression as before; GRIP4 variants carry ``_grip`` and an optional
``_r<codes>`` road-type segment.
"""

import json

import pytest

from gui import i18n

LOCALES = ("en", "es-ES", "fr", "pt-BR")
ALL_TYPES = ["highway", "primary", "secondary", "tertiary", "local"]
CODES = dict(zip(ALL_TYPES, "hpstl"))

ROAD_KEYS = (
    "vars.predefined.roads",
    "vars.predefined_info.roads",
    "vars.modal.param_roads_source",
    "vars.modal.param_roads_source_hint",
    "vars.modal.roads_source_osm",
    "vars.modal.roads_source_grip",
    "vars.modal.param_road_types",
    "vars.modal.param_road_types_hint",
    *(f"vars.modal.road_type_{value}" for value in ALL_TYPES),
)


def _flat(data, prefix=""):
    out = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flat(value, dotted))
        else:
            out[dotted] = value
    return out


def _vars_catalog(locale):
    path = i18n.MESSAGES_DIR / locale / "vars.json"
    return _flat(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("locale", LOCALES)
def test_roads_strings_exist_in_every_locale(locale):
    """Every roads key is declared, non-empty, in each shipped locale."""
    catalog = _vars_catalog(locale)
    missing = [key for key in ROAD_KEYS if not catalog.get(key)]
    assert not missing, (locale, missing)


@pytest.mark.parametrize("locale", LOCALES)
def test_roads_description_states_grip4_vintage_and_attribution(locale):
    """GRIP4's vintage, sparseness caveat and attribution reach every locale."""
    text = _vars_catalog(locale)["vars.predefined_info.roads"]
    for needle in (
        "GRIP4",
        "2018",
        "Meijer",
        "GLOBIO",
        "https://doi.org/10.1088/1748-9326/aabd42",
        "https://gee-community-catalog.org/projects/grip/",
        "https://www.openstreetmap.org/",
    ):
        assert needle in text, (locale, needle)


@pytest.mark.parametrize("locale", LOCALES)
def test_roads_label_names_both_sources(locale):
    """The dropdown label no longer claims the layer is OSM-only."""
    label = _vars_catalog(locale)["vars.predefined.roads"]
    assert "OSM" in label and "GRIP4" in label


# ---------------------------------------------------------------------------
# Builder (recording fake ``ee`` on the theme module that owns it)
# ---------------------------------------------------------------------------

import inspect  # noqa: E402
import types  # noqa: E402

import gui.scripts.predefined.infrastructure as infra  # noqa: E402

OSM_ASSET = (
    "projects/ee-andyarnellgee/assets/crosscutting/infrastructure"
    "/roads_osm/roadsAllImageOSM"
)
GRIP4_TABLE_IDS = [
    f"projects/sat-io/open-datasets/GRIP4/{region}"
    for region in (
        "Africa",
        "Central-South-America",
        "Europe",
        "Middle-East-Central-Asia",
        "North-America",
        "Oceania",
        "South-East-Asia",
    )
]


class _Geometry:
    """Fake ee.Geometry: only ``isinstance`` and ``bounds`` matter."""

    def bounds(self):
        return "BOUNDS"


class _Aoi:
    """Fake FeatureCollection AOI whose bounds are the ``"BOUNDS"`` sentinel."""

    def geometry(self):
        return _Geometry()


class _Rec:
    """Chainable ee stand-in recording ``(method, args, kwargs)``."""

    def __init__(self, calls):
        self._calls = calls

    def __getattr__(self, method):
        if method.startswith("__"):
            raise AttributeError(method)

        def _call(*args, **kwargs):
            self._calls.append((method, args, kwargs))
            return self

        return _call


@pytest.fixture
def fake_ee(monkeypatch):
    """Patch ``ee`` on infrastructure.py; return the shared call log."""
    calls = []
    rec = _Rec(calls)

    def _ctor(kind):
        def _make(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return rec

        return _make

    fake = types.SimpleNamespace(
        Image=_ctor("Image"),
        ImageCollection=_ctor("ImageCollection"),
        FeatureCollection=_ctor("FeatureCollection"),
        Filter=types.SimpleNamespace(
            inList=lambda prop, values: ("inList", prop, list(values))
        ),
        Geometry=_Geometry,
    )
    monkeypatch.setattr(infra, "ee", fake)
    return calls


def _osm_expression(aoi):
    return [
        ("Image", (OSM_ASSET,), {}),
        ("unmask", (), {}),
        ("clip", (aoi,), {}),
        ("toByte", (), {}),
    ]


def test_osm_default_is_the_pre_change_expression(fake_ee):
    """``roads`` must build exactly what it built before issue #47."""
    aoi = _Aoi()
    infra._get_roads(aoi)
    assert fake_ee == _osm_expression(aoi)


def test_osm_ignores_road_types(fake_ee):
    """The OSM asset has no type breakdown; a stray ``types`` changes nothing."""
    aoi = _Aoi()
    infra._get_roads(aoi, None, source="osm", types=["highway"])
    assert fake_ee == _osm_expression(aoi)


def test_grip_merges_all_seven_regional_tables(fake_ee):
    """AOIs straddling two GRIP4 regions (e.g. US-Mexico) need every table."""
    infra._get_roads(_Aoi(), None, source="grip", types=list(ALL_TYPES))
    tables = [
        call[1][0]
        for call in fake_ee
        if call[0] == "FeatureCollection" and isinstance(call[1][0], str)
    ]
    assert sorted(tables) == sorted(GRIP4_TABLE_IDS)
    assert ("flatten", (), {}) in fake_ee


def test_grip_filters_by_aoi_bounds_then_road_type(fake_ee):
    """Bounds (never the AOI) pre-filter first; GP_RTP codes are sorted ints."""
    infra._get_roads(_Aoi(), None, source="grip", types=["primary", "highway", "local"])
    assert [c for c in fake_ee if c[0] == "filterBounds"] == [
        ("filterBounds", ("BOUNDS",), {})
    ]
    assert [c for c in fake_ee if c[0] == "filter"] == [
        ("filter", (("inList", "GP_RTP", [1, 2, 5]),), {})
    ]
    names = [c[0] for c in fake_ee]
    assert names.index("filterBounds") < names.index("filter")


@pytest.mark.parametrize("value, code", [(v, i + 1) for i, v in enumerate(ALL_TYPES)])
def test_each_road_type_maps_to_its_gp_rtp_class(fake_ee, value, code):
    """highway=1, primary=2, secondary=3, tertiary=4, local=5."""
    infra._get_roads(_Aoi(), None, source="grip", types=[value])
    assert ("filter", (("inList", "GP_RTP", [code]),), {}) in fake_ee


def test_grip_paints_one_pixel_lines_on_a_zero_image(fake_ee):
    """Absent types must give 0 (not masked → 255 nodata); default width = 1 px."""
    aoi = _Aoi()
    infra._get_roads(aoi, None, source="grip", types=["highway"])
    names = [c[0] for c in fake_ee]
    i = names.index("paint")
    assert fake_ee[i - 1] == ("Image", (0,), {})
    method, args, kwargs = fake_ee[i]
    assert len(args) == 2 and args[1] == 1 and kwargs == {}
    assert fake_ee[i + 1 :] == [("clip", (aoi,), {}), ("toByte", (), {})]
    assert not {"selfMask", "updateMask", "mask"} & set(names)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "bing"},
        {"source": "grip", "types": []},
        {"source": "grip", "types": ["motorway"]},
    ],
)
def test_builder_rejects_unknown_values(fake_ee, kwargs):
    """Never fall back silently to OSM or to an empty filter."""
    with pytest.raises(ValueError):
        infra._get_roads(_Aoi(), None, **kwargs)


def test_builder_defaults_are_osm_and_all_types():
    """Calling with no params is the OSM layer; types default to all five."""
    sig = inspect.signature(infra._get_roads)
    assert sig.parameters["source"].default == "osm"
    assert sig.parameters["types"].default == ALL_TYPES
    assert [t["value"] for t in infra.ROAD_TYPES] == ALL_TYPES
    assert [t["code"] for t in infra.ROAD_TYPES] == list("hpstl")
    assert [t["gp_rtp"] for t in infra.ROAD_TYPES] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Param specs and the name grammar
# ---------------------------------------------------------------------------

from itertools import combinations  # noqa: E402

from gui.scripts.predefined import (  # noqa: E402
    PREDEFINED_CATALOGUE,
    advanced_param_specs,
    basic_param_specs,
    build_predefined_name,
    coerce_param_values,
    default_param_values,
    entry_for_name,
    param_specs,
    resolve_predefined,
    spec_segments,
)


def _subsets():
    for size in range(1, len(ALL_TYPES) + 1):
        for combo in combinations(ALL_TYPES, size):
            yield list(combo)


def _grip_name(subset):
    if subset == ALL_TYPES:
        return "roads_grip"
    return "roads_grip_r" + "".join(CODES[value] for value in subset)


def test_roads_declares_source_and_types_as_advanced():
    """Both knobs live in the Advanced section and are omitted at default."""
    specs = {spec["key"]: spec for spec in param_specs("roads")}
    assert list(specs) == ["source", "types"]
    assert basic_param_specs("roads") == []
    assert [s["key"] for s in advanced_param_specs("roads")] == ["source", "types"]

    source = specs["source"]
    assert source["type"] == "choice"
    assert source["default"] == "osm"
    assert source["options"] == ["osm", "grip"]
    assert source["suffix_prefix"] == ""
    assert source["omit_default"] is True
    assert source["option_label_key_prefix"] == "vars.modal.roads_source_"

    road_types = specs["types"]
    assert road_types["type"] == "multichoice"
    assert road_types["default"] == ALL_TYPES
    assert [(o["value"], o["code"]) for o in road_types["options"]] == list(
        CODES.items()
    )
    assert road_types["suffix_prefix"] == "r"
    assert road_types["omit_default"] is True
    assert road_types["depends_on"] == {"source": ["grip"]}
    assert road_types.get("min_selected", 1) == 1
    assert road_types["option_label_key_prefix"] == "vars.modal.road_type_"


def test_roads_option_labels_resolve():
    """Every option renders a translated label, not a raw dotted key."""
    for spec in param_specs("roads"):
        values = [
            option["value"] if isinstance(option, dict) else option
            for option in spec["options"]
        ]
        for value in values:
            key = spec["option_label_key_prefix"] + value
            assert i18n.t(key) != key, key


def test_builder_kwarg_defaults_equal_spec_defaults():
    """Single source of truth: a caller omitting kwargs gets the modal default."""
    sig = inspect.signature(PREDEFINED_CATALOGUE["roads"]["get_image"])
    for spec in param_specs("roads"):
        assert sig.parameters[spec["key"]].default == spec["default"], spec["key"]


def test_defaults_keep_the_bare_roads_name():
    """Saved ``roads`` layers keep resolving; defaults build the bare key."""
    assert default_param_values("roads") == {"source": "osm", "types": ALL_TYPES}
    assert build_predefined_name("roads", default_param_values("roads")) == "roads"
    assert resolve_predefined("roads") == ("roads", {})


def test_osm_with_default_types_round_trips_to_roads():
    """The only OSM name is the bare key."""
    values, bad = coerce_param_values("roads", {"source": "osm", "types": ALL_TYPES})
    assert bad is None
    assert values == {"source": "osm", "types": ALL_TYPES}
    assert build_predefined_name("roads", values) == "roads"


@pytest.mark.parametrize(
    "subset", list(_subsets()), ids=lambda s: "".join(CODES[v] for v in s)
)
def test_grip_round_trip_is_exact(subset):
    """Exhaustive: every non-empty type subset builds its name and parses back."""
    values, bad = coerce_param_values("roads", {"source": "grip", "types": subset})
    assert bad is None
    assert values == {"source": "grip", "types": subset}
    expected = _grip_name(subset)
    assert build_predefined_name("roads", values) == expected
    assert resolve_predefined(expected) == ("roads", values)


def test_canonical_names_are_distinct():
    """62 value combinations collapse to 32 names: ``roads`` + 31 GRIP variants."""
    names = set()
    for source in ("osm", "grip"):
        for subset in _subsets():
            values, bad = coerce_param_values(
                "roads", {"source": source, "types": subset}
            )
            assert bad is None
            names.add(build_predefined_name("roads", values))
    assert len(names) == 32
    assert "roads" in names and "roads_grip" in names


def test_form_order_is_canonicalised():
    """Ticking local before highway gives the same variable as the reverse."""
    values, bad = coerce_param_values(
        "roads", {"source": "grip", "types": ["local", "highway"]}
    )
    assert bad is None
    assert values["types"] == ["highway", "local"]
    assert build_predefined_name("roads", values) == "roads_grip_rhl"


def test_all_five_ticked_explicitly_is_plain_grip():
    """An explicit full selection is the default: no ``_r`` segment."""
    values, bad = coerce_param_values(
        "roads", {"source": "grip", "types": list(reversed(ALL_TYPES))}
    )
    assert bad is None
    assert build_predefined_name("roads", values) == "roads_grip"


def test_roads_segments_are_unambiguous_letters():
    """No segment of one param parses under the other's grammar."""
    source, road_types = param_specs("roads")
    source_segments = set(spec_segments(source))
    type_segments = set(spec_segments(road_types))
    assert "grip" in source_segments and source_segments <= {"osm", "grip"}
    assert {"rh", "rl", "rhpst"} <= type_segments
    for segment in type_segments:
        assert segment.startswith("r") and segment[1:]
        assert set(segment[1:]) <= set("hpstl")
    assert not source_segments & type_segments
    assert not any(segment.startswith("r") for segment in source_segments)
    for segment in source_segments | type_segments:
        assert segment.isalpha() and segment.islower(), segment


def test_grip_variants_keep_the_roads_style():
    """Map styling and legend resolve through the catalogue entry."""
    key, params, entry = entry_for_name("roads_grip_rhp")
    assert key == "roads"
    assert params["source"] == "grip"
    assert entry["vis_params"] == PREDEFINED_CATALOGUE["roads"]["vis_params"]
    assert entry["raster_type"] == "categorical"
    assert entry["temporal"] is False


# ---------------------------------------------------------------------------
# Hidden (inactive) conditional values — Review Focus 1 and 5
# ---------------------------------------------------------------------------

from gui.scripts.predefined import (  # noqa: E402
    active_param_specs,
    changed_advanced_count,
    describe_params,
)

OSM_VALUES = {"source": "osm", "types": ALL_TYPES}


def test_types_are_active_only_for_grip():
    """``depends_on`` gates the types field on the GRIP4 source."""
    assert [s["key"] for s in active_param_specs("roads", OSM_VALUES)] == ["source"]
    assert [
        s["key"]
        for s in active_param_specs("roads", {"source": "grip", "types": ALL_TYPES})
    ] == ["source", "types"]


# Group A — coercion ignores the hidden field's stale value.
@pytest.mark.parametrize(
    "stale",
    [["highway"], [], ["bogus"], None],
    ids=["narrowed", "cleared", "invalid", "absent"],
)
def test_hidden_types_never_block_or_leak_into_osm(stale):
    """GRIP → narrow/clear types → back to OSM submits plain ``roads``."""
    raw = {"source": "osm"}
    if stale is not None:
        raw["types"] = stale
    assert coerce_param_values("roads", raw) == (OSM_VALUES, None)


# Group B — build drops an inactive value even if a caller passes one.
def test_build_omits_an_inactive_types_segment():
    """Only the coerced form is normal input, but build must stay canonical."""
    assert build_predefined_name("roads", {"source": "osm", "types": ["highway"]}) == (
        "roads"
    )


# Group C — resolve rejects names carrying an inactive segment.
@pytest.mark.parametrize("name", ["roads_rh", "roads_rhpst", "roads_rl"])
def test_types_segment_without_grip_does_not_resolve(name):
    """``roads_rh`` is not producible: types only exist with GRIP4."""
    assert resolve_predefined(name) == (None, {})


# Group D — Part A canonical-name rules applied to roads.
@pytest.mark.parametrize(
    "name",
    [
        "roads_osm",  # explicit default source
        "roads_osm_rh",  # explicit default source + inactive types
        "roads_grip_rph",  # codes out of declared order
        "roads_grip_rhh",  # duplicate code
        "roads_grip_r",  # empty code list
        "roads_grip_rhpstl",  # explicit default types
        "roads_grip_rx",  # unknown code
        "roads_grip_grip",  # repeated source segment
        "roads_grip_rh_rp",  # two types segments
        "roads_grip_rh_2020",  # storage key, not a variable name
        "roads_GRIP",  # case matters
        "roads_",  # empty segment
    ],
)
def test_non_canonical_roads_names_do_not_resolve(name):
    """Hand-typed or imported names that build never emits fall through."""
    assert resolve_predefined(name) == (None, {})


# Validation of the active field.
@pytest.mark.parametrize(
    "raw",
    [
        {"source": "grip", "types": []},
        {"source": "grip", "types": ["highway", "bogus"]},
    ],
    ids=["empty", "unknown"],
)
def test_grip_types_are_validated(raw):
    """With GRIP4 the types field is live: empty or unknown is an error."""
    values, bad = coerce_param_values("roads", raw)
    assert values == {}
    assert bad is not None and bad["key"] == "types"


@pytest.mark.parametrize("source", ["bing", "", "GRIP", None])
def test_unknown_source_is_rejected(source):
    """The source must be one of the declared options, exactly."""
    values, bad = coerce_param_values("roads", {"source": source, "types": ALL_TYPES})
    assert values == {}
    assert bad is not None and bad["key"] == "source"


def test_changed_advanced_count_ignores_hidden_types():
    """The "(n changed)" badge counts only what the user can see."""
    assert changed_advanced_count("roads", OSM_VALUES) == 0
    # Types narrowed while hidden (source back to OSM) are not a change.
    assert changed_advanced_count("roads", {"source": "osm", "types": ["highway"]}) == 0
    assert changed_advanced_count("roads", {"source": "grip", "types": ALL_TYPES}) == 1
    assert (
        changed_advanced_count("roads", {"source": "grip", "types": ["highway"]}) == 2
    )


def test_describe_params_names_source_and_types():
    """The (unshown) summary helper names the source and the types."""
    assert describe_params("roads") == ""
    grip_only = describe_params("roads_grip")
    assert i18n.t("vars.modal.roads_source_grip") in grip_only
    text = describe_params("roads_grip_rhp")
    assert i18n.t("vars.modal.roads_source_grip") in text
    assert i18n.t("vars.modal.road_type_highway") in text
    assert i18n.t("vars.modal.road_type_primary") in text
    assert i18n.t("vars.modal.road_type_local") not in text


def test_details_rows_list_types_only_for_grip():
    """The Details dialog lists the source, plus the road types for GRIP4."""
    from gui.scripts.predefined import param_rows

    source_label = i18n.t("vars.modal.param_roads_source")
    types_label = i18n.t("vars.modal.param_road_types")
    assert param_rows("roads", OSM_VALUES) == [
        (source_label, i18n.t("vars.modal.roads_source_osm"))
    ]
    _, values = resolve_predefined("roads_grip_rhp")
    assert param_rows("roads", values) == [
        (source_label, i18n.t("vars.modal.roads_source_grip")),
        (
            types_label,
            "; ".join(
                i18n.t(f"vars.modal.road_type_{v}") for v in ("highway", "primary")
            ),
        ),
    ]
