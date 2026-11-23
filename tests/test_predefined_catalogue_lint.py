"""Catalogue-wide rules every predefined entry must satisfy.

Each later part adds layers and params; these checks keep the name grammar
unambiguous and the four-digit year contract intact without per-layer tests.
"""

import importlib
import inspect
import itertools
import re

import pytest
from _ee_fake import install

from gui import i18n
from gui.scripts.predefined import (
    DYNAMIC_ATTRS,
    GROUP_ORDER,
    PREDEFINED_CATALOGUE,
    resolved_entry,
    validate_entry,
)
from gui.scripts.predefined.params import (
    build_predefined_name,
    default_param_values,
    inactive_to_defaults,
    omits_default,
    param_specs,
    parse_segment,
    resolve_predefined,
    spec_segments,
)

LETTERS = re.compile(r"[a-z]+")
PREFIX = re.compile(r"[a-z]*")
KEYS = sorted(PREDEFINED_CATALOGUE)


@pytest.mark.parametrize("key", KEYS)
def test_entry_has_a_known_group(key):
    """Every entry declares a group listed in GROUP_ORDER."""
    assert PREDEFINED_CATALOGUE[key].get("group") in GROUP_ORDER


def test_every_group_has_an_entry_and_a_label():
    """No empty groups, and every group has an i18n label."""
    used = {meta["group"] for meta in PREDEFINED_CATALOGUE.values()}
    assert used == set(GROUP_ORDER)
    for group in GROUP_ORDER:
        label_key = f"vars.predefined_group.{group}"
        assert i18n.t(label_key) != label_key, group


@pytest.mark.parametrize("key", KEYS)
def test_spec_shapes(key):
    """Param specs are well-formed and can never emit a four-digit suffix."""
    specs = param_specs(key)
    assert len({s["key"] for s in specs}) == len(specs), "duplicate param keys"
    for spec in specs:
        kind = spec.get("type", "int")
        where = f"{key}.{spec['key']}"
        assert kind in ("int", "choice", "multichoice"), where
        assert PREFIX.fullmatch(spec["suffix_prefix"]), where
        if kind == "int":
            assert spec["max"] <= 999, f"{where} allows a four-digit suffix"
            assert spec["min"] <= spec["default"] <= spec["max"], where
        elif kind == "choice":
            for option in spec["options"]:
                assert LETTERS.fullmatch(option), f"{where}: {option!r}"
            assert spec["default"] in spec["options"], where
        else:
            codes = [o["code"] for o in spec["options"]]
            values = [o["value"] for o in spec["options"]]
            assert all(re.fullmatch(r"[a-z]", c) for c in codes), where
            assert len(set(codes)) == len(codes), f"{where}: duplicate codes"
            assert all(LETTERS.fullmatch(v) for v in values), where
            assert len(set(values)) == len(values), where
            default = spec["default"]
            assert default == [
                v for v in values if v in default
            ], f"{where}: default not in declared order"
            assert len(default) >= spec.get("min_selected", 1), where


@pytest.mark.parametrize("key", KEYS)
def test_no_segment_parses_under_two_specs(key):
    """The greedy resolver is only correct if segments are unambiguous."""
    specs = param_specs(key)
    for a, b in itertools.permutations(specs, 2):
        for segment in spec_segments(a):
            matched, _ = parse_segment(b, segment)
            assert not matched, (key, a["key"], b["key"], segment)


@pytest.mark.parametrize("key", KEYS)
def test_builder_kwarg_defaults_match_catalogue(key):
    """The builder's kwarg defaults equal the catalogue spec defaults."""
    sig = inspect.signature(PREDEFINED_CATALOGUE[key]["get_image"])
    for spec in param_specs(key):
        default = sig.parameters[spec["key"]].default
        if spec.get("type", "int") == "multichoice":
            default = list(default)
        assert default == spec["default"], (key, spec["key"])


@pytest.mark.parametrize("key", KEYS)
def test_param_labels_resolve(key):
    """Every param label, hint and option label has an i18n string."""
    for spec in param_specs(key):
        for label_key in (spec["label_key"], spec["hint_key"]):
            assert i18n.t(label_key) != label_key, (key, label_key)
        kind = spec.get("type", "int")
        if kind == "choice":
            options = spec["options"]
        elif kind == "multichoice":
            options = [o["value"] for o in spec["options"]]
        else:
            options = []
        for option in options:
            label_key = spec["option_label_key_prefix"] + option
            assert i18n.t(label_key) != label_key, (key, label_key)


def _value_space(spec):
    """Representative values: all options/subsets; int min, default, max."""
    kind = spec.get("type", "int")
    if kind == "choice":
        return list(spec["options"])
    if kind == "multichoice":
        return [parse_segment(spec, seg)[1] for seg in spec_segments(spec)]
    return sorted({spec["min"], spec["default"], spec["max"]})


@pytest.mark.parametrize("key", KEYS)
def test_exhaustive_roundtrip(key):
    """Build and resolve are inverses over the representative value space."""
    specs = param_specs(key)
    spaces = [_value_space(spec) for spec in specs]
    for combo in itertools.product(*spaces):
        values = inactive_to_defaults(
            key, {spec["key"]: value for spec, value in zip(specs, combo)}
        )
        name = build_predefined_name(key, values)
        assert not re.search(r"\d{4}", name), name
        if name == key:
            assert values == default_param_values(key)
            assert resolve_predefined(name) == (key, {})
        else:
            assert resolve_predefined(name) == (key, values), name


@pytest.mark.parametrize("key", KEYS)
def test_default_name_unchanged_for_advanced_only_entries(key):
    """An entry whose params are all omittable keeps its bare name at defaults."""
    specs = param_specs(key)
    if specs and all(omits_default(s) for s in specs):
        assert build_predefined_name(key, default_param_values(key)) == key


@pytest.mark.parametrize("key", KEYS)
def test_depends_on_shape(key):
    """A conditional param is omittable and controlled by an earlier choice."""
    specs = param_specs(key)
    for index, spec in enumerate(specs):
        depends_on = spec.get("depends_on") or {}
        where = f"{key}.{spec['key']}"
        if depends_on:
            assert omits_default(spec), f"{where}: depends_on needs omit_default"
        earlier = {s["key"]: s for s in specs[:index]}
        for controller, allowed in depends_on.items():
            assert controller in earlier, f"{where}: {controller!r} must come first"
            ctrl = earlier[controller]
            assert ctrl.get("type", "int") == "choice", where
            assert allowed, where
            assert set(allowed) <= set(ctrl["options"]), where


def _combos(key):
    """Every distinct full value combination (inactive params at default)."""
    specs = param_specs(key)
    seen = []
    for combo in itertools.product(*[_value_space(spec) for spec in specs]):
        values = inactive_to_defaults(
            key, {spec["key"]: value for spec, value in zip(specs, combo)}
        )
        if values not in seen:
            seen.append(values)
    return seen or [{}]


YEAR_MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


@pytest.mark.parametrize("key", KEYS)
def test_only_dynamic_attrs_are_callables(key):
    """Only dynamic attrs are callables."""
    for attr, value in PREDEFINED_CATALOGUE[key].items():
        if callable(value) and attr not in ("get_image", "validate"):
            assert attr in DYNAMIC_ATTRS, (key, attr)


@pytest.mark.parametrize("key", KEYS)
def test_dynamic_fields_resolve_for_every_combination(key):
    """Dynamic fields resolve for every combination."""
    for values in _combos(key):
        entry = resolved_entry(key, values)
        where = (key, values)
        assert entry["raster_type"] in ("categorical", "continuous"), where
        if entry.get("temporal"):
            years = entry.get("years")
            assert years, where
            assert all(isinstance(y, int) for y in years), where
            assert years == sorted(set(years)), where
        scale = entry.get("default_scale")
        assert scale is None or scale > 0, where
        vis = entry.get("vis_params")
        if vis is not None:
            assert isinstance(vis, dict) and vis.get("palette"), where
        unit = entry.get("unit_key")
        if unit:
            assert i18n.t(unit, value=1) != unit, where
        for class_key in entry.get("legend_class_keys") or ():
            assert i18n.t(class_key) != class_key, where


@pytest.mark.parametrize(
    "key", [k for k in KEYS if "availability" in PREDEFINED_CATALOGUE[k]]
)
def test_availability_shape(key):
    """Availability shape."""
    entry = PREDEFINED_CATALOGUE[key]
    windows = entry["availability"]
    by = {s["key"]: s for s in param_specs(key)}
    selector = next((by[k] for k in ("source", "sensor") if k in by), None)
    if selector is not None:
        assert set(windows) == set(selector["options"]), key
    else:
        assert len(windows) == 1, key
    for first, last in windows.values():
        assert YEAR_MONTH.fullmatch(first) and YEAR_MONTH.fullmatch(last), key
        assert first <= last, key
    assert callable(entry.get("validate")), key


@pytest.mark.parametrize(
    "key", [k for k in KEYS if "availability" in PREDEFINED_CATALOGUE[k]]
)
def test_offered_years_pass_validate(key):
    """The year list never offers a year validate refuses for lack of data."""
    for values in _combos(key):
        years = resolved_entry(key, values)["years"]
        for year in years:
            err = validate_entry(key, values, year)
            assert err is None or err[0] != "vars.modal.error_period_unavailable", (
                key,
                values,
                year,
            )
        err = validate_entry(key, values, years[-1] + 1)
        assert err is not None, (key, values, "the year after the list must fail")


@pytest.mark.parametrize(
    "key", [k for k in KEYS if "availability" in PREDEFINED_CATALOGUE[k]]
)
def test_every_combination_builds_against_a_fake_ee(monkeypatch, key):
    """Each period layer's builder runs for every value combination.

    ``ee`` is replaced on the module that owns the builder, so no Earth Engine
    session is needed; combinations the entry's ``validate`` refuses for a
    reason other than availability must raise ValueError instead.
    """
    entry = PREDEFINED_CATALOGUE[key]
    module = importlib.import_module(entry["get_image"].__module__)
    install(monkeypatch, module)
    for values in _combos(key):
        year = resolved_entry(key, values)["years"][-1]
        refused = validate_entry(key, values, year)
        if refused is not None:
            with pytest.raises(ValueError):
                entry["get_image"]("AOI", year, **values)
        else:
            entry["get_image"]("AOI", year, **values)
