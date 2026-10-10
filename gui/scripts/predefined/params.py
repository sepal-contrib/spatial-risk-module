"""Predefined-layer parameters: specs, coercion, and the name <-> params mapping."""

import re
from itertools import combinations

from gui.scripts.predefined.catalogue import PREDEFINED_CATALOGUE

_DIGITS = re.compile(r"\d+")

# ---------------------------------------------------------------------------
# Parameterised layers: name <-> params
# ---------------------------------------------------------------------------
#
# A layer may declare ``params`` (see forest_gfc). The chosen values are baked
# into the variable *name* — "forest_gfc" + {"tree_cover_threshold": 30} ->
# "forest_gfc_tc30" — so the storage key, the downloaded GeoTIFF, the processed
# variable and every derived name carry the parameterisation with no extra
# plumbing, and two settings coexist as separate variables.
#
# Nothing persists the values separately (raw GEEVars are session-only, and the
# Process step drops the ee.Image but keeps the name), so ``resolve_predefined``
# parses them back out of the name. Keep it an exact inverse of
# ``build_predefined_name``.


def param_specs(catalogue_key):
    """Parameter specs declared by a catalogue entry ([] when it has none)."""
    cat = PREDEFINED_CATALOGUE.get(catalogue_key or "")
    return list(cat.get("params", [])) if cat else []


def basic_param_specs(catalogue_key):
    """Specs rendered in the main body of the modal (``advanced`` False)."""
    return [s for s in param_specs(catalogue_key) if not s.get("advanced", False)]


def advanced_param_specs(catalogue_key):
    """Specs rendered inside the collapsed Advanced section."""
    return [s for s in param_specs(catalogue_key) if s.get("advanced", False)]


def omits_default(spec):
    """Whether the spec's name segment is dropped at its default value.

    Defaults to ``advanced``: advanced knobs leave today's names untouched,
    basic ones (``forest_gfc_tc30``) keep writing their segment.
    """
    return spec.get("omit_default", spec.get("advanced", False))


def _kind(spec):
    return spec.get("type", "int")


def _copy(value):
    return list(value) if isinstance(value, list) else value


def default_param_values(catalogue_key):
    """``{param key: default}`` for a catalogue entry ({} when unparameterised)."""
    return {spec["key"]: _copy(spec["default"]) for spec in param_specs(catalogue_key)}


def _canonical_multi(spec, selected):
    """Option values of ``selected`` in declared order, duplicates removed."""
    chosen = set(selected)
    return [o["value"] for o in spec["options"] if o["value"] in chosen]


def normalize_value(spec, value):
    """``value`` in the form compared against the spec default.

    Accepts typed values (from ``coerce_param_values`` / ``resolve_predefined``)
    and raw form values (int text, multichoice lists in any order), so the
    modal can count changed knobs before the form validates.
    """
    kind = _kind(spec)
    if kind == "multichoice":
        return _canonical_multi(spec, value or [])
    if kind == "int":
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return value
    return value


def _is_default(spec, value):
    return normalize_value(spec, value) == normalize_value(spec, spec["default"])


def coerce_param_values(catalogue_key, raw):
    """Validate raw form values for a catalogue entry.

    ``raw`` maps param key -> whatever the form holds (text fields hand back
    strings, multichoice selects hand back lists). Returns ``(values, None)``
    with every value coerced to its declared type, or ``({}, spec)`` naming
    the first param that fails validation — the caller turns that spec into a
    localized message. ``int`` params must be a whole number inside
    ``[min, max]``; ``choice`` params must be one of ``options``;
    ``multichoice`` params must be a list of option values with at least
    ``min_selected`` items, returned in declared order.
    """
    values = {}
    for spec in param_specs(catalogue_key):
        kind = _kind(spec)
        if kind == "multichoice":
            selected = raw.get(spec["key"])
            if not isinstance(selected, (list, tuple)):
                return {}, spec
            known = {o["value"] for o in spec["options"]}
            if any(item not in known for item in selected):
                return {}, spec
            canonical = _canonical_multi(spec, selected)
            if len(canonical) < spec.get("min_selected", 1):
                return {}, spec
            values[spec["key"]] = canonical
            continue
        text = str(raw.get(spec["key"], "")).strip()
        if kind == "choice":
            if text not in spec["options"]:
                return {}, spec
            values[spec["key"]] = text
            continue
        try:
            value = int(text)
        except (TypeError, ValueError):
            return {}, spec
        if not (spec["min"] <= value <= spec["max"]):
            return {}, spec
        values[spec["key"]] = value
    return values, None


def _segment(spec, value):
    """The name segment ``spec`` writes for ``value``."""
    if _kind(spec) == "multichoice":
        codes = {o["value"]: o["code"] for o in spec["options"]}
        body = "".join(codes[v] for v in _canonical_multi(spec, value))
    else:
        body = str(value)
    return f"{spec['suffix_prefix']}{body}"


def build_predefined_name(catalogue_key, values):
    """Variable name for a catalogue key plus its parameter values.

    ``("forest_gfc", {"tree_cover_threshold": 30})`` -> ``"forest_gfc_tc30"``.
    Segments follow spec order. An ``omit_default`` spec writes its segment only
    when its value differs from the default (a missing value counts as the
    default), so ``("rivers", defaults)`` stays ``"rivers"``. The year is NOT
    part of the name — ``entry_key`` appends it when forming the storage key.
    """
    parts = [catalogue_key]
    for spec in param_specs(catalogue_key):
        if omits_default(spec):
            value = values.get(spec["key"], spec["default"])
            if _is_default(spec, value):
                continue
        else:
            value = values[spec["key"]]
        parts.append(_segment(spec, value))
    return "_".join(parts)


def parse_segment(spec, segment):
    """Parse one name segment against ``spec``.

    Returns ``(True, value)`` when ``segment`` is one ``spec`` can emit, else
    ``(False, None)``.

    Multichoice bodies must list known codes once each, in declared order, and
    hold at least ``min_selected`` codes. Int bodies are plain digits (range is
    not checked here: the canonical rebuild in ``resolve_predefined`` and the
    catalogue lint keep names in range).
    """
    prefix = spec["suffix_prefix"]
    if not segment.startswith(prefix):
        return False, None
    body = segment[len(prefix) :]
    kind = _kind(spec)
    if kind == "choice":
        return (True, body) if body in spec["options"] else (False, None)
    if kind == "multichoice":
        codes = [o["code"] for o in spec["options"]]
        picked = [code for code in codes if code in body]
        if (
            not body
            or "".join(picked) != body
            or len(picked) < spec.get("min_selected", 1)
        ):
            return False, None
        return True, [o["value"] for o in spec["options"] if o["code"] in body]
    if _DIGITS.fullmatch(body) is None:
        return False, None
    return True, int(body)


def spec_segments(spec):
    """Every segment ``spec`` can emit (catalogue lint: ambiguity checks)."""
    prefix = spec["suffix_prefix"]
    kind = _kind(spec)
    if kind == "choice":
        return [f"{prefix}{option}" for option in spec["options"]]
    if kind == "multichoice":
        codes = [o["code"] for o in spec["options"]]
        out = []
        for size in range(spec.get("min_selected", 1), len(codes) + 1):
            out.extend(prefix + "".join(combo) for combo in combinations(codes, size))
        return out
    return [f"{prefix}{value}" for value in range(spec["min"], spec["max"] + 1)]


def _walk(specs, segments):
    """Values parsed from ``segments`` in spec order, or None on mismatch.

    A cursor moves over the segments. A segment matching the current spec is
    consumed; otherwise an ``omit_default`` spec takes its default and the
    walk moves on, and a required spec fails. Leftover segments fail. The
    catalogue lint guarantees no segment parses under two specs of one entry,
    so the greedy walk is unambiguous.
    """
    values = {}
    cursor = 0
    for spec in specs:
        if cursor < len(segments):
            matched, value = parse_segment(spec, segments[cursor])
            if matched:
                values[spec["key"]] = value
                cursor += 1
                continue
        if not omits_default(spec):
            return None
        values[spec["key"]] = _copy(spec["default"])
    return values if cursor == len(segments) else None


def resolve_predefined(name):
    """Split a variable name into its catalogue key and parameter values.

    The inverse of ``build_predefined_name``::

        "altitude"                       -> ("altitude", {})
        "rivers"                         -> ("rivers", {})
        "rivers_wlmc"                    -> ("rivers", {"classes": [...]})
        "forest_gfc_tc30"                -> ("forest_gfc", {"tree_cover_threshold": 30})
        "temperature_2m_median"          -> ("temperature_2m",
                                             {"aggregation": "median"})
        "loss_forest_gfc_tc30_2015_2020" -> (None, {})

    A bare catalogue key keeps the legacy ``(key, {})`` result (callers fill
    defaults with ``default_param_values``). Any other match returns every
    param value, defaults filled in for omitted segments.

    Returns ``(None, {})`` for anything that is not a catalogue variable —
    custom variables and post-process outputs — so they keep falling through to
    their own defaults. Use this instead of ``PREDEFINED_CATALOGUE.get(name)``
    anywhere a *variable* name (rather than a catalogue key) is looked up.

    Injective: the parsed values are rebuilt into a name and any mismatch is
    rejected, so only the canonical form resolves (no zero-padded ints, no
    out-of-order codes, no explicit default segments).
    """
    if not name:
        return None, {}
    if name in PREDEFINED_CATALOGUE:
        return name, {}

    # Longest key first: a shorter key must never shadow a longer one that also
    # prefixes the name.
    for key in sorted(PREDEFINED_CATALOGUE, key=len, reverse=True):
        specs = param_specs(key)
        if not specs or not name.startswith(f"{key}_"):
            continue
        values = _walk(specs, name[len(key) + 1 :].split("_"))
        if values is not None and build_predefined_name(key, values) == name:
            return key, values
    return None, {}


def changed_advanced_count(catalogue_key, values):
    """How many advanced params in ``values`` differ from their defaults.

    Accepts typed or raw form values; keys absent from ``values`` count as
    unchanged. Drives the "(n changed)" hint and auto-opening on edit.
    """
    return sum(
        1
        for spec in advanced_param_specs(catalogue_key)
        if spec["key"] in values and not _is_default(spec, values[spec["key"]])
    )


def _value_text(spec, value, t):
    kind = _kind(spec)
    if kind == "multichoice":
        return ", ".join(
            t(spec["option_label_key_prefix"] + v)
            for v in _canonical_multi(spec, value)
        )
    if kind == "choice":
        return t(spec["option_label_key_prefix"] + value)
    return str(value)


def param_rows(catalogue_key, values):
    """``(label, text)`` for every param of ``catalogue_key`` found in ``values``.

    The Details dialog lists them one field each, defaults included.
    """
    from gui.i18n import t

    return [
        (t(spec["label_key"]), _value_text(spec, values[spec["key"]], t))
        for spec in param_specs(catalogue_key)
        if spec["key"] in values
    ]


def describe_params(name):
    """Translated summary of the advanced values that differ from the defaults.

    ``"rivers_wlmc"`` -> ``"Water classes: Large lake or river, Major river,
    Canal"``. Empty for default settings and non-catalogue names.
    """
    from gui.i18n import t

    key, values = resolve_predefined(name)
    if key is None or not values:
        return ""
    parts = [
        f"{t(spec['label_key'])}: {_value_text(spec, values[spec['key']], t)}"
        for spec in advanced_param_specs(key)
        if not _is_default(spec, values[spec["key"]])
    ]
    return " · ".join(parts)
