"""Predefined-layer parameters: specs, coercion, and the name <-> params mapping."""

import re

from gui.scripts.predefined.catalogue import PREDEFINED_CATALOGUE

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


def default_param_values(catalogue_key):
    """``{param key: default}`` for a catalogue entry ({} when unparameterised)."""
    return {spec["key"]: spec["default"] for spec in param_specs(catalogue_key)}


def coerce_param_values(catalogue_key, raw):
    """Validate raw form values for a catalogue entry.

    ``raw`` maps param key -> whatever the form holds (text fields hand back
    strings). Returns ``(values, None)`` with every value coerced to its
    declared type, or ``({}, spec)`` naming the first param that fails
    validation — the caller turns that spec into a localized message. ``int``
    params must be a whole number inside ``[min, max]``; ``choice`` params
    must be one of ``options`` (kept as ``str``, no numeric coercion).
    """
    values = {}
    for spec in param_specs(catalogue_key):
        text = str(raw.get(spec["key"], "")).strip()
        if spec.get("type", "int") == "choice":
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


def build_predefined_name(catalogue_key, values):
    """Variable name for a catalogue key plus its parameter values.

    ``("forest_gfc", {"tree_cover_threshold": 30})`` -> ``"forest_gfc_tc30"``.
    An unparameterised layer keeps its bare key. The year is NOT part of the
    name — ``entry_key`` appends it when forming the storage key.
    """
    parts = [catalogue_key]
    for spec in param_specs(catalogue_key):
        parts.append(f"{spec['suffix_prefix']}{values[spec['key']]}")
    return "_".join(parts)


def resolve_predefined(name):
    """Split a variable name into its catalogue key and parameter values.

    The inverse of ``build_predefined_name``::

        "altitude"                       -> ("altitude", {})
        "forest_gfc_tc30"                -> ("forest_gfc", {"tree_cover_threshold": 30})
        "temperature_2m_median"         -> ("temperature_2m", {"aggregation": "median"})
        "loss_forest_gfc_tc30_2015_2020" -> (None, {})

    Returns ``(None, {})`` for anything that is not a catalogue variable —
    custom variables and post-process outputs — so they keep falling through to
    their own defaults. Use this instead of ``PREDEFINED_CATALOGUE.get(name)``
    anywhere a *variable* name (rather than a catalogue key) is looked up.

    The inverse property holds for every name ``build_predefined_name``
    produces; it is not injective over arbitrary strings, because a
    non-canonical zero-padded suffix parses to the same values as its canonical
    form (``forest_gfc_tc030`` -> the same params as ``forest_gfc_tc30``). No
    such name is reachable through the UI: values are coerced to ``int`` before
    the name is built, so only the canonical form is ever emitted.
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
        segments = name[len(key) + 1 :].split("_")
        if len(segments) != len(specs):
            continue
        values = {}
        for spec, segment in zip(specs, segments):
            if spec.get("type", "int") == "choice":
                if segment in spec["options"]:
                    values[spec["key"]] = segment
                    continue
                values = None
                break
            match = re.fullmatch(rf"{re.escape(spec['suffix_prefix'])}(\d+)", segment)
            if match is None:
                values = None
                break
            values[spec["key"]] = int(match.group(1))
        if values is not None:
            return key, values
    return None, {}
