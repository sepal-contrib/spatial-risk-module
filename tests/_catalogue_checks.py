"""Small shared helpers for the Part B2 layer-knob tests.

Entry lint (kwarg defaults, option grammar, depends_on, i18n) lives in
tests/test_predefined_catalogue_lint.py and covers every entry already;
locale key parity lives in tests/test_i18n.py.
"""

import itertools
import json
import re

from gui import i18n
from gui.scripts.predefined import build_predefined_name, resolve_predefined

LOCALES = ("en", "es-ES", "fr", "pt-BR")


def _flat(data, prefix=""):
    out = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flat(value, dotted))
        else:
            out[dotted] = value
    return out


def locale_messages(locale):
    """Every dotted key -> text of one locale, merged across its JSON files."""
    merged = {}
    for path in sorted((i18n.MESSAGES_DIR / locale).glob("*.json")):
        merged.update(_flat(json.loads(path.read_text())))
    return merged


def multichoice_subsets(spec):
    """Every selection a multichoice spec accepts, in declared option order."""
    values = [o["value"] for o in spec["options"]]
    for size in range(spec.get("min_selected", 1), len(values) + 1):
        for combo in itertools.combinations(values, size):
            yield list(combo)


def assert_roundtrip(catalogue_key, values):
    """Build a name from full ``values``, check its shape and that it resolves back.

    The all-defaults name is the bare key, which resolves to ``(key, {})``.
    """
    name = build_predefined_name(catalogue_key, values)
    assert re.fullmatch(r"[a-z0-9_]+", name), name
    assert not re.search(r"\d{4}", name), name
    expected = {} if name == catalogue_key else values
    assert resolve_predefined(name) == (catalogue_key, expected), (name, values)
    return name
