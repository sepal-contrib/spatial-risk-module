"""Internationalization for the Spatial Risk GUI.

``gui/messages/<locale>/*.json`` is bound once with pysepal's ``catalog()``.
Its ``msg()`` reads the kernel-scoped locale (``pysepal.i18n.current_locale``)
on every call, which the language selector that ``MapApp.element(locales=...)``
mounts writes. Reading it during a render subscribes the component, so a
language switch re-renders every component that calls t() in place. No string
may be resolved at import time — always call t()/plural() inside a component
render or an event handler in the same Solara context; a pool thread reads the
process default (English).
"""

import logging
from pathlib import Path

from pysepal.i18n import MessageFormatError, MissingMessageError, catalog

MESSAGES_DIR = Path(__file__).parent / "messages"

# Strict, so a missing key reaches t() as an exception; t() renders the key.
_messages = catalog(MESSAGES_DIR)
_log = logging.getLogger(__name__)


def t(key: str, /, **fmt) -> str:
    """Resolve a dotted key in the current locale (falling back to English).

    The value is interpolated with ``fmt``. A key missing from the catalog, or
    a placeholder left without a value, returns the key string so a gap
    degrades visibly instead of crashing the GUI.

    ``key`` is positional-only so a catalog value may carry a ``{key}``
    placeholder passed as a keyword (``key=...``) without colliding with this
    lookup parameter.
    """
    try:
        return _messages.msg(key, **fmt)
    except (MissingMessageError, MessageFormatError) as exc:
        _log.warning("i18n: %s", exc)
        return key


def plural(n: int, one_key: str, other_key: str, /, **fmt) -> str:
    """Pick the singular vs plural key for a count and interpolate n.

    The selector args are positional-only so a catalog value may carry an
    ``{n}``/``{one_key}``/``{other_key}`` placeholder without colliding.
    """
    fmt.setdefault("n", n)
    return t(one_key if n == 1 else other_key, **fmt)


def app_available_locales() -> list:
    """Locale codes shipped under gui/messages/ (drives the selector)."""
    return list(_messages.available_locales())
