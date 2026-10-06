"""Internationalization for the Spatial Risk GUI.

Reuses pysepal's Translator (JSON locale folders with es-ES->en fallback).
The active locale comes from ``pysepal.i18n`` and is kernel-scoped: the
language selector that ``MapApp.element(locales=...)`` mounts writes it, and
persists a pick in browser localStorage; a first load with no stored pick
follows navigator.language, then English. The translator is always built with
that locale as an explicit target, and ``use_app_locale()`` (called by Page)
swaps the translator reactive when the locale changes, live-re-rendering every
component that calls t() during render. No string may be resolved at import
time — always call t()/plural() inside a component render.

Keeping the Translator-backed t() rather than pysepal's ``catalog()``/``msg()``
is the smallest migration diff: t()'s missing-key fallback and the tests that
inject ``i18n._translator`` keep working. Moving to catalog()/msg() is a
feasible follow-up (a missing key would then raise MissingMessageError).
"""

from pathlib import Path

import solara
from pysepal.i18n import current_locale
from pysepal.translator import Translator

MESSAGES_DIR = Path(__file__).parent / "messages"

# Active translator for THIS session. Re-created on kernel start (reload-aware).
_translator = solara.reactive(None)


def _current_locale() -> str:
    return current_locale() or "en"


def get_translator() -> Translator:
    """Return this session's translator, building it on first use."""
    if _translator.value is None:
        _translator.value = Translator(MESSAGES_DIR, target=_current_locale())
    return _translator.value


def set_app_locale(code: str) -> None:
    """Swap the active translator to ``code`` (live, no reload)."""
    _translator.value = Translator(MESSAGES_DIR, target=code or "en")


def reset_translator() -> None:
    """Drop the cached translator.

    It lazily rebuilds in pysepal's current locale. Call from on_kernel_start.
    """
    _translator.value = None


def use_app_locale() -> str:
    """Render-time hook: re-create the translator when pysepal's locale changes.

    Reading ``current_locale()`` subscribes the calling component, so a switch
    re-renders it and the effect swaps the translator to the new locale.
    """
    locale = current_locale()

    def _swap():
        set_app_locale(locale)

    solara.use_effect(_swap, [locale])
    return locale


def t(key: str, /, **fmt) -> str:
    """Resolve a dotted key against the active catalog (es->en->key).

    The value is interpolated with str.format(**fmt). A key missing in both
    languages returns the key string so a gap degrades visibly instead of
    crashing the GUI.

    ``key`` is positional-only so a catalog value may carry a ``{key}``
    placeholder passed as a keyword (``key=...``) without colliding with this
    lookup parameter.
    """
    node = get_translator()
    try:
        for part in key.split("."):
            node = node[part]
        text = str(node)
        return text.format(**fmt) if fmt else text
    except Exception:
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
    return sorted(p.name for p in MESSAGES_DIR.glob("[!._]*") if p.is_dir())
