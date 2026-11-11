"""The GUI translator follows pysepal's kernel locale (``pysepal.i18n``).

pysepal 4 owns the locale: its header selector writes it with ``set_locale``
and ``current_locale()`` reads it. ``gui.i18n`` keeps its own
``Translator``-backed ``t()`` and only takes the locale from there — a lazy
rebuild reads it, and the ``use_app_locale()`` hook swaps the translator when
it changes, so every component that calls ``t()`` re-renders translated.

Every test restores English in a ``finally``: the locale is process-wide
outside a Solara kernel, so a leaked Spanish locale would translate the
English assertions of every later test.
"""

import json

import reacton
import solara
from pysepal.i18n import set_locale

from gui import i18n


def _catalog_title(locale: str) -> str:
    """Return ``app.title`` straight from a locale's shell catalogue."""
    shell = json.loads((i18n.MESSAGES_DIR / locale / "shell.json").read_text())
    return shell["app"]["title"]


def test_translator_follows_pysepal_locale():
    """A rebuilt translator targets whatever locale pysepal currently holds."""
    try:
        set_locale("es-ES")
        i18n.reset_translator()
        assert i18n.get_translator()._target == "es-ES"
        assert i18n.t("app.title") != _catalog_title("en")

        set_locale("en")
        i18n.reset_translator()
        assert i18n.get_translator()._target == "en"
    finally:
        set_locale("en")
        i18n.reset_translator()


def test_use_app_locale_retranslates_on_a_locale_switch():
    """A component using the hook re-renders in the locale pysepal switches to.

    The translator is cached, so only the hook's effect can swap it: without
    that effect the re-render the locale change triggers would still read the
    English translator built on the first render.
    """
    seen = []

    @solara.component
    def Probe():
        i18n.use_app_locale()
        seen.append(i18n.t("app.title"))
        return solara.Text(seen[-1])

    try:
        set_locale("en")
        i18n.reset_translator()
        _box, rc = reacton.render(Probe(), handle_error=False)
        try:
            assert seen[-1] == _catalog_title("en")

            set_locale("es-ES")

            assert seen[-1] == _catalog_title("es-ES")
        finally:
            rc.close()
    finally:
        set_locale("en")
        i18n.reset_translator()
