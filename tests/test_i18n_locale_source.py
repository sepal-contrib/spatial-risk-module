"""The GUI strings follow pysepal's kernel locale (``pysepal.i18n``).

pysepal 4 owns the locale: its header selector writes it with ``set_locale``
and ``current_locale()`` reads it. ``gui.i18n.t()`` resolves through pysepal's
``catalog()``, whose ``msg()`` reads that locale on every call and subscribes
the rendering component, so a switch re-renders every component that calls
``t()`` translated.

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


def test_t_defaults_to_english_without_a_locale():
    """With no locale set, t() is English: pysepal 4 reads no machine config."""
    assert i18n.t("app.title") == _catalog_title("en")


def test_t_follows_pysepal_locale():
    """t() resolves in whatever locale pysepal currently holds."""
    try:
        set_locale("es-ES")
        assert i18n.t("app.title") == _catalog_title("es-ES")
        assert i18n.t("app.title") != _catalog_title("en")

        set_locale("en")
        assert i18n.t("app.title") == _catalog_title("en")
    finally:
        set_locale("en")


def test_a_component_retranslates_on_a_locale_switch():
    """A component calling t() re-renders in the locale pysepal switches to."""
    seen = []

    @solara.component
    def Probe():
        seen.append(i18n.t("app.title"))
        return solara.Text(seen[-1])

    try:
        set_locale("en")
        _box, rc = reacton.render(Probe(), handle_error=False)
        try:
            assert seen[-1] == _catalog_title("en")

            set_locale("es-ES")

            assert seen[-1] == _catalog_title("es-ES")
        finally:
            rc.close()
    finally:
        set_locale("en")
