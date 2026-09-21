"""Harmonization reports through toasts; its job failure gets a localized key.

Before this change the harmonization catch stored a raw str(exc) with no i18n
key at all — the only failure path in the tile that was never translated.
"""

import inspect

from gui.i18n import t
from gui.tile.process_tile import ProcessTile


def test_process_tile_has_no_process_error_parameter():
    """ProcessTile signature no longer accepts a process_error parameter."""
    sig = inspect.signature(ProcessTile.f)
    assert "process_error" not in sig.parameters


def test_process_tile_toasts_every_failure_path():
    """Every toast ProcessTile raises is timed, and none uses process_error.

    ``warning`` counts alongside ``error``: a refused reference submit is not a
    failure but it is still a toast, and an untimed one would vanish at the
    default timeout — the thing ERROR_TOAST_TIMEOUT exists to prevent.
    """
    src = inspect.getsource(ProcessTile.f)
    assert "process_error" not in src
    assert (
        'error_format=lambda exc: t("tiles.process.error_processing", exc=exc)' in src
    )
    toasts = src.count("notifications.error(") + src.count("notifications.warning(")
    assert toasts == src.count("ERROR_TOAST_TIMEOUT")
    for key in (
        "tiles.process.error_download_first",
        "tiles.process.error_auto_utm",
        "tiles.process.error_set_base",
        "tiles.process.reference_already_running",
    ):
        assert key in src


def test_processing_error_key_resolves_and_interpolates():
    """The new i18n key exists and interpolates the exception text."""
    rendered = t("tiles.process.error_processing", exc="no CRS on source")
    assert "tiles.process.error_processing" != rendered  # key actually exists
    assert "no CRS on source" in rendered
