"""The reference dialog refuses input that would only fail inside GDAL."""

from gui.scripts.process_actions import validate_projection


def test_sentinels_cover_every_message_key():
    """Each sentinel validate_projection can return has a tiles.json message."""
    import json
    from pathlib import Path

    messages = json.loads(
        Path("gui/messages/en/tiles.json").read_text(encoding="utf-8")
    )["tiles"]["process"]
    for sentinel, key in (
        ("need_epsg", "error_need_epsg"),
        ("bad_epsg", "error_bad_epsg"),
        ("bad_resolution", "error_bad_resolution"),
        ("geographic_crs", "warn_geographic_crs"),
    ):
        assert key in messages, f"{sentinel} has no message key {key}"


def test_validate_reference_blocks_a_bogus_epsg():
    """The dialog's validate hook must return a message, not None."""
    error, _ = validate_projection("abcd", "30")
    assert error is not None
