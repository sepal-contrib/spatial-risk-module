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


def _blocking_sentinels() -> set:
    """Every non-``None`` first element ``validate_projection`` can return.

    Read out of the function's own AST rather than listed here: a list would
    only ever be as fresh as the last person who remembered to extend it, and
    forgetting is the failure this test exists to catch.
    """
    import ast
    import inspect

    from gui.scripts import process_actions

    tree = ast.parse(inspect.getsource(process_actions.validate_projection))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
            continue
        first = node.value.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    assert found, "no sentinels found — the AST walk stopped matching the function"
    return found


def _validation_message_map() -> dict:
    """``ProcessTile``'s ``_VALIDATION_MESSAGES``, which is a local of the component."""
    import ast
    import inspect

    from gui.tile import process_tile

    tree = ast.parse(inspect.getsource(process_tile.ProcessTile))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_VALIDATION_MESSAGES"
            for target in node.targets
        ):
            return {
                k.value: v.value for k, v in zip(node.value.keys, node.value.values)
            }
    raise AssertionError("_VALIDATION_MESSAGES is no longer defined in ProcessTile")


def test_every_blocking_sentinel_has_a_message():
    """``t(_VALIDATION_MESSAGES[error])`` is a raw subscript inside a click handler.

    A fourth blocking sentinel added to ``validate_projection`` without a
    message would raise ``KeyError`` there — inside ``process_kernel_messages``,
    where an escaping exception is a session-level failure rather than a
    refusal the user can read. Caught here, at the moment the sentinel is
    added, instead of defended with a ``.get()`` that would hide it at runtime.
    """
    from gui.i18n import t

    sentinels = _blocking_sentinels()
    mapped = _validation_message_map()
    missing = sorted(sentinels - set(mapped))
    assert not missing, (
        f"validate_projection can return {missing} with no entry in "
        "_VALIDATION_MESSAGES — the click handler would raise KeyError"
    )
    for sentinel, key in mapped.items():
        assert t(key, epsg="x") != key, f"{sentinel} maps to the untranslated {key!r}"
