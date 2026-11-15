"""Anchor SEPAL file-picker results to $HOME.

pysepal gap: 4.1.0's FileInput lists files through the user-files API, whose
entries are relative to /home/sepal-user, so a consumer gets
``downloads/x.tif`` — which Path() resolves against the read-only app CWD on
SEPAL.

Fixed upstream by openforis/pysepal#1078 and #1079 (absolute paths). Remove
this module and use ``Path(value)`` at its four call sites when the pin
includes them (4.2.0).
"""

from pathlib import Path


def resolve_picked_path(value: str | None, sepal_client) -> Path | None:
    """Anchor a file-picker value to $HOME when it came from a sepal_client.

    Args:
        value: the FileInput's raw ``v_model`` string, or None/empty.
        sepal_client: the client passed to the FileInputComponent, or None
            for a purely local picker.

    Returns:
        None when ``value`` is empty. An absolute ``value`` is returned as a
        ``Path`` untouched. A relative ``value`` is anchored under
        ``Path.home()`` only when ``sepal_client`` is set, since only the
        remote listing is home-relative. The result is idempotent: feeding
        the absolute path back in (as ``str(path)``) resolves to itself.
    """
    if not value:
        return None
    path = Path(value)
    if sepal_client is not None and not path.is_absolute():
        return Path.home() / path
    return path
