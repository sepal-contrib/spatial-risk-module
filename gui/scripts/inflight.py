"""Keys currently owned by a background worker, for per-item actions.

A user can fire the same per-item action twice (double-click, a re-click
while the first run is still going). The worker pattern in this app is one
``spawn_in_context`` thread per action, so the second click would start a
second worker on the same item — two GDAL processes writing one ``.tif``,
or a layer added to the map twice. ``InflightKeys`` is the guard: the handler
``claim``s the item key and skips when the claim fails; the worker
``release``s it in ``finally``.

Both operations run under a lock. Releases happen on worker threads, and the
unlocked ``reactive.set(reactive.value - {key})`` pattern is a read-modify-
write race: two workers finishing together can each publish a set that
still contains the other's key, so a button stays disabled forever.

The set is a ``frozenset`` reactive, so a component that reads ``.value``
re-renders on every claim/release (progress bar, per-row spinner).
"""

import threading
from typing import Optional

import solara


class InflightKeys:
    """Lock-backed ``frozenset`` reactive with atomic claim/release."""

    def __init__(self, key: Optional[str] = None):
        """Create an empty in-flight set, optionally under a stable key.

        An explicit key keeps a module-level reactive stable across solara
        hot reloads (see the solara-reload-reactive-key-aliasing memory).
        """
        self._keys = solara.Reactive(frozenset(), key=key)
        self._lock = threading.Lock()

    @property
    def value(self) -> frozenset:
        """Current snapshot; reading it inside a component subscribes it."""
        return self._keys.value

    def __contains__(self, key) -> bool:
        """Return True if ``key`` is currently claimed."""
        return key in self._keys.value

    def claim(self, *keys) -> bool:
        """Atomically take ownership of every key, or of none of them.

        Returns False (and changes nothing) when any key is already owned.
        """
        wanted = frozenset(keys)
        with self._lock:
            current = self._keys.value
            if current & wanted:
                return False
            self._keys.set(current | wanted)
            return True

    def release(self, *keys) -> None:
        """Give the keys back; unknown keys are ignored."""
        with self._lock:
            self._keys.set(self._keys.value - frozenset(keys))
