"""Two sample map toggles finishing together must not lose an on-map key.

``samples_on_map`` is updated by read-modify-write from the toggle workers
(and from the remove handler). Without a lock, worker B can read the set
between worker A's read and A's write, and one of the two keys is dropped:
the layer is drawn but its switch renders off. The other on-map sets
(variables, derived, predictions) are lock-guarded; this pins sampling to it.
"""

import threading
import types

import pytest
import solara

from gui.tile import sampling_tile

TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _drain_sampling_state():
    """Leave no claim or on-map key behind in sampling_tile's module-level state."""
    yield
    sampling_tile.samples_pending.release(*sampling_tile.samples_pending.value)
    sampling_tile.samples_on_map.set(set())


class _SlowFirstWrite:
    """Reactive stand-in whose *first* ``set`` blocks until released.

    Worker A reads the set, then parks inside ``set``; worker B is started
    meanwhile. Without the lock B reads the still-empty set and writes its own
    key, which A's delayed write then overwrites. With the lock B cannot read
    until A's write has landed.
    """

    def __init__(self):
        """Wrap a real reactive and arm the one-shot gate."""
        self._r = solara.reactive(set())
        self.gate = threading.Event()
        self.first_write_parked = threading.Event()
        self._count = 0
        self._count_lock = threading.Lock()

    @property
    def value(self):
        """Current set."""
        return self._r.value

    def set(self, new):
        """Park the first writer on the gate, then write through."""
        with self._count_lock:
            n = self._count
            self._count += 1
        if n == 0:
            self.first_write_parked.set()
            self.gate.wait(TIMEOUT)
        self._r.set(new)


class _FakeMap:
    """Stub map: nothing to remove."""

    def remove_layer(self, key, none_ok=False):
        """No-op."""


def test_overlapping_toggles_keep_both_keys_on_map(monkeypatch):
    """A parked write from toggle A must not swallow toggle B's key."""
    monkeypatch.setattr(
        "gui.scripts.pmtiles_map.add_sample_pmtiles_on_map",
        lambda *a, **k: None,
        raising=False,
    )
    slow = _SlowFirstWrite()
    monkeypatch.setattr(sampling_tile, "samples_on_map", slow)

    def _ss():
        return types.SimpleNamespace(
            points_path="/tmp/s.gpkg",
            pmtiles_path="/tmp/s.pmtiles",
            ensure_pmtiles=None,
        )

    project = solara.reactive(types.SimpleNamespace(samples={"a": _ss(), "b": _ss()}))
    assert sampling_tile.samples_pending.claim("a", "b")

    thread_a = threading.Thread(
        target=sampling_tile._toggle_sample_on_map,
        args=("a", project, _FakeMap(), True),
        daemon=True,
    )
    thread_a.start()
    assert slow.first_write_parked.wait(TIMEOUT), "toggle A never reached its write"

    thread_b = threading.Thread(
        target=sampling_tile._toggle_sample_on_map,
        args=("b", project, _FakeMap(), True),
        daemon=True,
    )
    thread_b.start()
    # Give B every chance to sneak its read in ahead of A's write.
    thread_b.join(0.3)

    slow.gate.set()
    thread_a.join(TIMEOUT)
    thread_b.join(TIMEOUT)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    assert slow.value == {"a", "b"}, f"lost an on-map key: {slow.value}"
    assert sampling_tile.samples_pending.value == frozenset()
