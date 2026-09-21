"""The sampling toggle's pending set is the locked InflightKeys."""

import pytest

from gui.scripts.inflight import InflightKeys
from gui.tile import sampling_tile


@pytest.fixture(autouse=True)
def _drain_sampling_inflight():
    """Leave no claim behind in sampling_tile's module-level state."""
    yield
    sampling_tile.samples_pending.release(*sampling_tile.samples_pending.value)


def test_samples_pending_is_an_inflight_set():
    """Verify that samples_pending is an InflightKeys instance."""
    assert isinstance(sampling_tile.samples_pending, InflightKeys)
    assert sampling_tile.samples_pending.value == frozenset()


def test_toggle_worker_releases_the_key_after_a_failure(monkeypatch):
    """Worker releases the key when an error occurs during map add."""
    import types

    import solara

    class FakeMap:
        """Stub map with no-op layer operations."""

        def remove_layer(self, key, none_ok=False):
            """No-op remove_layer."""
            pass

    def boom(*a, **k):
        """Raise an error to simulate a failed map operation."""
        raise RuntimeError("no tiles")

    monkeypatch.setattr(
        "gui.scripts.pmtiles_map.add_sample_pmtiles_on_map", boom, raising=False
    )
    monkeypatch.setattr(
        "gui.scripts.map_helpers.add_sample_points_on_map", boom, raising=False
    )
    ss = types.SimpleNamespace(
        points_path="/tmp/s.gpkg", pmtiles_path="/tmp/s.pmtiles", ensure_pmtiles=None
    )
    project = solara.reactive(types.SimpleNamespace(samples={"s": ss}))
    assert sampling_tile.samples_pending.claim("s")
    sampling_tile._toggle_sample_on_map("s", project, FakeMap(), True)
    assert "s" not in sampling_tile.samples_pending
