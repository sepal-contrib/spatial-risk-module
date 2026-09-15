"""InflightKeys: a lock-backed set of item keys currently owned by a worker.

Claim/release are atomic so two workers finishing at once cannot lose each
other's release (the read-modify-write `pending.set(pending.value - {key})`
pattern did exactly that).
"""

import threading

from gui.scripts.inflight import InflightKeys


def test_claim_is_exclusive_until_released():
    """A claimed key rejects a second claim until it is released."""
    inflight = InflightKeys()
    assert inflight.claim("a") is True
    assert inflight.claim("a") is False
    assert "a" in inflight
    inflight.release("a")
    assert "a" not in inflight
    assert inflight.claim("a") is True


def test_claim_is_all_or_nothing():
    """A multi-key claim fails entirely if any one key is already owned."""
    inflight = InflightKeys()
    assert inflight.claim("a")
    assert inflight.claim("a", "b") is False
    assert "b" not in inflight  # the partial claim must not leak
    assert inflight.claim("b", "c") is True
    assert inflight.value == frozenset({"a", "b", "c"})


def test_release_of_an_unknown_key_is_a_noop():
    """Releasing a key that was never claimed does nothing."""
    inflight = InflightKeys()
    inflight.release("never")
    assert inflight.value == frozenset()


def test_value_is_a_frozenset_reactive_snapshot():
    """``.value`` returns a frozen snapshot, unaffected by later releases."""
    inflight = InflightKeys()
    inflight.claim("a")
    snap = inflight.value
    inflight.release("a")
    assert snap == frozenset({"a"})
    assert inflight.value == frozenset()


def test_concurrent_releases_do_not_lose_updates():
    """8 threads releasing 200 keys concurrently must not drop any release."""
    inflight = InflightKeys()
    keys = [f"k{i}" for i in range(200)]
    assert inflight.claim(*keys)
    start = threading.Barrier(8)

    def worker(chunk):
        start.wait()
        for k in chunk:
            inflight.release(k)

    threads = [threading.Thread(target=worker, args=(keys[i::8],)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert inflight.value == frozenset()
