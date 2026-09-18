"""Surface geedim's tile-download progress as a plain Python callback.

geedim reports per-tile download progress only through a tqdm bar
(``geedim.tile.Tiler.map_tiles`` builds it via ``geedim.utils.auto_leave_tqdm``
with ``total=n_tiles`` and ticks it once per completed tile) — there is no
callback API. Rather than forking geedim, ``geedim_tile_progress`` rebinds the
``tqdm`` name in ``geedim.utils`` to a subclass whose ``update()`` also invokes
a thread-scoped ``callback(done, total)``.

The rebinding is installed once and left in place (mirroring the idempotent
handler install in ``gui/scripts/notify_bridge.py``); with no callback
registered the subclass behaves exactly like tqdm. Callbacks live in a
``threading.local`` keyed to the thread that *creates* the bar — geedim's
``AsyncRunner.run`` drives its event loop on the calling thread, so a download
job only ever sees its own callback and concurrent jobs cannot cross-talk.

``geedim_download_wait`` is the same thread-local pattern for a second event:
``download_ee_image`` serializes downloads on one lock (geedim's runner owns a
single event loop), and a download that finds the lock taken calls
``notify_download_wait`` once before blocking, so the UI can say it is queued.
"""

import threading
from contextlib import contextmanager

_local = threading.local()
_install_lock = threading.Lock()
_installed = False


def _install() -> None:
    """Rebind ``geedim.utils.tqdm`` to the callback-aware subclass (idempotent)."""
    global _installed
    with _install_lock:
        if _installed:
            return
        import geedim.utils as gd_utils

        base = gd_utils.tqdm

        class _CallbackTqdm(base):
            """tqdm that forwards ticks to the creating thread's callback."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._progress_cb = getattr(_local, "callback", None)

            def _emit(self, done):
                cb = self._progress_cb
                # Bars without a total (export monitors) carry no tile count.
                if cb is not None and self.total:
                    try:
                        cb(done, self.total)
                    except Exception:  # progress display must never kill a download
                        pass

            def __iter__(self):
                # geedim iterates the bar (`for task in bar`), and tqdm's
                # __iter__ throttles update() calls via miniters/mininterval —
                # hooking update() alone would drop ticks. Count yielded items
                # ourselves: with as_completed, the k-th yield = k tiles done.
                # (tqdm's __iter__ calls update() internally for display;
                # _in_iter stops that path double-emitting.)
                done = int(self.n or 0)
                self._in_iter = True
                try:
                    for obj in super().__iter__():
                        done += 1
                        self._emit(done)
                        yield obj
                finally:
                    self._in_iter = False

            def update(self, n=1):
                result = super().update(n)
                if not getattr(self, "_in_iter", False):
                    self._emit(int(self.n or 0))
                return result

        gd_utils.tqdm = _CallbackTqdm
        _installed = True


@contextmanager
def geedim_tile_progress(callback):
    """Route geedim tile-bar ticks on this thread to ``callback(done, total)``.

    Wrap each individual download call — every geedim bar created on this
    thread while the context is active reports through the callback, so keep
    one layer per ``with`` block to know which layer the counts belong to.
    Nesting restores the previous callback on exit.
    """
    _install()
    previous = getattr(_local, "callback", None)
    _local.callback = callback
    try:
        yield
    finally:
        _local.callback = previous


@contextmanager
def geedim_download_wait(callback):
    """Route "queued behind another download" notices on this thread to ``callback()``.

    ``download_ee_image`` fires it at most once per call, right before it
    blocks on the lock; a download that gets the lock straight away never
    fires it. Nesting restores the previous callback on exit.
    """
    previous = getattr(_local, "on_wait", None)
    _local.on_wait = callback
    try:
        yield
    finally:
        _local.on_wait = previous


def notify_download_wait() -> None:
    """Tell this thread's ``geedim_download_wait`` callback, if any, that we queue."""
    cb = getattr(_local, "on_wait", None)
    if cb is not None:
        try:
            cb()
        except Exception:  # a display hiccup must never kill a download
            pass
