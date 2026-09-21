"""Bridge the `spatial_risk` logger to pysepal notification TaskTrackers.

Each background job wraps its work in ``tracked_job(notifier, title)``, which
opens a pysepal ``notifier.track(...)`` task. While the job body runs, the
``TaskStepLogHandler`` attached to the ``spatial_risk`` logger forwards every
INFO+ record *emitted on the job's own thread* as a ``task.step(...)``
milestone — so granular library progress (``log_progress``'s
"Downloading layer 2/5", processing stage lines, …) surfaces in the official
notification pill without ``spatialrisk/`` knowing about the GUI.

Thread-scoping is what keeps concurrent jobs honest: two trainings running at
once each see only their own log lines. Completion / failure semantics come
from pysepal's tracker context manager (auto-complete on exit, mark FAILED +
error toast + re-raise on exception).

Threads without a Solara kernel context (``asyncio.to_thread`` pool threads)
cannot publish reactives to the browser — run those job bodies via
``solara_threads.to_thread_in_context`` so the tracker's bus updates land.
"""

import asyncio
import logging
import threading
from contextlib import contextmanager

_LOGGER_NAME = "spatial_risk"

ERROR_TOAST_TIMEOUT = 10.0
"""Seconds an error toast stays on screen. Successes keep pysepal's 3 s default.

Failures deserve a longer read than confirmations, but nothing is sticky: a
tracked job's failure also lands in the notification log, which outlives the
toast.
"""

# thread ident -> TaskTracker of the tracked job running on that thread.
_trackers = {}
_install_lock = threading.Lock()


class TaskStepLogHandler(logging.Handler):
    """Forward INFO+ records from tracked-job threads as task milestones.

    Records from threads with no registered tracker are ignored (file/console
    handlers still see them). The handler itself reads no reactive state, so a
    component that logs during its render cannot be auto-subscribed by it
    (``tracker.step`` does read the bus, but only job threads are registered,
    never a rendering thread).
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Step the emitting thread's tracker (if any) with the record message."""
        try:
            tracker = _trackers.get(threading.get_ident())
            if tracker is not None:
                tracker.step(record.getMessage())
        except Exception:  # a logging handler must never raise into caller code
            pass


def install_task_log_handler(level: int = logging.INFO) -> TaskStepLogHandler:
    """Attach a single ``TaskStepLogHandler`` to the ``spatial_risk`` logger.

    Idempotent: returns the existing handler if one is already attached (guards
    Solara hot-reload / repeated imports). Sets the handler level to ``INFO``
    but does NOT change the logger's own level (file/console keep DEBUG detail).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    with _install_lock:
        for h in logger.handlers:
            if isinstance(h, TaskStepLogHandler):
                return h
        handler = TaskStepLogHandler()
        handler.setLevel(level)
        logger.addHandler(handler)
        return handler


def layer_progress_reporter(
    task, format_title=None, format_detail=None, format_wait=None
):
    """Adapt ``materialize_raw_layers``'s ``on_progress`` events to a tracker.

    Returns an ``on_progress(layer_key, layer_idx, n_layers, done, total)``
    callable that drives the notification pill's alternating display:

    - layer start (zero tile counts): retitle the task to the run position
      via ``format_title(key, idx, n)`` — skipped for single-layer runs, whose
      task already carries a named title — and ``set_progress(None)`` so the
      ring returns to indeterminate instead of showing the previous layer's
      100%.
    - tile tick: ``set_progress(done/total, detail=...)`` with the layer's own
      fraction, throttled to whole-percent changes so a many-tile layer doesn't
      flood the reactive bus. The detail string comes from
      ``format_detail(key, done, total)`` (default: ``"{key} — tile {d}/{t}"``).
    - queued (``on_progress.on_wait(key, idx, n)``, wired to
      ``materialize_raw_layers``'s ``on_wait``): indeterminate ring with
      ``format_wait(key)`` as the detail, until the first tile tick replaces
      it. A no-op when no ``format_wait`` is given.

    ``format_*`` are injected so callers can localize without this module
    importing the GUI's translator. Progress is never derived from milestone
    step counts — the log handler's auto-increment would overshoot.

    Works against stock (unforked) pysepal too: its ``set_progress`` has no
    ``detail`` kwarg, so the reporter detects that once and publishes plain
    progress values instead — the stock pill ignores them and keeps today's
    indeterminate-spinner behavior, and the download never crashes.
    """
    import inspect

    try:
        detail_supported = "detail" in inspect.signature(task.set_progress).parameters
    except (TypeError, ValueError):  # builtins/mocks without introspectable signatures
        detail_supported = True

    def _set_progress(value, detail=None):
        if detail_supported:
            task.set_progress(value, detail=detail)
        else:
            task.set_progress(value)

    last_pct = None

    def on_progress(layer_key, layer_idx, n_layers, done, total):
        nonlocal last_pct
        if not total:  # layer-start event
            last_pct = None
            if n_layers > 1 and format_title is not None:
                task.update(format_title(layer_key, layer_idx, n_layers))
            _set_progress(None)
            return
        pct = int(done * 100 / total)
        if pct == last_pct:
            return
        last_pct = pct
        detail = (
            format_detail(layer_key, done, total)
            if format_detail is not None
            else f"{layer_key} — tile {done}/{total}"
        )
        _set_progress(done / total, detail=detail)

    def on_wait(layer_key, layer_idx, n_layers):
        if format_wait is not None:
            _set_progress(None, detail=format_wait(layer_key))

    on_progress.on_wait = on_wait
    return on_progress


@contextmanager
def tracked_job(notifier, title: str, total_steps=None, error_format=None):
    """Track a job as a pysepal notification task, echoing its log milestones.

    Must be entered on the thread that runs the job body — the log→milestone
    forwarding is keyed on that thread's ident. Nesting restores the previous
    tracker on exit. Yields the ``TaskTracker`` for explicit ``step()`` /
    ``set_progress()`` calls where log lines aren't enough. ``notifier=None``
    falls back to a no-op tracker so workers stay callable from plain tests.

    ``error_format`` (``Callable[[Exception], str]``) turns a failure into the
    user-facing message; without it the raw ``str(exc)`` is used. On failure the
    task is marked FAILED with that message *before* the toast is published:
    pysepal's ``_TaskTrackerContextManager.__exit__`` only raises its own
    bare-exception toast while the task is not already FAILED, so failing first
    is what keeps this to exactly one toast.
    """
    if notifier is None:
        from pysepal.solara.notifications.notifier import NoopNotifier

        notifier = NoopNotifier()
    install_task_log_handler()
    with notifier.track(title, total_steps=total_steps) as task:
        tid = threading.get_ident()
        previous = _trackers.get(tid)
        _trackers[tid] = task
        try:
            yield task
        except asyncio.CancelledError:
            raise  # pysepal's __exit__ marks the task CANCELLED; not a failure
        except Exception as exc:
            try:
                message = error_format(exc) if error_format is not None else str(exc)
            except Exception:  # a broken formatter must never mask the real failure
                logging.getLogger(_LOGGER_NAME).exception("error_format failed")
                message = str(exc)
            task.fail(message)
            notifier.error(message, timeout=ERROR_TOAST_TIMEOUT)
            raise
        finally:
            if previous is not None:
                _trackers[tid] = previous
            else:
                _trackers.pop(tid, None)
