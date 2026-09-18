"""notify_bridge: spatial_risk log milestones -> pysepal notification tasks.

The bridge replaces the old on-map LogConsole: instead of rendering raw log
records in a custom widget, each background job opens a pysepal TaskTracker
(``tracked_job``) and the ``TaskStepLogHandler`` forwards INFO+ records emitted
on the job's own thread as task milestones, so granular library progress
(e.g. ``log_progress``'s "Downloading layer 2/5") surfaces in the official
notification pill.
"""

import logging
import threading

import pytest
from pysepal.solara.notifications.bus import NotificationBus
from pysepal.solara.notifications.notifier import Notifier
from pysepal.solara.notifications.state import TaskStatus, ToastType

from gui.scripts import notify_bridge


def _fresh():
    """Real spatial_risk logger with the bridge handler + a private bus."""
    notify_bridge.install_task_log_handler()
    logger = logging.getLogger("spatial_risk")
    logger.setLevel(logging.DEBUG)
    bus = NotificationBus()
    return logger, bus, Notifier(bus)


def _milestones(bus, title):
    task = next(t for t in bus.tasks.value if t.title == title)
    return [m.message for m in task.milestones]


def _status(bus, title):
    return next(t for t in bus.tasks.value if t.title == title).status


def test_job_log_lines_become_milestones_and_task_completes():
    """INFO lines from the job thread land as milestones; exit completes."""
    logger, bus, notifier = _fresh()
    with notify_bridge.tracked_job(notifier, "My job"):
        logger.info("Step one")
        logger.info("Step %d", 2)
    assert _milestones(bus, "My job") == ["Step one", "Step 2"]
    assert _status(bus, "My job") == TaskStatus.COMPLETED


def test_debug_lines_are_not_forwarded():
    """DEBUG records stay out of the milestone log."""
    logger, bus, notifier = _fresh()
    with notify_bridge.tracked_job(notifier, "Quiet job"):
        logger.debug("hidden debug")
    assert _milestones(bus, "Quiet job") == []


def test_lines_from_unrelated_threads_are_not_forwarded():
    """Only the job's own thread feeds its tracker.

    A concurrent log line from elsewhere (another session, an unrelated
    worker) must not pollute the task's milestone list.
    """
    logger, bus, notifier = _fresh()
    with notify_bridge.tracked_job(notifier, "Mine"):
        other = threading.Thread(target=lambda: logger.info("not mine"))
        other.start()
        other.join()
        logger.info("mine")
    assert _milestones(bus, "Mine") == ["mine"]


def test_exception_marks_task_failed_with_error_toast_and_reraises():
    """A job exception fails the task, raises a toast, and re-raises."""
    logger, bus, notifier = _fresh()
    with pytest.raises(ValueError, match="boom"):
        with notify_bridge.tracked_job(notifier, "Doomed job"):
            logger.info("about to fail")
            raise ValueError("boom")
    task = next(t for t in bus.tasks.value if t.title == "Doomed job")
    assert task.status == TaskStatus.FAILED
    assert task.error_message == "boom"
    assert any(
        t.type == ToastType.ERROR and "boom" in t.message for t in bus.toasts.value
    )
    # The failed thread's registration must not leak into later log calls.
    logger.info("after the failure")
    assert "after the failure" not in _milestones(bus, "Doomed job")


def test_no_active_job_is_a_noop():
    """Log lines with no tracked job on the thread are ignored."""
    logger, bus, _notifier = _fresh()
    logger.info("nobody is tracking this")
    assert bus.tasks.value == []


def test_install_is_idempotent():
    """Repeated installs attach exactly one bridge handler."""
    notify_bridge.install_task_log_handler()
    notify_bridge.install_task_log_handler()
    handlers = [
        h
        for h in logging.getLogger("spatial_risk").handlers
        if isinstance(h, notify_bridge.TaskStepLogHandler)
    ]
    assert len(handlers) == 1


def test_concurrent_jobs_keep_milestones_separate():
    """Two jobs running at once each see only their own log lines."""
    logger, bus, notifier = _fresh()
    barrier = threading.Barrier(2, timeout=5)

    def job(title, line):
        with notify_bridge.tracked_job(notifier, title):
            barrier.wait()  # both jobs are tracking before either logs
            logger.info(line)
            barrier.wait()  # neither exits before both have logged

    t1 = threading.Thread(target=job, args=("Job A", "line A"))
    t2 = threading.Thread(target=job, args=("Job B", "line B"))
    t1.start(), t2.start()
    t1.join(), t2.join()
    assert _milestones(bus, "Job A") == ["line A"]
    assert _milestones(bus, "Job B") == ["line B"]


def test_to_thread_in_context_attaches_caller_context_to_pool_thread(monkeypatch):
    """Job bodies tracked inside ``asyncio.to_thread`` publish from a pool thread.

    Without the caller's kernel context attached first, those updates never
    reach the browser (same reason spawn_in_context exists).
    """
    import asyncio

    from solara.server import kernel_context

    from gui.scripts.solara_threads import to_thread_in_context

    sentinel = object()
    attached = []
    monkeypatch.setattr(kernel_context, "get_current_context", lambda: sentinel)
    monkeypatch.setattr(kernel_context, "has_current_context", lambda: False)
    monkeypatch.setattr(
        kernel_context,
        "set_context_for_thread",
        lambda ctx, thread: attached.append((ctx, thread)),
    )

    caller = threading.current_thread()

    def work(x):
        return (threading.current_thread(), x * 2)

    worker, result = asyncio.run(to_thread_in_context(work, 21))
    assert result == 42
    assert worker is not caller
    assert attached and attached[0] == (sentinel, worker)


def test_to_thread_in_context_without_context_still_runs(monkeypatch):
    """No kernel context (plain scripts/tests) must not break the helper."""
    import asyncio

    from solara.server import kernel_context

    from gui.scripts.solara_threads import to_thread_in_context

    def _raise():
        raise RuntimeError("no context")

    monkeypatch.setattr(kernel_context, "get_current_context", _raise)
    assert asyncio.run(to_thread_in_context(lambda: "ok")) == "ok"


class _RecordingTask:
    """Minimal TaskTracker stand-in that records adapter calls."""

    def __init__(self):
        self.titles = []
        self.progress_calls = []

    def update(self, title):
        self.titles.append(title)

    def set_progress(self, value, detail=None):
        self.progress_calls.append((value, detail))


def test_layer_reporter_start_updates_title_and_resets_progress():
    """A layer-start event retitles the task and resets the ring/detail."""
    _logger, bus, notifier = _fresh()
    with notifier.track("Downloading GEE layers") as task:
        report = notify_bridge.layer_progress_reporter(
            task, format_title=lambda k, i, n: f"Downloading layer {i + 1}/{n}: {k}"
        )
        report("rivers", 2, 5, 0, 0)
        t = bus.tasks.value[0]
        assert t.title == "Downloading layer 3/5: rivers"
        assert t.progress is None
        assert t.progress_detail is None


def test_layer_reporter_single_layer_keeps_original_title():
    """A user-triggered single download already has a named task title."""
    _logger, bus, notifier = _fresh()
    with notifier.track("Downloading layer 'rivers'") as task:
        report = notify_bridge.layer_progress_reporter(
            task, format_title=lambda k, i, n: f"Downloading layer {i + 1}/{n}: {k}"
        )
        report("rivers", 0, 1, 0, 0)
        assert bus.tasks.value[0].title == "Downloading layer 'rivers'"


def test_layer_reporter_tile_ticks_publish_progress_and_detail():
    """Tile ticks publish the layer fraction plus the tile-count detail."""
    _logger, bus, notifier = _fresh()
    with notifier.track("Downloading GEE layers") as task:
        report = notify_bridge.layer_progress_reporter(task)
        report("rivers", 0, 1, 0, 0)
        report("rivers", 0, 1, 10, 30)
        t = bus.tasks.value[0]
        assert t.progress == pytest.approx(10 / 30)
        assert t.progress_detail == "rivers — tile 10/30"
        # tile ticks are display-only: the milestone log stays clean
        assert t.milestones == ()


def test_layer_reporter_throttles_to_whole_percent_changes():
    """A 1000-tile layer must not publish 1000 reactive updates."""
    task = _RecordingTask()
    report = notify_bridge.layer_progress_reporter(task)
    report("big", 0, 1, 0, 0)
    for done in range(1, 11):
        report("big", 0, 1, done, 1000)
    # d=1 -> 0% (first emit), d=2..9 -> still 0% (skipped), d=10 -> 1%
    assert [v for v, _ in task.progress_calls if v is not None] == [
        pytest.approx(1 / 1000),
        pytest.approx(10 / 1000),
    ]


class _LegacyTask:
    """Upstream-pysepal TaskTracker: set_progress has no ``detail`` kwarg."""

    def __init__(self):
        self.titles = []
        self.progress_values = []

    def update(self, title):
        self.titles.append(title)

    def set_progress(self, value):
        self.progress_values.append(value)


def test_layer_reporter_degrades_on_stock_pysepal():
    """Against upstream pysepal (no progress_detail) nothing may crash.

    The reporter publishes plain progress and drops the detail, leaving the
    stock pill exactly as it behaves today.
    """
    task = _LegacyTask()
    report = notify_bridge.layer_progress_reporter(
        task, format_title=lambda k, i, n: f"Downloading layer {i + 1}/{n}: {k}"
    )
    report("rivers", 0, 2, 0, 0)
    report("rivers", 0, 2, 10, 30)
    assert task.titles == ["Downloading layer 1/2: rivers"]
    assert task.progress_values == [None, pytest.approx(10 / 30)]


def test_layer_reporter_custom_detail_formatting():
    """format_detail lets the caller localize the tile string."""
    task = _RecordingTask()
    report = notify_bridge.layer_progress_reporter(
        task, format_detail=lambda k, d, t: f"{k}: baldosa {d}/{t}"
    )
    report("rivers", 0, 1, 15, 30)
    assert task.progress_calls[-1] == (pytest.approx(0.5), "rivers: baldosa 15/30")


def test_nested_tracked_jobs_restore_the_outer_tracker():
    """Exiting an inner tracked job re-registers the outer one."""
    logger, bus, notifier = _fresh()
    with notify_bridge.tracked_job(notifier, "Outer"):
        with notify_bridge.tracked_job(notifier, "Inner"):
            logger.info("inner line")
        logger.info("outer line")
    assert _milestones(bus, "Inner") == ["inner line"]
    assert _milestones(bus, "Outer") == ["outer line"]


def test_error_format_localizes_the_failure_toast_and_task():
    """A formatter turns the raw exception into the user-facing message."""
    _logger, bus, notifier = _fresh()
    with pytest.raises(ValueError, match="boom"):
        with notify_bridge.tracked_job(
            notifier,
            "Harmonizing variables",
            error_format=lambda exc: f"Harmonization failed: {exc}",
        ):
            raise ValueError("boom")

    task = next(t for t in bus.tasks.value if t.title == "Harmonizing variables")
    assert task.status == TaskStatus.FAILED
    assert task.error_message == "Harmonization failed: boom"
    errors = [t for t in bus.toasts.value if t.type == ToastType.ERROR]
    assert [t.message for t in errors] == ["Harmonization failed: boom"]
    assert errors[0].effective_timeout() == notify_bridge.ERROR_TOAST_TIMEOUT


def test_broken_error_format_does_not_mask_the_original_exception():
    """A raising error_format must not replace the real failure.

    The task still fails and toasts with str(exc) of the ORIGINAL exception,
    and the ORIGINAL exception is what propagates — not the formatter's.
    """
    _logger, bus, notifier = _fresh()

    def broken_formatter(exc):
        raise RuntimeError("formatter is broken")

    with pytest.raises(ValueError, match="^boom$"):
        with notify_bridge.tracked_job(
            notifier, "Fragile job", error_format=broken_formatter
        ):
            raise ValueError("boom")

    task = next(t for t in bus.tasks.value if t.title == "Fragile job")
    assert task.status == TaskStatus.FAILED
    assert task.error_message == "boom"
    errors = [t for t in bus.toasts.value if t.type == ToastType.ERROR]
    assert [t.message for t in errors] == ["boom"]


def test_failure_publishes_exactly_one_error_toast():
    """Pysepal's own __exit__ toast must not fire on top of ours.

    bus.add_toast keeps only the newest ERROR toast, so the bus cannot show a
    duplicate — count the publications at the notifier instead.
    """
    _logger, _bus, notifier = _fresh()
    published = []
    original = notifier.error

    def spy(message, *, timeout=None):
        published.append((message, timeout))
        return original(message, timeout=timeout)

    notifier.error = spy
    with pytest.raises(RuntimeError):
        with notify_bridge.tracked_job(notifier, "One toast only"):
            raise RuntimeError("kaboom")

    assert published == [("kaboom", notify_bridge.ERROR_TOAST_TIMEOUT)]


def test_cancellation_is_not_reported_as_a_failure():
    """A cancelled job stays CANCELLED and raises no error toast."""
    import asyncio

    _logger, bus, notifier = _fresh()
    with pytest.raises(asyncio.CancelledError):
        with notify_bridge.tracked_job(notifier, "Cancelled job"):
            raise asyncio.CancelledError()

    task = next(t for t in bus.tasks.value if t.title == "Cancelled job")
    assert task.status == TaskStatus.CANCELLED
    assert [t for t in bus.toasts.value if t.type == ToastType.ERROR] == []


def test_layer_reporter_wait_publishes_indeterminate_detail():
    """A queued layer shows its waiting message on an indeterminate ring."""
    _logger, bus, notifier = _fresh()
    with notifier.track("Downloading layer 'rivers'") as task:
        report = notify_bridge.layer_progress_reporter(
            task, format_wait=lambda k: f"{k} — waiting for another download"
        )
        report("rivers", 0, 1, 0, 0)
        report.on_wait("rivers", 0, 1)
        t = bus.tasks.value[0]
        assert t.progress is None
        assert t.progress_detail == "rivers — waiting for another download"
        # the first tile tick replaces the waiting message
        report("rivers", 0, 1, 1, 4)
        assert bus.tasks.value[0].progress_detail == "rivers — tile 1/4"


def test_layer_reporter_wait_without_format_is_a_noop():
    """Callers that don't localize a wait message get a safe no-op hook."""
    _logger, bus, notifier = _fresh()
    with notifier.track("Downloading layer 'rivers'") as task:
        report = notify_bridge.layer_progress_reporter(task)
        report.on_wait("rivers", 0, 1)
        assert bus.tasks.value[0].progress_detail is None
