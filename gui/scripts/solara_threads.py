"""Helpers for running background work from Solara event handlers.

Which tool for which job:

* ``spawn_in_context`` — one thread per user-fired action (a job, a download,
  a map toggle). ALL continuation code (republish, on-map state, legends)
  goes inside the worker; guard re-clicks with ``gui.scripts.inflight.
  InflightKeys``. This is the default.
* ``solara.lab.use_task`` — a *latest-request-wins* slot. Re-invoking it
  cancels the in-flight coroutine at its ``await`` (a BaseException, so
  ``except Exception`` will not see it) while any thread it started finishes
  anyway. Use it only for dependency-driven values read via ``.value``, or
  for a single-shot action whose handler checks ``if task.pending: return``
  first. ``tests/test_use_task_sites_guarded.py`` enforces the latter.

Reactive updates (``some_reactive.set(...)``) made from a *bare*
``threading.Thread`` never reach the browser session, so status cards stay
stuck on "running" even after the backend has finished. Registering the render
thread's kernel context onto the worker thread — exactly what
``solara.lab.use_task`` does internally — lets those updates propagate.
"""

import asyncio
import threading


async def to_thread_in_context(fn, *args, **kwargs):
    """``asyncio.to_thread`` that propagates the caller's Solara kernel context.

    ``to_thread`` runs on a pool thread that has no kernel context, so any
    reactive publish made there (job status, notification-task milestones)
    never reaches the browser. This captures the caller's context (the
    ``use_task`` body runs with one) and attaches it to the pool thread before
    the work starts — the ``to_thread`` counterpart of ``spawn_in_context``.
    Pool threads are reused, so the binding persists; harmless under the app's
    single-user assumption (it is always the same session's context).
    """
    from solara.server import kernel_context

    try:
        ctx = kernel_context.get_current_context()
    except RuntimeError:
        ctx = None

    def _with_context():
        if ctx is not None and not kernel_context.has_current_context():
            kernel_context.set_context_for_thread(ctx, threading.current_thread())
        return fn(*args, **kwargs)

    return await asyncio.to_thread(_with_context)


_update_job_lock = threading.Lock()
"""Serialises the read-modify-write in :func:`update_job`.

Two workers finishing within microseconds of each other each read the same
list, build their own copy and publish it; without the lock the second
publish silently discards the first's status change and that row stays on
"running" for good.
"""


def update_job(jobs_reactive, job_id, *, skip_if_cancelled=True, **changes):
    """Immutably update one job dict by id and publish so the UI re-renders.

    Mutating a job dict in place and then calling ``reactive.set(list(...))``
    does NOT update the browser: the old and new lists share the same dict
    objects, so Solara's ``equals_extra(old, new)`` is True and ``set`` short-
    circuits without firing listeners — the status card stays stuck on a
    spinning "running" icon even after the background job has finished. Building
    a fresh dict for the changed job makes the new list genuinely differ so the
    update propagates.

    Parameters
    ----------
    jobs_reactive : solara.Reactive[list[dict]]
        The reactive holding the list of job dicts (each with an ``"id"``).
    job_id : str
        Id of the job to update.
    skip_if_cancelled : bool
        When True (default) a job the user already cancelled is left untouched,
        so a late-finishing thread can't resurrect it as completed/failed.
    **changes
        Fields to overwrite on the matching job dict.
    """
    with _update_job_lock:
        new_jobs = []
        for j in jobs_reactive.value:
            if j["id"] == job_id and not (
                skip_if_cancelled and j["status"] == "cancelled"
            ):
                new_jobs.append({**j, **changes})
            else:
                new_jobs.append(j)
        jobs_reactive.set(new_jobs)


def spawn_in_context(target, args=(), *, daemon=True):
    """Start a daemon thread that inherits the caller's Solara kernel context.

    Falls back to a plain thread when there is no active context (e.g. unit
    tests), so the function is usable outside a running app.

    Parameters
    ----------
    target : callable
        Function to run in the background thread.
    args : tuple
        Positional arguments forwarded to ``target``.
    daemon : bool
        Whether the thread is a daemon (default ``True``).

    Returns:
    -------
    threading.Thread
        The started thread.
    """
    from solara.server import kernel_context

    thread = threading.Thread(target=target, args=args, daemon=daemon)
    try:
        ctx = kernel_context.get_current_context()
    except RuntimeError:
        ctx = None
    if ctx is not None:
        kernel_context.set_context_for_thread(ctx, thread)
    thread.start()
    return thread


def publish_if_current(project_reactive, project) -> bool:
    """Publish a job's mutated project — unless it is no longer the open one.

    Background jobs capture a ``Project`` reference when they start and publish a
    fresh copy when they finish, so dependent tiles re-render. If the project was
    **deleted** meanwhile the reactive holds ``None``; if the user **switched
    projects** it holds a different one. Writing the captured reference back in
    either case resurrects a dead project into app state, and the auto-save inside
    those jobs re-creates the folder that was just deleted (``Project.save()``
    does ``mkdir(parents=True, exist_ok=True)``).

    Matching on ``project_name`` rather than object identity is deliberate: tiles
    routinely republish via ``project.set(p.model_copy())``, so by the time a long
    job finishes the live object is usually a *different instance* of the same
    project — and that job's result must still reach the UI.

    Returns True when the copy was published.
    """
    if project_reactive is None or project is None:
        return False
    live = project_reactive.value
    if live is None:
        return False  # project closed/deleted — never resurrect it
    if live.project_name != project.project_name:
        return False  # user switched projects — do not clobber the new one
    project_reactive.set(project.model_copy())
    return True
