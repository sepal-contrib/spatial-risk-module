"""A manually-invoked ``solara.lab.use_task`` is a single-shot slot.

``Task.__call__`` cancels the in-flight run when re-invoked, and the cancel
lands at the body's ``await`` as a BaseException — everything after the
``await`` is skipped while any thread it started finishes anyway. So a
``use_task(dependencies=None, ...)`` is acceptable only when its handler
refuses to re-invoke it while ``.pending`` (``process_task``, ``delete_task``).
Per-item actions must use one ``spawn_in_context`` worker per action instead
(see gui/scripts/solara_threads.py). This test fails on any new manual task
without that guard.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MANUAL_TASK = re.compile(
    r"use_task\(\s*dependencies=None[^)]*\)\s*\n\s*(?:async\s+)?def\s+(\w+)\("
)


def _manual_tasks(src: str):
    """Return the names of functions decorated with a manual ``use_task``."""
    return _MANUAL_TASK.findall(src)


def test_every_manual_use_task_is_guarded_on_pending():
    """Every manual single-shot task must refuse re-invocation while pending."""
    unguarded = []
    for path in sorted((ROOT / "gui").rglob("*.py")):
        src = path.read_text()
        for name in _manual_tasks(src):
            guard = re.search(rf"if\s+{name}\.pending\s*:\s*\n\s*return\b", src)
            if guard is None:
                unguarded.append(f"{path.relative_to(ROOT)}:{name}")
    assert not unguarded, (
        "manual use_task sites without an `if <task>.pending: return` guard "
        f"(re-invoking one drops its continuation): {unguarded}"
    )


def test_the_known_single_shot_tasks_are_still_detected():
    """Guards the regex itself: the two legitimate sites must be found."""
    found = {
        name
        for rel in ("gui/tile/process_tile.py", "gui/solara_app.py")
        for name in _manual_tasks((ROOT / rel).read_text())
    }
    assert {"process_task", "delete_task"} <= found
