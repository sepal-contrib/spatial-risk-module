"""Copy for the two dialogs that decide a file's fate.

Removing a variable and downloading over an existing raster are the only two
places where the app touches a user's files without being asked to, so both ask
first. The wording — and, for the delete question, whether the offer can be made
at all — is decided here rather than in the tiles: three tiles ask the delete
question and two buttons ask the overwrite one.

Solara-free by design (like the rest of ``gui/scripts``): these are pure
functions over a Project, so the tiles stay thin and the copy is tested once.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from gui.i18n import plural, t
from gui.scripts.process_actions import existing_download_targets
from gui.scripts.project_ui_helpers import format_size
from spatialrisk.variables.file_cleanup import FilePlan, plan_variable_files

logger = logging.getLogger("spatial_risk")


@dataclass(frozen=True)
class DeletePrompt:
    """What the Remove dialog adds when a variable has files on disk."""

    plan: FilePlan
    checkbox_label: Optional[str] = None  # None = no offer (see ``note``)
    note: Optional[str] = None  # why the files are being kept
    details: Tuple[str, ...] = ()  # the paths, one per line


@dataclass(frozen=True)
class OverwritePrompt:
    """The question asked before a download lands on files already on disk."""

    keys: Tuple[str, ...]  # the layers whose file is in the way
    title: str
    message: str
    note: str
    details: Tuple[str, ...] = ()


def _relative(project, path: Path) -> str:
    """Path as the user knows it: relative to the project folder when inside."""
    try:
        root = Path(project._project_dir()).resolve()
        return str(Path(path).resolve().relative_to(root))
    except (ValueError, OSError, AttributeError):
        return str(path)


def delete_prompt(project, key: str) -> DeletePrompt:
    """The checkbox (or the explanation for its absence) for removing *key*.

    A checkbox only appears when ticking it would really delete something;
    otherwise the dialog says, in one dimmed line, why the file stays.
    """
    plan = plan_variable_files(project, key)
    if plan.deletable:
        label = plural(
            len(plan.files),
            "widgets.file_cleanup.checkbox_one",
            "widgets.file_cleanup.checkbox_other",
            size=format_size(plan.total_bytes),
        )
        return DeletePrompt(
            plan=plan,
            checkbox_label=label,
            details=tuple(_relative(project, f) for f in plan.files),
        )

    if plan.blocked == "outside_project":
        first = plan.all_files[0] if plan.all_files else ""
        return DeletePrompt(
            plan=plan,
            note=t("widgets.file_cleanup.blocked_outside", path=str(first)),
        )
    if plan.blocked == "shared":
        return DeletePrompt(
            plan=plan,
            note=t("widgets.file_cleanup.blocked_shared", name=plan.blocked_by or ""),
        )
    # Nothing on disk yet (a cloud variable): the dialog stays as it was.
    return DeletePrompt(plan=plan)


def _file_line(project, path: Path) -> str:
    """One listed file: relative path, size and date.

    Size and date are what tell a stale or half-written file from the one you
    fetched on purpose — the whole reason for asking before overwriting it.
    """
    try:
        stat = Path(path).stat()
        size, when = format_size(stat.st_size), datetime.fromtimestamp(stat.st_mtime)
    except OSError:  # pragma: no cover - vanished between listing and rendering
        return _relative(project, path)
    return t(
        "widgets.file_cleanup.file_line",
        path=_relative(project, path),
        size=size,
        date=when.strftime("%Y-%m-%d"),
    )


def overwrite_prompt(project, keys: Optional[List[str]]) -> Optional[OverwritePrompt]:
    """Ask before downloading *keys* when their files are already on disk.

    ``keys`` is the download's own selection — one key from a row button, None
    for "download all". Returns None when nothing is in the way, which is the
    signal to start the download without asking.
    """
    conflicts = existing_download_targets(project, keys)
    if not conflicts:
        return None

    requested = (
        list(keys)
        if keys is not None
        else [
            k
            for k, v in (project.raw_variables or {}).items()
            if type(v).__name__ == "GEEVar"
        ]
    )
    n_conflicts, n_requested = len(conflicts), max(len(requested), len(conflicts))
    remaining = max(n_requested - n_conflicts, 0)

    if n_conflicts == 1 and n_requested == 1:
        key, _path = conflicts[0]
        title = t("tiles.variables.confirm_overwrite_title_one")
        message = t("tiles.variables.confirm_overwrite_message_one", name=key)
        note = t("tiles.variables.confirm_overwrite_note_one")
    else:
        title = t("tiles.variables.confirm_overwrite_title")
        message = t(
            "tiles.variables.confirm_overwrite_message",
            count=n_conflicts,
            total=n_requested,
        )
        note = plural(
            remaining,
            "tiles.variables.confirm_overwrite_note_one_left",
            "tiles.variables.confirm_overwrite_note",
            remaining=remaining,
        )

    return OverwritePrompt(
        keys=tuple(key for key, _ in conflicts),
        title=title,
        message=message,
        note=note,
        details=tuple(_file_line(project, path) for _, path in conflicts),
    )
