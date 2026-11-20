"""Pure, UI-agnostic helpers for the project panel.

Kept free of Solara/ipyvuetify so the display and validation logic can be
unit-tested without a render harness.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from gui.i18n import plural, t

_NAME_RE = re.compile(r"[A-Za-z0-9 _-]+")


def compute_app_title(project, dirty: bool, base: str = None) -> str:
    """Header title: always '<base>', plus a '●' when there are unsaved changes.

    The project name is shown elsewhere, not in the header. Plain text (header
    cannot host a widget).
    """
    base = base if base is not None else t("app.title")
    suffix = t("app.title_dirty_suffix") if dirty and project is not None else ""
    return f"{base}{suffix}"


def format_relative(when: datetime, now: datetime) -> str:
    """Coarse relative time: 'just now' / 'N min ago' / 'N hr ago' / 'N days ago'."""
    seconds = max(0, int((now - when).total_seconds()))
    if seconds < 60:
        return t("time.just_now")
    if seconds < 3600:
        m = int(seconds // 60)
        return plural(m, "time.minutes_ago_one", "time.minutes_ago_other")
    if seconds < 86400:
        h = int(seconds // 3600)
        return plural(h, "time.hours_ago_one", "time.hours_ago_other")
    d = int(seconds // 86400)
    return plural(d, "time.days_ago_one", "time.days_ago_other")


def format_last_saved(when: Optional[datetime], now: datetime) -> str:
    """Return 'never saved' when None, else 'saved <relative>'."""
    if when is None:
        return t("project.never_saved")
    return t("project.saved_relative", time_ago=format_relative(when, now))


@dataclass
class CountChip:
    """One count chip for the load list.

    A label plus whether it's accented (colored) rather than neutral grey.
    """

    label: str
    accent: bool


def project_count_chips(info) -> list[CountChip]:
    """Chip specs summarising one saved project's contents.

    In display order: raw, processed, models, predictions.

    Raw/processed are always neutral. The models chip appends ``(K trained)``
    when any models exist and is accented when at least one is trained. The
    predictions chip is accented when there is at least one prediction.
    """
    models_label = plural(info.model_count, "chips.models_one", "chips.models_other")
    if info.model_count:
        models_label += t(
            "project.chip_models_trained", trained=info.trained_model_count
        )
    return [
        CountChip(plural(info.raw_count, "chips.raw_one", "chips.raw_other"), False),
        CountChip(
            plural(
                info.processed_count, "chips.processed_one", "chips.processed_other"
            ),
            False,
        ),
        CountChip(models_label, info.trained_model_count >= 1),
        CountChip(
            plural(
                info.prediction_count,
                "chips.predictions_one",
                "chips.predictions_other",
            ),
            info.prediction_count > 0,
        ),
    ]


@dataclass
class NameValidation:
    """Outcome of checking a project name."""

    valid: bool
    cleaned: str
    exists: bool
    error: Optional[str] = None


def validate_project_name(name: str, existing_names: list[str]) -> NameValidation:
    """Validate a new project name.

    Non-empty and filesystem-safe ([A-Za-z0-9 _-]). ``exists`` is informational
    (a warning, not an error).
    """
    cleaned = name.strip()
    if not cleaned:
        return NameValidation(False, "", False, t("project.validation_empty_name"))
    if not _NAME_RE.fullmatch(cleaned):
        return NameValidation(
            False,
            cleaned,
            False,
            t("project.validation_invalid_chars"),
        )
    return NameValidation(True, cleaned, cleaned in existing_names, None)


def overwrite_needed(
    name: str, last_saved: Optional[datetime], existing_names: list[str]
) -> bool:
    """True when saving would clobber a different project.

    That is: a new/never-saved-this-session project whose name already exists
    on disk.
    """
    return last_saved is None and name in existing_names


def manage_projects_label(count: Optional[int]) -> str:
    """Label for the empty-state 'manage projects' button.

    ``count`` is the number of saved projects on disk; ``None`` when the scan
    failed. 0 or None → a neutral invite to create one instead; otherwise the
    count is surfaced so the button is worth a click before opening the dialog.
    """
    if not count:  # 0 or None
        return t("project.manage_none")
    return t("project.manage_count", count=count)


def filter_project_infos(infos: list, query: str) -> list:
    """Saved projects whose name contains ``query`` (case-insensitive).

    A blank query returns every project. Input order is preserved — the caller
    has already sorted them.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return list(infos)
    return [i for i in infos if needle in i.name.lower()]


def format_size(num_bytes: int) -> str:
    """Human size for the delete-confirm message: '512 B', '2.9 MB', '3.2 GB'.

    Binary steps (1024) with conventional labels — it matches what ``du -h``
    reports, which is what the user sees if they check the folder themselves.
    """
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def delete_confirm_valid(typed: str, name: str) -> bool:
    """True when the user has typed the project's exact name.

    Deleting is irreversible and can discard GBs, so the match is exact and
    case-sensitive; only surrounding whitespace is forgiven.
    """
    return (typed or "").strip() == name


def aoi_project_name(aoi_name: str, when: datetime) -> str:
    """Auto-name for a project created from an AOI selection.

    ``<aoi>_<yyyymmdd>`` (e.g. ``San Marino_20260623``). The date is the creation
    moment, so the same AOI picked on different days yields distinct,
    non-colliding project names.
    """
    return f"{aoi_name}_{when:%Y%m%d}"
