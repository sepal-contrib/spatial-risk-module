"""Project load/save helpers for the Spatial Risk GUI."""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from spatialrisk.project import DATA_DIR, Project

logger = logging.getLogger("spatial_risk")


def load_project(project_name: str) -> Project:
    """Load a project from disk by name."""
    return Project.load(project_name)


def save_project(project: Project) -> Path:
    """Save a project to disk, returns the saved path."""
    return project.save()


def list_projects(data_dir: Path) -> list[str]:
    """Return names of all saved projects found under data_dir."""
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and (p / f"{p.name}_project.json").exists()
    )


@dataclass
class ProjectInfo:
    """Lightweight metadata for one saved project, for the load picker."""

    name: str
    raw_count: int
    processed_count: int
    model_count: int
    modified: Optional[datetime]
    readable: bool
    trained_model_count: int = 0
    prediction_count: int = 0
    error: Optional[str] = None


def list_project_infos(data_dir: Path) -> list[ProjectInfo]:
    """Scan ``data_dir`` and return metadata for each saved project.

    A directory counts as a project only if it contains
    ``{name}_project.json``. The JSON is parsed for variable/model counts and
    the file mtime; a corrupt/unreadable file yields ``readable=False`` with an
    ``error`` reason rather than being dropped silently.
    """
    infos: list[ProjectInfo] = []
    if not data_dir.exists():
        return infos
    for child in sorted(data_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        json_path = child / f"{child.name}_project.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            modified = datetime.fromtimestamp(json_path.stat().st_mtime)
            models = data.get("models", {})
            trained_model_count = sum(
                1 for m in models.values() if isinstance(m, dict) and m.get("trained")
            )
            infos.append(
                ProjectInfo(
                    name=child.name,
                    raw_count=len(data.get("raw_variables", {})),
                    processed_count=len(data.get("processed_variables", {})),
                    model_count=len(models),
                    modified=modified,
                    readable=True,
                    trained_model_count=trained_model_count,
                    prediction_count=len(data.get("predictions", {})),
                )
            )
        except Exception as exc:  # corrupt JSON, permissions, etc.
            infos.append(
                ProjectInfo(
                    name=child.name,
                    raw_count=0,
                    processed_count=0,
                    model_count=0,
                    modified=None,
                    readable=False,
                    error=str(exc),
                )
            )
    return infos


def _project_dir(name: str, data_dir: Path) -> Path:
    """Resolve a saved project's folder, refusing anything that is not one.

    ``delete_project`` calls ``shutil.rmtree`` on the result, so this is the
    security boundary: the target must be a direct child of ``data_dir``.
    Symlinks resolve out of ``data_dir`` and are therefore refused — we do not
    follow a link out of the data dir.
    """
    cleaned = (name or "").strip()
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid project name: {name!r}")
    if "/" in cleaned or "\\" in cleaned or Path(cleaned).name != cleaned:
        raise ValueError(f"Invalid project name: {name!r}")

    try:
        target = (data_dir / cleaned).resolve()
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Invalid project path: {name!r}") from exc
    if target.parent != data_dir.resolve():
        raise ValueError(f"Refusing to delete outside the data directory: {target}")
    return target


def project_dir_size(name: str, data_dir: Path = DATA_DIR) -> int:
    """Total bytes on disk for a saved project's folder (0 when it does not exist).

    Only stats files — never reads them — so it stays cheap even for a multi-GB
    project. Entries that vanish or deny access mid-walk are skipped.
    """
    try:
        folder = _project_dir(name, data_dir)
    except ValueError:
        return 0
    if not folder.is_dir():
        return 0
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:  # pragma: no cover - races, broken links
            continue
    return total


def delete_project(name: str, data_dir: Path = DATA_DIR) -> bool:
    """Permanently delete a saved project's folder. False when it does not exist.

    Removes everything: the manifest, downloaded rasters, samples, trained models
    and predictions. There is no undo — the caller confirms.

    Raises ``ValueError`` when the target is not a project folder directly inside
    ``data_dir`` (see ``_project_dir``). A folder counts as a project when it
    contains ``{name}_project.json``; that file merely has to exist, so *corrupt*
    projects stay deletable — the case users most want to clean up.

    The manifest is unlinked **last**, so a delete that fails part-way stays
    recoverable — see below.
    """
    folder = _project_dir(name, data_dir)
    if not folder.exists():
        return False
    if not folder.is_dir():
        raise ValueError(f"Not a project folder: {folder}")
    manifest = folder / f"{folder.name}_project.json"
    if not manifest.exists():
        raise ValueError(f"Not a project folder (no manifest): {folder}")

    # Contents first, manifest last — deliberately NOT a plain rmtree(folder).
    # rmtree unlinks in os.scandir order, which routinely removes the manifest
    # before a deep subdirectory. If that subdirectory then fails (permissions, a
    # raster still held open by a reader), what is left is a multi-GB folder that no
    # longer looks like a project: list_project_infos drops it from the Manage list
    # and this function refuses it ("no manifest") — undeletable through the UI, and
    # invisible. Keeping the manifest until everything else is gone means a partial
    # failure still lists, and a retry can still finish the job.
    #
    # Symlink handling matches rmtree's: a symlinked directory is unlinked, never
    # followed. (`_project_dir` has already refused a symlinked project root.)
    for child in list(folder.iterdir()):  # materialised: we mutate the directory
        if child == manifest:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    # The snapshot above is taken once, up front. If anything lands in the folder
    # afterwards — a writer that slipped the mark, an NFS silly-rename of a
    # top-level file still held open by a reader — the loop never saw it and so
    # never removed it. Unlinking the manifest anyway and letting rmdir() fail on
    # that leftover would already have done the one thing "manifest last" exists
    # to prevent: the project must stay a project (manifest present) until the
    # folder is verified genuinely empty, not merely believed to be.
    leftover = [c for c in folder.iterdir() if c != manifest]
    if leftover:
        raise OSError(f"Refusing to finish the delete, folder not empty: {leftover[0]}")
    manifest.unlink()
    folder.rmdir()
    logger.info("Deleted project folder: %s", folder)
    return True
