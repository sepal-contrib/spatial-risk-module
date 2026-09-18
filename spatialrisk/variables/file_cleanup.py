"""What a variable owns on disk, and whether it may be deleted.

Removing a variable unregisters it; the files it was pointing at are a separate
question, and a destructive one. This module answers that question *before*
anything is unlinked, so the Remove dialog can say exactly what would go and
:meth:`Project.delete_variable_files` can act on the same answer.

Two situations mean "delete nothing":

``outside_project``
    The path is not inside the project folder — a raster the user picked from
    their own drive through the file input. The app deletes only files it wrote.

``shared``
    Another registered variable (or the reference raster) points at the same
    file. Reprojected copies and re-registrations routinely alias one raster,
    and the survivor must keep working.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Shapefile companions: a lone .shp is unreadable, so they travel together.
# Anything named "<main file>.<something>" (slope.tif.ovr, slope.tif.aux.xml)
# is picked up separately, by prefix.
_SHAPEFILE_COMPANIONS = (
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
    ".qpj",
    ".sbn",
    ".sbx",
)


@dataclass(frozen=True)
class FilePlan:
    """The files a variable would take with it, and why it may not."""

    files: Tuple[Path, ...] = ()
    total_bytes: int = 0
    blocked: Optional[str] = None  # "outside_project" | "shared" | None
    blocked_by: Optional[str] = None  # the variable still using the file
    all_files: Tuple[Path, ...] = field(default=())  # what exists, blocked or not

    @property
    def deletable(self) -> bool:
        """True when ticking the box would actually remove something."""
        return bool(self.files)


def variable_files(var) -> List[Path]:
    """Every file on disk that belongs to *var* — the main one and its sidecars.

    Empty for a cloud variable: ``GEEVar.path`` is an Earth Engine asset id, not
    a filesystem path, and resolving it as one would be meaningless at best.
    """
    if var is None or type(var).__name__ == "GEEVar":
        return []
    raw_path = getattr(var, "path", None)
    if not raw_path:
        return []
    try:
        main = Path(raw_path)
        exists = main.is_file()
    except (OSError, ValueError):  # pragma: no cover - malformed path
        return []
    if not exists:
        return []

    found = [main]
    parent, prefix = main.parent, main.name + "."
    try:
        siblings = list(parent.iterdir())
    except OSError:  # pragma: no cover - vanished or unreadable folder
        siblings = []
    # ".ovr" / ".aux.xml" and friends hang off the full name, never the stem.
    found += [s for s in siblings if s.name.startswith(prefix) and s.is_file()]
    if main.suffix.lower() == ".shp":
        found += [
            c
            for ext in _SHAPEFILE_COMPANIONS
            for c in (main.with_suffix(ext),)
            if c.is_file()
        ]
    # dict.fromkeys: de-duplicate (a .shp.xml matches both rules) in order.
    return list(dict.fromkeys(found))


def _registered_variables(project):
    """(key, var) for everything that can hold a path: both registries + base."""
    for attr in ("raw_variables", "processed_variables"):
        for key, var in (getattr(project, attr, None) or {}).items():
            yield key, var
    base = getattr(project, "base_raster", None)
    if base is not None:
        yield getattr(base, "name", "base_raster"), base


def _find(project, key: str):
    """The variable stored under *key*, in either registry (None if unknown)."""
    for attr in ("raw_variables", "processed_variables"):
        var = (getattr(project, attr, None) or {}).get(key)
        if var is not None:
            return var
    return None


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:  # pragma: no cover - races
        return 0


def plan_variable_files(project, key: str) -> FilePlan:
    """What deleting the variable stored under *key* would remove from disk.

    Pure: nothing is unlinked, nothing is created (in particular this never
    touches ``project.folders``, whose getter builds the whole folder tree).
    """
    var = _find(project, key)
    files = variable_files(var)
    if not files:
        return FilePlan()
    found = tuple(files)

    try:
        project_dir = Path(project._project_dir()).resolve()
        inside = all(project_dir in f.resolve().parents for f in found)
    except (OSError, RuntimeError, AttributeError):  # pragma: no cover
        inside = False
    if not inside:
        return FilePlan(blocked="outside_project", all_files=found)

    mine = {f.resolve() for f in found}
    for other_key, other in _registered_variables(project):
        if other_key == key or other is var:
            continue
        if mine & {f.resolve() for f in variable_files(other)}:
            return FilePlan(blocked="shared", blocked_by=other_key, all_files=found)

    return FilePlan(
        files=found,
        total_bytes=sum(_size(f) for f in found),
        all_files=found,
    )
