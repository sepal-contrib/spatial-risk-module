"""rename_project moves the folder, the manifest and the paths inside it."""

import json

import pytest

from gui.scripts.project_io import list_projects, rename_project


def _make(root, name):
    folder = root.resolve() / name
    (folder / "data_raw").mkdir(parents=True)
    (folder / "data_raw" / "a.tif").write_bytes(b"x")
    manifest = {
        "project_name": name,
        # a layer that happens to share the project's name keeps it
        "raw_variables": {
            name: {"name": name, "path": str(folder / "data_raw" / "a.tif")}
        },
        "models": {
            "m": {"project_name": name, "model_path": str(folder / "far_rf" / "m.pkl")}
        },
        "elsewhere": f"/somewhere/else/{name}/b.tif",
        "lookalike": str(folder) + "_copy/b.tif",
    }
    (folder / f"{name}_project.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_rename_moves_folder_manifest_and_paths(tmp_path):
    """The folder, the manifest name and every path inside follow the new name."""
    _make(tmp_path, "old")
    rename_project("old", "new", tmp_path)

    new = tmp_path.resolve() / "new"
    assert list_projects(tmp_path) == ["new"]
    assert sorted(p.name for p in new.iterdir()) == ["data_raw", "new_project.json"]
    data = json.loads((new / "new_project.json").read_text(encoding="utf-8"))
    assert data["project_name"] == "new"
    assert data["models"]["m"] == {
        "project_name": "new",
        "model_path": str(new / "far_rf" / "m.pkl"),
    }
    assert data["raw_variables"]["old"] == {
        "name": "old",
        "path": str(new / "data_raw" / "a.tif"),
    }
    # paths outside the project folder are not ours to touch
    assert data["elsewhere"] == "/somewhere/else/old/b.tif"
    assert data["lookalike"].endswith("old_copy/b.tif")


def test_rename_refuses_a_taken_name_and_leaves_both(tmp_path):
    """A taken name is refused and nothing on disk changes."""
    _make(tmp_path, "a")
    (tmp_path / "b").mkdir()  # taken even without a manifest
    with pytest.raises(FileExistsError):
        rename_project("a", "b", tmp_path)
    assert list_projects(tmp_path) == ["a"]
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == [
        "a_project.json",
        "data_raw",
    ]


def test_rename_refuses_what_is_not_a_project(tmp_path):
    """A folder without a manifest, or a missing one, is not renamed."""
    (tmp_path / "bare").mkdir()
    with pytest.raises(FileNotFoundError):
        rename_project("bare", "x", tmp_path)
    with pytest.raises(FileNotFoundError):
        rename_project("gone", "x", tmp_path)


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", ""])
def test_rename_refuses_unsafe_names(tmp_path, bad):
    """Names that leave the data directory are refused on both sides."""
    _make(tmp_path, "ok")
    for old, new in ((bad, "fine"), ("ok", bad)):
        with pytest.raises(ValueError):
            rename_project(old, new, tmp_path)
    assert list_projects(tmp_path) == ["ok"]
