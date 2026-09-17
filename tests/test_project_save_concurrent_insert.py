"""Project.save() must tolerate a worker registering a variable mid-save.

Every published Project copy shares one ``processed_variables`` dict (pydantic's
``model_copy`` is shallow), and derived-layer / download workers now run
concurrently. Serialising with a plain ``for k, v in dict.items()`` raises
``RuntimeError: dictionary changed size during iteration`` when another worker
inserts — which used to abort the job's continuation. The loops snapshot with
``list(...)`` instead.
"""

from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar


def _var(p, name):
    return LocalRasterVar.model_construct(
        name=name,
        data_type="raster",
        raster_type="continuous",
        path=None,
        project=p,
        processing_history=[],
    )


def test_save_tolerates_an_insert_during_serialisation(tmp_path, monkeypatch):
    """save() must not raise when a worker inserts into a registry mid-loop."""
    monkeypatch.setenv("SPATIAL_RISK_DATA_DIR", str(tmp_path))
    p = Project(project_name="race")
    p.processed_variables["first"] = _var(p, "first")

    original_dump = LocalRasterVar.model_dump

    def dump_and_insert(self, *args, **kwargs):
        # Stands in for a worker thread registering its output while save()
        # is walking the registry.
        if "late" not in p.processed_variables:
            p.processed_variables["late"] = _var(p, "late")
        return original_dump(self, *args, **kwargs)

    monkeypatch.setattr(LocalRasterVar, "model_dump", dump_and_insert)

    path = p.save()  # must not raise RuntimeError
    assert path.exists()
