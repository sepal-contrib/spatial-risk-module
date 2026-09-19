"""Allocation job runner, row builder and transactional delete."""

from pathlib import Path

import pytest

from gui.scripts.allocation_runner import (
    AllocationForm,
    BordersSelection,
    allocation_rows,
    delete_allocation_run,
    resolve_borders_file,
    run_allocation,
    validate_form,
)
from spatialrisk.allocations import AllocationRun
from spatialrisk.project import Project


def _form(**kw):
    """Build an AllocationForm with a complete, runnable set of inputs."""
    base = dict(
        name="reserve_north",
        prediction_key="icar_run",
        user_defrate_path=None,
        borders=BordersSelection(method="FILE", file_path="/data/borders.gpkg"),
        mask_file=None,
        defor_juris_ha=20000.0,
        years_forecast=4,
        density_extent=None,
    )
    base.update(kw)
    return AllocationForm(**base)


def _project(monkeypatch, tmp_path, name):
    """Project rooted at *tmp_path* (the repo's downloads_folder convention)."""
    import spatialrisk.project as project_module

    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    project = Project(project_name=name)
    project.initialize_folders()
    return project


def _record(run_dir, key="reserve_abc123"):
    """A saved AllocationRun pointing at *run_dir*."""
    return AllocationRun(
        name="reserve",
        run_id="abc123",
        borders_file="/b.gpkg",
        defor_juris_ha=1.0,
        years_forecast=4,
        annual_ha=2.0,
        total_ha=8.0,
        out_dir=str(run_dir),
        csv_path=str(Path(run_dir) / "defor_project.csv"),
    )


def test_validate_form_rejects_empty_name():
    """A run without a name cannot be saved or foldered."""
    assert "name" in validate_form(None, _form(name="  ")).lower()


def test_validate_form_rejects_missing_risk_map():
    """No prediction selected means nothing to allocate over.

    External risk maps are NOT an allocation-side input: they enter the
    project through the inference tab and arrive here as predictions.
    """
    msg = validate_form(None, _form(prediction_key=None))
    assert "prediction" in msg.lower()


def test_validate_form_rejects_non_positive_years():
    """A forecast period of zero years would divide by zero downstream."""
    assert "year" in validate_form(None, _form(years_forecast=0)).lower()


def test_validate_form_rejects_negative_hectares():
    """Negative expected deforestation is meaningless."""
    assert "hectare" in validate_form(None, _form(defor_juris_ha=-1)).lower()


def test_validate_form_accepts_a_complete_form(tmp_path):
    """A fully specified form whose files exist passes validation."""
    borders = tmp_path / "borders.gpkg"
    borders.write_text("")
    selection = BordersSelection(method="FILE", file_path=str(borders))
    assert validate_form(None, _form(borders=selection)) is None


def test_validate_form_rejects_a_borders_file_that_is_not_there(tmp_path):
    """Paths are checked before the run starts, not deep inside GDAL."""
    selection = BordersSelection(method="FILE", file_path=str(tmp_path / "gone.gpkg"))
    msg = validate_form(None, _form(borders=selection))
    assert "does not exist" in msg


def test_validate_form_rejects_no_borders_at_all():
    """The core cannot crop a risk map without a project area."""
    assert "borders" in validate_form(None, _form(borders=None)).lower()


def test_validate_form_rejects_an_unknown_borders_method():
    """A typo must fail loudly, not resolve to some default."""
    selection = BordersSelection(method="ADMIN", admin_code="3431")
    assert "borders" in validate_form(None, _form(borders=selection)).lower()


def test_validate_form_rejects_an_admin_selection_with_no_code():
    """Picking 'Region' without picking a region is incomplete."""
    selection = BordersSelection(method="ADMIN1")
    assert "administrative" in validate_form(None, _form(borders=selection)).lower()


def test_validate_form_rejects_an_asset_filter_with_no_value():
    """AssetSelectComponent publishes {column: X, value: None} mid-filter.

    Accepting it would silently allocate over the whole unfiltered collection.
    """
    selection = BordersSelection(
        method="ASSET",
        asset={
            "asset_id": "users/me/t",
            "type": "TABLE",
            "column": "adm1",
            "value": None,
        },
    )
    assert "adm1" in validate_form(None, _form(borders=selection))


def test_validate_form_accepts_an_admin_selection():
    """ADMIN borders are materialized at run time, so no file check applies."""
    selection = BordersSelection(method="ADMIN1", admin_code="3431")
    assert validate_form(None, _form(borders=selection)) is None


def test_resolve_borders_file_rewrites_a_file_selection(tmp_path):
    """FILE is canonicalized too: one meaning for borders_file, sidecars gone."""
    import geopandas as gpd
    from shapely.geometry import box

    src = tmp_path / "src.geojson"
    gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(src)

    out = tmp_path / "run"
    path = resolve_borders_file(
        BordersSelection(method="FILE", file_path=str(src)), out
    )

    assert path == out / "project_borders.gpkg"
    assert path.exists()
    assert len(gpd.read_file(path)) == 1


def test_resolve_borders_file_creates_the_run_directory(tmp_path):
    """Resolution now runs before the core creates out_dir."""
    import geopandas as gpd
    from shapely.geometry import box

    src = tmp_path / "src.geojson"
    gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(src)

    out = tmp_path / "not" / "there" / "yet"
    resolve_borders_file(BordersSelection(method="FILE", file_path=str(src)), out)

    assert out.is_dir()


def test_run_allocation_registers_a_record(tmp_path, monkeypatch):
    """A successful run lands in project.allocations with its provenance."""
    import gui.scripts.allocation_runner as runner
    from spatialrisk.allocation import AllocationResult
    from spatialrisk.predictions.prediction import Prediction

    project = _project(monkeypatch, tmp_path, "alloc_run")
    # A real Prediction, not a stub: run_allocation autosaves, and save()
    # serializes every registered prediction via model_dump().
    project.predictions["icar_run"] = Prediction(
        path=tmp_path / "prob.tif",
        model_key="icar",
        dataset_name="forecast",
    )

    monkeypatch.setattr(
        runner,
        "resolve_defrate_table",
        lambda *a, **k: runner.DefrateSource(
            path=tmp_path / "r.csv", provenance="computed"
        ),
    )

    def fake_allocate(**kwargs):
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return AllocationResult(
            annual_ha=312.4,
            total_ha=1249.6,
            out_dir=out,
            csv_path=out / "defor_project.csv",
            defrate_path=out / "defrate.csv",
            cropped_riskmap_path=out / "project_riskmap.tif",
        )

    monkeypatch.setattr(runner, "_allocate", fake_allocate)
    monkeypatch.setattr(
        runner,
        "resolve_borders_file",
        lambda selection, out_dir: Path(out_dir) / "b.gpkg",
    )

    record = run_allocation(project, _form(), job_id="job1")

    assert isinstance(record, AllocationRun)
    assert record.storage_key() in project.allocations
    assert record.annual_ha == pytest.approx(312.4)
    assert record.prediction_key == "icar_run"
    assert record.prediction_snapshot["model_key"] == "icar"
    assert Path(record.out_dir).parent.name == "allocation"
    assert record.borders_file.endswith("b.gpkg")
    assert record.borders_source["method"] == "FILE"


def test_form_has_no_external_riskmap_channel():
    """External risk maps are imported on the inference tab, never here."""
    import dataclasses

    fields = [f.name for f in dataclasses.fields(AllocationForm)]
    assert "external_riskmap" not in fields


def test_allocation_rows_merge_records_and_jobs(tmp_path, monkeypatch):
    """In-flight jobs sort ahead of saved records in the list."""
    project = _project(monkeypatch, tmp_path, "alloc_rows")
    project.allocations["reserve_abc123"] = _record("/o")
    jobs = [{"id": "j1", "name": "pending_run", "status": "running"}]

    rows = allocation_rows(project, jobs)

    assert [r["kind"] for r in rows] == ["job", "record"]
    assert rows[0]["status"] == "running"
    assert rows[1]["annual_ha"] == pytest.approx(2.0)
    assert rows[1]["key"] == "reserve_abc123"


def test_completed_job_row_is_superseded_by_its_saved_record(tmp_path, monkeypatch):
    """A finished job stops rendering once its record is registered.

    Without the skip the run shows up twice: a stale "completed" job row with
    no figures, and the real record row right under it.
    """
    project = _project(monkeypatch, tmp_path, "alloc_supersede")
    project.allocations["reserve_abc123"] = _record("/o")
    jobs = [
        {
            "id": "j1",
            "name": "reserve",
            "status": "completed",
            "record_key": "reserve_abc123",
        }
    ]

    rows = allocation_rows(project, jobs)

    assert [r["kind"] for r in rows] == ["record"]


def test_completed_job_row_survives_until_its_record_lands(tmp_path, monkeypatch):
    """The window between 'completed' and the registry publish still shows a row.

    The worker stamps the key and flips the status before the project reactive
    republishes, so a completed job whose record is not in the registry yet
    must keep its row rather than vanishing from the list.
    """
    project = _project(monkeypatch, tmp_path, "alloc_window")
    jobs = [
        {
            "id": "j1",
            "name": "reserve",
            "status": "completed",
            "record_key": "reserve_abc123",
        }
    ]

    rows = allocation_rows(project, jobs)

    assert [r["kind"] for r in rows] == ["job"]


def test_delete_allocation_run_removes_record_and_folder(tmp_path, monkeypatch):
    """Deleting drops the registry entry and the run's output folder."""
    project = _project(monkeypatch, tmp_path, "alloc_del")
    run_dir = Path(project.folders.project_folder) / "allocation" / "reserve_abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "defor_project.csv").write_text("x")
    project.allocations["reserve_abc123"] = _record(run_dir)

    assert delete_allocation_run(project, "reserve_abc123") is True
    assert "reserve_abc123" not in project.allocations
    assert not run_dir.exists()


def test_delete_allocation_run_restores_record_when_save_fails(tmp_path, monkeypatch):
    """A failed manifest save must not leave a record pointing at deleted files."""
    project = _project(monkeypatch, tmp_path, "alloc_del_fail")
    run_dir = Path(project.folders.project_folder) / "allocation" / "reserve_abc123"
    run_dir.mkdir(parents=True)
    project.allocations["reserve_abc123"] = _record(run_dir)

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    # Patch the class: pydantic models reject setting non-field attributes.
    monkeypatch.setattr(Project, "save", boom)

    with pytest.raises(OSError):
        delete_allocation_run(project, "reserve_abc123")

    assert "reserve_abc123" in project.allocations
    assert run_dir.exists()


def test_delete_allocation_run_refuses_paths_outside_the_project(tmp_path, monkeypatch):
    """A record whose out_dir escapes <project>/allocation never gets removed."""
    project = _project(monkeypatch, tmp_path, "alloc_del_evil")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    project.allocations["evil_abc123"] = AllocationRun(
        name="evil",
        run_id="abc123",
        borders_file="/b.gpkg",
        defor_juris_ha=1.0,
        years_forecast=4,
        annual_ha=1.0,
        total_ha=1.0,
        out_dir=str(outside),
        csv_path=str(outside / "c.csv"),
    )

    delete_allocation_run(project, "evil_abc123")

    assert outside.exists()  # never removed: outside <project>/allocation


# --- form-time helpers: rate-table preview and mask choices -------------


def _preview_pred(**kw):
    from types import SimpleNamespace

    base = dict(
        path=Path("/data/p/inference/forecast/prob.tif"),
        model_key="icar",
        dataset_name="forecast",
        window=None,
        defrate_path=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _preview_project(predictions=None, processed_variables=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        predictions=predictions or {},
        processed_variables=processed_variables or {},
    )


def test_preview_shows_a_persisted_table_without_computing(tmp_path):
    """A JNR/MW run's own table is previewed by name, with the JNR caveat."""
    from gui.scripts.allocation_runner import preview_defrate_source

    csv = tmp_path / "defrate_cat_bm_forecast.csv"
    csv.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _preview_project(
        {"jnr_run": _preview_pred(model_key="jnr", defrate_path=csv)}
    )

    src = preview_defrate_source(project, "jnr_run")

    assert src.path == csv
    assert src.provenance == "persisted"
    assert src.caveat is not None


def test_preview_of_a_far_prediction_names_the_future_table(tmp_path):
    """FAR families preview the to-be-computed path; nothing is computed."""
    from gui.scripts.allocation_runner import preview_defrate_source

    pred = _preview_pred(path=tmp_path / "prob.tif")
    project = _preview_project({"icar_run": pred})

    src = preview_defrate_source(project, "icar_run")

    assert src.provenance == "computed"
    assert src.path == tmp_path / "defrate_cat_icar_forecast.csv"
    assert not src.path.exists()  # preview must not compute it


def test_preview_reports_an_unresolvable_table_instead_of_raising(tmp_path):
    """An MW run with no sibling table previews as unavailable, with the reason."""
    from gui.scripts.allocation_runner import preview_defrate_source

    prob = tmp_path / "prob_mw_11_forecast.tif"
    prob.write_bytes(b"")
    project = _preview_project(
        {"mw_run": _preview_pred(model_key="mw", window=11, path=prob)}
    )

    src = preview_defrate_source(project, "mw_run")

    assert src.provenance == "unavailable"
    assert src.path is None
    assert "rate table" in (src.caveat or "")


def test_preview_honours_the_user_override(tmp_path):
    """An explicit table short-circuits the preview like it does the run."""
    from gui.scripts.allocation_runner import preview_defrate_source

    csv = tmp_path / "mine.csv"
    csv.write_text("cat\n1\n")

    src = preview_defrate_source(_preview_project(), "nope", user_path=csv)

    assert src.provenance == "user"
    assert src.path == csv


def test_resolve_returns_an_existing_computed_table_even_with_compute_off(tmp_path):
    """compute=False forbids computing, not reusing a table already on disk."""
    from gui.scripts.allocation_runner import resolve_defrate_table

    pred = _preview_pred(path=tmp_path / "prob.tif")
    existing = tmp_path / "defrate_cat_icar_forecast.csv"
    existing.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _preview_project({"icar_run": pred})

    src = resolve_defrate_table(project, "icar_run", compute=False)

    assert src.path == existing
    assert src.provenance == "computed"


def test_mask_items_lists_processed_raster_variables(tmp_path):
    """The mask choices are the project's processed rasters, not free files."""
    from types import SimpleNamespace

    from gui.scripts.allocation_runner import mask_items

    project = _preview_project(
        processed_variables={
            "forest_gfc_tc30": SimpleNamespace(path=tmp_path / "forest.tif"),
            "roads": SimpleNamespace(path=tmp_path / "roads.gpkg"),  # vector: out
            "elevation": SimpleNamespace(path=tmp_path / "elev.vrt"),
        }
    )

    items = mask_items(project)

    assert [i["text"] for i in items] == ["elevation", "forest_gfc_tc30"]
    assert items[1]["value"] == str(tmp_path / "forest.tif")


def test_mask_items_of_an_empty_project_is_empty():
    """No processed variables (or no project) means no mask choices."""
    from gui.scripts.allocation_runner import mask_items

    assert mask_items(None) == []
    assert mask_items(_preview_project()) == []


def test_allocation_rows_carry_the_run_source(tmp_path, monkeypatch):
    """Record rows say what the run was computed from (model — dataset)."""
    project = _project(monkeypatch, tmp_path, "alloc_rows_src")
    record = _record(tmp_path / "run")
    record.prediction_snapshot = {
        "model_key": "icar",
        "dataset_name": "forecast",
        "window": None,
        "year": None,
        "path": "/p.tif",
    }
    project.allocations["reserve_abc123"] = record

    rows = allocation_rows(project)

    assert rows[0]["source"] == "ICAR · icar — forecast"


def test_allocation_rows_source_is_none_for_external_maps(tmp_path, monkeypatch):
    """An external-map run has no prediction to name; the widget labels it."""
    project = _project(monkeypatch, tmp_path, "alloc_rows_ext")
    project.allocations["reserve_abc123"] = _record(tmp_path / "run")

    rows = allocation_rows(project)

    assert rows[0]["source"] is None


def _square_gdf(crs="EPSG:4326"):
    """A one-polygon GeoDataFrame — the shape a borders selection must yield."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=crs)


def test_resolve_borders_file_fetches_an_admin_boundary(tmp_path, monkeypatch):
    """ADMIN goes through pysepal's non-GEE WFS path and lands as the canonical file."""
    import geopandas as gpd

    import gui.scripts.allocation_runner as runner

    monkeypatch.setattr(runner, "_admin_gdf", lambda method, code: _square_gdf())

    out = tmp_path / "run"
    path = resolve_borders_file(
        BordersSelection(method="ADMIN1", admin_code="3431"), out
    )

    assert path == out / "project_borders.gpkg"
    assert len(gpd.read_file(path)) == 1


def test_resolve_borders_file_rejects_a_non_polygonal_selection(tmp_path, monkeypatch):
    """A TABLE asset may be points; that cannot crop a risk map."""
    import geopandas as gpd
    from shapely.geometry import Point

    import gui.scripts.allocation_runner as runner

    points = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    monkeypatch.setattr(runner, "_asset_gdf", lambda asset, out_dir: points)

    selection = BordersSelection(method="ASSET", asset={"asset_id": "users/me/t"})
    with pytest.raises(runner.AllocationResolveError, match="polygon"):
        resolve_borders_file(selection, tmp_path / "run")


def test_resolve_borders_file_rejects_borders_without_a_crs(tmp_path, monkeypatch):
    """Without a CRS the core cannot reproject them onto the risk map's grid."""
    import gui.scripts.allocation_runner as runner

    monkeypatch.setattr(runner, "_admin_gdf", lambda m, c: _square_gdf(crs=None))

    with pytest.raises(runner.AllocationResolveError, match="CRS"):
        resolve_borders_file(
            BordersSelection(method="ADMIN0", admin_code="62"), tmp_path / "run"
        )


def test_resolve_borders_file_rejects_an_empty_selection(tmp_path, monkeypatch):
    """An admin code that matches nothing must not produce an empty cutline."""
    import geopandas as gpd

    import gui.scripts.allocation_runner as runner

    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    monkeypatch.setattr(runner, "_admin_gdf", lambda m, c: empty)

    with pytest.raises(runner.AllocationResolveError, match="no features"):
        resolve_borders_file(
            BordersSelection(method="ADMIN0", admin_code="62"), tmp_path / "run"
        )


def test_resolve_borders_file_rejects_all_null_geometries(tmp_path, monkeypatch):
    """Attribute-only rows must not silently pass as an empty-but-valid layer.

    ``set(gdf.geom_type.dropna().unique())`` is the empty set when every
    geometry is null, and ``set() <= _POLYGONAL`` is True — so this used to
    sail through the check, get written to project_borders.gpkg, and only
    fail later as a confusing gdal.Warp cutline error.
    """
    import geopandas as gpd

    import gui.scripts.allocation_runner as runner

    all_null = gpd.GeoDataFrame({"col": [1, 2]}, geometry=[None, None], crs="EPSG:4326")
    monkeypatch.setattr(runner, "_admin_gdf", lambda m, c: all_null)

    with pytest.raises(runner.AllocationResolveError, match="null"):
        resolve_borders_file(
            BordersSelection(method="ADMIN0", admin_code="62"), tmp_path / "run"
        )


def test_resolve_borders_file_rejects_all_empty_polygons(tmp_path, monkeypatch):
    """An empty (but non-null) Polygon geometry is just as unusable as null."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    import gui.scripts.allocation_runner as runner

    all_empty = gpd.GeoDataFrame(geometry=[Polygon(), Polygon()], crs="EPSG:4326")
    monkeypatch.setattr(runner, "_admin_gdf", lambda m, c: all_empty)

    with pytest.raises(runner.AllocationResolveError, match="null"):
        resolve_borders_file(
            BordersSelection(method="ADMIN0", admin_code="62"), tmp_path / "run"
        )


def test_resolve_borders_file_rejects_mixed_null_and_polygon_geometries(
    tmp_path, monkeypatch
):
    """A partially null layer is corrupted data, not a smaller-but-valid AOI.

    Silently dropping the null rows would let a run quietly crop against an
    incomplete boundary with no indication anything was wrong.
    """
    import geopandas as gpd
    from shapely.geometry import box

    import gui.scripts.allocation_runner as runner

    mixed = gpd.GeoDataFrame(geometry=[None, box(0, 0, 1, 1)], crs="EPSG:4326")
    monkeypatch.setattr(runner, "_admin_gdf", lambda m, c: mixed)

    with pytest.raises(runner.AllocationResolveError, match="null"):
        resolve_borders_file(
            BordersSelection(method="ADMIN0", admin_code="62"), tmp_path / "run"
        )


def test_bad_borders_leave_no_file_behind(tmp_path, monkeypatch):
    """Validation runs before the write, so a rejected selection writes nothing."""
    import geopandas as gpd
    from shapely.geometry import Point

    import gui.scripts.allocation_runner as runner

    monkeypatch.setattr(
        runner,
        "_asset_gdf",
        lambda asset, out_dir: gpd.GeoDataFrame(
            geometry=[Point(0, 0)], crs="EPSG:4326"
        ),
    )

    out = tmp_path / "run"
    with pytest.raises(runner.AllocationResolveError):
        resolve_borders_file(
            BordersSelection(method="ASSET", asset={"asset_id": "users/me/t"}), out
        )

    assert not (out / "project_borders.gpkg").exists()


def test_asset_export_temp_file_is_removed_when_the_export_raises(
    tmp_path, monkeypatch
):
    """A partial TABLE export must not survive a failed/timed-out download.

    The export call used to sit outside the try/finally that unlinks the temp
    GeoJSON, so a large asset that wrote a partial file and then raised (e.g.
    on timeout) left that file behind forever.
    """
    import gui.scripts.allocation_runner as runner

    def fake_export(fc, filename, selectors=None, verbose=True, **kw):
        # Simulate geemap writing a partial file before the export fails.
        Path(filename).write_text("{}")
        raise TimeoutError("export timed out")

    # init_ee reads ~/.config/earthengine/credentials and raises when they
    # carry no project: keep the test independent of the machine's EE setup.
    monkeypatch.setattr("pysepal.scripts.utils.init_ee", lambda: None)
    monkeypatch.setattr(runner, "_ee_export_vector", fake_export)
    monkeypatch.setattr(runner, "_build_asset_fc", lambda asset: object())

    out = tmp_path / "run"
    out.mkdir(parents=True)

    with pytest.raises(TimeoutError):
        runner._asset_gdf({"asset_id": "users/me/t"}, out)

    assert not list(out.glob("*.geojson"))


def test_run_allocation_removes_the_run_directory_it_created_on_a_failed_resolution(
    tmp_path, monkeypatch
):
    """A directory created for a run must not be orphaned when resolution fails.

    Cleanup used to only remove an EMPTY out_dir, so a partial EE export (or
    any other file resolution wrote before failing) left the directory behind
    with no registered AllocationRun able to ever clean it up.
    """
    import gui.scripts.allocation_runner as runner
    from spatialrisk.predictions.prediction import Prediction

    project = _project(monkeypatch, tmp_path, "alloc_orphan")
    project.predictions["icar_run"] = Prediction(
        path=tmp_path / "prob.tif", model_key="icar", dataset_name="forecast"
    )

    class _FixedUUID:
        hex = "deadbeef"

    monkeypatch.setattr(runner.uuid, "uuid4", lambda: _FixedUUID())
    monkeypatch.setattr(
        runner,
        "resolve_defrate_table",
        lambda *a, **k: runner.DefrateSource(
            path=tmp_path / "r.csv", provenance="computed"
        ),
    )

    def fake_resolve(selection, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "project_borders_export.geojson").write_text("{}")
        raise runner.AllocationResolveError("export timed out")

    monkeypatch.setattr(runner, "resolve_borders_file", fake_resolve)

    with pytest.raises(runner.AllocationResolveError):
        run_allocation(project, _form(), job_id="job1")

    out_dir = (
        Path(project.folders.project_folder) / "allocation" / "reserve_north_deadbeef"
    )
    assert not out_dir.exists()


def test_run_allocation_does_not_delete_a_pre_existing_directory(tmp_path, monkeypatch):
    """Cleanup must only ever remove the directory THIS call created."""
    import gui.scripts.allocation_runner as runner
    from spatialrisk.predictions.prediction import Prediction

    project = _project(monkeypatch, tmp_path, "alloc_orphan_safe")
    project.predictions["icar_run"] = Prediction(
        path=tmp_path / "prob.tif", model_key="icar", dataset_name="forecast"
    )

    class _FixedUUID:
        hex = "deadbeef"

    monkeypatch.setattr(runner.uuid, "uuid4", lambda: _FixedUUID())
    monkeypatch.setattr(
        runner,
        "resolve_defrate_table",
        lambda *a, **k: runner.DefrateSource(
            path=tmp_path / "r.csv", provenance="computed"
        ),
    )

    out_dir = (
        Path(project.folders.project_folder) / "allocation" / "reserve_north_deadbeef"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "keep.txt").write_text("mine")

    monkeypatch.setattr(
        runner,
        "resolve_borders_file",
        lambda selection, out_dir: (_ for _ in ()).throw(
            runner.AllocationResolveError("boom")
        ),
    )

    with pytest.raises(runner.AllocationResolveError):
        run_allocation(project, _form(), job_id="job1")

    assert out_dir.exists()
    assert (out_dir / "keep.txt").exists()


def test_run_allocation_cleanup_failure_does_not_mask_the_original_error(
    tmp_path, monkeypatch
):
    """A cleanup OSError must never replace the real resolution failure."""
    import gui.scripts.allocation_runner as runner
    from spatialrisk.predictions.prediction import Prediction

    project = _project(monkeypatch, tmp_path, "alloc_mask")
    project.predictions["icar_run"] = Prediction(
        path=tmp_path / "prob.tif", model_key="icar", dataset_name="forecast"
    )

    def fake_resolve(selection, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        raise runner.AllocationResolveError("original failure")

    def boom_rmtree(*args, **kwargs):
        raise OSError("Directory not empty")

    monkeypatch.setattr(
        runner,
        "resolve_defrate_table",
        lambda *a, **k: runner.DefrateSource(
            path=tmp_path / "r.csv", provenance="computed"
        ),
    )
    monkeypatch.setattr(runner, "resolve_borders_file", fake_resolve)
    monkeypatch.setattr(runner.shutil, "rmtree", boom_rmtree)

    with pytest.raises(runner.AllocationResolveError, match="original failure"):
        run_allocation(project, _form(), job_id="job1")


def test_asset_export_target_is_geojson_not_gpkg(tmp_path, monkeypatch):
    """Geemap 0.36 allows csv/geojson/json/kml/kmz/shp only — .gpkg raises."""
    import gui.scripts.allocation_runner as runner

    seen = {}

    def fake_export(fc, filename, selectors=None, verbose=True, **kw):
        seen["filename"] = filename
        seen["selectors"] = selectors
        _square_gdf().to_file(filename, driver="GeoJSON")

    monkeypatch.setattr("pysepal.scripts.utils.init_ee", lambda: None)
    monkeypatch.setattr(runner, "_ee_export_vector", fake_export)
    monkeypatch.setattr(runner, "_build_asset_fc", lambda asset: object())

    out = tmp_path / "run"
    out.mkdir(parents=True)
    gdf = runner._asset_gdf({"asset_id": "users/me/t"}, out)

    assert seen["filename"].endswith(".geojson")
    assert seen["selectors"] == []  # geometry only: a cutline needs no attributes
    assert len(gdf) == 1
    assert not list(out.glob("*.geojson"))  # the temp export is cleaned up


def test_job_rows_carry_their_submission_entry_and_id():
    """allocation_rows passes the launch entry through so edit can prefill."""
    entry = _form(name="allocation_1")
    jobs = [
        {
            "id": "j1",
            "name": "allocation_1",
            "status": "failed",
            "error": "boom",
            "entry": entry,
        }
    ]
    rows = allocation_rows(None, jobs)
    assert rows[0]["job_id"] == "j1"
    assert rows[0]["entry"] is entry


def test_job_rows_without_an_entry_report_none():
    """A job launched before this change still renders; it just cannot be edited."""
    jobs = [{"id": "j1", "name": "allocation_1", "status": "failed", "error": "boom"}]
    rows = allocation_rows(None, jobs)
    assert rows[0]["job_id"] == "j1"
    assert rows[0]["entry"] is None


def test_run_allocation_passes_the_density_extent_to_the_core(tmp_path, monkeypatch):
    """The form's extent reaches the core call and lands on the record."""
    import gui.scripts.allocation_runner as runner
    from spatialrisk.allocation import AllocationResult
    from spatialrisk.predictions.prediction import Prediction

    project = _project(monkeypatch, tmp_path, "alloc_density")
    project.predictions["icar_run"] = Prediction(
        path=tmp_path / "prob.tif",
        model_key="icar",
        dataset_name="forecast",
    )
    monkeypatch.setattr(
        runner,
        "resolve_defrate_table",
        lambda *a, **k: runner.DefrateSource(
            path=tmp_path / "r.csv", provenance="computed"
        ),
    )
    seen = {}

    def fake_allocate(**kwargs):
        seen.update(kwargs)
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return AllocationResult(
            annual_ha=1.0,
            total_ha=4.0,
            out_dir=out,
            csv_path=out / "defor_project.csv",
            defrate_path=out / "defrate.csv",
            cropped_riskmap_path=out / "project_riskmap.tif",
            density_map_path=out / "deforestation_density_map.tif",
        )

    monkeypatch.setattr(runner, "_allocate", fake_allocate)
    monkeypatch.setattr(
        runner,
        "resolve_borders_file",
        lambda selection, out_dir: Path(out_dir) / "b.gpkg",
    )

    record = run_allocation(project, _form(density_extent="project"), job_id="j1")

    assert seen["defor_density_map"] == "project"
    assert record.density_extent == "project"
