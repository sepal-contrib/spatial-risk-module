"""Derived-layer rows: registry products + session-job overlay with suppression."""

import types

from gui.scripts.product_rows import derived_rows


def _proj(processed=None):
    return types.SimpleNamespace(processed_variables=processed or {})


def _var(name, year=None):
    return types.SimpleNamespace(name=name, year=year)


def _job(job_id="a1", name="loss_forest_2010_2015", output_key=None, **kw):
    base = {
        "id": job_id,
        "name": name,
        "output_key": output_key if output_key is not None else name,
        "status": "running",
        "error": None,
    }
    base.update(kw)
    return base


def test_derived_rows_products_only():
    """With no jobs, every registered derived variable is a ready row."""
    p = _proj({"loss_forest_2010_2015": _var("loss_forest_2010_2015")})
    rows = derived_rows(p, [])
    assert [r["kind"] for r in rows] == ["variable"]
    assert rows[0]["key"] == "loss_forest_2010_2015"
    assert rows[0]["name"] == "loss_forest_2010_2015"
    assert rows[0]["status"] == "ready"
    assert rows[0]["error"] is None


def test_derived_rows_running_job_listed_before_products():
    """In-flight work sits above the finished layers."""
    p = _proj({"loss_forest_2010_2015": _var("loss_forest_2010_2015")})
    rows = derived_rows(p, [_job(name="forest_dist")])
    assert [r["kind"] for r in rows] == ["job", "variable"]
    assert rows[0]["key"] == "job_a1"
    assert rows[0]["job_id"] == "a1"
    assert rows[0]["name"] == "forest_dist"
    assert rows[0]["status"] == "running"


def test_derived_rows_newest_job_first():
    """The most recent submission leads, as in the other job lists."""
    jobs = [_job("a1", name="first"), _job("a2", name="second")]
    rows = derived_rows(_proj(), jobs)
    assert [r["name"] for r in rows] == ["second", "first"]


def test_derived_rows_completed_job_suppressed_once_registered():
    """The product row supersedes the job row that produced it."""
    p = _proj({"forest_dist": _var("forest_dist")})
    rows = derived_rows(p, [_job(name="forest_dist", status="completed")])
    assert [r["kind"] for r in rows] == ["variable"]


def test_derived_rows_completed_job_kept_when_registration_missing():
    """A run whose registration never landed must not silently vanish."""
    rows = derived_rows(_proj(), [_job(name="forest_dist", status="completed")])
    assert [r["kind"] for r in rows] == ["job"]
    assert rows[0]["status"] == "completed"


def test_derived_rows_suppression_matches_the_storage_key_not_the_name():
    """edge/dist inherits the source's year, so the key is name_year."""
    p = _proj({"forest_dist_2010": _var("forest_dist", year=2010)})
    job = _job(name="forest_dist", output_key="forest_dist_2010", status="completed")
    rows = derived_rows(p, [job])
    assert [r["kind"] for r in rows] == ["variable"]


def test_derived_rows_failed_job_is_kept_with_its_error():
    """A failure stays on screen until the user dismisses it."""
    job = _job(name="forest_dist", status="failed", error="gdal exploded")
    rows = derived_rows(_proj(), [job])
    assert [r["kind"] for r in rows] == ["job"]
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "gdal exploded"


def test_derived_rows_keys_filter_restricts_products():
    """Harmonization outputs stay out of the derived list."""
    p = _proj(
        {
            "loss_forest_2010_2015": _var("loss_forest_2010_2015"),
            "forest_2010": _var("forest", year=2010),
        }
    )
    rows = derived_rows(p, [], keys=["loss_forest_2010_2015"])
    assert [r["key"] for r in rows] == ["loss_forest_2010_2015"]


def test_derived_rows_keys_filter_does_not_hide_jobs():
    """A job's output is not in the registry yet, so it can't be in ``keys``."""
    p = _proj({"loss_forest_2010_2015": _var("loss_forest_2010_2015")})
    rows = derived_rows(p, [_job(name="forest_dist")], keys=["loss_forest_2010_2015"])
    assert [r["kind"] for r in rows] == ["job", "variable"]


def test_derived_rows_handles_a_missing_project():
    """No open project means no rows, not a crash."""
    assert derived_rows(None, []) == []
