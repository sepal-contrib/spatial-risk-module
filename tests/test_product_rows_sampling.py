"""sample_rows(): session-job overlay for the Samples list."""
from types import SimpleNamespace

from gui.scripts.product_rows import sample_rows


def _project(**samples):
    return SimpleNamespace(samples=samples)


def _sample(strategy="random", allocation=None, n_total=10, class_counts=None):
    return SimpleNamespace(
        strategy=strategy, allocation=allocation,
        n_total=n_total, class_counts=class_counts or {},
    )


def test_registered_samples_render_as_ready_rows():
    rows = sample_rows(_project(random_1=_sample()), [])
    assert len(rows) == 1
    assert rows[0]["kind"] == "sample"
    assert rows[0]["key"] == "random_1"
    assert rows[0]["status"] == "ready"


def test_running_job_renders_after_products():
    jobs = [{"id": "j1", "name": "random_2", "strategy": "random",
             "status": "running", "error": None,
             "n_total": None, "class_counts": None}]
    rows = sample_rows(_project(random_1=_sample()), jobs)
    assert [r["kind"] for r in rows] == ["sample", "job"]
    assert rows[1]["status"] == "running" and rows[1]["job_id"] == "j1"


def test_completed_job_suppressed_once_registered():
    jobs = [{"id": "j1", "name": "random_1", "strategy": "random",
             "status": "completed", "error": None,
             "n_total": 10, "class_counts": None}]
    rows = sample_rows(_project(random_1=_sample()), jobs)
    # the product row supersedes the completed job row
    assert [r["kind"] for r in rows] == ["sample"]


def test_completed_job_kept_if_registration_missing():
    jobs = [{"id": "j1", "name": "gone", "strategy": "random",
             "status": "completed", "error": None,
             "n_total": 10, "class_counts": None}]
    rows = sample_rows(_project(), jobs)
    assert [r["kind"] for r in rows] == ["job"]


def test_failed_job_carries_error():
    jobs = [{"id": "j1", "name": "random_2", "strategy": "random",
             "status": "failed", "error": "boom",
             "n_total": None, "class_counts": None}]
    rows = sample_rows(_project(), jobs)
    assert rows[0]["status"] == "failed" and rows[0]["error"] == "boom"
