"""Suite-wide collection-time setup.

Reacton's first ``t()`` call *during* a first render corrupts its internal
widget map and surfaces later as a ``use_event`` KeyError. It only reproduces
in isolated single-test runs, which makes it an unpleasant intermittent rather
than an honest failure.

Seven test modules already guard against this with an import-time
``t("common.cancel")`` plus a trail of ``# noqa: E402`` on the component
imports that must follow it. Five render modules do not
(test_evaluation, test_echarts_adapter, test_pipeline_header,
test_summary_tile_reactivity, test_postprocess_tile_threading). None of them
trips the bug today; doing the warm-up once here makes that a property of
the suite rather than a coincidence of which components each module happens
to mount.

This must stay an import-time call rather than an autouse fixture: conftest is
imported before any test module, whereas fixtures run only after the test
module — and its component imports — have already been evaluated.
"""

import sys

import pytest

from gui.i18n import t

t("common.cancel")


# Module-level InflightKeys, one per tile that runs per-item workers. A test
# that claims a key through a stubbed worker and fails before its own teardown
# leaves the key claimed for every later test in the process — the button it
# gates ("Harmonize all" on ``reference_inflight``, a row spinner elsewhere)
# then stays dead, as a failure in an unrelated test that moves with collection
# order. The tile modules' own autouse fixtures drain their key on the happy
# path; this one is the backstop, and it only touches modules a test already
# imported so pure-library tests never pull the GUI in.
_INFLIGHT_KEYS = (
    ("gui.tile.derived_map", "derived_toggle_inflight"),
    ("gui.tile.inference_tile", "preds_inflight"),
    ("gui.tile.postprocess_tile", "derived_inflight"),
    ("gui.tile.process_tile", "reference_inflight"),
    ("gui.tile.sampling_tile", "samples_pending"),
    ("gui.tile.variables_tile", "download_inflight"),
    ("gui.tile.variables_tile", "vars_inflight"),
)


@pytest.fixture(autouse=True)
def _drain_inflight_keys():
    """Release every module-level in-flight claim a test left behind."""
    yield
    for module_name, attr in _INFLIGHT_KEYS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        keys = getattr(module, attr)
        keys.release(*keys.value)
