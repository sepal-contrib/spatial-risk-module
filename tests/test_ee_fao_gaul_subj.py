"""``get_fao_gaul_subj`` must select GAUL features without inlining the AOI geometry.

``filterBounds(aoi)`` / ``filterBounds(aoi.geometry())`` put the AOI's
computed geometry into every download request built on the result; a large
table asset then fails with "Description length exceeds maximum" (reproduced
2026-09-11, ~1.1M-coordinate asset). A bounding-box filter alone over-selects
(345 vs 106 districts for that AOI) and the codes are exported as a byte, so
the selection has to stay exact: bbox pre-filter, then a spatial join against
the AOI collection.
"""

import types

import pytest

import spatialrisk.gee.ee_fao_gaul as gaul_module
from spatialrisk.gee.ee_fao_gaul import get_fao_gaul_subj


class _Rec:
    """Chainable stand-in recording method calls on a shared log."""

    def __init__(self, log, label):
        self._log = log
        self._label = label

    def __getattr__(self, method):
        def _call(*args, **kwargs):
            self._log.append((self._label, method, args, kwargs))
            return _Rec(self._log, f"{self._label}.{method}")

        return _call


class _Geometry:
    """Fake ee.Geometry: only ``bounds`` matters."""

    def bounds(self):
        return "BOUNDS"


@pytest.fixture()
def fake_ee(monkeypatch):
    """Stub ``ee`` with recording fakes; returns ``(log, fake_ee)``."""
    log = []

    class _FeatureCollection:
        """Fake ee.FeatureCollection: the GAUL asset and the AOI alike."""

        def __init__(self, *args):
            pass

        def geometry(self):
            return _Geometry()

        def filterBounds(self, arg):
            log.append(("gaul", "filterBounds", (arg,), {}))
            return _Rec(log, "gaul.filterBounds")

    class _Join:
        @staticmethod
        def simple():
            return _Rec(log, "join")

    class _Filter:
        @staticmethod
        def intersects(**kwargs):
            log.append(("filter", "intersects", (), kwargs))
            return "INTERSECTS"

    fake = types.SimpleNamespace(
        FeatureCollection=_FeatureCollection,
        Feature=type("Feature", (), {}),
        Geometry=_Geometry,
        Join=_Join,
        Filter=_Filter,
    )
    monkeypatch.setattr(gaul_module, "ee", fake)
    return log, fake


def test_selection_is_bbox_prefilter_then_join_on_the_aoi_collection(fake_ee):
    """Bbox pre-filter on the AOI bounds, then a join whose secondary is the AOI."""
    log, fake = fake_ee
    aoi = fake.FeatureCollection("projects/x/assets/aoi")

    get_fao_gaul_subj(2, aoi)

    bbox = [e for e in log if e[1] == "filterBounds"]
    assert bbox and bbox[0][2] == ("BOUNDS",), "pre-filter must use the AOI bounds"

    joins = [e for e in log if e[0] == "join" and e[1] == "apply"]
    assert len(joins) == 1
    _, _, (primary, secondary, condition), _ = joins[0]
    assert secondary is aoi, "the AOI collection is the join's secondary side"
    assert condition == "INTERSECTS"

    (intersects,) = [e for e in log if e[1] == "intersects"]
    assert intersects[3]["leftField"] == ".geo"
    assert intersects[3]["rightField"] == ".geo"


def test_rejects_invalid_level(fake_ee):
    """Levels other than 1 and 2 are refused before touching EE."""
    _, fake = fake_ee
    with pytest.raises(ValueError):
        get_fao_gaul_subj(3, fake.FeatureCollection("projects/x/assets/aoi"))
