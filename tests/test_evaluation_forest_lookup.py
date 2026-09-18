"""The library-layer forest lookup tolerates a parameterised layer name.

``spatialrisk.evaluation.resolve_layers`` recovers the "forest at period start"
raster from a prediction's dataset. It matched ``FOREST_VAR`` exactly, which
misses every Hansen layer created from the GUI since the tree-cover threshold
became part of the variable name (``forest_gfc_tc30``).

The resolver that parses those names lives in ``gui/``, and ``spatialrisk/``
must never import ``gui/`` (documented in spatialrisk/evaluation.py), so the
library matches the prefix instead. These tests pin both halves of that: the
suffixed name is found, and names that merely contain the key are not.
"""

import types

from spatialrisk import evaluation as ev


def _project(feature_names):
    target = types.SimpleNamespace(name="forest_loss_2015_2020", path="/data/defor.tif")
    features = [
        types.SimpleNamespace(name=n, path=f"/data/{n}.tif") for n in feature_names
    ]
    dataset = types.SimpleNamespace(
        name="calibration", target=target, features=features
    )
    pred = types.SimpleNamespace(
        model_key="glm_glm_v1",
        window=None,
        dataset_name="calibration",
        path="/data/risk.tif",
    )
    project = types.SimpleNamespace(get_dataset=lambda n: dataset)
    return project, pred


def test_resolve_layers_finds_a_parameterised_forest_feature():
    """Regression: forest_gfc_tc30 is the forest layer, not a stranger."""
    project, pred = _project(["altitude", "forest_gfc_tc30"])
    layers = ev.resolve_layers(project, pred)
    assert layers["forest_file"] == "/data/forest_gfc_tc30.tif"


def test_resolve_layers_still_finds_the_bare_legacy_name():
    """Projects predating the threshold feature keep working unchanged."""
    project, pred = _project(["forest_gfc"])
    layers = ev.resolve_layers(project, pred)
    assert layers["forest_file"] == "/data/forest_gfc.tif"


def test_resolve_layers_ignores_names_that_merely_contain_the_key():
    """The prefix match is anchored: a suffix or an infix does not count."""
    project, pred = _project(["my_forest_gfc", "forest_gfcx", "forest_tmf"])
    try:
        ev.resolve_layers(project, pred)
    except ValueError as exc:
        assert "forest_gfc" in str(exc)
    else:
        raise AssertionError("expected ValueError for a dataset with no forest layer")


def test_resolve_layers_keeps_first_match_semantics():
    """Several forest layers: the library path takes the first, as it always did.

    Choosing between them is a GUI concern (the Predict dialog asks); this is
    the notebook/library path, which must stay non-interactive.
    """
    project, pred = _project(["forest_gfc_tc75", "forest_gfc_tc30"])
    layers = ev.resolve_layers(project, pred)
    assert layers["forest_file"] == "/data/forest_gfc_tc75.tif"


def test_resolve_layers_uses_an_explicit_forest_file_over_the_scan():
    """A caller who knows the forest layer (the allocation form) names it.

    Datasets built on a non-Hansen forest layer ("geobosques") have nothing
    for the prefix scan to find, so the explicit path must short-circuit it.
    """
    project, pred = _project(["altitude", "geobosques"])
    layers = ev.resolve_layers(project, pred, forest_file="/data/geobosques.tif")
    assert layers["forest_file"] == "/data/geobosques.tif"


def test_resolve_layers_explicit_forest_file_wins_over_a_hansen_feature():
    """The explicit choice is authoritative even when the scan would succeed."""
    project, pred = _project(["forest_gfc_tc30"])
    layers = ev.resolve_layers(project, pred, forest_file="/data/other.tif")
    assert layers["forest_file"] == "/data/other.tif"
