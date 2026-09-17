"""What a submitted derived-layer entry will register: display name and key.

The job row needs both halves. They differ for edge/dist, because
``add_as_processed`` stores a year-bearing variable under ``{name}_{year}`` and
``_create_post_var`` inherits its source's year — so keying a job on the name
alone would treat the layer as never registered (a row stuck on "completed"
beside its own product) and would conflate two same-named sources from
different years.
"""

from types import SimpleNamespace

from gui.scripts.process_actions import derived_output


def _project(**processed):
    return SimpleNamespace(processed_variables=processed)


def _var(name, year=None):
    return SimpleNamespace(name=name, year=year)


def _entry(op, start_key="", end_key="", pp_key=""):
    return {"op": op, "start_key": start_key, "end_key": end_key, "pp_key": pp_key}


def test_derived_output_of_a_change_layer_keys_on_its_name():
    """Change layers carry no year, so name and key coincide."""
    p = _project(a2010=_var("forest", 2010), a2020=_var("forest", 2020))
    out = derived_output(p, _entry("loss", "a2010", "a2020"))
    assert out.name == "loss_forest_2010_2020"
    assert out.key == "loss_forest_2010_2020"


def test_derived_output_of_edge_dist_on_a_yearless_source_keys_on_its_name():
    """No year to inherit means no year suffix on the key."""
    p = _project(k=_var("loss_forest_2010_2020"))
    out = derived_output(p, _entry("dist", pp_key="k"))
    assert out.name == "loss_forest_2010_2020_dist"
    assert out.key == "loss_forest_2010_2020_dist"


def test_derived_output_of_edge_dist_carries_the_sources_year_into_the_key():
    """add_as_processed stores a year-bearing variable under name_year."""
    p = _project(forest_2010=_var("forest", 2010))
    out = derived_output(p, _entry("edge", pp_key="forest_2010"))
    assert out.name == "forest_edge"
    assert out.key == "forest_edge_2010"


def test_derived_output_keys_differ_for_the_same_name_in_two_years():
    """Two same-named sources yield one display name but distinct outputs."""
    p = _project(forest_2010=_var("forest", 2010), forest_2015=_var("forest", 2015))
    a = derived_output(p, _entry("dist", pp_key="forest_2010"))
    b = derived_output(p, _entry("dist", pp_key="forest_2015"))
    assert a.name == b.name == "forest_dist"
    assert a.key != b.key


def test_derived_output_is_none_for_an_invalid_entry():
    """Nothing to claim or list when the entry cannot produce a layer."""
    p = _project(a=_var("tmf", 2010), b=_var("gfc", 2020))
    assert derived_output(p, _entry("loss", "b", "a")) is None  # start >= end
    assert derived_output(p, _entry("dist", pp_key="missing")) is None
