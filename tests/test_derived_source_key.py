"""Source-guess for derived variables must see through loss_/gain_ prefixes."""

from types import SimpleNamespace

from gui.widget.variable_list import derived_source_key


class _P:
    def __init__(self):
        self.raw_variables = {
            "forest_gfc_2015": SimpleNamespace(name="forest_gfc"),
            "rivers": SimpleNamespace(name="rivers"),
        }


def test_plain_derived_name_matches_raw():
    """A post-processed layer resolves to the raw variable it extends."""
    assert derived_source_key(_P(), "rivers_dist", "?") == "rivers"


def test_loss_prefix_stripped():
    """A loss layer resolves to its start layer, not to "unknown"."""
    assert (
        derived_source_key(_P(), "loss_forest_gfc_2015_2020", "?") == "forest_gfc_2015"
    )


def test_gain_prefix_stripped():
    """Same for a gain layer — both operation prefixes are stripped."""
    assert (
        derived_source_key(_P(), "gain_forest_gfc_2015_2020", "?") == "forest_gfc_2015"
    )


def test_unknown_falls_back():
    """A name matching no raw variable yields the caller's fallback."""
    assert derived_source_key(_P(), "loss_mangrove_2015_2020", "?") == "?"


class _Temporal:
    """One layer, two years — same ``name``, distinct raw keys."""

    def __init__(self):
        self.raw_variables = {
            "forest_2000": SimpleNamespace(name="forest", year=2000),
            "forest_2010": SimpleNamespace(name="forest", year=2010),
        }


def test_same_named_years_resolve_by_year():
    """A reprojected copy keeps its source's name AND year — match on both."""
    assert derived_source_key(_Temporal(), "forest", "?", year=2010) == "forest_2010"
    assert derived_source_key(_Temporal(), "forest", "?", year=2000) == "forest_2000"


def test_yearless_derived_still_matches_by_name():
    """Change layers carry no year (it lives in ``tags``): keep prefix matching."""
    assert (
        derived_source_key(_P(), "loss_forest_gfc_2015_2020", "?") == "forest_gfc_2015"
    )


class _Params:
    """A default and a parameterised setting of the same layer."""

    def __init__(self):
        self.raw_variables = {
            "rivers": SimpleNamespace(name="rivers"),
            "rivers_wlmc": SimpleNamespace(name="rivers_wlmc"),
        }


def test_longest_raw_name_wins():
    """``rivers_wlmc_dist`` comes from ``rivers_wlmc``, not from ``rivers``."""
    assert derived_source_key(_Params(), "rivers_wlmc_dist", "?") == "rivers_wlmc"
    assert derived_source_key(_Params(), "rivers_dist", "?") == "rivers"


def test_match_requires_a_name_boundary():
    """``riversx`` is not derived from ``rivers``."""
    assert derived_source_key(_Params(), "riversx", "?") == "?"


class _PrefixAndYear:
    """A shorter name with a matching year next to the longer exact name."""

    def __init__(self):
        self.raw_variables = {
            "forest_tmf_2020": SimpleNamespace(name="forest_tmf", year=2020),
            "forest_tmf_degraded": SimpleNamespace(
                name="forest_tmf_degraded", year=2015
            ),
        }


def test_longest_name_beats_year_of_a_shorter_name():
    """The year only breaks ties among entries with the longest matching name."""
    assert (
        derived_source_key(_PrefixAndYear(), "forest_tmf_degraded_dist", "?", year=2020)
        == "forest_tmf_degraded"
    )
