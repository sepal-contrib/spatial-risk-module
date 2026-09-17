"""The reference CRS / pixel-size validator (Solara-free, no I/O)."""

import pytest

from gui.scripts.process_actions import validate_projection


@pytest.mark.parametrize(
    "epsg",
    ["abcd", "EPSG:99999", "4326x", "EPSG:", "-1"],
)
def test_unparseable_epsg_is_a_blocking_error(epsg):
    """A CRS pyproj cannot resolve must never reach GDAL."""
    error, warning = validate_projection(epsg, "30")
    assert error == "bad_epsg"
    assert warning is None


def test_projected_crs_is_clean():
    """UTM 18N is the shape auto_utm_epsg produces — no error, no warning."""
    assert validate_projection("EPSG:32618", "30") == (None, None)


def test_bare_code_is_accepted():
    """The field's placeholder invites a bare code, so it must parse."""
    assert validate_projection("32618", "30") == (None, None)


def test_geographic_crs_warns_but_does_not_block():
    """Resolution is metres everywhere else, so degrees needs saying — not blocking."""
    error, warning = validate_projection("EPSG:4326", "30")
    assert error is None
    assert warning == "geographic_crs"


@pytest.mark.parametrize("resolution", ["", "abc", "0", "-5"])
def test_non_positive_resolution_is_a_blocking_error(resolution):
    """float() used to sit in a try/except that silently substituted 30.0."""
    error, _ = validate_projection("EPSG:32618", resolution)
    assert error == "bad_resolution"


def test_epsg_error_takes_precedence_over_resolution_error():
    """One message at a time; the CRS is the field the user was editing."""
    error, _ = validate_projection("abcd", "abc")
    assert error == "bad_epsg"
