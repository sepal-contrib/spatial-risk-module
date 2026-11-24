"""burned_area: MODIS / VIIRS monthly burned-area products as a 0/1 mask."""

from _ee_fake import install

from gui.scripts.predefined import (
    GROUP_ORDER,
    build_predefined_name,
    entry_for_name,
    resolve_predefined,
    resolved_entry,
    validate_entry,
)


def _fire():
    import gui.scripts.predefined.fire as fire

    return fire


def test_fire_group_sits_after_forest():
    """Fire group sits after forest."""
    assert GROUP_ORDER == (
        "terrain",
        "infrastructure",
        "water",
        "forest",
        "forest_change",
        "fire",
        "climate",
        "admin",
        "mapbiomas",
    )


def test_default_entry():
    """Default entry."""
    key, params, entry = entry_for_name("burned_area")
    assert key == "burned_area"
    assert params == {"period": "year", "sensor": "modis"}
    assert entry["group"] == "fire"
    assert entry["raster_type"] == "categorical"
    assert entry["temporal"] is True
    assert entry["default_scale"] == 463
    assert entry["vis_params"] == {"palette": ["ffffff", "d84315"], "min": 0, "max": 1}
    assert entry["legend_class_keys"] == (
        "legend.class.unburned",
        "legend.class.burned",
    )


def test_names_round_trip():
    """Names round trip."""
    assert build_predefined_name(
        "burned_area", {"period": "sep", "sensor": "modis"}
    ) == ("burned_area_sep")
    assert resolve_predefined("burned_area_sep_viirs") == (
        "burned_area",
        {"period": "sep", "sensor": "viirs"},
    )
    assert resolve_predefined("burned_area_modis") == (None, {})  # default not written


def test_builder_makes_a_byte_mask_of_any_burn_in_the_period(monkeypatch):
    """Builder makes a byte mask of any burn in the period."""
    fire = _fire()
    fake = install(monkeypatch, fire)
    fire._get_burned_area("AOI", 2024, period="sep")
    assert fake.calls == [
        ("ImageCollection", ("MODIS/061/MCD64A1",)),
        ("filterDate", ("2024-09-01", "2024-10-01")),
        ("select", ("BurnDate",)),
        ("max", ()),
        ("gt", (0,)),
        ("unmask", (0,)),  # unburned pixels are masked in the product
        ("toByte", ()),
        ("clip", ("AOI",)),
        ("rename", ("B1",)),
    ]


def test_viirs_reads_its_own_band(monkeypatch):
    """Viirs reads its own band."""
    fire = _fire()
    fake = install(monkeypatch, fire)
    fire._get_burned_area("AOI", 2020, sensor="viirs")
    assert fake.calls[:3] == [
        ("ImageCollection", ("NASA/VIIRS/002/VNP64A1",)),
        ("filterDate", ("2020-01-01", "2021-01-01")),
        ("select", ("Burn_Date",)),
    ]


def test_burned_area_years_follow_the_sensor():
    """Burned area years follow the sensor."""

    def years(**params):
        return resolved_entry("burned_area", params)["years"]

    assert years() == list(range(2001, 2026))  # MCD64A1 ends 2026-07
    assert years(period="jul")[-1] == 2026
    assert years(period="aug")[-1] == 2025
    assert years(sensor="viirs") == list(range(2013, 2026))  # VNP64A1 from 2012-03
    assert years(period="aug", sensor="viirs") == list(range(2012, 2027))


def test_validate_names_the_missing_months():
    """Validate names the missing months."""
    err = validate_entry("burned_area", {}, 2026)
    assert err == (
        "vars.modal.error_period_unavailable",
        {
            "year": 2026,
            "start": "2026-01",
            "end": "2026-12",
            "first": "2000-11",
            "last": "2026-07",
        },
    )
    assert validate_entry("burned_area", {"period": "jul"}, 2026) is None
