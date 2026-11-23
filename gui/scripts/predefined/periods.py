"""Calendar periods and declared data availability for period-based layers.

Part D layers take a ``period`` param: the whole year (default, left out of the
name) or one calendar month (``burned_area_sep``). ``year`` stays the only
temporal axis, so a monthly study builds one model per calendar month.

Two families use these helpers:

* period aggregates cover ``period_range(year, period)``;
* antecedent indicators end at ``period_start(year, period)`` and look back a
  window (``lookback_days`` / ``lookback_years``), so a September target never
  sees September data.

Availability is declared, not discovered. Each entry lists, per source, the
first and last *complete* month the source covers (``"YYYY-MM"``). The modal
offers only the years whose data interval lies inside that window, and the
entry's ``validate`` refuses the rest before any Earth Engine call. No helper
here talks to Earth Engine: they run inside Solara renders.
"""

from datetime import date, timedelta

MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)

#: First year a period layer offers. Every temporal layer started at 2001
#: before Part D; keeping the floor keeps today's year lists unchanged.
YEAR_FLOOR = 2001


def period_param(advanced=False):
    """The shared ``period`` param: ``year`` (default, omitted) or a month."""
    return {
        "key": "period",
        "label_key": "vars.modal.param_period",
        "hint_key": "vars.modal.param_period_hint",
        "type": "choice",
        "default": "year",
        "options": ["year", *MONTHS],
        "option_label_key_prefix": "vars.modal.period_",
        "suffix_prefix": "",
        "advanced": advanced,
        "omit_default": True,
    }


def month_index(period):
    """1-12 for a month value, None for ``"year"``; ValueError otherwise."""
    if period == "year":
        return None
    if period not in MONTHS:
        raise ValueError(f"Unknown period {period!r}")
    return MONTHS.index(period) + 1


def _start_date(year, period):
    return date(int(year), month_index(period) or 1, 1)


def _end_date(year, period):
    month = month_index(period)
    if month is None:
        return date(int(year) + 1, 1, 1)
    if month == 12:
        return date(int(year) + 1, 1, 1)
    return date(int(year), month + 1, 1)


def period_start(year, period):
    """ISO date the period starts on: 1 January or the 1st of the month."""
    return _start_date(year, period).isoformat()


def period_range(year, period):
    """``(start, end)`` ISO dates of the period, end exclusive: ``[start, end)``.

    ``(2020, "year")`` gives ``("2020-01-01", "2021-01-01")``: the exact strings
    the pre-Part-D builders passed to ``filterDate``, so default names keep
    building the same expression.
    """
    return _start_date(year, period).isoformat(), _end_date(year, period).isoformat()


def lookback_days(year, period, days):
    """``[period start - days, period start)`` as ISO dates."""
    end = _start_date(year, period)
    return (end - timedelta(days=int(days))).isoformat(), end.isoformat()


def lookback_years(year, period, years):
    """``[period start - years, period start)``, anchored on the same month."""
    end = _start_date(year, period)
    return end.replace(year=end.year - int(years)).isoformat(), end.isoformat()


def months_spanned(start, end):
    """First and last ``"YYYY-MM"`` touched by the ISO interval ``[start, end)``."""
    first = date.fromisoformat(start)
    last = date.fromisoformat(end) - timedelta(days=1)
    return f"{first:%Y-%m}", f"{last:%Y-%m}"


def window_covered(window, needed):
    """Does the ``(first, last)`` month window contain the ``needed`` months?"""
    return window[0] <= needed[0] and needed[1] <= window[1]


def source_window(availability, params):
    """The availability window ``params`` select.

    Keyed by the value of the entry's ``source`` param, else its ``sensor``
    param. An entry with a single source has one key and needs neither.
    """
    for key in ("source", "sensor"):
        value = params.get(key)
        if value in availability:
            return availability[value]
    if len(availability) == 1:
        return next(iter(availability.values()))
    raise KeyError(f"No availability window for params {params!r}")


def available_years(availability, needs, params, floor=YEAR_FLOOR):
    """Years whose data interval ``needs(year, params)`` the source covers.

    The scan runs one year past the window's last year: a January look-back
    only needs the previous December.
    """
    window = source_window(availability, params)
    last = int(window[1][:4])
    return [
        year
        for year in range(floor, last + 2)
        if window_covered(window, months_spanned(*needs(year, params)))
    ]


def availability_error(availability, needs, params, year):
    """``(i18n key, kwargs)`` when ``year`` falls outside the source, else None."""
    if year is None:
        return None
    window = source_window(availability, params)
    needed = months_spanned(*needs(int(year), params))
    if window_covered(window, needed):
        return None
    return (
        "vars.modal.error_period_unavailable",
        {
            "year": year,
            "start": needed[0],
            "end": needed[1],
            "first": window[0],
            "last": window[1],
        },
    )


def period_needs(year, params):
    """Data interval of a period aggregate: the period itself."""
    return period_range(year, params["period"])


def availability_fields(availability, needs=period_needs, floor=YEAR_FLOOR):
    """The ``availability`` / ``years`` / ``validate`` fields of a period layer.

    ``years`` and ``validate`` read the entry's own copy of the windows, so a
    data bump is a one-place edit of ``entry["availability"]``.
    """
    windows = dict(availability)
    return {
        "availability": windows,
        "years": lambda params: available_years(windows, needs, params, floor),
        "validate": lambda params, year: availability_error(
            windows, needs, params, year
        ),
    }
