"""
ksmesopy.utils
==============
Derived agrometeorology variables built on top of ksmesopy.core.

    growing_degree_days(tmin, tmax, base, ceiling)  -> ndarray
    heat_index(temp, rh)                            -> ndarray
    wind_chill(temp, wspd)                          -> ndarray
    temperature_humidity_index(temp, rh)            -> ndarray
"""

from __future__ import annotations

from typing import Union

import numpy as np


def growing_degree_days(
    tmin:    Union[float, np.ndarray],
    tmax:    Union[float, np.ndarray],
    base:    float = 10.0,
    ceiling: float | None = 30.0,
) -> np.ndarray:
    """
    Daily growing degree days (GDD): mean of clamped tmin/tmax, minus base.

    Both tmin and tmax are individually clamped to [base, ceiling] before
    averaging (the standard agronomic approach). Common base values: maize
    10 °C, wheat 0 °C, soybean 10 °C. Pass ceiling=None to disable the cap.
    Returns GDD (°C day⁻¹), >= 0.
    """
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)

    if ceiling is not None:
        tmin = np.minimum(tmin, ceiling)
        tmax = np.minimum(tmax, ceiling)

    tmin = np.maximum(tmin, base)
    tmax = np.maximum(tmax, base)

    return np.maximum((tmin + tmax) / 2.0 - base, 0.0)


def heat_index(
    temp: Union[float, np.ndarray],
    rh:   Union[float, np.ndarray],
) -> np.ndarray:
    """
    NOAA/NWS heat index (apparent temperature), °C rounded to 1 dp.

    Uses the Rothfusz regression for hot, humid conditions and Steadman's
    simpler estimate elsewhere (also the fallback below 27 °C).
    temp in °C, rh in % (0–100).
    """
    t  = np.asarray(temp, dtype=float)
    rh = np.asarray(rh,   dtype=float)
    tf = t * 9.0 / 5.0 + 32.0  # °F for NWS formula

    hi_simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)

    hi_full_f = (
        -42.379
        + 2.04901523  * tf
        + 10.14333127 * rh
        - 0.22475541  * tf * rh
        - 0.00683783  * tf ** 2
        - 0.05481717  * rh ** 2
        + 0.00122874  * tf ** 2 * rh
        + 0.00085282  * tf * rh ** 2
        - 0.00000199  * tf ** 2 * rh ** 2
    )

    # NWS adjustments, applied only inside specific tf/rh windows
    sqrt_arg = np.maximum((17.0 - np.abs(tf - 95.0)) / 17.0, 0.0)
    adj_low  = ((13.0 - rh) / 4.0) * np.sqrt(sqrt_arg)
    adj_high = ((rh - 85.0) / 10.0) * ((87.0 - tf) / 5.0)

    hi_full_f = np.where((rh < 13.0) & (tf >= 80.0) & (tf <= 112.0), hi_full_f - adj_low,  hi_full_f)
    hi_full_f = np.where((rh > 85.0) & (tf >= 80.0) & (tf <= 87.0),  hi_full_f + adj_high, hi_full_f)

    hi_full = (hi_full_f - 32.0) * 5.0 / 9.0

    hi = np.where((hi_simple >= 27.0) & (t >= 27.0), hi_full, hi_simple)
    return np.round(hi, 1)


def wind_chill(
    temp: Union[float, np.ndarray],
    wspd: Union[float, np.ndarray],
) -> np.ndarray:
    """
    NWS wind chill temperature, °C rounded to 1 dp.

    Valid for temp <= 10 °C and wind >= 4.8 km h⁻¹ (1.33 m s⁻¹); outside
    that range the observed temperature is returned unchanged.
    temp in °C, wspd (10 m) in m s⁻¹.
    """
    t     = np.asarray(temp, dtype=float)
    v_kmh = np.asarray(wspd, dtype=float) * 3.6
    wc    = 13.12 + 0.6215 * t - 11.37 * v_kmh ** 0.16 + 0.3965 * t * v_kmh ** 0.16
    valid = (t <= 10.0) & (v_kmh >= 4.8)
    return np.round(np.where(valid, wc, t), 1)


def temperature_humidity_index(
    temp: Union[float, np.ndarray],
    rh:   Union[float, np.ndarray],
) -> np.ndarray:
    """
    Temperature-Humidity Index (THI) for livestock heat stress.

    THI = Tf - (0.55 - 0.0055 * RH) * (Tf - 58), with Tf in °F.
    Dairy-cattle thresholds (Mader et al. 2006, J. Anim. Sci. 84:1924):
    <68 none, 68–72 mild, 72–80 moderate, 80–90 severe, >90 dangerous.
    temp in °C, rh in % (0–100). Returns THI (dimensionless), 1 dp.
    """
    tf = np.asarray(temp, dtype=float) * 9.0 / 5.0 + 32.0
    rh = np.asarray(rh,   dtype=float)
    return np.round(tf - (0.55 - 0.0055 * rh) * (tf - 58.0), 1)
