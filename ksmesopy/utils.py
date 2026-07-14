"""
ksmesopy.utils
==============
    Derived agrometeorology variables built on top of ksmesopy.core.
    
    Soil processing
    ---------------
    calibrate_vwc(df, vwc_cols)          -> pd.DataFrame
    compute_soil_water_storage(df)        -> pd.DataFrame
    
    Atmospheric helpers
    -------------------
    srad_to_mj(srad, period)
    atmospheric_pressure(elev)
    saturation_vapor_pressure(temp)
    actual_vapor_pressure(temp, rh)
    vapor_pressure_deficit(temp, rh)
    slope_saturation_vapor_pressure(temp)
    psychrometric_constant(elev)
    extraterrestrial_radiation(doy, lat)
    net_radiation(srad_mj, tmin, tmax, ea, elev, doy, lat, *, alpha)
    
    Reference evapotranspiration
    -----------------------------
    reference_et_penman_monteith(doy, lat, elev, tmin, tmax, srad, wspd,
                                  rhmin, rhmax, *, vpd, ea, wind_height)  -> ndarray
    reference_et_hargreaves(doy, lat, tmin, tmax, *, tmean)              -> ndarray

    Others
    ------
    growing_degree_days(tmin, tmax, base, ceiling)  -> ndarray
    heat_index(temp, rh)                            -> ndarray
    wind_chill(temp, wspd)                          -> ndarray
    temperature_humidity_index(temp, rh)            -> ndarray
"""

from __future__ import annotations
from typing import Union
import numpy as np


# ---------------------------------------------------------------------------
# Soil processing
# ---------------------------------------------------------------------------

def calibrate_vwc(df: pd.DataFrame, vwc_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Apply the KSU CS655 calibration equation, overwriting the raw VWC values.

    VWC = max(-0.115 + 0.0989 * sqrt(Ka) - 0.0572 * EC, 0)

    The Mesonet API returns VWC computed by the CS655 firmware equation. This
    function replaces those values with the KSU site-specific calibration for
    any depth where Ka (SOILKA*CM) and EC (SOILEC*CM) are present. Depths
    without Ka/EC are skipped with a warning. All other columns are unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain SOILKA*CM and SOILEC*CM columns for each depth to calibrate.
    vwc_cols : list[str] or None
        VWC columns to calibrate. Any subset of
        ['VWC5CM', 'VWC10CM', 'VWC20CM', 'VWC50CM'].
        Defaults to all depths found with matching Ka/EC columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with VWC values replaced by calibrated equivalents.

    Reference
    ---------
    Patrignani, A., Ochsner, T. E., Feng, L., Dyer, D., & Rossini, P. R. (2022). 
    Calibration and validation of soil water reflectometers. 
    Vadose Zone Journal, 21(3), e20190. https://doi.org/10.1002/vzj2.20190
    """
    df = df.copy()
    if vwc_cols is None:
        vwc_cols = _ALL_VWC

    for col in vwc_cols:
        if col not in _VWC_DEPS:
            raise ValueError(f"{col!r} is not a valid VWC column. Options: {_ALL_VWC}")
        ka_col, ec_col = _VWC_DEPS[col]
        if ka_col not in df.columns or ec_col not in df.columns:
            logger.warning("Skipping %s calibration — %s or %s not found.", col, ka_col, ec_col)
            continue
        df[col] = np.round(
            np.maximum(-0.115 + 0.0989 * np.sqrt(df[ka_col]) - 0.0572 * df[ec_col], 0), 4
        )

    return df


def compute_soil_water_storage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute soil water storage in the top 50 cm (mm) using the trapezoidal rule.

    VWC is integrated over depth nodes [0, 5, 10, 20, 50] cm (i.e. [0, 50,
    100, 200, 500] mm), with the 5 cm sensor assigned to both the surface
    (0 cm) and 5 cm nodes. Rows with any NaN VWC produce NaN storage.

    Requires all four VWC depth columns. Call calibrate_vwc() first if you
    want storage computed from calibrated values — the VWC column names are
    the same either way.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain VWC5CM, VWC10CM, VWC20CM, and VWC50CM columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with an added STORAGE_MM column.
    """
    missing = [c for c in _ALL_VWC if c not in df.columns]
    if missing:
        raise ValueError(f"Missing VWC columns: {missing}")

    df = df.copy()
    depths  = np.array([0, 50, 100, 200, 500], dtype=float)  # mm
    vwc_arr = df[["VWC5CM", "VWC5CM", "VWC10CM", "VWC20CM", "VWC50CM"]].values
    trapz   = getattr(np, "trapezoid", None) or np.trapz
    df["STORAGE_MM"] = np.where(
        np.any(np.isnan(vwc_arr), axis=1),
        np.nan,
        np.round([trapz(row, depths) for row in vwc_arr], 1),
    )
    return df


# ---------------------------------------------------------------------------
# Atmospheric helpers
# ---------------------------------------------------------------------------

def srad_to_mj(srad: Union[float, np.ndarray], period: Union[int, float]) -> Union[float, np.ndarray]:
    """
    Convert mean solar irradiance (W m⁻²) to total energy (MJ m⁻²).

    Parameters
    ----------
    srad : float or array-like
        Mean solar irradiance (W m⁻²).
    period : int or float
        Integration period in seconds (e.g. 86400 for one day).
    """
    return np.asarray(srad) * period / 1_000_000


def atmospheric_pressure(elev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Atmospheric pressure from elevation (FAO-56 Eq. 7).

    Parameters
    ----------
    elev : float or array-like
        Elevation above sea level (m).

    Returns
    -------
    float or np.ndarray
        Pressure (kPa).
    """
    return 101.3 * ((293.0 - 0.0065 * np.asarray(elev, dtype=float)) / 293.0) ** 5.26


def saturation_vapor_pressure(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Saturation vapor pressure (FAO-56 Eq. 11).

    Parameters
    ----------
    temp : float or array-like
        Air temperature (°C).

    Returns
    -------
    float or np.ndarray
        es (kPa).
    """
    t = np.asarray(temp, dtype=float)
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def actual_vapor_pressure(temp: Union[float, np.ndarray], rh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Actual vapor pressure (FAO-56 Eq. 17).

    Parameters
    ----------
    temp : float or array-like
        Air temperature (°C).
    rh : float or array-like
        Relative humidity (%, 0–100).

    Returns
    -------
    float or np.ndarray
        ea (kPa).
    """
    return saturation_vapor_pressure(temp) * (np.asarray(rh, dtype=float) / 100.0)


def vapor_pressure_deficit(temp: Union[float, np.ndarray], rh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vapor pressure deficit, clipped to zero (FAO-56 Eq. 15).

    Parameters
    ----------
    temp : float or array-like
        Air temperature (°C).
    rh : float or array-like
        Relative humidity (%, 0–100).

    Returns
    -------
    float or np.ndarray
        VPD (kPa), >= 0.
    """
    return np.maximum(saturation_vapor_pressure(temp) - actual_vapor_pressure(temp, rh), 0.0)


def slope_saturation_vapor_pressure(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Slope of the saturation vapor pressure curve, Delta (FAO-56 Eq. 13).

    Parameters
    ----------
    temp : float or array-like
        Air temperature (°C).

    Returns
    -------
    float or np.ndarray
        Delta (kPa °C⁻¹).
    """
    t = np.asarray(temp, dtype=float)
    return 4098.0 * saturation_vapor_pressure(t) / (t + 237.3) ** 2


def psychrometric_constant(elev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Psychrometric constant gamma from elevation (FAO-56 Eq. 8).

    Parameters
    ----------
    elev : float or array-like
        Elevation above sea level (m).

    Returns
    -------
    float or np.ndarray
        gamma (kPa °C⁻¹).
    """
    cp      = 1.013e-3   # MJ kg⁻¹ °C⁻¹
    epsilon = 0.622
    lam     = 2.45       # MJ kg⁻¹
    return (cp * atmospheric_pressure(elev)) / (epsilon * lam)


def extraterrestrial_radiation(
    doy: Union[int, np.ndarray],
    lat: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Daily extraterrestrial radiation Ra (FAO-56 Eq. 21).

    Parameters
    ----------
    doy : int or array-like
        Day of year (1–365/366).
    lat : float or array-like
        Latitude (decimal degrees, positive north).

    Returns
    -------
    float or np.ndarray
        Ra (MJ m⁻² day⁻¹).
    """
    doy = np.asarray(doy, dtype=float)
    phi = np.pi / 180.0 * np.asarray(lat, dtype=float)
    dr  = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    d   = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    ws  = np.arccos(-np.tan(phi) * np.tan(d))
    return 24.0 * 60.0 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(ws))


def net_radiation(
    srad_mj: Union[float, np.ndarray],
    tmin:    Union[float, np.ndarray],
    tmax:    Union[float, np.ndarray],
    ea:      Union[float, np.ndarray],
    elev:    Union[float, np.ndarray],
    doy:     Union[int, np.ndarray],
    lat:     Union[float, np.ndarray],
    *,
    alpha: float = 0.23,
) -> Union[float, np.ndarray]:
    """
    Daily net radiation Rn (FAO-56 Eqs. 38–40).

    Parameters
    ----------
    srad_mj : float or array-like
        Incoming shortwave radiation (MJ m⁻² day⁻¹).
    tmin : float or array-like
        Daily minimum temperature (°C).
    tmax : float or array-like
        Daily maximum temperature (°C).
    ea : float or array-like
        Actual vapor pressure (kPa).
    elev : float or array-like
        Elevation above sea level (m).
    doy : int or array-like
        Day of year (1–365/366).
    lat : float or array-like
        Latitude (decimal degrees, positive north).
    alpha : float, default 0.23
        Surface albedo (0.23 for the FAO-56 grass reference).

    Returns
    -------
    float or np.ndarray
        Rn (MJ m⁻² day⁻¹).
    """
    srad_mj = np.asarray(srad_mj, dtype=float)
    tmin    = np.asarray(tmin,    dtype=float)
    tmax    = np.asarray(tmax,    dtype=float)
    ea      = np.asarray(ea,      dtype=float)

    Ra  = extraterrestrial_radiation(doy, lat)
    Rso = (0.75 + 2e-5 * np.asarray(elev, dtype=float)) * Ra  # clear-sky radiation, Eq. 37
    Rns = (1.0 - alpha) * srad_mj                              # net shortwave,       Eq. 38
    Rnl = (
        4.903e-9                                                # Stefan-Boltzmann [MJ m⁻² K⁻⁴ day⁻¹]
        * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * np.minimum(srad_mj / Rso, 1.0) - 0.35)
    )                                                           # net longwave,        Eq. 39
    return Rns - Rnl                                            # Eq. 40


# ---------------------------------------------------------------------------
# Reference evapotranspiration
# ---------------------------------------------------------------------------

def reference_et_penman_monteith(
    doy:   Union[int,   np.ndarray],
    lat:   Union[float, np.ndarray],
    elev:  Union[float, np.ndarray],
    tmin:  Union[float, np.ndarray],
    tmax:  Union[float, np.ndarray],
    srad:  Union[float, np.ndarray],
    wspd:  Union[float, np.ndarray],
    rhmin: Union[float, np.ndarray],
    rhmax: Union[float, np.ndarray],
    *,
    vpd: Union[float, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Daily reference evapotranspiration by the FAO-56 Penman-Monteith method.

    Parameters
    ----------
    doy : int or array-like
        Day of year (1–365/366).
    lat : float or array-like
        Latitude (decimal degrees, positive north).
    elev : float or array-like
        Elevation above sea level (m).
    tmin : float or array-like
        Daily minimum temperature (°C).
    tmax : float or array-like
        Daily maximum temperature (°C).
    srad : float or array-like
        Mean solar irradiance (W m⁻²); converted to MJ m⁻² day⁻¹ internally.
    wspd : float or array-like
        Wind speed at 2 m (m s⁻¹).
    rhmin : float or array-like
        Daily minimum relative humidity (%).
    rhmax : float or array-like
        Daily maximum relative humidity (%).
    vpd : float or array-like, keyword-only, optional
        Vapor pressure deficit (kPa). When provided, rhmin and rhmax are
        still used for net longwave radiation but VPD is taken directly.
        A scaling factor of 0.84 is applied when VPD is estimated from
        rhmin and rhmax (FAO-56 Eq. 17).

    Returns
    -------
    ETo : np.ndarray
        Reference ET (mm day⁻¹), rounded to 2 decimal places.
    """
    tmin  = np.asarray(tmin,  dtype=float)
    tmax  = np.asarray(tmax,  dtype=float)
    srad  = np.asarray(srad,  dtype=float)
    wspd  = np.asarray(wspd,  dtype=float)
    rhmin = np.asarray(rhmin, dtype=float)
    rhmax = np.asarray(rhmax, dtype=float)
    tavg  = (tmin + tmax) / 2.0

    # Wind speed correction to 2 m (Mesonet sensor is already at 2 m → factor = 1.0)
    u2 = wspd * (4.87 / np.log(67.8 * 2.0 - 5.42))

    srad_mj = srad_to_mj(srad, 86_400)

    es_min = saturation_vapor_pressure(tmin)
    es_max = saturation_vapor_pressure(tmax)
    es     = (es_min + es_max) / 2.0

    # Actual vapor pressure from rhmin/rhmax (FAO-56 Eq. 17)
    ea = (es_min * rhmax / 100.0 + es_max * rhmin / 100.0) / 2.0

    # VPD: use supplied value
    if vpd is not None:
        vpd = np.maximum(np.asarray(vpd, dtype=float), 0.0)
    else:
        vpd = np.maximum((es - ea), 0.0)

    Ra = extraterrestrial_radiation(doy, lat)
    Rn = net_radiation(srad_mj, tmin, tmax, ea, elev, doy, lat)

    Delta = slope_saturation_vapor_pressure(tavg)
    gamma = psychrometric_constant(elev)

    ETo = (
        (0.408 * Delta * Rn + gamma * (900.0 / (tavg + 273.0)) * u2 * vpd)
        / (Delta + gamma * (1.0 + 0.34 * u2))
    )

    return np.round(ETo, 2)


def reference_et_hargreaves(
    doy:   Union[int, np.ndarray],
    lat:   Union[float, np.ndarray],
    tmin:  Union[float, np.ndarray],
    tmax:  Union[float, np.ndarray],
    *,
    tmean: Union[float, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Daily reference evapotranspiration by the Hargreaves-Samani (1985) method.

    Requires only temperature and location; useful when humidity, radiation,
    and wind data are unavailable.

    ETo = 0.0023 * Ra * (tmean + 17.8) * sqrt(tmax - tmin) * 0.408

    Parameters
    ----------
    doy : int or array-like
        Day of year (1–365/366).
    lat : float or array-like
        Latitude (decimal degrees, positive north).
    tmin : float or array-like
        Daily minimum temperature (°C).
    tmax : float or array-like
        Daily maximum temperature (°C).
    tmean : float or array-like, keyword-only, optional
        Daily mean temperature (°C). Defaults to (tmin + tmax) / 2.

    Returns
    -------
    np.ndarray
        Reference ET (mm day⁻¹), rounded to 2 decimal places.

    See Also
    --------
    extraterrestrial_radiation : compute Ra separately if you need it.
    """
    tmin  = np.asarray(tmin,  dtype=float)
    tmax  = np.asarray(tmax,  dtype=float)
    tmean = np.asarray(tmean, dtype=float) if tmean is not None else (tmin + tmax) / 2.0
    Ra    = extraterrestrial_radiation(doy, lat)
    ETo   = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0)) * 0.408
    return np.round(ETo, 2)

    
# ------
# Others
# ------
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
