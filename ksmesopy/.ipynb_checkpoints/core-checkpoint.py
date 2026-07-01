"""
ksmesopy.core
=============
Core API for the Kansas Mesonet (https://mesonet.k-state.edu).

Data retrieval
--------------
get_stations()                       -> pd.DataFrame or list[str]
get_stations_active()                -> pd.DataFrame
list_variables(interval)             -> list[dict]
request_data(station, start, end, interval, variables, *, verbose, sleep)
                                     -> pd.DataFrame
request_data_multi(stations, ...)    -> dict[str, pd.DataFrame]
rename_columns(df, preset)           -> pd.DataFrame

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
                              rhmin, rhmax, *, vpd, ea, wind_height)
                                     -> tuple[ndarray, ndarray]
reference_et_hargreaves(doy, lat, tmin, tmax, *, tmean)
                                     -> tuple[ndarray, ndarray]
"""

from __future__ import annotations

import logging
import time
from typing import Literal, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variable catalogue
# Each entry: (api_name, snake_name, description, available_intervals)
# ---------------------------------------------------------------------------

_ALL   = frozenset({"5min", "hour", "day"})
_DONLY = frozenset({"day"})
_D5    = frozenset({"5min", "day"})

VARIABLES: list[tuple[str, str, str, frozenset]] = [
    # Atmospheric
    ("TEMP2MAVG",       "t2m",              "Air temperature 2 m avg (°C)",           _ALL),
    ("TEMP2MMIN",       "t2m_min",          "Air temperature 2 m min (°C)",           _DONLY),
    ("TEMP2MMAX",       "t2m_max",          "Air temperature 2 m max (°C)",           _DONLY),
    ("TEMP10MAVG",      "t10m",             "Air temperature 10 m avg (°C)",          _ALL),
    ("TEMP10MMIN",      "t10m_min",         "Air temperature 10 m min (°C)",          _DONLY),
    ("TEMP10MMAX",      "t10m_max",         "Air temperature 10 m max (°C)",          _DONLY),
    ("RELHUM2MAVG",     "rh",               "Relative humidity 2 m avg (%)",          _ALL),
    ("RELHUM2MMIN",     "rh_min",           "Relative humidity 2 m min (%)",          _DONLY),
    ("RELHUM2MMAX",     "rh_max",           "Relative humidity 2 m max (%)",          _DONLY),
    ("VPDEFAVG",        "vpd",              "Vapor pressure deficit avg (kPa)",       _ALL),
    ("PRESSUREAVG",     "pres",             "Atmospheric pressure avg (kPa)",         _ALL),
    ("PRECIP",          "precip",           "Precipitation gauge 1 (mm)",             _ALL),
    ("PRECIP2",         "precip2",          "Precipitation gauge 2 (mm)",             _ALL),
    ("SRAVG",           "srad",             "Solar radiation avg (W m⁻²)",            _ALL),
    ("WSPD2MAVG",       "wspd",             "Wind speed 2 m avg (m s⁻¹)",            _ALL),
    ("WSPD2MMAX",       "wspd_max",         "Wind speed 2 m max (m s⁻¹)",            _D5),
    ("WDIR2M",          "wdir",             "Wind direction 2 m (°)",                 _ALL),
    ("WDIR2MSTD",       "wdir_std",         "Wind direction 2 m std dev (°)",         _ALL),
    ("WSPD10MAVG",      "wspd10m",          "Wind speed 10 m avg (m s⁻¹)",           _ALL),
    ("WSPD10MMAX",      "wspd10m_max",      "Wind speed 10 m max (m s⁻¹)",           _D5),
    ("WDIR10M",         "wdir10m",          "Wind direction 10 m (°)",                _ALL),
    ("WDIR10MSTD",      "wdir10m_std",      "Wind direction 10 m std dev (°)",        _ALL),
    # Soil temperature — dedicated probes
    ("SOILTMP5AVG",     "tsoil_5cm",        "Soil temperature 5 cm avg (°C)",         _ALL),
    ("SOILTMP5MIN",     "tsoil_5cm_min",    "Soil temperature 5 cm min (°C)",         _DONLY),
    ("SOILTMP5MAX",     "tsoil_5cm_max",    "Soil temperature 5 cm max (°C)",         _DONLY),
    ("SOILTMP10AVG",    "tsoil_10cm",       "Soil temperature 10 cm avg (°C)",        _ALL),
    ("SOILTMP10MIN",    "tsoil_10cm_min",   "Soil temperature 10 cm min (°C)",        _DONLY),
    ("SOILTMP10MAX",    "tsoil_10cm_max",   "Soil temperature 10 cm max (°C)",        _DONLY),
    # Soil temperature — CS655 sensors
    ("SOILTMP5AVG655",  "tsoil_5cm_655",    "Soil temperature 5 cm CS655 (°C)",       _ALL),
    ("SOILTMP10AVG655", "tsoil_10cm_655",   "Soil temperature 10 cm CS655 (°C)",      _ALL),
    ("SOILTMP20AVG655", "tsoil_20cm_655",   "Soil temperature 20 cm CS655 (°C)",      _ALL),
    ("SOILTMP50AVG655", "tsoil_50cm_655",   "Soil temperature 50 cm CS655 (°C)",      _ALL),
    # Soil dielectric constant (Ka) — CS655
    ("SOILKA5CM",       "ka_5cm",           "Soil dielectric constant 5 cm (CS655)",  _ALL),
    ("SOILKA10CM",      "ka_10cm",          "Soil dielectric constant 10 cm (CS655)", _ALL),
    ("SOILKA20CM",      "ka_20cm",          "Soil dielectric constant 20 cm (CS655)", _ALL),
    ("SOILKA50CM",      "ka_50cm",          "Soil dielectric constant 50 cm (CS655)", _ALL),
    # Soil electrical conductivity (EC) — CS655
    ("SOILEC5CM",       "ec_5cm",           "Soil EC 5 cm (dS m⁻¹)",                 _ALL),
    ("SOILEC10CM",      "ec_10cm",          "Soil EC 10 cm (dS m⁻¹)",                _ALL),
    ("SOILEC20CM",      "ec_20cm",          "Soil EC 20 cm (dS m⁻¹)",                _ALL),
    ("SOILEC50CM",      "ec_50cm",          "Soil EC 50 cm (dS m⁻¹)",                _ALL),
    # Soil VWC — CS655 firmware equation (use calibrate_vwc for KSU equation)
    ("VWC5CM",          "vwc_5cm",          "Soil VWC 5 cm (m³ m⁻³)",                _ALL),
    ("VWC10CM",         "vwc_10cm",         "Soil VWC 10 cm (m³ m⁻³)",               _ALL),
    ("VWC20CM",         "vwc_20cm",         "Soil VWC 20 cm (m³ m⁻³)",               _ALL),
    ("VWC50CM",         "vwc_50cm",         "Soil VWC 50 cm (m³ m⁻³)",               _ALL),
]

_SNAKE_MAP: dict[str, str] = {v[0]: v[1] for v in VARIABLES}
_VALID_FOR: dict[str, set] = {
    intv: {v[0] for v in VARIABLES if intv in v[3]}
    for intv in ("5min", "hour", "day")
}

# VWC column -> (Ka column, EC column)
_VWC_DEPS: dict[str, tuple[str, str]] = {
    "VWC5CM":  ("SOILKA5CM",  "SOILEC5CM"),
    "VWC10CM": ("SOILKA10CM", "SOILEC10CM"),
    "VWC20CM": ("SOILKA20CM", "SOILEC20CM"),
    "VWC50CM": ("SOILKA50CM", "SOILEC50CM"),
}
_ALL_VWC: list[str] = list(_VWC_DEPS.keys())

RENAME_PRESET: dict[str, str] = {"TIMESTAMP": "timestamp", **_SNAKE_MAP}

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_BASE_URL     = "http://mesonet.k-state.edu/rest"
_TS_FMT       = "%Y%m%d%H%M%S"
_MAX_RECORDS  = 3_000
_MAX_RETRIES  = 3
_RETRY_SLEEP  = 2      # seconds between retry attempts
_POLITE_SLEEP = 0.8    # seconds between successful chunk requests


# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------

def get_stations(names_only=False) -> list[str]:
    """Return a sorted DataFrame or list of all Kansas Mesonet station names or metadata."""
    url = f"{_BASE_URL}/stationnames/"
    try:
        df = pd.read_csv(url)
        if names_only:
            names = sorted(df.iloc[:, 0].dropna().unique().tolist())
            if not names:
                raise RuntimeError("API returned an empty station list.")
            return names  
        else:
            return df
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve stations: {exc}") from exc


def get_stations_active() -> pd.DataFrame:
    """
    Return the station availability table from the Mesonet REST API.

    Returns
    -------
    pd.DataFrame
        Columns include STATION, OBS_INTERVAL (seconds), START, and END.
    """
    url = f"{_BASE_URL}/stationactive/"
    try:
        return pd.read_csv(url, parse_dates=["START", "END"])
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve station availability: {exc}") from exc


# ---------------------------------------------------------------------------
# Variable catalogue
# ---------------------------------------------------------------------------

def list_variables(
    interval: Literal["5min", "hour", "day"] | None = None,
) -> list[dict]:
    """
    Return the variable catalogue, optionally filtered by interval.

    Parameters
    ----------
    interval : {'5min', 'hour', 'day'} or None
        If None, all variables are returned.

    Returns
    -------
    list[dict]
        Each dict has keys: api_name, snake_name, description, intervals.
    """
    rows = VARIABLES if interval is None else [v for v in VARIABLES if interval in v[3]]
    return [
        {"api_name": v[0], "snake_name": v[1], "description": v[2], "intervals": sorted(v[3])}
        for v in rows
    ]


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

def request_data(
    station:   str,
    start:     Union[str, pd.Timestamp],
    end:       Union[str, pd.Timestamp],
    interval:  Literal["5min", "hour", "day"],
    variables: list[str],
    *,
    verbose:   bool = True,
    sleep:     float = _POLITE_SLEEP,
) -> pd.DataFrame:
    """
    Retrieve observed data for a single station.

    Fetches in chunks of up to 3 000 records into a NaN-pre-allocated frame,
    so gaps where a sensor was not yet installed appear as NaN rather than
    missing rows.

    For daily data the Mesonet API stores each day's values at 00:00 of the
    following calendar day; this function shifts timestamps back to the
    observation date automatically.

    Parameters
    ----------
    station : str
        Station name as returned by get_stations().
    start : str or pd.Timestamp
        Start of the requested period (inclusive).
    end : str or pd.Timestamp
        End of the requested period (inclusive).
    interval : {'5min', 'hour', 'day'}
        Temporal resolution.
    variables : list[str]
        API variable names.  Use list_variables() to see what is available.
    verbose : bool, default True
        Log progress at INFO level.
    sleep : float, default 0.8
        Seconds to wait between chunk requests.

    Returns
    -------
    pd.DataFrame
        Columns: TIMESTAMP, then the requested variables in order.
    """
    _delta_map = {"day": "1D", "hour": "1h", "5min": "5min"}
    if interval not in _delta_map:
        raise ValueError(f"interval must be '5min', 'hour', or 'day'; got {interval!r}")

    delta    = pd.Timedelta(_delta_map[interval])
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    if interval == "day":
        start_dt = start_dt.normalize()
        end_dt   = end_dt.normalize()
    elif interval == "hour":
        start_dt = start_dt.floor("h")
        end_dt   = end_dt.floor("h")

    # Daily API timestamps are stored at 00:00 of the next day; work in that
    # space during the fetch and shift back at the end.
    if interval == "day":
        api_start = start_dt + delta
        api_end   = end_dt   + delta
    else:
        api_start = start_dt
        api_end   = end_dt

    full_idx  = pd.date_range(start=api_start, end=api_end, freq=delta)
    df_master = pd.DataFrame({"TIMESTAMP": full_idx}).set_index("TIMESTAMP")
    for var in variables:
        df_master[var] = np.nan

    cur = api_start
    while cur <= api_end:
        chunk_end = min(cur + delta * _MAX_RECORDS, api_end)
        url = (
            f"{_BASE_URL}/stationdata/"
            f"?stn={station}&int={interval}"
            f"&t_start={cur.strftime(_TS_FMT)}"
            f"&t_end={chunk_end.strftime(_TS_FMT)}"
            f"&vars={','.join(variables)}"
        ).replace(" ", "%20")

        log_cur = (cur - delta) if interval == "day" else cur
        log_end = (chunk_end - delta) if interval == "day" else chunk_end
        _log_chunk(station, log_cur, log_end, interval, verbose)

        df_chunk = _fetch_chunk(url, station, verbose)
        if not df_chunk.empty:
            df_chunk.set_index("TIMESTAMP", inplace=True)
            for col in df_chunk.columns:
                if col in df_master.columns:
                    df_master.loc[df_chunk.index, col] = df_chunk[col]

        cur = chunk_end + delta
        time.sleep(sleep)

    df_master.reset_index(inplace=True)

    if interval == "day":
        df_master["TIMESTAMP"] = df_master["TIMESTAMP"] - delta

    if verbose:
        logger.info("Done — %s | %d rows × %d variables", station, len(df_master), len(variables))

    cols = ["TIMESTAMP"] + [v for v in variables if v in df_master.columns]
    return df_master[cols]


def _fetch_chunk(url: str, station: str, verbose: bool) -> pd.DataFrame:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return pd.read_csv(url, na_values="M", parse_dates=["TIMESTAMP"])
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Request failed for {station!r} after {_MAX_RETRIES} attempts: {exc}"
                ) from exc
            if verbose:
                logger.warning("Attempt %d/%d failed for %s: %s — retrying", attempt, _MAX_RETRIES, station, exc)
            time.sleep(_RETRY_SLEEP)


def _log_chunk(station: str, start: pd.Timestamp, end: pd.Timestamp, interval: str, verbose: bool) -> None:
    if not verbose:
        return
    fmt = "%Y-%m-%d" if interval == "day" else "%Y-%m-%d %H:%M"
    logger.info("  %s  |  %s -> %s", station, start.strftime(fmt), end.strftime(fmt))


def request_data_multi(
    stations:  list[str],
    start:     Union[str, pd.Timestamp],
    end:       Union[str, pd.Timestamp],
    interval:  Literal["5min", "hour", "day"],
    variables: list[str],
    *,
    verbose:   bool = True,
    sleep:     float = _POLITE_SLEEP,
) -> dict[str, pd.DataFrame]:
    """
    Retrieve data for multiple stations, one at a time.

    Parameters
    ----------
    stations : list[str]
        Station names.
    start, end, interval, variables, verbose, sleep
        Same as request_data().

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are station names.
    """
    results: dict[str, pd.DataFrame] = {}
    for stn in stations:
        if verbose:
            logger.info("Station %s (%d/%d)", stn, len(results) + 1, len(stations))
        results[stn] = request_data(stn, start, end, interval, variables, verbose=verbose, sleep=sleep)
    return results


def rename_columns(
    df:     pd.DataFrame,
    preset: Union[Literal["snake"], dict] = "snake",
) -> pd.DataFrame:
    """
    Rename DataFrame columns using a preset or a custom mapping.

    Parameters
    ----------
    preset : 'snake' or dict, default 'snake'
        'snake' applies the built-in API-name -> snake_case mapping.
        Pass a dict for a custom mapping.

    Returns
    -------
    pd.DataFrame
        Copy with renamed columns.
    """
    if isinstance(preset, str):
        if preset != "snake":
            raise ValueError(f"Unknown preset {preset!r}. Use 'snake' or pass a dict.")
        mapping = RENAME_PRESET
    else:
        mapping = preset
    return df.rename(columns={c: mapping[c] for c in df.columns if c in mapping})


# ---------------------------------------------------------------------------
# Soil processing
# ---------------------------------------------------------------------------

def calibrate_vwc(df: pd.DataFrame, vwc_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Apply the KSU CS655 calibration equation and drop the raw Ka/EC columns.

    VWC = max(-0.115 + 0.0989 * sqrt(Ka) - 0.0572 * EC, 0)

    The raw VWC columns returned by the Mesonet API use the CS655 firmware
    equation; this function replaces them with values from the KSU site-specific
    calibration.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain SOILKA*CM and SOILEC*CM columns for each depth requested.
    vwc_cols : list[str] or None
        Subset of ['VWC5CM', 'VWC10CM', 'VWC20CM', 'VWC50CM'].
        Defaults to all four depths.

    Returns
    -------
    pd.DataFrame
        Copy of df with calibrated VWC values and Ka/EC columns removed.
    """
    df = df.copy()
    if vwc_cols is None:
        vwc_cols = _ALL_VWC

    to_drop: list[str] = []
    for col in vwc_cols:
        if col not in _VWC_DEPS:
            raise ValueError(f"{col!r} is not a valid VWC column. Options: {_ALL_VWC}")
        ka_col, ec_col = _VWC_DEPS[col]
        if ka_col not in df.columns or ec_col not in df.columns:
            logger.warning("Skipping %s calibration — %s or %s not found.", col, ka_col, ec_col)
            continue
        df[col] = np.round(np.maximum(-0.115 + 0.0989 * np.sqrt(df[ka_col]) - 0.0572 * df[ec_col], 0), 4)
        to_drop.extend([ka_col, ec_col])

    df.drop(columns=list(set(to_drop)), inplace=True, errors="ignore")
    return df


def compute_soil_water_storage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute soil water storage in the top 50 cm (mm) using the trapezoidal rule.

    The 5 cm sensor is assigned to both the surface (0 cm) and the 5 cm node;
    integration nodes are [0, 5, 10, 20, 50] cm. Rows with any NaN VWC
    produce NaN storage.

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
    doy:   Union[int, np.ndarray],
    lat:   Union[float, np.ndarray],
    elev:  Union[float, np.ndarray],
    tmin:  Union[float, np.ndarray],
    tmax:  Union[float, np.ndarray],
    srad:  Union[float, np.ndarray],
    wspd:  Union[float, np.ndarray],
    rhmin: Union[float, np.ndarray] | None = None,
    rhmax: Union[float, np.ndarray] | None = None,
    *,
    vpd:         Union[float, np.ndarray] | None = None,
    ea:          Union[float, np.ndarray] | None = None,
    wind_height: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Daily reference evapotranspiration by the FAO-56 Penman-Monteith method.

    Actual vapor pressure must be supplied via one of three paths (checked in
    this order): ea directly, vpd (ea is derived as es - vpd), or rhmin +
    rhmax (ea = (es_min * rhmax + es_max * rhmin) / 2, FAO-56 Eq. 17).

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
        Wind speed at wind_height (m s⁻¹).
    rhmin : float or array-like, optional
        Daily minimum relative humidity (%).
    rhmax : float or array-like, optional
        Daily maximum relative humidity (%).
    vpd : float or array-like, keyword-only, optional
        Vapor pressure deficit (kPa).
    ea : float or array-like, keyword-only, optional
        Actual vapor pressure (kPa).
    wind_height : float, default 2.0
        Anemometer height (m). The Mesonet 2 m sensor gives a correction of 1.0.

    Returns
    -------
    ETo : np.ndarray
        Reference ET (mm day⁻¹), rounded to 2 decimal places.
    Ra : np.ndarray
        Extraterrestrial radiation (MJ m⁻² day⁻¹), rounded to 2 decimal places.
    """
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    srad = np.asarray(srad, dtype=float)
    wspd = np.asarray(wspd, dtype=float)
    tavg = (tmin + tmax) / 2.0

    u2      = wspd * (4.87 / np.log(67.8 * wind_height - 5.42))   # wind at 2 m, Eq. 47
    srad_mj = srad_to_mj(srad, 86_400)

    es_min = saturation_vapor_pressure(tmin)
    es_max = saturation_vapor_pressure(tmax)
    es     = (es_min + es_max) / 2.0

    if ea is not None:
        ea = np.asarray(ea, dtype=float)
    elif vpd is not None:
        ea = np.maximum(es - np.asarray(vpd, dtype=float), 0.0)
    elif rhmin is not None and rhmax is not None:
        ea = (es_min * np.asarray(rhmax, dtype=float) / 100.0
              + es_max * np.asarray(rhmin, dtype=float) / 100.0) / 2.0
    else:
        raise ValueError("Provide ea, vpd, or rhmin + rhmax.")

    Ra    = extraterrestrial_radiation(doy, lat)
    Rn    = net_radiation(srad_mj, tmin, tmax, ea, elev, doy, lat)
    Delta = slope_saturation_vapor_pressure(tavg)
    gamma = psychrometric_constant(elev)
    vpd_c = np.maximum(es - ea, 0.0)

    ETo = (
        (0.408 * Delta * Rn + gamma * (900.0 / (tavg + 273.0)) * u2 * vpd_c)
        / (Delta + gamma * (1.0 + 0.34 * u2))
    )
    return np.round(ETo, 2), np.round(Ra, 2)


def reference_et_hargreaves(
    doy:   Union[int, np.ndarray],
    lat:   Union[float, np.ndarray],
    tmin:  Union[float, np.ndarray],
    tmax:  Union[float, np.ndarray],
    *,
    tmean: Union[float, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Daily reference evapotranspiration by the Hargreaves-Samani (1985) method.

    Requires only temperature and location; useful when humidity, radiation,
    and wind data are unavailable.

    ETo = 0.0023 * Ra * (tmean + 17.8) * sqrt(tmax - tmin)

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
    ETo : np.ndarray
        Reference ET (mm day⁻¹), rounded to 2 decimal places.
    Ra : np.ndarray
        Extraterrestrial radiation (MJ m⁻² day⁻¹), rounded to 2 decimal places.
    """
    tmin  = np.asarray(tmin,  dtype=float)
    tmax  = np.asarray(tmax,  dtype=float)
    tmean = np.asarray(tmean, dtype=float) if tmean is not None else (tmin + tmax) / 2.0
    Ra    = extraterrestrial_radiation(doy, lat)
    ETo   = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0))
    return np.round(ETo, 2), np.round(Ra, 2)
