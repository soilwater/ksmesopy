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
    ("TEMP2MAVG",       "tair_2m_avg",      "Air temperature 2 m avg (°C)",           _ALL),
    ("TEMP2MMIN",       "tair_2m_min",      "Air temperature 2 m min (°C)",           _DONLY),
    ("TEMP2MMAX",       "tair_2m_max",      "Air temperature 2 m max (°C)",           _DONLY),
    ("TEMP10MAVG",      "tair_10m_avg",     "Air temperature 10 m avg (°C)",          _ALL),
    ("TEMP10MMIN",      "tair_10m_min",     "Air temperature 10 m min (°C)",          _DONLY),
    ("TEMP10MMAX",      "tair_10m_max",     "Air temperature 10 m max (°C)",          _DONLY),
    ("RELHUM2MAVG",     "rh_2m_avg",        "Relative humidity 2 m avg (%)",          _ALL),
    ("RELHUM2MMIN",     "rh_2m_min",        "Relative humidity 2 m min (%)",          _DONLY),
    ("RELHUM2MMAX",     "rh_2m_max",        "Relative humidity 2 m max (%)",          _DONLY),
    ("VPDEFAVG",        "vpd_avg",          "Vapor pressure deficit avg (kPa)",       _ALL),
    ("PRESSUREAVG",     "pressure_avg",     "Atmospheric pressure avg (kPa)",         _ALL),
    ("PRECIP",          "precip",           "Precipitation gauge 1 (mm)",             _ALL),
    ("PRECIP2",         "precip2",          "Precipitation gauge 2 (mm)",             _ALL),
    ("SRAVG",           "srad",             "Solar radiation avg (W m⁻²)",            _ALL),
    ("WSPD2MAVG",       "wspd_2m_avg",      "Wind speed 2 m avg (m s⁻¹)",            _ALL),
    ("WSPD2MMAX",       "wspd_2m_max",      "Wind speed 2 m max (m s⁻¹)",            _D5),
    ("WDIR2M",          "wdir_2m",          "Wind direction 2 m (°)",                 _ALL),
    ("WDIR2MSTD",       "wdir_2m_std",      "Wind direction 2 m std dev (°)",         _ALL),
    ("WSPD10MAVG",      "wspd_10m_avg",     "Wind speed 10 m avg (m s⁻¹)",           _ALL),
    ("WSPD10MMAX",      "wspd_10m_max",     "Wind speed 10 m max (m s⁻¹)",           _D5),
    ("WDIR10M",         "wdir_10m",         "Wind direction 10 m (°)",                _ALL),
    ("WDIR10MSTD",      "wdir_10m_std",     "Wind direction 10 m std dev (°)",        _ALL),
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

def get_stations(names_only: bool = False) -> pd.DataFrame | list[str]:
    """
    Return station metadata or a plain list of station names.

    Parameters
    ----------
    names_only : bool, default False
        If False, return the full metadata DataFrame from the API.
        If True, return a sorted list of station name strings.

    Returns
    -------
    pd.DataFrame or list[str]
    """
    url = f"{_BASE_URL}/stationnames/"
    try:
        df = pd.read_csv(url)
        if names_only:
            names = sorted(df.iloc[:, 0].dropna().unique().tolist())
            if not names:
                raise RuntimeError("API returned an empty station list.")
            return names
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

