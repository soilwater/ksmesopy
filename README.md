# ksmesopy

Python package for downloading and processing data from the [Kansas Mesonet](https://mesonet.k-state.edu), an environmental monitoring network operated by Kansas State University.

## Installation

```bash
pip install git+https://github.com/<your-username>/ksmesopy.git
```

Dependencies: `numpy`, `pandas`, `requests`. The desktop app additionally requires `matplotlib` and `guile`.

## Quick start

```python
import ksmesopy as ms

# Download daily temperature and precipitation for one station
df = ms.request_data(
    station="Manhattan",
    start="2024-01-01",
    end="2024-12-31",
    interval="day",
    variables=["TEMP2MAVG", "TEMP2MMIN", "TEMP2MMAX", "PRECIP"],
)

# Optional: rename columns to snake_case
df = ms.rename_columns(df)
# TIMESTAMP → timestamp, TEMP2MAVG → t2m, PRECIP → precip, …
```

## Desktop app

```bash
python ksmesoapp.py
```

A GUI for selecting stations, date ranges, variables, and intervals, with a tabular view and time-series chart. Exports to CSV and PNG.

---

## API reference

### Stations

| Function | Returns | Description |
|---|---|---|
| `get_stations()` | `list[str]` | Sorted list of all station names |
| `get_stations_active()` | `DataFrame` | Availability table with columns `STATION`, `OBS_INTERVAL` (s), `START`, `END` |

### Data retrieval

```python
ms.request_data(station, start, end, interval, variables, *, verbose=True, sleep=0.8)
```

- **`interval`** — `"day"`, `"hour"`, or `"5min"`
- **`variables`** — list of API names from the table below
- Returns a `DataFrame` with `TIMESTAMP` plus the requested columns
- Daily timestamps are corrected for the Mesonet convention of storing each day's values at 00:00 of the following day

```python
ms.request_data_multi(stations, start, end, interval, variables, ...)
# -> dict[str, DataFrame], one entry per station
```

```python
ms.list_variables(interval=None)
# -> list of dicts with keys: api_name, snake_name, description, intervals
```

### Soil processing

```python
# Apply the KSU site-specific CS655 calibration equation.
# Requires Ka (SOILKA*CM) and EC (SOILEC*CM) to be fetched alongside VWC*CM.
# Drops the raw Ka/EC columns from the output.
df = ms.calibrate_vwc(df, vwc_cols=["VWC5CM", "VWC10CM", "VWC20CM", "VWC50CM"])

# Trapezoidal soil water storage in the top 50 cm (mm).
# Requires all four VWC depths. Adds a STORAGE_MM column.
df = ms.compute_soil_water_storage(df)
```

### Derived variables (utils)

```python
# Growing degree days — clips tmin/tmax to [base, ceiling] before averaging
gdd = ms.growing_degree_days(tmin, tmax, base=10.0, ceiling=30.0)

# NOAA/NWS heat index (apparent temperature, °C)
hi = ms.heat_index(temp, rh)

# NWS wind chill (°C); valid for temp <= 10 °C, wind >= 1.3 m s⁻¹
wc = ms.wind_chill(temp, wspd)

# Temperature-Humidity Index for livestock heat stress (dimensionless)
thi = ms.temperature_humidity_index(temp, rh)
# Thresholds (dairy): <68 none · 68–72 mild · 72–80 moderate · 80–90 severe · >90 dangerous
```

### Reference evapotranspiration

```python
# FAO-56 Penman-Monteith — supply actual vapour pressure via one of:
#   ea=     actual vapour pressure (kPa)
#   vpd=    vapour pressure deficit (kPa)
#   rhmin= + rhmax=   daily RH min/max (%)
ETo, Ra = ms.reference_et_penman_monteith(
    doy, lat, elev, tmin, tmax, srad, wspd,
    rhmin=rh_min, rhmax=rh_max,
)

# Hargreaves–Samani — temperature only, no humidity or radiation needed
ETo, Ra = ms.reference_et_hargreaves(doy, lat, tmin, tmax)

# Both return (ETo [mm day⁻¹], Ra [MJ m⁻² day⁻¹])
# srad is supplied as W m⁻² (as reported by the Mesonet); converted internally
```

### Atmospheric helpers

| Function | Output |
|---|---|
| `saturation_vapor_pressure(temp)` | es (kPa) |
| `actual_vapor_pressure(temp, rh)` | ea (kPa) |
| `vapor_pressure_deficit(temp, rh)` | VPD (kPa, ≥ 0) |
| `slope_saturation_vapor_pressure(temp)` | Δ (kPa °C⁻¹) |
| `atmospheric_pressure(elev)` | P (kPa) |
| `psychrometric_constant(elev)` | γ (kPa °C⁻¹) |
| `extraterrestrial_radiation(doy, lat)` | Ra (MJ m⁻² day⁻¹) |
| `net_radiation(srad_mj, tmin, tmax, ea, elev, doy, lat)` | Rn (MJ m⁻² day⁻¹) |
| `srad_to_mj(srad, period)` | energy (MJ m⁻²) |

All functions accept scalars or NumPy arrays.

---

## Variable catalogue

Intervals: **D** = daily only · **H** = hourly and daily · **A** = 5-min, hourly, and daily

### Atmospheric

| API name | snake_case | Description | Unit | Intervals |
|---|---|---|---|---|
| `TEMP2MAVG` | `t2m` | Air temperature 2 m avg | °C | A |
| `TEMP2MMIN` | `t2m_min` | Air temperature 2 m min | °C | D |
| `TEMP2MMAX` | `t2m_max` | Air temperature 2 m max | °C | D |
| `TEMP10MAVG` | `t10m` | Air temperature 10 m avg | °C | A |
| `TEMP10MMIN` | `t10m_min` | Air temperature 10 m min | °C | D |
| `TEMP10MMAX` | `t10m_max` | Air temperature 10 m max | °C | D |
| `RELHUM2MAVG` | `rh` | Relative humidity 2 m avg | % | A |
| `RELHUM2MMIN` | `rh_min` | Relative humidity 2 m min | % | D |
| `RELHUM2MMAX` | `rh_max` | Relative humidity 2 m max | % | D |
| `VPDEFAVG` | `vpd` | Vapor pressure deficit avg | kPa | A |
| `PRESSUREAVG` | `pres` | Atmospheric pressure avg | kPa | A |
| `PRECIP` | `precip` | Precipitation gauge 1 | mm | A |
| `PRECIP2` | `precip2` | Precipitation gauge 2 | mm | A |
| `SRAVG` | `srad` | Solar radiation avg | W m⁻² | A |
| `WSPD2MAVG` | `wspd` | Wind speed 2 m avg | m s⁻¹ | A |
| `WSPD2MMAX` | `wspd_max` | Wind speed 2 m max | m s⁻¹ | H¹ |
| `WDIR2M` | `wdir` | Wind direction 2 m | ° | A |
| `WDIR2MSTD` | `wdir_std` | Wind direction 2 m std dev | ° | A |
| `WSPD10MAVG` | `wspd10m` | Wind speed 10 m avg | m s⁻¹ | A |
| `WSPD10MMAX` | `wspd10m_max` | Wind speed 10 m max | m s⁻¹ | H¹ |
| `WDIR10M` | `wdir10m` | Wind direction 10 m | ° | A |
| `WDIR10MSTD` | `wdir10m_std` | Wind direction 10 m std dev | ° | A |

¹ Available at 5-min and daily only (not hourly).

### Soil temperature — dedicated probes

| API name | snake_case | Description | Unit | Intervals |
|---|---|---|---|---|
| `SOILTMP5AVG` | `tsoil_5cm` | Soil temperature 5 cm avg | °C | A |
| `SOILTMP5MIN` | `tsoil_5cm_min` | Soil temperature 5 cm min | °C | D |
| `SOILTMP5MAX` | `tsoil_5cm_max` | Soil temperature 5 cm max | °C | D |
| `SOILTMP10AVG` | `tsoil_10cm` | Soil temperature 10 cm avg | °C | A |
| `SOILTMP10MIN` | `tsoil_10cm_min` | Soil temperature 10 cm min | °C | D |
| `SOILTMP10MAX` | `tsoil_10cm_max` | Soil temperature 10 cm max | °C | D |

### Soil — CS655 sensors

| API name | snake_case | Description | Unit | Intervals |
|---|---|---|---|---|
| `SOILTMP5AVG655` | `tsoil_5cm_655` | Soil temperature 5 cm | °C | A |
| `SOILTMP10AVG655` | `tsoil_10cm_655` | Soil temperature 10 cm | °C | A |
| `SOILTMP20AVG655` | `tsoil_20cm_655` | Soil temperature 20 cm | °C | A |
| `SOILTMP50AVG655` | `tsoil_50cm_655` | Soil temperature 50 cm | °C | A |
| `SOILKA5CM` | `ka_5cm` | Dielectric constant 5 cm | — | A |
| `SOILKA10CM` | `ka_10cm` | Dielectric constant 10 cm | — | A |
| `SOILKA20CM` | `ka_20cm` | Dielectric constant 20 cm | — | A |
| `SOILKA50CM` | `ka_50cm` | Dielectric constant 50 cm | — | A |
| `SOILEC5CM` | `ec_5cm` | Electrical conductivity 5 cm | dS m⁻¹ | A |
| `SOILEC10CM` | `ec_10cm` | Electrical conductivity 10 cm | dS m⁻¹ | A |
| `SOILEC20CM` | `ec_20cm` | Electrical conductivity 20 cm | dS m⁻¹ | A |
| `SOILEC50CM` | `ec_50cm` | Electrical conductivity 50 cm | dS m⁻¹ | A |
| `VWC5CM` | `vwc_5cm` | Volumetric water content 5 cm | m³ m⁻³ | A |
| `VWC10CM` | `vwc_10cm` | Volumetric water content 10 cm | m³ m⁻³ | A |
| `VWC20CM` | `vwc_20cm` | Volumetric water content 20 cm | m³ m⁻³ | A |
| `VWC50CM` | `vwc_50cm` | Volumetric water content 50 cm | m³ m⁻³ | A |

> **VWC note:** the Mesonet API returns firmware-equation VWC. Call `calibrate_vwc()` to apply the KSU site-specific equation, which requires fetching `SOILKA*CM` and `SOILEC*CM` alongside the VWC columns. The desktop app does this automatically.

---

## Notes

**Precipitation.** The Mesonet operates dual tipping-bucket rain gauges at most stations. `request_data()` returns both (`PRECIP`, `PRECIP2`); the desktop app merges them to the row-wise maximum automatically. For scripted use, merge manually:

```python
df["PRECIP"] = df[["PRECIP", "PRECIP2"]].max(axis=1)
df.drop(columns="PRECIP2", inplace=True)
```

**Daily timestamps.** The Mesonet API stores each day's aggregated values at 00:00 of the following calendar day. `request_data()` corrects for this automatically; the returned `TIMESTAMP` always reflects the observation date.

**Missing values.** The API encodes missing observations as `"M"`. These are converted to `NaN` on read. Periods before a station or sensor was installed are pre-filled with `NaN` rather than omitted.

---

## License

MIT
