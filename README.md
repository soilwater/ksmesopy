# ksmesopy

Python package for downloading and processing data from the [Kansas Mesonet](https://mesonet.k-state.edu), an environmental monitoring network operated by Kansas State University.

## Installation

```bash
pip install git+https://github.com/<your-username>/ksmesopy.git
```

Dependencies: `numpy`, `pandas`, `matplotlib`. The desktop app additionally requires `guile`.

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
| `get_stations()` | `DataFrame` | Full station metadata table |
| `get_stations(names_only=True)` | `list[str]` | Sorted list of station names only |
| `get_stations_active()` | `DataFrame` | Availability table: `STATION`, `OBS_INTERVAL` (s), `START`, `END` |

### Data retrieval

| Function | Returns | Description |
|---|---|---|
| `request_data(station, start, end, interval, variables, *, verbose, sleep)` | `DataFrame` | Download data for one station. `interval` is `"day"`, `"hour"`, or `"5min"`. Daily timestamps are corrected from the Mesonet's next-day convention. |
| `request_data_multi(stations, start, end, interval, variables, *, verbose, sleep)` | `dict[str, DataFrame]` | Same as above for a list of stations; returns one DataFrame per station. |
| `list_variables(interval=None)` | `list[dict]` | Variable catalogue filtered by interval, or all variables if `None`. Each entry has keys `api_name`, `snake_name`, `description`, `intervals`. |
| `rename_columns(df, preset="snake")` | `DataFrame` | Rename API column names to snake_case (e.g. `TEMP2MAVG` → `t2m`). Pass a `dict` for a custom mapping. |

### Soil processing

| Function | Returns | Description |
|---|---|---|
| `calibrate_vwc(df, vwc_cols=None)` | `DataFrame` | Replace firmware VWC values with the KSU site-specific calibration. Fetch `SOILKA*CM` and `SOILEC*CM` alongside `VWC*CM` in the same `request_data` call. Works on any subset of depths. |
| `compute_soil_water_storage(df)` | `DataFrame` | Trapezoidal soil water storage in the top 50 cm (mm). Requires all four VWC depths; adds a `STORAGE_MM` column. Call `calibrate_vwc()` first for calibrated storage. |

### Derived variables

| Function | Returns | Description |
|---|---|---|
| `growing_degree_days(tmin, tmax, base=10.0, ceiling=30.0)` | `ndarray` | Daily GDD. Both tmin/tmax clipped to `[base, ceiling]` before averaging. |
| `heat_index(temp, rh)` | `ndarray` | NOAA/NWS apparent temperature (°C). |
| `wind_chill(temp, wspd)` | `ndarray` | NWS wind chill (°C); valid for temp ≤ 10 °C, wind ≥ 1.3 m s⁻¹. |
| `temperature_humidity_index(temp, rh)` | `ndarray` | THI for livestock heat stress. Thresholds (dairy): <68 none · 68–72 mild · 72–80 moderate · 80–90 severe · >90 dangerous. |

### Reference evapotranspiration

| Function | Returns | Description |
|---|---|---|
| `reference_et_penman_monteith(doy, lat, elev, tmin, tmax, srad, wspd, rhmin, rhmax, *, vpd, ea, wind_height)` | `(ETo, Ra)` | FAO-56 Penman-Monteith. Supply vapour pressure via `ea=`, `vpd=`, or `rhmin=`+`rhmax=`. `srad` in W m⁻², converted internally. |
| `reference_et_hargreaves(doy, lat, tmin, tmax, *, tmean)` | `(ETo, Ra)` | Hargreaves–Samani. Temperature only — no humidity, radiation, or wind needed. |

Both return `(ETo [mm day⁻¹], Ra [MJ m⁻² day⁻¹])`.

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

### Charts

Each function draws onto a Matplotlib `Axes` supplied by the caller, so panels compose freely inside any figure layout. All functions accept API column names (`TEMP2MAVG`) or snake_case names (`t2m`) interchangeably, and return the axes they drew on so they can be further customised.

```python
import matplotlib.pyplot as plt
import ksmesopy as ms

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 14))

ms.plot_temperature(axes[0], df, ["TEMP2MAVG", "TEMP2MMIN", "TEMP2MMAX"])
ms.plot_precip(axes[1], df, "PRECIP")
ms.plot_humidity(axes[2], df, "RELHUM2MAVG")
ms.plot_solar_radiation(axes[3], df, ["SRAVG", "Ra"])  # Ra drawn dashed
ms.plot_vwc(axes[4], df)                               # auto-detects VWC columns

plt.tight_layout()
plt.savefig("meteogram.png", dpi=150)
```

plt.tight_layout()
plt.savefig("meteogram.png", dpi=150)
```

All functions accept API column names (`TEMP2MAVG`) or snake_case names (`t2m`) interchangeably, and return the axes they drew on so they can be further customised.

| Function | Key behaviour |
|---|---|
| `plot_temperature(ax, df, variables, *, band, ylabel, legend)` | Shaded band when min/avg/max triplet detected (`band=True`); plain lines otherwise. Works for air and soil temperature. |
| `plot_precip(ax, df, variable, *, ylabel, color)` | Bar chart, bar width inferred from timestamp spacing. |
| `plot_humidity(ax, df, variables, *, ylabel, legend)` | Lines, y-axis fixed 0–100 %. |
| `plot_vpd(ax, df, variables, *, ylabel, legend)` | Filled area + line, y-axis starts at 0. |
| `plot_solar_radiation(ax, df, variables, *, ylabel, legend)` | Filled area for observed; dashed line for Ra columns (detected by name). |
| `plot_wind(ax, df, speed, direction, *, ylabel, legend)` | Speed as line; direction overlaid as scatter on a twin y-axis with N/E/S/W ticks. |
| `plot_vwc(ax, df, variables, *, ylabel, legend)` | Sequential colormap shallow→deep; auto-detects VWC columns if `variables=None`. |
| `plot_et(ax, df, variables, *, bar, ylabel, legend)` | Line by default; `bar=True` for daily totals. |

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

> **VWC note:** the Mesonet API returns VWC computed by the CS655 firmware equation. Requesting any `VWC*CM` column will print a warning reminding you that a site-specific calibration is available. To apply it, fetch the corresponding `SOILKA*CM` and `SOILEC*CM` columns in the same `request_data` call and then pass the DataFrame to `calibrate_vwc()`. Works for any subset of depths independently.

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
