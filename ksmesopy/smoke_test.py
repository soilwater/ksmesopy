"""
smoke_test.py
=============
Lightweight smoke tests for ksmesopy: each pure function is checked against a
trivial or well-known reference value. No network access is required — the
data-retrieval functions (request_data, get_stations, ...) are exercised only
for import/signature sanity, not called.

Run:
    python smoke_test.py          # plain, exits non-zero on failure
    pytest smoke_test.py          # also works under pytest

References
----------
FAO-56  : Allen et al. (1998), Irrigation & Drainage Paper 56.
NWS HI  : NOAA Rothfusz regression / heat-index tables.
NWS WC  : NOAA wind-chill formula (2001).
THI     : Mader et al. (2006), J. Anim. Sci. 84:1924.
"""

import numpy as np
import pandas as pd

import core as ms
import utils
import charts

# Absolute tolerance for float comparisons (values are rounded to <= 4 dp).
TOL = 1e-2


def approx(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# core.py — vapor pressure & atmospheric helpers
# ---------------------------------------------------------------------------

def test_saturation_vapor_pressure():
    # FAO-56: es(20 °C) = 2.338 kPa
    assert approx(ms.saturation_vapor_pressure(20), 2.338)
    # es(0 °C) = 0.6108 kPa by definition of the Tetens form
    assert approx(ms.saturation_vapor_pressure(0), 0.6108)


def test_actual_vapor_pressure():
    # 50 % RH halves the saturation value
    assert approx(ms.actual_vapor_pressure(20, 50), 2.338 / 2)
    # 100 % RH -> ea == es
    assert approx(ms.actual_vapor_pressure(20, 100), ms.saturation_vapor_pressure(20))


def test_vapor_pressure_deficit():
    # VPD at 50 % RH equals the actual vapor pressure at 20 °C (es - es/2)
    assert approx(ms.vapor_pressure_deficit(20, 50), 1.1691)
    # Saturated air -> zero deficit
    assert approx(ms.vapor_pressure_deficit(20, 100), 0.0)


def test_atmospheric_pressure():
    # FAO-56 Eq. 7: sea level -> 101.3 kPa
    assert approx(ms.atmospheric_pressure(0), 101.3)
    # Pressure decreases with elevation
    assert ms.atmospheric_pressure(1000) < ms.atmospheric_pressure(0)


def test_slope_saturation_vapor_pressure():
    # FAO-56 Eq. 13: Delta(20 °C) ~ 0.1447 kPa/°C
    assert approx(ms.slope_saturation_vapor_pressure(20), 0.1447)


def test_psychrometric_constant():
    # FAO-56 Eq. 8: gamma at sea level ~ 0.0673 kPa/°C
    assert approx(ms.psychrometric_constant(0), 0.0673)


def test_srad_to_mj():
    # 300 W/m^2 sustained over one day = 300 * 86400 / 1e6 = 25.92 MJ/m^2
    assert approx(ms.srad_to_mj(300, 86400), 25.92)
    # Zero irradiance -> zero energy
    assert approx(ms.srad_to_mj(0, 86400), 0.0)


def test_extraterrestrial_radiation():
    # FAO-56 Example 8: doy=246, lat=-20 deg -> Ra = 32.2 MJ/m^2/day
    assert approx(ms.extraterrestrial_radiation(246, -20), 32.2, tol=0.05)


def test_net_radiation_positive_daytime():
    # A clear summer day should yield positive net radiation.
    rn = ms.net_radiation(srad_mj=22.0, tmin=15, tmax=30, ea=1.4,
                          elev=300, doy=196, lat=39)
    assert float(rn) > 0.0


# ---------------------------------------------------------------------------
# core.py — reference evapotranspiration
# ---------------------------------------------------------------------------

def test_reference_et_penman_monteith_summer():
    # Typical clear summer day -> ETo in the physically sensible 4-9 mm range.
    eto = ms.reference_et_penman_monteith(
        doy=196, lat=39, elev=300, tmin=18, tmax=32,
        srad=300, wspd=2, rhmin=30, rhmax=80,
    )
    assert 4.0 <= float(eto) <= 9.0


def test_reference_et_penman_monteith_ea_paths_agree():
    # Supplying ea directly vs via vpd should give the same ETo.
    es = (ms.saturation_vapor_pressure(18) + ms.saturation_vapor_pressure(32)) / 2
    ea = 1.2
    eto_ea = ms.reference_et_penman_monteith(
        196, 39, 300, 18, 32, 300, 2, ea=ea)
    eto_vpd = ms.reference_et_penman_monteith(
        196, 39, 300, 18, 32, 300, 2, vpd=float(es - ea))
    assert approx(eto_ea, eto_vpd)


def test_reference_et_hargreaves():
    # Hargreaves-Samani: ETo = 0.0023 * Ra * (tmean + 17.8) * sqrt(tmax - tmin)
    eto = ms.reference_et_hargreaves(doy=196, lat=40, tmin=10, tmax=25)
    ra = ms.extraterrestrial_radiation(196, 40)
    tmean = (10 + 25) / 2
    expected = 0.0023 * float(ra) * (tmean + 17.8) * np.sqrt(25 - 10)
    assert approx(eto, round(expected, 2))


# ---------------------------------------------------------------------------
# core.py — soil processing
# ---------------------------------------------------------------------------

def test_calibrate_vwc():
    # VWC = max(-0.115 + 0.0989*sqrt(Ka) - 0.0572*EC, 0)
    # Ka=10, EC=0.2 -> 0.1863
    df = pd.DataFrame({
        "VWC5CM":    [np.nan],
        "SOILKA5CM": [10.0],
        "SOILEC5CM": [0.2],
    })
    out = ms.calibrate_vwc(df, ["VWC5CM"])
    assert approx(out["VWC5CM"].iloc[0], 0.1863, tol=1e-3)


def test_calibrate_vwc_clips_negative():
    # Very low Ka drives the raw equation negative -> clipped to 0.
    df = pd.DataFrame({"VWC5CM": [np.nan], "SOILKA5CM": [1.0], "SOILEC5CM": [0.0]})
    out = ms.calibrate_vwc(df, ["VWC5CM"])
    assert approx(out["VWC5CM"].iloc[0], 0.0)


def test_compute_soil_water_storage_uniform():
    # Uniform VWC of 0.3 over the top 500 mm -> 0.3 * 500 = 150 mm.
    df = pd.DataFrame({
        "VWC5CM":  [0.3], "VWC10CM": [0.3],
        "VWC20CM": [0.3], "VWC50CM": [0.3],
    })
    out = ms.compute_soil_water_storage(df)
    assert approx(out["STORAGE_MM"].iloc[0], 150.0)


def test_compute_soil_water_storage_nan_propagates():
    # Any missing depth -> NaN storage.
    df = pd.DataFrame({
        "VWC5CM":  [np.nan], "VWC10CM": [0.3],
        "VWC20CM": [0.3],    "VWC50CM": [0.3],
    })
    out = ms.compute_soil_water_storage(df)
    assert np.isnan(out["STORAGE_MM"].iloc[0])


# ---------------------------------------------------------------------------
# core.py — catalogue & renaming (no network)
# ---------------------------------------------------------------------------

def test_list_variables_filtering():
    all_vars = ms.list_variables()
    day_vars = ms.list_variables("day")
    hour_vars = ms.list_variables("hour")
    # Min/max columns exist only at the daily interval.
    assert len(day_vars) == len(all_vars)
    assert len(hour_vars) < len(day_vars)
    names = {v["api_name"] for v in hour_vars}
    assert "TEMP2MMIN" not in names        # daily-only
    assert "TEMP2MAVG" in names            # all intervals


def test_rename_columns_snake():
    df = pd.DataFrame(columns=["TIMESTAMP", "TEMP2MAVG", "PRECIP"])
    out = ms.rename_columns(df)
    assert list(out.columns) == ["timestamp", "tair_2m_avg", "precip"]


# ---------------------------------------------------------------------------
# utils.py — agrometeorology
# ---------------------------------------------------------------------------

def test_growing_degree_days():
    # Standard case: (15+25)/2 - 10 = 10
    assert approx(utils.growing_degree_days(15, 25), 10.0)
    # Cold night clamped to base: tmin 2 -> 10, (10+20)/2 - 10 = 5
    assert approx(utils.growing_degree_days(2, 20), 5.0)
    # Entirely below base -> 0
    assert approx(utils.growing_degree_days(5, 8), 0.0)
    # Heat wave clamped to ceiling: tmax 40 -> 30, (30+30)/2 - 10 = 20
    assert approx(utils.growing_degree_days(35, 40), 20.0)


def test_heat_index():
    # NWS table: 90 F / 70 % RH ~ 106 F. 90 F = 32.22 C, 106 F = 41.1 C.
    assert approx(utils.heat_index(32.22, 70), 41.1, tol=0.5)
    # Cool conditions pass through roughly unchanged (< 27 C branch).
    assert approx(utils.heat_index(20, 50), 20.0, tol=6.0)


def test_wind_chill():
    # NWS: 0 F / 15 mph -> -19 F. 0 F = -17.78 C, 6.7056 m/s = 15 mph.
    # -19 F = -28.3 C.
    assert approx(utils.wind_chill(-17.78, 6.7056), -28.3, tol=0.5)
    # Above 10 C the observed temperature is returned unchanged.
    assert approx(utils.wind_chill(15, 5), 15.0)


def test_temperature_humidity_index():
    # THI(30 C, 50 %) ~ 78.3 (moderate stress band).
    assert approx(utils.temperature_humidity_index(30, 50), 78.3, tol=0.5)


def test_vectorized_inputs():
    # Functions should accept arrays and return element-wise results.
    gdd = utils.growing_degree_days(np.array([15, 2]), np.array([25, 20]))
    assert gdd.shape == (2,)
    assert approx(gdd[0], 10.0) and approx(gdd[1], 5.0)


# ---------------------------------------------------------------------------
# charts.py — smoke render (headless Agg backend)
# ---------------------------------------------------------------------------

def _sample_frame():
    ts = pd.date_range("2024-06-01", periods=5, freq="D")
    return pd.DataFrame({
        "TIMESTAMP":   ts,
        "TEMP2MAVG":   [18, 19, 20, 21, 22],
        "TEMP2MMIN":   [12, 13, 14, 15, 16],
        "TEMP2MMAX":   [24, 25, 26, 27, 28],
        "RELHUM2MAVG": [60, 62, 58, 55, 50],
        "VPDEFAVG":    [1.0, 1.1, 1.2, 1.3, 1.4],
        "SRAVG":       [200, 220, 240, 210, 230],
        "PRECIP":      [0, 5, 0, 12, 3],
        "WSPD2MAVG":   [2.0, 2.5, 3.0, 2.2, 1.8],
        "WDIR2M":      [180, 200, 90, 270, 45],
        "VWC5CM":      [0.30, 0.31, 0.29, 0.28, 0.30],
        "VWC10CM":     [0.32, 0.33, 0.31, 0.30, 0.32],
        "VWC20CM":     [0.34, 0.34, 0.33, 0.33, 0.34],
        "VWC50CM":     [0.36, 0.36, 0.35, 0.35, 0.36],
        "ETo":         [4.1, 4.5, 5.0, 4.2, 4.8],
    })


def test_charts_render_without_error():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _sample_frame()
    ax = charts.plot_temperature(None, df, ["TEMP2MMIN", "TEMP2MMAX", "TEMP2MAVG"])
    assert ax is not None
    charts.plot_precip(None, df, "PRECIP")
    charts.plot_humidity(None, df, "RELHUM2MAVG")
    charts.plot_vpd(None, df, "VPDEFAVG")
    charts.plot_solar_radiation(None, df, "SRAVG")
    charts.plot_wind(None, df, "WSPD2MAVG", "WDIR2M")
    charts.plot_vwc(None, df)   # auto-discovers the four VWC depths
    charts.plot_et(None, df, "ETo", bar=True)
    plt.close("all")


def test_charts_missing_column_raises():
    df = _sample_frame()
    try:
        charts.plot_precip(None, df, "NOT_A_COLUMN")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a missing column.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - smoke runner reports all
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
