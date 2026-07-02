"""
ksmesopy GUI
============
Desktop app for downloading and exploring Kansas Mesonet data.

Tabs
----
⚙  Inputs  — Station, date range, interval, and variable selection
📋  Table   — Downloaded data with CSV export
📈  Chart   — Time-series plot with PNG export

Run
---
    python -m ksmesopy.app
    # or directly:
    python ksmesoapp.py
"""

import io
import os
import base64
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import guile as gui
import ksmesopy.core as core

# ---------------------------------------------------------------------------
# GUI variable catalogue
# Extends core.VARIABLES with a display label and group for the checkbox UI.
# Ka/EC are excluded — fetched transparently when VWC is requested.
# ---------------------------------------------------------------------------

_ALL   = {"5min", "hour", "day"}
_DONLY = {"day"}
_D5    = {"5min", "day"}

# (api_name, label, group, intervals)
_CATALOGUE = [
    # Atmospheric
    ("TEMP2MAVG",       "Air temp avg (°C)",              "Atmospheric", _ALL),
    ("TEMP2MMIN",       "Air temp min (°C)",              "Atmospheric", _DONLY),
    ("TEMP2MMAX",       "Air temp max (°C)",              "Atmospheric", _DONLY),
    ("TEMP10MAVG",      "Air temp 10 m avg (°C)",         "Atmospheric", _ALL),
    ("TEMP10MMIN",      "Air temp 10 m min (°C)",         "Atmospheric", _DONLY),
    ("TEMP10MMAX",      "Air temp 10 m max (°C)",         "Atmospheric", _DONLY),
    ("RELHUM2MAVG",     "Rel. humidity avg (%)",          "Atmospheric", _ALL),
    ("RELHUM2MMIN",     "Rel. humidity min (%)",          "Atmospheric", _DONLY),
    ("RELHUM2MMAX",     "Rel. humidity max (%)",          "Atmospheric", _DONLY),
    ("VPDEFAVG",        "Vapor pressure deficit (kPa)",   "Atmospheric", _ALL),
    ("PRESSUREAVG",     "Atm. pressure avg (kPa)",        "Atmospheric", _ALL),
    ("PRECIP",          "Precipitation (mm)",             "Atmospheric", _ALL),
    ("SRAVG",           "Solar radiation avg (W/m²)",     "Atmospheric", _ALL),
    ("WSPD2MAVG",       "Wind speed 2 m avg (m/s)",       "Atmospheric", _ALL),
    ("WSPD2MMAX",       "Wind speed 2 m max (m/s)",       "Atmospheric", _D5),
    ("WDIR2M",          "Wind direction 2 m (°)",         "Atmospheric", _ALL),
    ("WDIR2MSTD",       "Wind dir 2 m std (°)",           "Atmospheric", _ALL),
    ("WSPD10MAVG",      "Wind speed 10 m avg (m/s)",      "Atmospheric", _ALL),
    ("WSPD10MMAX",      "Wind speed 10 m max (m/s)",      "Atmospheric", _D5),
    ("WDIR10M",         "Wind direction 10 m (°)",        "Atmospheric", _ALL),
    ("WDIR10MSTD",      "Wind dir 10 m std (°)",          "Atmospheric", _ALL),
    # Soil temperature — dedicated probes
    ("SOILTMP5AVG",     "Soil temp 5 cm avg (°C)",        "Soil", _ALL),
    ("SOILTMP5MIN",     "Soil temp 5 cm min (°C)",        "Soil", _DONLY),
    ("SOILTMP5MAX",     "Soil temp 5 cm max (°C)",        "Soil", _DONLY),
    ("SOILTMP10AVG",    "Soil temp 10 cm avg (°C)",       "Soil", _ALL),
    ("SOILTMP10MIN",    "Soil temp 10 cm min (°C)",       "Soil", _DONLY),
    ("SOILTMP10MAX",    "Soil temp 10 cm max (°C)",       "Soil", _DONLY),
    # Soil temperature — CS655
    ("SOILTMP5AVG655",  "Soil temp 5 cm CS655 (°C)",      "Soil", _ALL),
    ("SOILTMP10AVG655", "Soil temp 10 cm CS655 (°C)",     "Soil", _ALL),
    ("SOILTMP20AVG655", "Soil temp 20 cm CS655 (°C)",     "Soil", _ALL),
    ("SOILTMP50AVG655", "Soil temp 50 cm CS655 (°C)",     "Soil", _ALL),
    # Soil VWC (Ka + EC fetched automatically; KSU calibration applied)
    ("VWC5CM",          "Soil VWC 5 cm (m³/m³)",          "Soil", _ALL),
    ("VWC10CM",         "Soil VWC 10 cm (m³/m³)",         "Soil", _ALL),
    ("VWC20CM",         "Soil VWC 20 cm (m³/m³)",         "Soil", _ALL),
    ("VWC50CM",         "Soil VWC 50 cm (m³/m³)",         "Soil", _ALL),
]

def _catalogue_for_interval(intv: str) -> list:
    return [row for row in _CATALOGUE if intv in row[3]]


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_stations_list   = gui.state([])
_station         = gui.state("")
_station_active  = gui.state({})   # (station_name, intv_str) -> (start_ts, end_ts)
_start_date      = gui.state("2024-01-01T00:00")
_end_date        = gui.state("2024-12-31T00:00")
_interval        = gui.state("day")
_selected_vars   = gui.state([])
_compute_storage = gui.state(False)
_snake_names     = gui.state(False)

_raw_df     = gui.state(None)   # DataFrame with API column names
_display_df = gui.state(None)   # Renamed copy shown in the Table tab
_status     = gui.state([])
_loading    = gui.state(False)

_active_tab = gui.state("⚙  Inputs")
_chart_var  = gui.state("")
_chart_b64  = gui.state("")

_INTV_SECS = {300: "5min", 3600: "hour", 86400: "day"}

_FALLBACK_STATIONS = sorted([
    "Ashland Bottoms", "Belleville", "Colby", "Dodge City", "Elkhart",
    "Emporia", "Garden City", "Hays", "Hill City", "Hutchinson 10SW",
    "Lakin", "Lawrence", "Liberal", "Manhattan", "Meade", "Oberlin",
    "Pratt", "Russell", "Salina", "Scott City", "Smith Center",
    "Topeka", "Tribune", "Wichita",
])


# ---------------------------------------------------------------------------
# Startup: load station list and availability in background
# ---------------------------------------------------------------------------

def _load_stations():
    try:
        names = core.get_stations()
        _stations_list.set(names)
        _station.set(names[0])
    except Exception as exc:
        _log(f"Could not load station list ({exc}); using built-in list.")
        _stations_list.set(_FALLBACK_STATIONS)
        _station.set(_FALLBACK_STATIONS[0])

    try:
        af = core.get_stations_active()
        lookup = {}
        for _, row in af.iterrows():
            intv_str = _INTV_SECS.get(int(row["OBS_INTERVAL"]))
            if intv_str:
                lookup[(row["STATION"], intv_str)] = (
                    pd.Timestamp(row["START"]),
                    pd.Timestamp(row["END"]),
                )
        _station_active.set(lookup)
    except Exception:
        pass  # advisory only; silently skip if unavailable

threading.Thread(target=_load_stations, daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str):
    _status.set(list(_status.value) + [msg])


def _parse_date(s: str) -> pd.Timestamp:
    s = s.strip()
    if len(s) == 16:
        s += ":00"
    return pd.to_datetime(s)


def _check_availability(stn: str, intv: str, start: str, end: str):
    lookup = _station_active.value
    if not lookup:
        return
    key = (stn, intv)
    if key not in lookup:
        _log(f"⚠  No availability record for {stn} at {intv} interval.")
        return
    avail_start, avail_end = lookup[key]
    req_start = _parse_date(start)
    req_end   = _parse_date(end)
    if req_end < avail_start or req_start > avail_end:
        _log(f"⚠  Requested period is entirely outside {stn}'s availability "
             f"({avail_start.date()} – {avail_end.date()}). All values will be NaN.")
    else:
        if req_start < avail_start:
            _log(f"⚠  Start date is before {stn}'s earliest record ({avail_start.date()}).")
        if req_end > avail_end:
            _log(f"⚠  End date is after {stn}'s latest record ({avail_end.date()}).")


def _build_display(df: pd.DataFrame) -> pd.DataFrame:
    return core.rename_columns(df.copy()) if _snake_names.value else df.copy()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _on_interval_change(new_intv: str):
    valid = core._VALID_FOR[new_intv]
    _selected_vars.set([v for v in _selected_vars.value if v in valid])
    _interval.set(new_intv)


def fetch():
    if _loading.value:
        return
    if not _selected_vars.value:
        _log("⚠  Select at least one variable.")
        return

    _loading.set(True)
    _status.set([f"Requesting data from {_station.value}…"])
    _raw_df.set(None)
    _display_df.set(None)
    _chart_var.set("")
    _chart_b64.set("")

    _check_availability(_station.value, _interval.value, _start_date.value, _end_date.value)

    def _run():
        try:
            intv      = _interval.value
            user_vars = list(_selected_vars.value)

            # Build the fetch list.
            # For VWC columns: fetch the VWC column itself AND its Ka/EC deps
            # so calibrate_vwc() has everything it needs.
            # For PRECIP: also fetch PRECIP2 for dual-gauge merging.
            fetch_vars    = []
            vwc_requested = []
            for v in user_vars:
                if v not in fetch_vars:
                    fetch_vars.append(v)
                if v in core._VWC_DEPS:
                    vwc_requested.append(v)
                    for dep in core._VWC_DEPS[v]:
                        if dep not in fetch_vars:
                            fetch_vars.append(dep)
                elif v == "PRECIP" and "PRECIP2" not in fetch_vars:
                    fetch_vars.append("PRECIP2")

            df = core.request_data(
                _station.value, _start_date.value, _end_date.value,
                intv, fetch_vars,
                verbose=False,
            )

            # Progress is visible via the status log; pipe core logger to _log
            # (core uses logging.INFO; we use verbose=False and log manually)
            _log(f"  {_station.value}  |  {_start_date.value[:10]} → {_end_date.value[:10]}")

            # Merge dual rain gauges: row-wise maximum, then drop PRECIP2
            if "PRECIP" in df.columns and "PRECIP2" in df.columns:
                df["PRECIP"] = df[["PRECIP", "PRECIP2"]].max(axis=1)
                df.drop(columns="PRECIP2", inplace=True)

            # Apply KSU CS655 calibration — overwrites VWC columns with calibrated values
            if vwc_requested:
                df = core.calibrate_vwc(df, vwc_requested)

            # Optional soil water storage (requires all four VWC depths)
            if _compute_storage.value and set(core._ALL_VWC).issubset(set(user_vars)):
                df = core.compute_soil_water_storage(df)

            # Column order: TIMESTAMP, user-requested vars, then Ka/EC deps
            # (fetched silently for calibration), then STORAGE_MM if computed.
            ka_ec_cols = [dep for v in vwc_requested
                          for dep in core._VWC_DEPS[v]
                          if dep in df.columns]
            ordered = (["TIMESTAMP"]
                       + [v for v in user_vars if v in df.columns]
                       + [c for c in ka_ec_cols if c not in user_vars]
                       + (["STORAGE_MM"] if "STORAGE_MM" in df.columns else []))
            df = df[ordered]

            _raw_df.set(df)
            _display_df.set(_build_display(df))
            _log(f"✓ Done — {len(df):,} rows · {len(df.columns)} columns")

            num_cols = [c for c in df.columns
                        if c != "TIMESTAMP" and pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                _chart_var.set(num_cols[0])
                _make_chart(df, num_cols[0])

            _active_tab.set("📋  Table")

        except Exception as exc:
            _log(f"✗ Error: {exc}")
        finally:
            _loading.set(False)

    threading.Thread(target=_run, daemon=True).start()


def _toggle_snake(v: bool):
    _snake_names.set(v)
    df = _raw_df.value
    if df is not None:
        _display_df.set(_build_display(df))


def _save_csv(path: str):
    df = _display_df.value
    if df is not None and path:
        df.to_csv(path, index=False)
        gui.notify(f"Saved: {os.path.basename(path)}", variant="success")


def _save_chart(path: str):
    b64 = _chart_b64.value
    if b64 and path:
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64))
        gui.notify(f"Figure saved: {os.path.basename(path)}", variant="success")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _make_chart(df: pd.DataFrame, var: str):
    if df is None or var not in df.columns or "TIMESTAMP" not in df.columns:
        _chart_b64.set("")
        return
    series = df[["TIMESTAMP", var]].dropna()
    if series.empty:
        _chart_b64.set("")
        return

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(series["TIMESTAMP"], series[var], linewidth=1.2, color="#2563eb", alpha=0.9)
    ax.set_ylabel(var, fontsize=10)
    ax.set_title(f"{_station.value}  ·  {var}", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="none", transparent=True)
    buf.seek(0)
    _chart_b64.set(base64.b64encode(buf.read()).decode())
    plt.close(fig)


def _update_chart():
    df = _raw_df.value
    if df is not None and _chart_var.value:
        _make_chart(df, _chart_var.value)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _col_chunks(items: list, n: int) -> list[list]:
    k, rem = divmod(len(items), n)
    out, i = [], 0
    for c in range(n):
        sz = k + (1 if c < rem else 0)
        out.append(items[i : i + sz])
        i += sz
    return out


def _var_checkbox(api: str, label: str):
    def _toggle(checked, name=api):
        cur = list(_selected_vars.value)
        if checked:
            if name not in cur:
                cur.append(name)
        else:
            cur = [x for x in cur if x != name]
            if name in core._ALL_VWC:
                _compute_storage.set(False)
        _selected_vars.set(cur)
    gui.checkbox(label, value=api in _selected_vars.value, on_change=_toggle, key=f"v_{api}")


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

TAB_LABELS = ["⚙  Inputs", "📋  Table", "📈  Chart"]


def _inputs_tab():
    intv = _interval.value

    with gui.col(gap=16, padding="20px"):

        with gui.row(gap=12, align="flex-start", wrap=True):
            with gui.col(gap=4, style="width:220px;flex-shrink:0"):
                gui.select(_stations_list.value or ["Loading…"], "Station",
                           value=_station, on_change=_station.set, key="stn")
            with gui.col(gap=4, style="flex:1;min-width:170px;max-width:210px"):
                gui.datetime_input("Start", value=_start_date,
                                   on_change=_start_date.set, key="sd")
            with gui.col(gap=4, style="flex:1;min-width:170px;max-width:210px"):
                gui.datetime_input("End", value=_end_date,
                                   on_change=_end_date.set, key="ed")
            with gui.col(gap=4, style="width:130px;flex-shrink:0"):
                gui.select(
                    [("day", "Daily"), ("hour", "Hourly"), ("5min", "5-minute")],
                    "Interval", value=_interval,
                    on_change=_on_interval_change, key="intv",
                )

        gui.divider()

        valid = _catalogue_for_interval(intv)
        atm   = [(a, lbl) for a, lbl, g, _ in valid if g == "Atmospheric"]
        soil  = [(a, lbl) for a, lbl, g, _ in valid if g == "Soil"]

        with gui.row(gap=28, align="flex-start"):
            for i, chunk in enumerate(_col_chunks(atm, 2)):
                with gui.col(gap=5, style="min-width:210px"):
                    if i == 0:
                        gui.text("Atmospheric", bold=True, size="sm",
                                 style="color:var(--primary);margin-bottom:4px")
                    else:
                        gui.spacer(h=19)
                    for api, lbl in chunk:
                        _var_checkbox(api, lbl)

            gui.html('<div style="width:1px;background:var(--border);'
                     'align-self:stretch;margin:0 4px"></div>')

            for i, chunk in enumerate(_col_chunks(soil, 2)):
                with gui.col(gap=5, style="min-width:210px"):
                    if i == 0:
                        gui.text("Soil", bold=True, size="sm",
                                 style="color:var(--primary);margin-bottom:4px")
                    else:
                        gui.spacer(h=19)
                    for api, lbl in chunk:
                        _var_checkbox(api, lbl)

        gui.divider()

        with gui.row(gap=24, align="center", wrap=True):
            all_vwc_selected = set(core._ALL_VWC).issubset(set(_selected_vars.value))
            if all_vwc_selected:
                gui.checkbox("Calculate soil water storage 0–50 cm (mm)",
                             value=_compute_storage,
                             on_change=_compute_storage.set, key="storage_opt")
            else:
                gui.text("Select all 4 VWC depths to enable soil water storage",
                         muted=True, size="sm", italic=True)
            gui.checkbox("Rename columns to snake_case",
                         value=_snake_names, on_change=_toggle_snake, key="snake_opt")

        n   = len(_selected_vars.value)
        lbl = ("⏳ Requesting…" if _loading.value
               else f"▶  Download  ({n} variable{'s' if n != 1 else ''})")
        gui.button(lbl, on_click=fetch,
                   disabled=_loading.value or n == 0,
                   size="lg", style="align-self:flex-start")

        if _status.value:
            gui.spacer(h=4)
            with gui.col(gap=2,
                         style="background:var(--surface-2);border-radius:var(--r-sm);"
                               "padding:10px 14px;max-height:160px;overflow-y:auto"):
                for line in _status.value:
                    color = ("var(--success)" if line.startswith("✓") else
                             "var(--danger)"  if line.startswith("✗") else
                             "var(--warning)" if line.startswith("⚠") else
                             "var(--text-2)")
                    gui.text(line, size="sm", mono=True, style=f"color:{color};white-space:pre")


def _table_tab():
    df = _display_df.value
    with gui.col(gap=12, padding="20px", fill=True):
        if df is None:
            with gui.col(align="center", justify="center", style="height:300px"):
                gui.text("No data yet — go to Inputs and click Download.", muted=True)
            return
        with gui.row(gap=10, justify="space-between", align="center"):
            with gui.row(gap=8):
                gui.badge(f"{len(df):,} rows",      variant="primary")
                gui.badge(f"{len(df.columns)} cols", variant="neutral")
                gui.badge(_station.value,            variant="success")
            gui.file_picker("💾  Save CSV", save=True,
                            file_types=("CSV Files (*.csv)",),
                            on_change=_save_csv, key="save_csv")
        with gui.scroll(max_height=520):
            gui.table(df)


def _chart_tab():
    df = _raw_df.value
    with gui.col(gap=14, padding="20px", fill=True):
        if df is None:
            with gui.col(align="center", justify="center", style="height:300px"):
                gui.text("No data yet — go to Inputs and click Download.", muted=True)
            return

        num_cols = [c for c in df.columns
                    if c != "TIMESTAMP" and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            gui.text("No numeric columns to plot.", muted=True)
            return

        cur_var = _chart_var.value if _chart_var.value in num_cols else num_cols[0]

        with gui.row(gap=12, align="flex-end"):
            gui.select(num_cols, "Variable", value=cur_var,
                       on_change=lambda v: (_chart_var.set(v), _update_chart()),
                       key="chart_sel", style="max-width:300px")
            gui.button("Refresh", on_click=_update_chart, variant="secondary", size="sm")
            gui.spacer(fill=True)
            gui.file_picker("💾  Save PNG", save=True,
                            file_types=("PNG Image (*.png)",),
                            disabled=not _chart_b64.value,
                            on_change=_save_chart, key="save_png")

        if _chart_b64.value:
            gui.html(f'<img src="data:image/png;base64,{_chart_b64.value}"'
                     f' style="width:100%;border-radius:var(--r-sm);display:block">')
        else:
            gui.text("Select a variable and click Refresh.", muted=True, size="sm")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@gui.app("Kansas Mesonet", width=1040, height=740, resizable=True)
def ui():
    gui.theme("light", primary="#2563eb")

    with gui.col(gap=0, fill=True, style="min-height:100vh"):
        with gui.row(gap=12, padding="12px 20px",
                     style="background:var(--surface);"
                           "border-bottom:1px solid var(--border);flex-shrink:0"):
            with gui.col(gap=1, justify="center"):
                gui.title("Kansas Mesonet", size="lg", style="line-height:1")
                gui.text("Environmental Monitoring Network", muted=True, size="xs")

        tab = gui.tabs(TAB_LABELS, value=_active_tab,
                       on_change=_active_tab.set, key="main_tabs")

        with gui.col(fill=True, scroll=True):
            if   tab == "⚙  Inputs": _inputs_tab()
            elif tab == "📋  Table":  _table_tab()
            elif tab == "📈  Chart":  _chart_tab()


def main():
    ui()


if __name__ == "__main__":
    main()
