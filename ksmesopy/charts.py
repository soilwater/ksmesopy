"""
ksmesopy.charts
===============
Axes-level plot functions for Kansas Mesonet data. Each function draws onto a
Matplotlib Axes supplied by the caller, so panels compose freely:

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    ms.plot_temperature(axes[0], df, ["TEMP2MAVG", "TEMP2MMIN", "TEMP2MMAX"])
    ms.plot_precip(axes[1], df, "PRECIP")
    ms.plot_humidity(axes[2], df, "RELHUM2MAVG")
    ms.plot_solar_radiation(axes[3], df, "SRAVG")

Column names may be API names (TEMP2MAVG) or snake_case (tair_2m_avg) —
whichever is present after an optional rename_columns() call.
"""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Distinct, colorblind-friendly cycle
_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d"]

# Display labels covering both API and snake_case column names
_LABELS: dict[str, str] = {
    "TEMP2MAVG":       "T 2 m avg",   "tair_2m_avg":     "T 2 m avg",
    "TEMP2MMIN":       "T 2 m min",   "tair_2m_min":     "T 2 m min",
    "TEMP2MMAX":       "T 2 m max",   "tair_2m_max":     "T 2 m max",
    "TEMP10MAVG":      "T 10 m avg",  "tair_10m_avg":    "T 10 m avg",
    "TEMP10MMIN":      "T 10 m min",  "tair_10m_min":    "T 10 m min",
    "TEMP10MMAX":      "T 10 m max",  "tair_10m_max":    "T 10 m max",
    "SOILTMP5AVG":     "Ts 5 cm",     "tsoil_5cm":       "Ts 5 cm",
    "SOILTMP10AVG":    "Ts 10 cm",    "tsoil_10cm":      "Ts 10 cm",
    "SOILTMP5AVG655":  "Ts 5 cm",     "tsoil_5cm_655":   "Ts 5 cm",
    "SOILTMP10AVG655": "Ts 10 cm",    "tsoil_10cm_655":  "Ts 10 cm",
    "SOILTMP20AVG655": "Ts 20 cm",    "tsoil_20cm_655":  "Ts 20 cm",
    "SOILTMP50AVG655": "Ts 50 cm",    "tsoil_50cm_655":  "Ts 50 cm",
    "PRESSUREAVG":     "P",           "pressure_avg":    "P",
    "RELHUM2MAVG":     "RH avg",      "rh_2m_avg":       "RH avg",
    "RELHUM2MMIN":     "RH min",      "rh_2m_min":       "RH min",
    "RELHUM2MMAX":     "RH max",      "rh_2m_max":       "RH max",
    "VPDEFAVG":        "VPD",         "vpd_avg":         "VPD",
    "VPD_calc":        "VPD (calc)",
    "SRAVG":           "Rs",          "srad":            "Rs",
    "Rs_MJ":           "Rs",
    "Ra_Wm2":          "Ra",          "Ra_MJ":           "Ra",
    "WSPD2MAVG":       "u 2 m",       "wspd_2m_avg":     "u 2 m",
    "WSPD2MMAX":       "u 2 m max",   "wspd_2m_max":     "u 2 m max",
    "WSPD10MAVG":      "u 10 m",      "wspd_10m_avg":    "u 10 m",
    "WSPD10MMAX":      "u 10 m max",  "wspd_10m_max":    "u 10 m max",
    "WDIR2M":          "Dir 2 m",     "wdir_2m":         "Dir 2 m",
    "WDIR10M":         "Dir 10 m",    "wdir_10m":        "Dir 10 m",
    "PRECIP":          "Precip",      "precip":          "Precip",
    "VWC5CM":          "VWC 5 cm",    "vwc_5cm":         "VWC 5 cm",
    "VWC10CM":         "VWC 10 cm",   "vwc_10cm":        "VWC 10 cm",
    "VWC20CM":         "VWC 20 cm",   "vwc_20cm":        "VWC 20 cm",
    "VWC50CM":         "VWC 50 cm",   "vwc_50cm":        "VWC 50 cm",
}


def _label(col: str) -> str:
    return _LABELS.get(col, col)


def _ts(df: pd.DataFrame) -> pd.Series:
    """Return the timestamp column regardless of its name case."""
    for name in ("TIMESTAMP", "timestamp"):
        if name in df.columns:
            return df[name]
    raise ValueError("DataFrame has no TIMESTAMP or timestamp column.")


def _setup_ax(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(axis="x", rotation=30)


def _get_ax(ax: plt.Axes | None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    return ax


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in DataFrame: {missing}")


def _as_list(v: Union[str, list[str]]) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_temperature(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    variables: Union[str, list[str]] = None,
    *,
    band:    bool = True,
    ylabel:  str = "Temperature (°C)",
    legend:  bool = True,
) -> plt.Axes:
    """
    Plot one or more temperature series on *ax* (air or soil).

    When band=True and a min/max pair is present alongside an avg column for
    the same sensor, draw a shaded band between min and max with the avg line
    on top. Call twice to overlay air and soil on the same axes.
    """
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    plotted: set[str] = set()

    if band:
        pairs: list[tuple[str, str, str | None]] = []  # (min_col, max_col, avg_col)
        remaining = list(cols)

        def _strip_suffix(c: str, suffixes: tuple[str, ...]) -> str | None:
            for s in suffixes:
                if c.endswith(s):
                    return c[: -len(s)]
            return None

        min_suffixes = ("MIN", "_min")
        max_suffixes = ("MAX", "_max")

        for c in list(remaining):
            base = _strip_suffix(c, min_suffixes)
            if base is None:
                continue
            max_col = next(
                (x for x in remaining if any(x == base + s for s in max_suffixes)),
                None,
            )
            if max_col:
                avg_col = next(
                    (x for x in remaining
                     if x not in (c, max_col) and x.startswith(base)
                     and not any(x.endswith(s) for s in min_suffixes + max_suffixes)),
                    None,
                )
                pairs.append((c, max_col, avg_col))

        for i, (mn, mx, avg) in enumerate(pairs):
            color = _COLORS[i % len(_COLORS)]
            mask = df[mn].notna() & df[mx].notna()
            ax.fill_between(t[mask], df[mn][mask], df[mx][mask],
                            alpha=0.18, color=color, linewidth=0)
            ax.plot(t[mask], df[mn][mask], linewidth=0.6, color=color, alpha=0.5)
            ax.plot(t[mask], df[mx][mask], linewidth=0.6, color=color, alpha=0.5)
            if avg and avg in df.columns:
                mask_avg = df[avg].notna()
                ax.plot(t[mask_avg], df[avg][mask_avg],
                        linewidth=1.4, color=color, label=_label(avg))
                plotted.update({mn, mx, avg})
            else:
                ax.plot([], [], color=color, linewidth=1.4,
                        label=_label(mn).replace(" min", ""))
                plotted.update({mn, mx})

    remaining_cols = [c for c in cols if c not in plotted]
    for i, col in enumerate(remaining_cols):
        color = _COLORS[(len(plotted) // 3 + i) % len(_COLORS)]
        mask = df[col].notna()
        ax.plot(t[mask], df[col][mask], linewidth=1.4, color=color, label=_label(col))

    ax.set_ylabel(ylabel)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    if legend:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    return ax


def plot_precip(
    ax:       plt.Axes | None = None,
    df:       pd.DataFrame = None,
    variable: str = "PRECIP",
    *,
    ylabel: str = "Precipitation (mm)",
    color:  str = "#2563eb",
) -> plt.Axes:
    """Plot precipitation as a bar chart on *ax*."""
    _require(df, [variable])
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    mask = df[variable].notna()
    # Bar width from timestamp spacing, default 0.8 days
    if mask.sum() > 1:
        dt = (t[mask].iloc[1] - t[mask].iloc[0]).total_seconds() / 86400
        width = dt * 0.8
    else:
        width = 0.8
    ax.bar(t[mask], df[variable][mask], width=width, color=color, alpha=0.75, align="center")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    return ax


def plot_humidity(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    variables: Union[str, list[str]] = "RELHUM2MAVG",
    *,
    ylabel: str = "Relative humidity (%)",
    legend: bool = True,
) -> plt.Axes:
    """Plot relative humidity on *ax*."""
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    for i, col in enumerate(cols):
        color = _COLORS[i % len(_COLORS)]
        mask = df[col].notna()
        ax.plot(t[mask], df[col][mask], linewidth=1.4, color=color, label=_label(col))

    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel)
    if legend and len(cols) > 1:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    return ax


def plot_vpd(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    variables: Union[str, list[str]] = "VPDEFAVG",
    *,
    ylabel: str = "VPD (kPa)",
    legend: bool = True,
) -> plt.Axes:
    """Plot vapor pressure deficit on *ax* (measured or computed)."""
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    multi = len(cols) > 1
    for i, col in enumerate(cols):
        color = _COLORS[i % len(_COLORS)]
        mask = df[col].notna()
        if not multi:
            ax.fill_between(t[mask], df[col][mask], alpha=0.15, color=color, linewidth=0)
        ax.plot(t[mask], df[col][mask], linewidth=1.4, color=color, label=_label(col))

    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if legend and multi:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    return ax


def plot_solar_radiation(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    variables: Union[str, list[str]] = "SRAVG",
    *,
    ylabel: str = "Solar radiation (W m⁻²)",
    legend: bool = True,
) -> plt.Axes:
    """
    Plot solar radiation on *ax* as a filled area.

    Observed radiation (SRAVG / srad) is filled; extraterrestrial radiation
    (Ra columns, detected by name) is drawn as a dashed envelope.
    """
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    def _is_ra(col: str) -> bool:
        c = col.lower()
        return c == "ra" or c.startswith("ra_") or "extra" in c

    for i, col in enumerate(cols):
        color = _COLORS[i % len(_COLORS)]
        mask = df[col].notna()
        if _is_ra(col):
            ax.plot(t[mask], df[col][mask], linewidth=1.2, color=color,
                    linestyle="--", alpha=0.7, label=_label(col))
        else:
            ax.fill_between(t[mask], df[col][mask], alpha=0.25, color=color, linewidth=0)
            ax.plot(t[mask], df[col][mask], linewidth=1.4, color=color, label=_label(col))

    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if legend and len(cols) > 1:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    return ax


def plot_wind(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    speed:     str = "WSPD2MAVG",
    direction: str | None = None,
    *,
    ylabel: str = "Wind speed (m s⁻¹)",
    legend: bool = True,
) -> plt.Axes:
    """
    Plot wind speed as a line and, optionally, direction as a scatter.

    Direction is overlaid on a twin y-axis (0–360°) so both signals share the
    x-axis without distorting the speed scale. Returns the speed axes.
    """
    _require(df, [speed] + ([direction] if direction else []))
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    mask_spd = df[speed].notna()
    ax.plot(t[mask_spd], df[speed][mask_spd],
            linewidth=1.4, color=_COLORS[0], label=_label(speed))
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)

    if direction:
        ax_dir = ax.twinx()
        mask_dir = df[direction].notna()
        ax_dir.scatter(t[mask_dir], df[direction][mask_dir],
                       s=6, color=_COLORS[1], alpha=0.45, label=_label(direction),
                       zorder=3)
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.set_yticklabels(["N", "E", "S", "W", "N"], fontsize=7)
        ax_dir.set_ylabel("")
        if legend:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dir.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, loc="upper left", framealpha=0.7)
    elif legend:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    return ax


def plot_vwc(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    variables: Union[str, list[str]] = None,
    *,
    ylabel: str = "VWC (m³ m⁻³)",
    legend: bool = True,
) -> plt.Axes:
    """
    Plot volumetric water content on *ax*.

    Defaults to all four standard VWC depths present in the DataFrame. When
    multiple depths are plotted, a sequential colormap maps shallow to deep.
    """
    _VWC_ORDER = [
        "VWC5CM", "VWC10CM", "VWC20CM", "VWC50CM",
        "vwc_5cm", "vwc_10cm", "vwc_20cm", "vwc_50cm",
    ]
    if variables is None:
        cols = [c for c in _VWC_ORDER if c in df.columns]
        if not cols:
            raise ValueError("No VWC columns found in DataFrame.")
    else:
        cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    # Depth colormap: shallow -> deep
    cmap = plt.get_cmap("YlOrBr", len(cols) + 1)
    colors = [cmap(i + 1) for i in range(len(cols))]

    for col, color in zip(cols, colors):
        mask = df[col].notna()
        ax.plot(t[mask], df[col][mask], linewidth=1.4, color=color, label=_label(col))

    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    return ax


def plot_et(
    ax:        plt.Axes | None = None,
    df:        pd.DataFrame = None,
    variables: Union[str, list[str]] = None,
    *,
    bar:    bool = False,
    ylabel: str = "ET (mm day⁻¹)",
    legend: bool = True,
) -> plt.Axes:
    """
    Plot reference evapotranspiration on *ax*.

    Set bar=True to draw daily totals as bars instead of a line.
    """
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    for i, col in enumerate(cols):
        color = _COLORS[i % len(_COLORS)]
        mask = df[col].notna()
        if bar:
            if mask.sum() > 1:
                dt = (t[mask].iloc[1] - t[mask].iloc[0]).total_seconds() / 86400
                width = dt * 0.8
            else:
                width = 0.8
            ax.bar(t[mask], df[col][mask], width=width,
                   color=color, alpha=0.7, label=_label(col), align="center")
        else:
            ax.plot(t[mask], df[col][mask],
                    linewidth=1.4, color=color, label=_label(col))

    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if legend and len(cols) > 1:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    return ax
