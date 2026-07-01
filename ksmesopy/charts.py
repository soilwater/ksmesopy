"""
ksmesopy.charts
===============
Axes-level plot functions for Kansas Mesonet data.

Each function draws onto a Matplotlib Axes object supplied by the caller,
so panels compose freely inside any figure layout:

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    ms.plot_temperature(axes[0], df, ["TEMP2MAVG", "TEMP2MMIN", "TEMP2MMAX"])
    ms.plot_precip(axes[1], df, "PRECIP")
    ms.plot_humidity(axes[2], df, "RELHUM2MAVG")
    ms.plot_solar_radiation(axes[3], df, "SRAVG")
    plt.tight_layout()

Column names can be API names (TEMP2MAVG) or snake_case (t2m) — whichever
is present in the DataFrame after an optional rename_columns() call.

Functions
---------
plot_temperature(ax, df, variables, *, band, ylabel, legend)
plot_precip(ax, df, variable, *, ylabel, color)
plot_humidity(ax, df, variables, *, ylabel, legend)
plot_vpd(ax, df, variables, *, ylabel, legend)
plot_solar_radiation(ax, df, variables, *, ylabel, legend)
plot_wind(ax, df, speed, direction, *, ylabel, legend)
plot_vwc(ax, df, variables, *, ylabel, legend)
plot_et(ax, df, variables, *, bar, ylabel, legend)
"""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Default color cycle — distinct, colorblind-friendly
_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d"]

# Column label lookup: covers both API names and snake_case equivalents
_LABELS: dict[str, str] = {
    # Temperature
    "TEMP2MAVG":       "T 2 m avg",    "t2m":             "T 2 m avg",
    "TEMP2MMIN":       "T 2 m min",    "t2m_min":         "T 2 m min",
    "TEMP2MMAX":       "T 2 m max",    "t2m_max":         "T 2 m max",
    "TEMP10MAVG":      "T 10 m avg",   "t10m":            "T 10 m avg",
    "TEMP10MMIN":      "T 10 m min",   "t10m_min":        "T 10 m min",
    "TEMP10MMAX":      "T 10 m max",   "t10m_max":        "T 10 m max",
    "SOILTMP5AVG":     "Ts 5 cm",      "tsoil_5cm":       "Ts 5 cm",
    "SOILTMP10AVG":    "Ts 10 cm",     "tsoil_10cm":      "Ts 10 cm",
    "SOILTMP5AVG655":  "Ts 5 cm",      "tsoil_5cm_655":   "Ts 5 cm",
    "SOILTMP10AVG655": "Ts 10 cm",     "tsoil_10cm_655":  "Ts 10 cm",
    "SOILTMP20AVG655": "Ts 20 cm",     "tsoil_20cm_655":  "Ts 20 cm",
    "SOILTMP50AVG655": "Ts 50 cm",     "tsoil_50cm_655":  "Ts 50 cm",
    # Humidity / pressure
    "RELHUM2MAVG":     "RH avg",       "rh":              "RH avg",
    "RELHUM2MMIN":     "RH min",       "rh_min":          "RH min",
    "RELHUM2MMAX":     "RH max",       "rh_max":          "RH max",
    "VPDEFAVG":        "VPD",          "vpd":             "VPD",
    # Radiation
    "SRAVG":           "Rs",           "srad":            "Rs",
    # Wind
    "WSPD2MAVG":       "u 2 m",        "wspd":            "u 2 m",
    "WSPD2MMAX":       "u 2 m max",    "wspd_max":        "u 2 m max",
    "WSPD10MAVG":      "u 10 m",       "wspd10m":         "u 10 m",
    "WSPD10MMAX":      "u 10 m max",   "wspd10m_max":     "u 10 m max",
    "WDIR2M":          "Dir 2 m",      "wdir":            "Dir 2 m",
    "WDIR10M":         "Dir 10 m",     "wdir10m":         "Dir 10 m",
    # Precipitation
    "PRECIP":          "Precip",       "precip":          "Precip",
    # VWC
    "VWC5CM":          "VWC 5 cm",     "vwc_5cm":         "VWC 5 cm",
    "VWC10CM":         "VWC 10 cm",    "vwc_10cm":        "VWC 10 cm",
    "VWC20CM":         "VWC 20 cm",    "vwc_20cm":        "VWC 20 cm",
    "VWC50CM":         "VWC 50 cm",    "vwc_50cm":        "VWC 50 cm",
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
    """Apply shared grid and date-formatting defaults."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(axis="x", rotation=30)


def _get_ax(ax: plt.Axes | None) -> plt.Axes:
    """Return ax if supplied, otherwise create and return a new figure's axes."""
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
    Plot one or more temperature series on *ax*.

    Works for air temperature (TEMP2MAVG, TEMP2MMIN, TEMP2MMAX, …) and
    soil temperature (SOILTMP*AVG, SOILTMP*AVG655) columns. Call twice to
    overlay air and soil temperatures on the same axes.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and all requested columns.
    variables : str or list[str]
        Column names to plot.
    band : bool, default True
        When a min/max pair is detected alongside an avg column for the same
        sensor (e.g. TEMP2MMIN + TEMP2MMAX + TEMP2MAVG), draw a shaded band
        between min and max. The avg line is drawn on top.
    ylabel : str
        Y-axis label.
    legend : bool, default True
        Show a legend.

    Returns
    -------
    plt.Axes
        The axes that was drawn on.
    """
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    # Detect min/max/avg triplets for the band
    # Canonical pairing: a col ending in MIN and one in MAX share a prefix
    # Works for both API names (TEMP2MMIN/MAX) and snake names (t2m_min/max)
    plotted: set[str] = set()

    if band:
        pairs: list[tuple[str, str, str | None]] = []  # (min_col, max_col, avg_col)
        remaining = list(cols)

        def _strip_suffix(c: str, suffixes: list[str]) -> str | None:
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
                (x for x in remaining
                 if any(x == base + s for s in max_suffixes)),
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
                # No avg — label the band by its min column's base name
                ax.plot([], [], color=color, linewidth=1.4, label=_label(mn).replace(" min", ""))
                plotted.update({mn, mx})

    # Remaining columns plotted as plain lines
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
    """
    Plot precipitation as a bar chart on *ax*.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and the precipitation column.
    variable : str, default 'PRECIP'
        Column name (PRECIP or precip).
    ylabel : str
        Y-axis label.
    color : str
        Bar fill color.

    Returns
    -------
    plt.Axes
    """
    _require(df, [variable])
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    mask = df[variable].notna()
    # Bar width: infer from timestamp spacing, default to 0.8 days
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
    """
    Plot relative humidity on *ax*.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and all requested columns.
    variables : str or list[str], default 'RELHUM2MAVG'
        Column name(s) (RELHUM2MAVG, rh, RELHUM2MMIN, …).
    ylabel : str
        Y-axis label.
    legend : bool, default True
        Show a legend when more than one column is plotted.

    Returns
    -------
    plt.Axes
    """
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
    """
    Plot vapor pressure deficit on *ax*.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and all requested columns.
    variables : str or list[str], default 'VPDEFAVG'
        Column name(s). Accepts measured VPD (VPDEFAVG / vpd) or values
        computed by vapor_pressure_deficit().
    ylabel : str
        Y-axis label.
    legend : bool, default True
        Show a legend when more than one column is plotted.

    Returns
    -------
    plt.Axes
    """
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    for i, col in enumerate(cols):
        color = _COLORS[i % len(_COLORS)]
        mask = df[col].notna()
        ax.fill_between(t[mask], df[col][mask], alpha=0.15, color=color, linewidth=0)
        ax.plot(t[mask], df[col][mask], linewidth=1.4, color=color, label=_label(col))

    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if legend and len(cols) > 1:
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

    Can plot observed (SRAVG / srad) and extraterrestrial radiation (Ra)
    together on the same axes — pass both column names and Ra will appear
    as a dashed envelope above the observed signal.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and all requested columns.
    variables : str or list[str], default 'SRAVG'
        Column name(s). Ra columns are detected by name and drawn dashed.
    ylabel : str
        Y-axis label.
    legend : bool, default True
        Show a legend when more than one column is plotted.

    Returns
    -------
    plt.Axes
    """
    cols = _as_list(variables)
    _require(df, cols)
    t = _ts(df)
    ax = _get_ax(ax)
    _setup_ax(ax)

    # Heuristic: columns named Ra (case-insensitive) or containing "extra"
    # are drawn as dashed envelopes rather than filled areas
    def _is_ra(col: str) -> bool:
        c = col.lower()
        return c in ("ra",) or "extra" in c

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
    Plot wind speed as a line and, optionally, wind direction as a scatter.

    Direction is overlaid on a twin y-axis (0–360°) so the two signals share
    the same x-axis without distorting the speed scale. Direction markers are
    small and semi-transparent to avoid cluttering the speed line.

    Parameters
    ----------
    ax : plt.Axes
        Target axes (used for wind speed).
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and the speed column.
    speed : str, default 'WSPD2MAVG'
        Wind speed column (WSPD2MAVG, wspd, WSPD10MAVG, …).
    direction : str or None, default None
        Wind direction column (WDIR2M, wdir, …). If None, direction is
        not plotted.
    ylabel : str
        Y-axis label for the speed axis.
    legend : bool, default True
        Show a legend.

    Returns
    -------
    plt.Axes
        The speed axes (not the twin direction axes).
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

    Defaults to all four standard VWC depths if *variables* is None and
    those columns are present in the DataFrame. When multiple depths are
    plotted, a sequential colormap (blue → brown) maps shallow to deep,
    giving an intuitive depth gradient.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and all requested columns.
    variables : str or list[str] or None
        Column name(s). Defaults to all VWC*CM / vwc_* columns found
        in the DataFrame.
    ylabel : str
        Y-axis label.
    legend : bool, default True
        Show a legend.

    Returns
    -------
    plt.Axes
    """
    # Default: discover VWC columns present in the DataFrame
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

    # Depth colormap: shallow = blue, deep = brown
    cmap = cm.get_cmap("YlOrBr", len(cols) + 1)
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

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    df : pd.DataFrame
        Must contain TIMESTAMP (or timestamp) and all requested columns.
        Typically populated from reference_et_penman_monteith() or
        reference_et_hargreaves().
    variables : str or list[str]
        Column name(s) holding ETo values.
    bar : bool, default False
        Draw as a bar chart (appropriate for daily totals) instead of a line.
    ylabel : str
        Y-axis label.
    legend : bool, default True
        Show a legend when more than one column is plotted.

    Returns
    -------
    plt.Axes
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
