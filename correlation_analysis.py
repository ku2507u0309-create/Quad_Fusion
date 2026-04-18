"""
correlation_analysis.py
-----------------------
Functions to compute and visualise correlations between air-quality sensor
readings and industrial-activity indicators in the fused GSPCB dataset.

Capabilities
------------
* Pearson and Spearman correlation matrices
* Time-lagged (cross-correlation) analysis between any two columns
* Correlation heat-maps  (seaborn)
* Time-series overlay plots
* Scatter plots with regression lines
* Statistical significance (p-value) table
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib
matplotlib.use("Agg")          # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Default column sets
# ---------------------------------------------------------------------------

POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "SO2", "CO"]
ACTIVITY_COLS = ["industrial_activity_score", "is_industrial_hour", "hour",
                 "day_of_week", "is_weekend"]


# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute a correlation matrix for the selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : sequence of str, optional
        Subset of columns to include.  Defaults to all numeric columns.
    method : str
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    subset = df[list(columns)].copy()
    # Cast boolean columns to int so scipy handles them correctly
    for col in subset.columns:
        if subset[col].dtype == bool:
            subset[col] = subset[col].astype(int)
    return subset.corr(method=method)


def compute_pvalue_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute a matrix of p-values for pairwise correlations.

    Parameters
    ----------
    df : pd.DataFrame
    columns : sequence of str, optional
    method : str  ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    pd.DataFrame
        p-value matrix (same shape as the correlation matrix).
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    subset = df[list(columns)].dropna()
    for col in subset.columns:
        if subset[col].dtype == bool:
            subset[col] = subset[col].astype(int)

    n = len(columns)
    pvals = np.ones((n, n))
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr

    for i in range(n):
        for j in range(i + 1, n):
            try:
                _, p = corr_fn(subset.iloc[:, i], subset.iloc[:, j])
            except Exception:
                p = np.nan
            pvals[i, j] = pvals[j, i] = p

    return pd.DataFrame(pvals, index=columns, columns=columns)


# ---------------------------------------------------------------------------
# Time-lagged correlations
# ---------------------------------------------------------------------------

def compute_lagged_correlation(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    max_lag: int = 24,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute the correlation between *col_x* and lagged versions of *col_y*.

    A positive lag means *col_y* is shifted **forward** in time (i.e. *col_x*
    leads *col_y* by *lag* periods).  This is useful for detecting delayed
    pollution effects after an industrial-activity event.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *col_x* and *col_y*.
    col_x : str
        The reference (leading) variable, e.g. ``"industrial_activity_score"``.
    col_y : str
        The lagged (following) variable, e.g. ``"PM2.5"``.
    max_lag : int
        Maximum number of periods to shift (both positive and negative).
    method : str
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``lag``, ``correlation``, ``p_value``.
    """
    x = df[col_x].dropna()
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr

    records = []
    for lag in range(-max_lag, max_lag + 1):
        y_shifted = df[col_y].shift(lag)
        valid = pd.concat([x, y_shifted], axis=1).dropna()
        if len(valid) < 10:
            records.append({"lag": lag, "correlation": np.nan, "p_value": np.nan})
            continue
        try:
            r, p = corr_fn(valid.iloc[:, 0], valid.iloc[:, 1])
        except Exception:
            r, p = np.nan, np.nan
        records.append({"lag": lag, "correlation": r, "p_value": p})

    return pd.DataFrame(records)


def find_optimal_lag(lagged_df: pd.DataFrame) -> dict:
    """
    Return the lag with the highest absolute correlation.

    Parameters
    ----------
    lagged_df : pd.DataFrame
        Output of :func:`compute_lagged_correlation`.

    Returns
    -------
    dict
        Keys: ``lag``, ``correlation``, ``p_value``.
    """
    idx = lagged_df["correlation"].abs().idxmax()
    row = lagged_df.loc[idx]
    return {"lag": int(row["lag"]), "correlation": row["correlation"], "p_value": row["p_value"]}


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap",
    pvalue_matrix: pd.DataFrame | None = None,
    significance_level: float = 0.05,
    output_path: str | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Draw a colour-coded correlation heat-map.

    If *pvalue_matrix* is provided, cells that are **not** statistically
    significant (p ≥ *significance_level*) are masked with a cross.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix.
    title : str
    pvalue_matrix : pd.DataFrame, optional
    significance_level : float
    output_path : str, optional
        If given, save the figure to this path.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    if pvalue_matrix is not None:
        sig_mask = pvalue_matrix >= significance_level
        for i in range(len(corr_matrix)):
            for j in range(i):  # lower triangle only
                if sig_mask.iloc[i, j]:
                    ax.text(
                        j + 0.5, i + 0.5, "✗",
                        ha="center", va="center",
                        fontsize=14, color="grey", alpha=0.6,
                    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_correlation_heatmap] Saved → {output_path}")
    return fig


def plot_lagged_correlation(
    lagged_df: pd.DataFrame,
    col_x: str,
    col_y: str,
    output_path: str | None = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Bar chart of cross-correlation at each lag.

    Parameters
    ----------
    lagged_df : pd.DataFrame
        Output of :func:`compute_lagged_correlation`.
    col_x, col_y : str
        Column names (for the plot title / axis labels).
    output_path : str, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["steelblue" if v >= 0 else "tomato" for v in lagged_df["correlation"]]
    ax.bar(lagged_df["lag"], lagged_df["correlation"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (periods)", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title(f"Lagged Correlation: {col_x} → {col_y}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate the peak lag
    best = find_optimal_lag(lagged_df)
    ax.axvline(best["lag"], color="red", linestyle="--", linewidth=1.2,
               label=f"Best lag = {best['lag']} (r={best['correlation']:.2f})")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_lagged_correlation] Saved → {output_path}")
    return fig


def plot_time_series(
    df: pd.DataFrame,
    columns: Sequence[str],
    timestamp_col: str = "From Date",
    title: str = "Pollutant Time Series",
    output_path: str | None = None,
    figsize: tuple = (14, 6),
    date_sample: int | None = 500,
) -> plt.Figure:
    """
    Multi-line time-series plot for the selected columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : sequence of str
        Numeric columns to plot.
    timestamp_col : str
    title : str
    output_path : str, optional
    figsize : tuple
    date_sample : int, optional
        Down-sample to this many evenly spaced rows for performance.  Pass
        ``None`` to plot all rows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df.sort_values(timestamp_col)
    if date_sample and len(plot_df) > date_sample:
        step = len(plot_df) // date_sample
        plot_df = plot_df.iloc[::step]

    fig, ax = plt.subplots(figsize=figsize)
    for col in columns:
        if col in plot_df.columns:
            ax.plot(plot_df[timestamp_col], plot_df[col], label=col, linewidth=0.9, alpha=0.85)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Concentration (µg/m³ or ppb)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_time_series] Saved → {output_path}")
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str | None = None,
    output_path: str | None = None,
    figsize: tuple = (8, 6),
    sample_n: int = 2000,
) -> plt.Figure:
    """
    Scatter plot of *x_col* vs *y_col* with an OLS regression line.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
    color_col : str, optional
        Column used to colour the points (must be numeric or boolean).
    title : str, optional
    output_path : str, optional
    figsize : tuple
    sample_n : int
        Maximum number of points to plot (random sample).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=figsize)

    if color_col:
        scatter = ax.scatter(
            plot_df[x_col], plot_df[y_col],
            c=plot_df[color_col], cmap="plasma", alpha=0.5, s=15,
        )
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.4, s=15, color="steelblue")

    # Regression line
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, r, p, _ = stats.linregress(plot_df[x_col], plot_df[y_col])
    x_range = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 200)
    ax.plot(x_range, slope * x_range + intercept, color="red", linewidth=1.5,
            label=f"OLS  r={r:.2f}  p={p:.3f}")

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(title or f"{x_col} vs {y_col}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_scatter] Saved → {output_path}")
    return fig


def plot_hourly_pollution_profile(
    df: pd.DataFrame,
    pollutant_cols: Sequence[str] | None = None,
    hour_col: str = "hour",
    output_path: str | None = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Box plot showing the distribution of each pollutant for each hour of day.

    Useful for identifying diurnal patterns that coincide with industrial
    activity hours.

    Parameters
    ----------
    df : pd.DataFrame
    pollutant_cols : sequence of str, optional
    hour_col : str
    output_path : str, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pollutant_cols is None:
        pollutant_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    n = len(pollutant_cols)
    if n == 0:
        raise ValueError("No pollutant columns found in DataFrame.")

    cols_per_row = min(3, n)
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(figsize[0], figsize[1] * rows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for i, col in enumerate(pollutant_cols):
        hourly = df.groupby(hour_col)[col].median().reset_index()
        axes[i].bar(hourly[hour_col], hourly[col], color="teal", alpha=0.75)
        axes[i].set_title(f"Median {col} by Hour", fontsize=11)
        axes[i].set_xlabel("Hour of Day")
        axes[i].set_ylabel(col)
        axes[i].axvspan(8, 20, alpha=0.08, color="red",
                        label="Industrial hours (08–20)")
        axes[i].legend(fontsize=8)
        axes[i].grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Diurnal Pollution Profile (Median)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_hourly_pollution_profile] Saved → {output_path}")
    return fig


# ---------------------------------------------------------------------------
# High-level summary function
# ---------------------------------------------------------------------------

def run_full_correlation_analysis(
    df: pd.DataFrame,
    pollutant_cols: Sequence[str] | None = None,
    activity_col: str = "industrial_activity_score",
    method: str = "pearson",
    max_lag: int = 24,
    output_dir: str = ".",
) -> dict:
    """
    Run the complete correlation analysis and save all visualisations.

    Parameters
    ----------
    df : pd.DataFrame
        Fused dataset (output of :func:`data_fusion.build_fused_dataset`).
    pollutant_cols : sequence of str, optional
        Pollutant columns to include.  Defaults to available columns from
        :data:`POLLUTANT_COLS`.
    activity_col : str
        Industrial-activity column to use in lagged-correlation analysis.
    method : str  ``"pearson"`` or ``"spearman"``.
    max_lag : int
        Maximum lag (in periods) for cross-correlation.
    output_dir : str
        Directory where PNG figures are written.

    Returns
    -------
    dict
        Keys:
        ``"correlation_matrix"``, ``"pvalue_matrix"``,
        ``"lagged_correlations"`` (dict keyed by pollutant column name),
        ``"optimal_lags"`` (dict keyed by pollutant column name).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if pollutant_cols is None:
        pollutant_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    analysis_cols = [c for c in pollutant_cols + [activity_col] if c in df.columns]

    # 1. Correlation matrix
    corr = compute_correlation_matrix(df, columns=analysis_cols, method=method)
    pvals = compute_pvalue_matrix(df, columns=analysis_cols, method=method)

    plot_correlation_heatmap(
        corr, pvalue_matrix=pvals,
        title=f"Pollutant × Industrial Activity Correlation ({method.title()})",
        output_path=os.path.join(output_dir, "correlation_heatmap.png"),
    )

    # 2. Pollutant-only heatmap
    poll_corr = compute_correlation_matrix(df, columns=pollutant_cols, method=method)
    plot_correlation_heatmap(
        poll_corr,
        title=f"Inter-Pollutant Correlation ({method.title()})",
        output_path=os.path.join(output_dir, "pollutant_heatmap.png"),
    )

    # 3. Time series
    plot_time_series(
        df,
        columns=pollutant_cols,
        output_path=os.path.join(output_dir, "time_series.png"),
    )

    # 4. PM2.5 vs industrial activity scatter
    if "PM2.5" in df.columns and activity_col in df.columns:
        plot_scatter(
            df, x_col=activity_col, y_col="PM2.5",
            title="PM2.5 vs Industrial Activity Score",
            output_path=os.path.join(output_dir, "pm25_vs_activity_scatter.png"),
        )

    # 5. Hourly profile
    if "hour" in df.columns:
        plot_hourly_pollution_profile(
            df, pollutant_cols=pollutant_cols,
            output_path=os.path.join(output_dir, "hourly_profile.png"),
        )

    # 6. Lagged correlations for PM2.5 and SO2
    lagged = {}
    optimal = {}
    for pol in [c for c in ["PM2.5", "SO2"] if c in df.columns and activity_col in df.columns]:
        lag_df = compute_lagged_correlation(df, activity_col, pol, max_lag=max_lag, method=method)
        lagged[pol] = lag_df
        optimal[pol] = find_optimal_lag(lag_df)
        plot_lagged_correlation(
            lag_df, col_x=activity_col, col_y=pol,
            output_path=os.path.join(output_dir, f"lagged_corr_{pol.replace('.', '')}.png"),
        )

    plt.close("all")

    return {
        "correlation_matrix": corr,
        "pvalue_matrix": pvals,
        "lagged_correlations": lagged,
        "optimal_lags": optimal,
    }
