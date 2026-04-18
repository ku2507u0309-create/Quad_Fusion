"""
main.py
-------
End-to-end integration script for the GSPCB Air Quality
Data Fusion and Correlation Analysis Pipeline.

Usage
-----
    python main.py

    # Or with custom paths:
    python main.py --sensor TS-PS9-2.csv --registry TS-PS9-1.csv --out results/

Run ``python main.py --help`` for all options.
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def process_data_pipeline(
    sensor_csv: str = "TS-PS9-2.csv",
    registry_csv: str = "TS-PS9-1.csv",
    station_name: str = "Maninagar, Ahmedabad - GPCB",
    resample_freq: str = "1h",
    correlation_method: str = "pearson",
    max_lag: int = 24,
    output_dir: str = "results",
    remove_outliers: bool = True,
    fill_method: str = "interpolate",
    save_csv: bool = True,
) -> dict:
    """
    Run the complete data fusion and correlation analysis pipeline.

    Parameters
    ----------
    sensor_csv : str
        Path to the sensor time-series CSV (TS-PS9-2.csv).
    registry_csv : str
        Path to the station-registry CSV (TS-PS9-1.csv).
    station_name : str
        Station to look up in the registry for metadata enrichment.
    resample_freq : str
        Pandas offset alias for resampling (e.g. ``"1h"``, ``"1D"``).
        Pass ``""`` to skip resampling.
    correlation_method : str
        ``"pearson"`` or ``"spearman"``.
    max_lag : int
        Maximum time-lag (in resampled periods) for cross-correlation analysis.
    output_dir : str
        Directory where results (CSV files, PNG plots) are saved.
    remove_outliers : bool
        Whether to apply IQR-based outlier removal during cleaning.
    fill_method : str
        Missing-value fill strategy: ``"interpolate"``, ``"forward_fill"``,
        or ``"drop"``.
    save_csv : bool
        If ``True``, save intermediate and final DataFrames as CSV files.

    Returns
    -------
    dict
        Keys:
        ``"cleaned_sensor"``, ``"cleaned_registry"``, ``"fused"``,
        ``"correlation_matrix"``, ``"pvalue_matrix"``,
        ``"lagged_correlations"``, ``"optimal_lags"``.
    """
    from data_cleaning import clean_sensor_data, clean_station_registry, validate_sensor_data
    from data_fusion import build_fused_dataset
    from correlation_analysis import run_full_correlation_analysis

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – Data Cleaning
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: DATA CLEANING")
    print("=" * 60)

    print(f"\n→ Cleaning sensor data from: {sensor_csv}")
    sensor_df = clean_sensor_data(
        sensor_csv,
        remove_outliers=remove_outliers,
        fill_method=fill_method,
    )
    validation = validate_sensor_data(sensor_df)
    print(f"\n  Validation summary:")
    print(f"    Rows      : {validation['rows']:,}")
    print(f"    Date range: {validation['date_range'][0]} → {validation['date_range'][1]}")
    print(f"    Missing % : {validation['missing_pct']}")

    print(f"\n→ Cleaning station registry from: {registry_csv}")
    registry_df = clean_station_registry(registry_csv)
    print(f"  Registry shape: {registry_df.shape}")

    if save_csv:
        path = os.path.join(output_dir, "cleaned_sensor.csv")
        sensor_df.to_csv(path, index=False)
        print(f"  Saved cleaned sensor data → {path}")

        path = os.path.join(output_dir, "cleaned_registry.csv")
        registry_df.to_csv(path, index=False)
        print(f"  Saved cleaned registry   → {path}")

    # ------------------------------------------------------------------
    # Step 2 – Data Fusion
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: DATA FUSION")
    print("=" * 60)

    freq = resample_freq if resample_freq else None
    fused_df = build_fused_dataset(
        sensor_df,
        registry_df,
        station_name=station_name,
        resample_freq=freq,
    )
    print(f"\n  Fused dataset columns: {fused_df.columns.tolist()}")

    if save_csv:
        path = os.path.join(output_dir, "fused_dataset.csv")
        fused_df.to_csv(path, index=False)
        print(f"  Saved fused dataset      → {path}")

    # ------------------------------------------------------------------
    # Step 3 – Correlation Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: CORRELATION ANALYSIS")
    print("=" * 60)

    analysis = run_full_correlation_analysis(
        fused_df,
        method=correlation_method,
        max_lag=max_lag,
        output_dir=output_dir,
    )

    print("\n  Correlation matrix (PM2.5, SO2, CO vs activity):")
    highlight_cols = [c for c in ["PM2.5", "SO2", "CO", "industrial_activity_score"]
                      if c in analysis["correlation_matrix"].columns]
    if highlight_cols:
        print(analysis["correlation_matrix"].loc[highlight_cols, highlight_cols].round(3).to_string())

    print("\n  Optimal lags (industrial activity → pollutant):")
    for pol, lag_info in analysis["optimal_lags"].items():
        print(f"    {pol:6s}: lag={lag_info['lag']:+3d} periods, "
              f"r={lag_info['correlation']:.3f}, p={lag_info['p_value']:.4f}")

    if save_csv:
        path = os.path.join(output_dir, "correlation_matrix.csv")
        analysis["correlation_matrix"].to_csv(path)
        print(f"\n  Saved correlation matrix → {path}")

        path = os.path.join(output_dir, "pvalue_matrix.csv")
        analysis["pvalue_matrix"].to_csv(path)
        print(f"  Saved p-value matrix     → {path}")

        for pol, lag_df in analysis["lagged_correlations"].items():
            path = os.path.join(output_dir, f"lagged_corr_{pol.replace('.', '')}.csv")
            lag_df.to_csv(path, index=False)
            print(f"  Saved lagged correlation → {path}")

    print("\n" + "=" * 60)
    print(f"Pipeline complete.  All outputs saved in: {os.path.abspath(output_dir)}/")
    print("=" * 60 + "\n")

    return {
        "cleaned_sensor": sensor_df,
        "cleaned_registry": registry_df,
        "fused": fused_df,
        "correlation_matrix": analysis["correlation_matrix"],
        "pvalue_matrix": analysis["pvalue_matrix"],
        "lagged_correlations": analysis["lagged_correlations"],
        "optimal_lags": analysis["optimal_lags"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="GSPCB Air Quality Data Fusion & Correlation Analysis Pipeline"
    )
    parser.add_argument("--sensor", default="TS-PS9-2.csv",
                        help="Path to sensor time-series CSV  [default: TS-PS9-2.csv]")
    parser.add_argument("--registry", default="TS-PS9-1.csv",
                        help="Path to station registry CSV  [default: TS-PS9-1.csv]")
    parser.add_argument("--station", default="Maninagar, Ahmedabad - GPCB",
                        help="Station name to look up in registry")
    parser.add_argument("--resample", default="1h",
                        help="Resample frequency (pandas offset alias, e.g. '1h', '1D'). "
                             "Pass empty string to skip.  [default: 1h]")
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman"],
                        help="Correlation method  [default: pearson]")
    parser.add_argument("--max-lag", type=int, default=24,
                        help="Maximum lag periods for cross-correlation  [default: 24]")
    parser.add_argument("--out", default="results",
                        help="Output directory for plots and CSV files  [default: results/]")
    parser.add_argument("--no-outliers", action="store_true",
                        help="Skip IQR-based outlier removal")
    parser.add_argument("--fill", default="interpolate",
                        choices=["interpolate", "forward_fill", "drop"],
                        help="Missing-value fill method  [default: interpolate]")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not save intermediate CSV files")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    for path in [args.sensor, args.registry]:
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    process_data_pipeline(
        sensor_csv=args.sensor,
        registry_csv=args.registry,
        station_name=args.station,
        resample_freq=args.resample,
        correlation_method=args.method,
        max_lag=args.max_lag,
        output_dir=args.out,
        remove_outliers=not args.no_outliers,
        fill_method=args.fill,
        save_csv=not args.no_csv,
    )
