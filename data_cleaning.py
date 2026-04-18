"""
data_cleaning.py
----------------
Functions to load, clean, and preprocess the two GSPCB CSV files:
  - TS-PS9-1.csv : CAAQMS station registry (state / city / station name / status)
  - TS-PS9-2.csv : Continuous ambient air-quality sensor time series
                   (PM2.5, PM10, NO, NO2, NOx, SO2, CO at 15-minute intervals)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TS-PS9-1.csv  –  Station registry
# ---------------------------------------------------------------------------

def load_station_registry(filepath: str) -> pd.DataFrame:
    """
    Load the CAAQMS station registry CSV.

    The file has 3 metadata header rows before the actual column names, and may
    contain a UTF-8 BOM character.  State and City columns use forward-fill
    because consecutive rows for the same state/city are left blank.

    Parameters
    ----------
    filepath : str
        Path to TS-PS9-1.csv (or equivalent).

    Returns
    -------
    pd.DataFrame
        Raw station registry with columns:
        S.No., State, City, Station Name, Status
    """
    df = pd.read_csv(filepath, skiprows=3, encoding="utf-8-sig")
    # Forward-fill State and City (merged cells represented as blanks)
    df["State"] = df["State"].ffill()
    df["City"] = df["City"].ffill()
    return df


def clean_station_registry(filepath: str) -> pd.DataFrame:
    """
    Load and fully clean the station registry.

    Cleaning steps
    --------------
    1. Load with :func:`load_station_registry`.
    2. Strip leading/trailing whitespace from all string columns.
    3. Standardise *Status* column to title-case.
    4. Drop rows where *Station Name* is missing.
    5. Reset index.

    Parameters
    ----------
    filepath : str
        Path to TS-PS9-1.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned station registry.
    """
    df = load_station_registry(filepath)

    str_cols = ["State", "City", "Station Name", "Status"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    df["Status"] = df["Status"].str.title()

    # Drop rows without a station name (NaN or empty string / any case variant)
    invalid = {"", "nan", "none", "n/a", "na"}
    df = df[df["Station Name"].notna() & ~df["Station Name"].str.lower().isin(invalid)]
    df = df.drop_duplicates(subset=["Station Name"])
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# TS-PS9-2.csv  –  Sensor time series
# ---------------------------------------------------------------------------

SENSOR_NUMERIC_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "SO2", "CO"]


def load_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Load the CPCB continuous air-quality sensor CSV.

    The file contains 16 metadata / header rows before the data starts.  The
    string ``"None"`` is used in place of missing numeric values in some
    exports; pandas' ``na_values`` argument handles this transparently.

    Parameters
    ----------
    filepath : str
        Path to TS-PS9-2.csv (or equivalent).

    Returns
    -------
    pd.DataFrame
        Raw sensor data with columns:
        From Date, To Date, PM2.5, PM10, NO, NO2, NOx, SO2, CO
    """
    df = pd.read_csv(
        filepath,
        skiprows=16,
        encoding="utf-8-sig",
        na_values=["None", "none", "NA", "N/A", "--", ""],
    )
    return df


def parse_sensor_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse *From Date* and *To Date* columns to ``datetime64``.

    Expected format: ``DD-MM-YYYY HH:MM``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with string date columns (output of :func:`load_sensor_data`).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with datetime columns; rows where parsing fails are
        dropped.
    """
    df = df.copy()
    df["From Date"] = pd.to_datetime(df["From Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["To Date"] = pd.to_datetime(df["To Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    before = len(df)
    df = df.dropna(subset=["From Date"])
    dropped = before - len(df)
    if dropped:
        print(f"[parse_sensor_timestamps] Dropped {dropped} rows with unparseable timestamps.")
    return df


def remove_duplicates(df: pd.DataFrame, timestamp_col: str = "From Date") -> pd.DataFrame:
    """
    Remove duplicate rows based on the timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
    timestamp_col : str
        Column to use as the uniqueness key.

    Returns
    -------
    pd.DataFrame
    """
    before = len(df)
    df = df.drop_duplicates(subset=[timestamp_col])
    dropped = before - len(df)
    if dropped:
        print(f"[remove_duplicates] Dropped {dropped} duplicate rows on '{timestamp_col}'.")
    return df.sort_values(timestamp_col).reset_index(drop=True)


def remove_outliers_iqr(df: pd.DataFrame,
                        columns: list | None = None,
                        factor: float = 3.0) -> pd.DataFrame:
    """
    Replace values outside ``[Q1 - factor*IQR, Q3 + factor*IQR]`` with NaN.

    A generous *factor* of 3.0 is used by default so that genuine pollution
    spikes are retained rather than discarded.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list, optional
        Numeric columns to check.  Defaults to :data:`SENSOR_NUMERIC_COLS`.
    factor : float
        IQR multiplier for the fence.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in SENSOR_NUMERIC_COLS if c in df.columns]
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        count = mask.sum()
        if count:
            print(f"[remove_outliers_iqr] '{col}': flagged {count} outlier(s) as NaN.")
        df.loc[mask, col] = np.nan
    return df


def fill_missing_values(df: pd.DataFrame,
                        columns: list | None = None,
                        method: str = "interpolate") -> pd.DataFrame:
    """
    Fill missing numeric values in sensor data.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list, optional
        Columns to fill.  Defaults to :data:`SENSOR_NUMERIC_COLS`.
    method : str
        ``"interpolate"`` – linear time-based interpolation (recommended for
        time series).
        ``"forward_fill"`` – propagate last valid observation forward.
        ``"drop"`` – drop rows that still have any NaN after no filling.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in SENSOR_NUMERIC_COLS if c in df.columns]

    if method == "interpolate":
        df[columns] = df[columns].interpolate(method="linear", limit_direction="both")
    elif method == "forward_fill":
        df[columns] = df[columns].ffill().bfill()
    elif method == "drop":
        df = df.dropna(subset=columns)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'interpolate', 'forward_fill', or 'drop'.")

    return df


def clean_sensor_data(filepath: str,
                      remove_outliers: bool = True,
                      fill_method: str = "interpolate") -> pd.DataFrame:
    """
    Full cleaning pipeline for the sensor time-series CSV.

    Steps
    -----
    1. Load raw data.
    2. Parse timestamps.
    3. Remove duplicate timestamps.
    4. Optionally flag outliers as NaN (IQR method).
    5. Fill missing values.

    Parameters
    ----------
    filepath : str
        Path to TS-PS9-2.csv.
    remove_outliers : bool
        If ``True``, run IQR-based outlier detection.
    fill_method : str
        Passed to :func:`fill_missing_values`.

    Returns
    -------
    pd.DataFrame
        Cleaned sensor data indexed from 0, sorted by *From Date*.
    """
    df = load_sensor_data(filepath)
    df = parse_sensor_timestamps(df)
    df = remove_duplicates(df, timestamp_col="From Date")
    if remove_outliers:
        df = remove_outliers_iqr(df)
    df = fill_missing_values(df, method=fill_method)
    print(f"[clean_sensor_data] Final shape: {df.shape}")
    return df


def validate_sensor_data(df: pd.DataFrame) -> dict:
    """
    Run basic validation checks on cleaned sensor data.

    Returns
    -------
    dict
        Summary with keys: ``rows``, ``date_range``, ``missing_pct``,
        ``negative_values``.
    """
    num_cols = [c for c in SENSOR_NUMERIC_COLS if c in df.columns]
    missing_pct = {c: round(df[c].isna().mean() * 100, 2) for c in num_cols}
    negative = {c: int((df[c] < 0).sum()) for c in num_cols}
    return {
        "rows": len(df),
        "date_range": (df["From Date"].min(), df["From Date"].max()),
        "missing_pct": missing_pct,
        "negative_values": negative,
    }
