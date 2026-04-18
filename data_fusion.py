"""
data_fusion.py
--------------
Functions to merge the cleaned station-registry dataset with the sensor
time-series dataset and enrich the result with derived features useful for
industrial-activity correlation analysis.

Design overview
---------------
* The sensor file (TS-PS9-2.csv) records measurements from a single station
  ("Maninagar, Ahmedabad - GPCB").
* The station registry (TS-PS9-1.csv) provides metadata (State, City, Status)
  for every CAAQMS station in India, including the one above.
* **Data fusion** joins the station metadata onto the sensor rows so that
  downstream analyses can be filtered or grouped by State, City, or Status.
* In the absence of a separate industrial-schedule file, a derived boolean
  column *is_industrial_hour* serves as a proxy for industrial activity:
  weekday hours 08:00–20:00 are treated as typical operational windows for
  the manufacturing corridor (Vapi–Ankleshwar–Valsad / Ahmedabad GIDC areas).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_station_name(station_full: str) -> str:
    """
    Normalise a station-name string for fuzzy matching.

    Example: ``"Maninagar, Ahmedabad - GPCB"`` → ``"maninagar ahmedabad gpcb"``
    """
    return (
        station_full.lower()
        .replace(",", " ")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


# ---------------------------------------------------------------------------
# Core fusion function
# ---------------------------------------------------------------------------

def fuse_sensor_with_registry(
    sensor_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    station_name: str = "Maninagar, Ahmedabad - GPCB",
) -> pd.DataFrame:
    """
    Attach station-registry metadata to the sensor time series.

    The function looks up *station_name* in the registry (case-insensitive,
    punctuation-tolerant) and adds its metadata columns (State, City,
    Station Name, Status) as constant columns on every sensor row.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Cleaned sensor time series (output of
        :func:`data_cleaning.clean_sensor_data`).
    registry_df : pd.DataFrame
        Cleaned station registry (output of
        :func:`data_cleaning.clean_station_registry`).
    station_name : str
        Full station name to look up in the registry.  Must match one of the
        values in the *Station Name* column (case/punctuation tolerant).

    Returns
    -------
    pd.DataFrame
        Sensor data with four extra columns:
        ``State``, ``City``, ``Station Name``, ``Status``.
    """
    norm_target = _extract_station_name(station_name)
    registry_df = registry_df.copy()
    registry_df["_norm"] = registry_df["Station Name"].apply(_extract_station_name)

    match = registry_df[registry_df["_norm"] == norm_target]
    if match.empty:
        # Partial match fallback – use the first substring match
        match = registry_df[registry_df["_norm"].str.contains(norm_target.split()[0], na=False)]

    if match.empty:
        print(
            f"[fuse_sensor_with_registry] WARNING: '{station_name}' not found in "
            "registry. Metadata columns will be 'Unknown'."
        )
        meta = {"State": "Unknown", "City": "Unknown", "Station Name": station_name, "Status": "Unknown"}
    else:
        row = match.iloc[0]
        meta = {
            "State": row["State"],
            "City": row["City"],
            "Station Name": row["Station Name"],
            "Status": row["Status"],
        }

    fused = sensor_df.copy()
    for key, val in meta.items():
        fused[key] = val

    registry_df.drop(columns=["_norm"], inplace=True)
    return fused


# ---------------------------------------------------------------------------
# Temporal feature engineering
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame, timestamp_col: str = "From Date") -> pd.DataFrame:
    """
    Derive time-based features from the measurement timestamp.

    Added columns
    -------------
    * ``hour``           – hour of day (0–23)
    * ``day_of_week``    – Monday=0 … Sunday=6
    * ``day_name``       – e.g. "Monday"
    * ``month``          – month number (1–12)
    * ``year``           – year
    * ``is_weekend``     – True on Saturday/Sunday
    * ``is_working_hour``– True between 09:00 and 18:00 on weekdays
    * ``is_industrial_hour`` – True between 08:00 and 20:00 on weekdays
                               (proxy for typical GIDC factory operation windows)
    * ``time_of_day``    – categorical: "Night" / "Morning" / "Afternoon" / "Evening"
    * ``season``         – categorical: "Winter" / "Summer" / "Monsoon" / "Post-Monsoon"

    Parameters
    ----------
    df : pd.DataFrame
    timestamp_col : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    ts = df[timestamp_col]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_name"] = ts.dt.day_name()
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year

    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_working_hour"] = (~df["is_weekend"]) & df["hour"].between(9, 17)
    df["is_industrial_hour"] = (~df["is_weekend"]) & df["hour"].between(8, 19)

    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )

    # Gujarat seasons (approximate): map each month explicitly to avoid
    # duplicate-label issues with pd.cut
    _season_map = {
        1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
        12: "Winter",
    }
    df["season"] = df["month"].map(_season_map).astype("category")

    return df


def add_industrial_activity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a continuous *industrial_activity_score* (0–1) as a proxy for the
    intensity of industrial operations at any given time.

    The score combines three factors derived from the timestamp:

    1. **Day-of-week weight** – weekdays score 1.0, Saturday 0.5, Sunday 0.0.
    2. **Hour-of-day weight** – Gaussian-like peak centred at 13:00 (shift
       midpoint), falling to 0 outside 05:00–23:00.
    3. **Seasonal weight** – Summer (high energy demand) = 1.0,
       Monsoon (partial shutdowns) = 0.7, Winter/Post-Monsoon = 0.85.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *hour*, *day_of_week*, *month* columns (added by
        :func:`add_temporal_features`).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an extra ``industrial_activity_score`` column.
    """
    import numpy as np

    df = df.copy()

    # Day-of-week weight (Monday=0 … Friday=4 are full working days;
    # Saturday=5 is half-day; Sunday=6 is closed)
    dow_map = {
        0: 1.0,  # Monday
        1: 1.0,  # Tuesday
        2: 1.0,  # Wednesday
        3: 1.0,  # Thursday
        4: 1.0,  # Friday
        5: 0.5,  # Saturday (half-day)
        6: 0.0,  # Sunday (closed)
    }
    dow_weight = df["day_of_week"].map(dow_map)

    # Hour weight – bell curve centred at 13:00, σ≈5 h
    hour_weight = np.exp(-0.5 * ((df["hour"] - 13) / 5) ** 2)

    # Seasonal weight
    season_map = {
        1: 0.85, 2: 0.85,          # Winter
        3: 1.0,  4: 1.0,  5: 1.0,  # Summer
        6: 0.70, 7: 0.70,          # Monsoon
        8: 0.70, 9: 0.70,          # Monsoon
        10: 0.85, 11: 0.85,        # Post-Monsoon
        12: 0.85,                  # Winter
    }
    season_weight = df["month"].map(season_map)

    df["industrial_activity_score"] = (dow_weight * hour_weight * season_weight).round(4)
    return df


# ---------------------------------------------------------------------------
# Resampling / aggregation
# ---------------------------------------------------------------------------

def resample_sensor_data(
    df: pd.DataFrame,
    freq: str = "1h",
    timestamp_col: str = "From Date",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Resample the 15-minute sensor data to a coarser time resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Sensor data with a datetime *timestamp_col*.
    freq : str
        Pandas offset alias, e.g. ``"1h"`` (hourly), ``"1D"`` (daily).
    timestamp_col : str
        Name of the datetime column used as the time index.
    agg : str
        Aggregation function: ``"mean"``, ``"median"``, ``"max"``, ``"min"``.

    Returns
    -------
    pd.DataFrame
        Resampled data with *timestamp_col* as the index, reset afterward.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    result = (
        df.set_index(timestamp_col)[numeric_cols]
        .resample(freq)
        .agg(agg)
        .reset_index()
    )
    result.rename(columns={result.columns[0]: timestamp_col}, inplace=True)
    return result


# ---------------------------------------------------------------------------
# Time-window merge (for multiple stations / schedules)
# ---------------------------------------------------------------------------

def merge_on_nearest_timestamp(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_ts: str = "From Date",
    right_ts: str = "From Date",
    tolerance: str = "15min",
    direction: str = "nearest",
) -> pd.DataFrame:
    """
    Merge two time-series DataFrames on their nearest matching timestamps.

    Useful when the two sources have different sampling frequencies or slight
    clock offsets.

    Parameters
    ----------
    left : pd.DataFrame
        Primary dataset.
    right : pd.DataFrame
        Secondary dataset to merge onto *left*.
    left_ts, right_ts : str
        Timestamp column names in each DataFrame.
    tolerance : str
        Maximum allowed time difference (pandas Timedelta string).
    direction : str
        ``"backward"``, ``"forward"``, or ``"nearest"``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame; rows in *left* with no match within *tolerance* will
        have NaN for columns from *right*.
    """
    left = left.sort_values(left_ts).copy()
    right = right.sort_values(right_ts).copy()

    if right_ts != left_ts:
        right = right.rename(columns={right_ts: left_ts})

    merged = pd.merge_asof(
        left,
        right,
        on=left_ts,
        tolerance=pd.Timedelta(tolerance),
        direction=direction,
        suffixes=("", "_right"),
    )
    return merged


# ---------------------------------------------------------------------------
# Full fusion pipeline
# ---------------------------------------------------------------------------

def build_fused_dataset(
    sensor_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    station_name: str = "Maninagar, Ahmedabad - GPCB",
    resample_freq: str | None = "1h",
) -> pd.DataFrame:
    """
    Run the complete data-fusion pipeline.

    Steps
    -----
    1. Fuse station metadata onto sensor rows.
    2. Add temporal features.
    3. Add industrial-activity score.
    4. Optionally resample to a lower frequency.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Output of :func:`data_cleaning.clean_sensor_data`.
    registry_df : pd.DataFrame
        Output of :func:`data_cleaning.clean_station_registry`.
    station_name : str
        Station to look up in the registry.
    resample_freq : str or None
        If provided, resample after adding features.  Pass ``None`` to keep
        the original 15-minute resolution.

    Returns
    -------
    pd.DataFrame
        Fully fused and enriched dataset ready for correlation analysis.
    """
    fused = fuse_sensor_with_registry(sensor_df, registry_df, station_name)
    fused = add_temporal_features(fused)
    fused = add_industrial_activity_score(fused)

    if resample_freq:
        # Resample only numeric columns; re-attach categorical after
        fused = resample_sensor_data(fused, freq=resample_freq)
        # Re-add temporal features on the resampled index
        fused = add_temporal_features(fused)
        fused = add_industrial_activity_score(fused)

    print(f"[build_fused_dataset] Fused dataset shape: {fused.shape}")
    return fused
