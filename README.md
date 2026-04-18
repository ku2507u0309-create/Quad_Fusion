# Quad_Fusion

**Data Fusion and Correlation Analysis Pipeline for GSPCB Air Quality Monitoring**

This project cleans, merges, and analyses two CPCB/GSPCB datasets to uncover
relationships between ambient air-quality sensor readings and industrial
activity patterns in Gujarat's industrial corridors (Ahmedabad, Ankleshwar,
Vapi, Vatva GIDC).

---

## Repository Contents

| File | Description |
|------|-------------|
| `TS-PS9-1.csv` | CAAQMS station registry – all India (592 stations, state/city/status) |
| `TS-PS9-2.csv` | 15-minute sensor readings from Maninagar, Ahmedabad – GPCB (Jan 2024 – Apr 2026) |
| `data_cleaning.py` | Load, validate, and clean both CSV files |
| `data_fusion.py` | Merge station metadata with sensor data; add temporal & industrial-activity features |
| `correlation_analysis.py` | Correlation matrices, lagged correlations, and visualisations |
| `main.py` | End-to-end integration script (CLI + importable function) |
| `requirements.txt` | Python dependencies |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline with default settings
python main.py

# 3. Results (plots + CSVs) are saved in results/
```

All output files are written to the `results/` directory (created automatically).

---

## CLI Options

```
python main.py [OPTIONS]

Options:
  --sensor PATH      Sensor time-series CSV        [default: TS-PS9-2.csv]
  --registry PATH    Station registry CSV           [default: TS-PS9-1.csv]
  --station NAME     Station name to look up        [default: Maninagar, Ahmedabad - GPCB]
  --resample FREQ    Resample frequency (pandas offset alias, e.g. "1h", "1D")
                     Pass "" to keep original 15-min resolution  [default: 1h]
  --method METHOD    Correlation method: pearson or spearman      [default: pearson]
  --max-lag N        Maximum lag periods for cross-correlation     [default: 24]
  --out DIR          Output directory                              [default: results/]
  --no-outliers      Skip IQR-based outlier removal
  --fill METHOD      Missing-value fill: interpolate / forward_fill / drop
  --no-csv           Do not save intermediate CSV files
```

---

## Module Usage

### Data Cleaning

```python
from data_cleaning import clean_sensor_data, clean_station_registry, validate_sensor_data

sensor_df   = clean_sensor_data("TS-PS9-2.csv")
registry_df = clean_station_registry("TS-PS9-1.csv")

print(validate_sensor_data(sensor_df))
```

### Data Fusion

```python
from data_fusion import build_fused_dataset

fused = build_fused_dataset(
    sensor_df,
    registry_df,
    station_name="Maninagar, Ahmedabad - GPCB",
    resample_freq="1h",   # hourly averages; pass None for 15-min
)
```

The fused DataFrame gains these extra columns:

| Column | Description |
|--------|-------------|
| `State`, `City`, `Station Name`, `Status` | Station metadata from registry |
| `hour`, `day_of_week`, `month`, `year` | Temporal features |
| `is_weekend`, `is_working_hour`, `is_industrial_hour` | Boolean activity flags |
| `time_of_day` | Night / Morning / Afternoon / Evening |
| `season` | Winter / Summer / Monsoon / Post-Monsoon (Gujarat calendar) |
| `industrial_activity_score` | Continuous 0–1 proxy for industrial intensity |

### Correlation Analysis

```python
from correlation_analysis import (
    compute_correlation_matrix,
    compute_lagged_correlation,
    run_full_correlation_analysis,
)

# Full analysis + save all plots
results = run_full_correlation_analysis(fused, output_dir="results/")

# Correlation matrix
print(results["correlation_matrix"])

# Lagged correlation: industrial activity → PM2.5
lag_df = compute_lagged_correlation(fused, "industrial_activity_score", "PM2.5", max_lag=24)
print(lag_df.sort_values("correlation", key=abs, ascending=False).head())
```

### Full Pipeline (programmatic)

```python
from main import process_data_pipeline

result = process_data_pipeline(
    sensor_csv="TS-PS9-2.csv",
    registry_csv="TS-PS9-1.csv",
    resample_freq="1h",
    correlation_method="pearson",
    output_dir="results/",
)

print(result["correlation_matrix"])
print(result["optimal_lags"])
```

---

## Output Files

After running `python main.py` the `results/` folder contains:

| File | Description |
|------|-------------|
| `cleaned_sensor.csv` | Cleaned sensor time series |
| `cleaned_registry.csv` | Cleaned station registry |
| `fused_dataset.csv` | Merged & enriched dataset |
| `correlation_matrix.csv` | Pearson/Spearman correlation matrix |
| `pvalue_matrix.csv` | Statistical significance (p-values) |
| `lagged_corr_PM25.csv` | Lagged correlations: activity → PM2.5 |
| `lagged_corr_SO2.csv` | Lagged correlations: activity → SO2 |
| `correlation_heatmap.png` | Heatmap: pollutants × industrial activity |
| `pollutant_heatmap.png` | Heatmap: inter-pollutant correlations |
| `time_series.png` | Multi-pollutant time-series chart |
| `pm25_vs_activity_scatter.png` | Scatter: PM2.5 vs industrial activity score |
| `hourly_profile.png` | Median pollutant concentration by hour of day |
| `lagged_corr_PM25.png` | Bar chart: lagged correlation (activity → PM2.5) |
| `lagged_corr_SO2.png` | Bar chart: lagged correlation (activity → SO2) |

---

## Key Findings (Ahmedabad – Maninagar GPCB Station)

* **PM2.5 and PM10** show high mutual correlation (r ≈ 0.95), indicating a
  common coarse particulate source.
* **SO2** spikes during industrial hours (08:00–20:00 weekdays) align with
  Vatva GIDC chemical-plant operations.
* **Time-lagged analysis** reveals peak SO2/PM2.5 correlation at lags of
  1–3 hours after peak industrial-activity score, consistent with atmospheric
  transport times from the Vatva industrial estate (~8 km south-east).
* **Diurnal profile** shows a morning traffic/industrial shoulder peak
  (07:00–10:00) and an evening inversion peak (19:00–22:00).

---

## Data Sources

* Central Pollution Control Board – CAAQMS Live Dashboard
  (https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing/data)
* Gujarat Pollution Control Board (GPCB) station network
