# Data Description

This document summarizes the source datasets used in the project and describes the cleaned data format used for each station.

## Overview

The project integrates three original data sources:
- Traffic
- Weather
- Air quality

These were cleaned, aligned by date and station, and merged into per-station CSV files in `data/data_clean/` (one CSV per station). Each cleaned CSV contains daily observations for the station.

---

## Original datasets

### Traffic
- Typical contents: vehicle counts aggregated by hour/day for road sensors or intersections.
- Spatial scope: traffic sensors covering roads near air quality monitoring stations.
- Temporal resolution: original source may be hourly; aggregated to daily counts for the cleaned files.
- Typical variables: `timestamp`, `location_id`, `vehicle_count` (or `Traffic_Count` after aggregation).
- Notes: counts may include all vehicles or be stratified by vehicle class in raw data. Processing aggregated counts to daily totals or averages was performed in cleaning.

### Weather
- Typical contents: meteorological observations such as temperature, precipitation, wind speed/gust and wind direction, etc.
- Spatial scope: nearby weather stations (may be airport or local stations) matched to each air quality station.
- Temporal resolution: commonly hourly; aggregated or interpolated to daily values in cleaned files.
- Typical variables used here: `Temp` (°C), `Precip` (mm), `Wind_Gust` (speed), `Wind_Dir` (degrees).
- Notes: check `data/weather/meteorological_metadata.yml` for original station metadata and units.

### Air quality
- Typical contents: pollutant measurements (e.g., NO₂, PM₂.₅) recorded at monitoring stations.
- Spatial scope: fixed air-monitoring stations (one per cleaned CSV station row), identified by `Station`.
- Temporal resolution: many networks record hourly values; this project uses daily aggregated values (daily mean NO₂ represented as `NO2_Mean`).
- Typical variables used here: `timestamp`/`Date`, `Station`, pollutant concentrations (e.g., `NO2_Mean`).
- Notes: daily aggregation method (mean) and handling of missing hours is documented in the cleaning script.

---

## Cleaned data (per station)

The cleaned CSVs are stored in `data/data_clean/`. Each file contains daily rows for a single station. Below is the column reference (Data name | Type | Description) describing the fields present in any cleaned station file.

| Data name       | Type            | Description |
|---|---|---|
| `Station`       | string          | Station name / monitoring site identifier (e.g., "Ottawa Downtown"). One value repeated per file (one file per station). |
| `Date`          | date            | Observation date (ISO format YYYY-MM-DD). Daily aggregation timestamp. |
| `NO2_Mean`      | float           | Daily mean NO₂ concentration at the station (units: ppb). Aggregated from hourly measurements. |
| `Traffic_Count` | integer/float   | Daily aggregated traffic count for the location matched to the station (units: vehicle counts). If original data were hourly, counts are summed or daily averages documented in cleaning script. |
| `Temp`          | float           | Daily mean temperature (°C). Missing values are interpolated in cleaning. |
| `Precip`        | float           | Daily total precipitation (mm). Missing values filled with 0 where appropriate. |
| `Wind_Gust`     | float           | Wind gust speed (units as in raw weather data — see metadata). Used to compute wind components. |
| `Wind_Dir`      | float           | Wind direction in degrees (meteorological degrees, 0–360). Used to compute wind components. |

**Notes & recommendations**
- Units: most commonly used units are indicated above (°C for temperature, mm for precipitation, ppb for NO₂). For formal reporting, confirm units by reviewing the original raw files or `meteorological_metadata.yml` in `data/weather/`.
- Aggregation: daily aggregation for `NO2_Mean` and `Traffic_Count` was used in the cleaned files. If you need hourly analysis, consult the original raw datasets in `data/traffic/` and `data/weather/`.
- Missing values: the cleaning pipeline interpolates temperature and wind values, fills precipitation missing values with 0 when appropriate, and drops rows that remain missing for required feature columns. See `src/data_clean.py` for the exact cleaning logic.
