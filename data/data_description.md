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
- Contents: vehicle counts aggregated by day for road sensors at intersections.
- Variables: `timestamp`, `location_id`, `traffic_count` (after aggregation).
- Link to dataset: https://www.airqualityontario.com/history/index.php

### Weather
- Contents: meteorological observations such as temperature, precipitation, wind speed/gust and wind direction, etc.
- Variables: `Temp` (°C), `Precip` (mm), `Wind_Gust` (speed), `Wind_Dir` (degrees).
- Link to dataset: https://climate.weather.gc.ca

### Air quality
- Contents: pollutant measurements (e.g., NO₂, PM₂.₅) recorded at monitoring stations.
- Variables: `timestamp`/`Date`, `Station`, pollutant concentrations (e.g., `NO2_Mean`).
- Link to dataset: https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2022018-eng.htm


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

**Notes
- Units: most commonly used units are indicated above (°C for temperature, mm for precipitation, ppb for NO₂). For formal reporting, confirm units by reviewing the original raw files or `meteorological_metadata.yml` in `data/weather/`.
- Aggregation: daily aggregation for `NO2_Mean` and `Traffic_Count` was used in the cleaned files.
- Missing values: the cleaning pipeline interpolates temperature and wind values, fills precipitation missing values with 0 when appropriate, and drops rows that remain missing for required feature columns. See `src/data_clean.py` for the exact cleaning logic.
