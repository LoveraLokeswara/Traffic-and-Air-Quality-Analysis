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
- Variables: `Date`, `Traffic_Source`, `Traffic_Count`.
- Link to dataset: https://www.airqualityontario.com/history/index.php

### Weather
- Contents: meteorological observations such as temperature, precipitation, wind speed/gust and wind direction, etc.
- Variables: `Temp` (°C), `Precip` (mm), `Wind_Gust` (speed), `Wind_Dir` (degrees).
- Link to dataset: https://climate.weather.gc.ca

### Air quality
- Contents: pollutant measurements (e.g., NO₂) recorded at monitoring stations.
- Variables: `Date`,`Hour`, `Station`, pollutant concentrations (e.g., `NO2_Mean`).
- Link to dataset: https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2022018-eng.htm


---

## Cleaned data (per station)

The cleaned CSVs are stored in `data/data_clean/`. Each file contains daily rows for a single station. Below is the column reference (Data name | Type | Description) describing the fields present in any cleaned station file.

| Data name       | Type            | Description |
|---|---|---|
| `Station`       | string          | Station name / monitoring site identifier (e.g., "Ottawa Downtown"). One value repeated per file (one file per station). |
| `Date`          | date            | Observation date (ISO format YYYY-MM-DD). Daily aggregation timestamp. |
| `NO2_Mean`      | float           | Daily mean NO₂ concentration at the station (ppb: parts per billion). Aggregated from hourly measurements. |
| `Traffic_Count` | integer   | Daily traffic count for the location matched to the station (units: vehicle counts) |
| `Temp`          | float           | Daily mean temperature (°C). |
| `Precip`        | float           | Daily total precipitation (mm). |
| `Wind_Gust`     | float           | Wind gust speed (km).|
| `Wind_Dir`      | float           | Wind direction in degrees (meteorological degrees, 0–360). |

**Notes
- Units: most commonly used units are indicated above (°C for temperature, mm for precipitation, ppb for NO₂)
- Aggregation: daily aggregation for `NO2_Mean` and `Traffic_Count` was used in the cleaned files.
- Missing values: missing values were filled using Scikit-learn’s IterativeImputer
