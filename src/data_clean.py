"""
Data Cleaning Pipeline
======================
Reads station configurations from stations.txt, loads raw Air Quality, Traffic,
and Weather data, merges them on a common (Station, Date) index, and performs
advanced Multiple Imputation using scikit-learn's IterativeImputer.

"""

# ====================================================================
# IMPORTS & LIBRARY SETUP
# ====================================================================
import glob  # Used for pattern matching to find files (e.g., *.csv)
import re  # Regular expressions for string pattern matching (e.g., extracting station names)
from pathlib import Path  # Object-oriented file path handling across platforms

import numpy as np  # For numerical operations and NaN handling
import pandas as pd  # DataFrame manipulation and CSV I/O

# Enable IterativeImputer (still experimental in scikit-learn)
# This is required before importing IterativeImputer from sklearn.impute
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
# IterativeImputer performs MICE (Multivariate Imputation by Chained Equations)
# to fill missing values by modeling each feature with missing values as a function
# of other features in an iterative manner
from sklearn.impute import IterativeImputer

# ====================================================================
# CONFIGURATION & FILE PATHS
# ====================================================================
# Define file paths using pathlib for cross-platform compatibility
SCRIPT_DIR = Path(__file__).resolve().parent  # Current script's directory: data/
PROJECT_ROOT = SCRIPT_DIR.parent  # Parent directory: repo root
DATA_DIR = PROJECT_ROOT / 'data'  # Data directory: data/
DATA_CLEAN_DIR = DATA_DIR / 'data_clean'  # Directory for cleaned data
STATIONS_FILE = DATA_CLEAN_DIR / "stations.txt"  # Config file in data/data_clean


# Define timeframe for data filtering
START_DATE = pd.Timestamp("2022-02-02")  # Beginning of the data collection period
END_DATE = pd.Timestamp("2024-12-31")  # End of the data collection period


# ===================================================================
# 1. PARSE STATIONS CONFIGURATION
# ===================================================================
def parse_stations(filepath: Path) -> list[dict]:
    """Parse ``stations.txt`` into a list of linked station configurations.

    CLEANING PURPOSE:
    - Reads a structured configuration file that maps three independently-managed
      datasets (Air Quality, Traffic, Weather) to common station identifiers
    - Creates a unified station configuration that allows merging data from
      different sources by matching them to the same logical station
    
    STATION ENTRY FORMAT:
    - Stations can be specified on a single line (comma-separated)
    - Or on separate lines (one station per line)
    - The parser flexibly handles both formats

    The file contains named sections (``air_quality:``, ``traffic:``,
    ``weather:``) each followed by one or more station entries.
    Entries at the same *positional index* across sections are 
    treated as belonging to the same logical station group.

    Returns
    -------
    list[dict]
        Each dict has keys ``aq_name``, ``traffic_id``, ``weather_file`` that
        link Air Quality station names, Traffic WKT coordinates, and Weather
        file stems to a single logical station.
    """
    sections: dict[str, list[str]] = {}  # Dictionary to store each section's entries
    current_section: str | None = None  # Track which section is being parsed

    # Read and parse the stations.txt file sequentially
    with open(filepath, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip blank lines
                continue
            # Section header (e.g. "air_quality:") indicates a new data source
            if line.endswith(":"):
                current_section = line[:-1].strip()  # Extract section name (remove colon)
                sections.setdefault(current_section, [])  # Initialize list for this section
            elif current_section is not None:  # Add station entries to current section
                # Split by comma to handle both single-line (comma-separated) and multi-line formats
                # Strip whitespace from each entry to normalize them
                entries = [entry.strip() for entry in line.split(",")]
                sections[current_section].extend(entries)

    # Extract each section's entries (handles missing sections gracefully)
    aq_entries = sections.get("air_quality", [])  # Air Quality station names
    traffic_entries = sections.get("traffic", [])  # Traffic WKT coordinates
    weather_entries = sections.get("weather", [])  # Weather file names

    # Determine how many station groups to create (use longest section, minimum 1)
    n = max(len(aq_entries), len(traffic_entries), len(weather_entries), 1)

    # Create list of station configs by matching entries at same positional index
    configs: list[dict] = []
    for i in range(n):
        configs.append(
            {
                "aq_name": aq_entries[i] if i < len(aq_entries) else None,  # Air Quality source identifier
                "traffic_id": traffic_entries[i] if i < len(traffic_entries) else None,  # Traffic WKT key
                "weather_file": weather_entries[i] if i < len(weather_entries) else None,  # Weather file stem
            }
        )
    return configs


# ===================================================================
# 2. LOAD & CLEAN AIR QUALITY DATA
# ===================================================================
def load_air_quality(data_dir: Path, station_configs: list[dict]) -> pd.DataFrame:
    """Load and pre-process Air Quality CSV files for configured stations.

    CLEANING STEPS:
    1. Extracts station names from metadata headers in each AQ file
    2. Matches stations against the configuration to filter relevant files
    3. Replaces sentinel values (9999, -999) with NaN (marks invalid pollution readings)
    4. Aggregates hourly measurements (H01-H24) into a single daily NO2_Mean value
    5. Deduplicates if a station appears in multiple files or has duplicate dates

    Each AQ file has a 10-line metadata header. Line 2 (0-indexed line 1)
    contains the station name in the form ``Station,Toronto Downtown (31129)``.
    The base name (``Toronto Downtown``) is extracted and matched against the
    ``aq_name`` values from *station_configs*.

    Hourly columns ``H01``–``H24`` are averaged into a daily ``NO2_Mean``.
    Sentinel values ``9999`` / ``-999`` are replaced with ``NaN``.
    """
    # Extract set of target station names we should load from config
    target_names: set[str] = {
        cfg["aq_name"] for cfg in station_configs if cfg["aq_name"]
    }

    # Find all Air Quality CSV files in the data directory
    files = sorted(glob.glob(str(data_dir / "air_quality" / "aq_data" / "*.csv")))
    print(f"  Scanning {len(files)} Air Quality file(s)...")

    all_frames: list[pd.DataFrame] = []
    for file_path in files:
        fname = Path(file_path).name  # Get just the filename for logging
        try:
            # Read the first 10 lines to extract metadata (station name is on line 2)
            with open(file_path, "r", encoding="latin1") as fh:
                meta_lines = [next(fh) for _ in range(10)]
            raw_name = meta_lines[1].split(",")[1].strip()  # Extract station from "Station,Toronto Downtown (31129)"

            # Remove station ID suffix in parentheses using regex
            # "Toronto Downtown (31129)" -> "Toronto Downtown"
            station_name = re.sub(r"\s*\(\d+\)\s*$", "", raw_name).strip()

            # Skip files for stations not in our configuration
            if station_name not in target_names:
                print(f"    Skipping {fname} ({station_name}) — not in stations.txt")
                continue

            # Load the actual data (skip 10-line metadata header)
            df = pd.read_csv(file_path, skiprows=10, index_col=False)

            # Replace sentinel/error values with NaN for missing data imputation later
            # 9999/-999 are common placeholder values for invalid measurements
            df.replace([9999, -999, 9999.0, -999.0], np.nan, inplace=True)

            # Convert Date column to datetime, remove rows with invalid dates
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date"], inplace=True)

            # Aggregate 24 hourly NO2 measurements (H01, H02, ..., H24) into daily mean
            # This reduces noise and provides one value per day per station
            hour_cols = [c for c in df.columns if c.startswith("H") and c[1:].isdigit()]
            df["NO2_Mean"] = df[hour_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

            # Keep only the essential columns for merging later
            df = df[["Date", "NO2_Mean"]].copy()
            df["Station"] = station_name  # Add station identifier for merging
            all_frames.append(df)
            print(f"    Loaded {fname} -> {station_name} ({len(df)} rows)")

        except Exception as exc:
            print(f"    Error reading {fname}: {exc}")

    # Handle case where no valid Air Quality files were found
    if not all_frames:
        print("  WARNING: No Air Quality data loaded.")
        return pd.DataFrame(columns=["Date", "Station", "NO2_Mean"])

    # Combine all station data into one dataframe
    combined = pd.concat(all_frames, ignore_index=True)

    # Handle duplicate (Station, Date) rows by averaging them
    # This occurs when the same station has multiple readings on the same date
    df_clean = combined.groupby(["Station", "Date"], as_index=False)["NO2_Mean"].mean()

    print(f"  AQ loaded: {len(df_clean)} rows, {df_clean['Station'].nunique()} station(s)")
    return df_clean


# ===================================================================
# 3. LOAD & CLEAN TRAFFIC DATA
# ===================================================================
def load_traffic(data_dir: Path, station_configs: list[dict]) -> pd.DataFrame:
    """Load traffic counts for configured stations.

    CLEANING STEPS:
    1. Loads traffic data stored in wide format (one row per camera, many date columns)
    2. Maps WKT geographic coordinates to station names using the configuration
    3. Converts wide format (xYYYY_MM_DD columns) to long format (Date rows)
    4. Averages traffic counts if multiple cameras map to the same station+date
       (NaN values are ignored in the averaging calculation)

    The traffic CSV stores one row per camera location with a ``WKT`` column
    (e.g. ``POINT (-79.388845 43.643857)``) and many date-valued columns
    (``x2022_02_02``, …).  The ``traffic_id`` from *station_configs* is matched
    against the ``WKT`` column to filter rows.  The wide format is then melted
    into a long ``(Station, Date, Traffic_Count)`` table.
    
    AVERAGING LOGIC:
    - If multiple cameras with the same WKT have data for the same station+date,
      their counts are averaged
    - Missing values (NaN) are automatically excluded from the averaging
    - Example: 3 cameras with values [100, NaN, 150] → average = (100+150)/2 = 125
    """
    # Construct file path to traffic data
    traffic_path = data_dir / "traffic" / "tf-ft-eng.csv"
    if not traffic_path.exists():
        print("  WARNING: Traffic file not found.")
        return pd.DataFrame(columns=["Date", "Station", "Traffic_Count"])

    print(f"  Loading {traffic_path.name}...")
    df = pd.read_csv(traffic_path)  # Load wide-format traffic data

    # Build mapping from WKT geographic coordinates to canonical station names
    # WKT values are geospatial coordinates stored in Well-Known Text format
    wkt_to_station: dict[str, str] = {}
    for cfg in station_configs:
        if cfg["traffic_id"] and cfg["aq_name"]:  # Only map if both traffic ID and station name exist
            wkt_to_station[cfg["traffic_id"].strip()] = cfg["aq_name"]

    # Apply WKT -> station name mapping to all rows
    # This allows us to group traffic by logical station rather than by individual camera WKT
    df["Station"] = df["WKT"].str.strip().map(wkt_to_station)
    df = df.dropna(subset=["Station"])  # Remove rows that don't match any configured station

    if df.empty:
        print("  WARNING: No traffic rows matched station configs.")
        return pd.DataFrame(columns=["Date", "Station", "Traffic_Count"])

    # Identify all date columns (follow pattern: xYYYY_MM_DD)
    # These columns store traffic counts for each date
    date_col_re = re.compile(r"^x\d{4}_\d{2}_\d{2}$")
    date_cols = [c for c in df.columns if date_col_re.match(c)]

    # Convert from wide format to long format using melt()
    # Before: [Station, WKT, x2022_02_02, x2022_02_03, ...]
    # After:  [Station, Date, Traffic_Count] with one row per station-date pair
    df_long = df.melt(
        id_vars=["Station"],
        value_vars=date_cols,
        var_name="Date_Str",  # Temporary column with column names like "x2022_02_02"
        value_name="Traffic_Count",  # The actual traffic count values
    )

    # Parse the date string (remove leading 'x' and convert to datetime)
    # "x2022_02_02" -> datetime(2022, 02, 02)
    df_long["Date"] = pd.to_datetime(
        df_long["Date_Str"].str[1:], format="%Y_%m_%d", errors="coerce"
    )
    # Remove rows with invalid dates
    df_long = df_long.dropna(subset=["Date"])
    
    # Convert Traffic_Count to numeric, replacing non-numeric values with NaN
    df_long["Traffic_Count"] = pd.to_numeric(df_long["Traffic_Count"], errors="coerce")

    # If multiple cameras/roads map to the same station on the same date, average their counts
    # pandas .mean() automatically ignores NaN values and averages only the valid entries
    # Example: if 3 rows for same (Station, Date) have values [100, NaN, 150],
    # the result will be (100 + 150) / 2 = 125
    df_final = df_long.groupby(["Station", "Date"], as_index=False)["Traffic_Count"].mean()

    print(f"  Traffic loaded: {len(df_final)} rows, {df_final['Station'].nunique()} station(s)")
    return df_final


# ===================================================================
# 4. LOAD & CLEAN WEATHER DATA
# ===================================================================
def load_weather(data_dir: Path, station_configs: list[dict]) -> pd.DataFrame:
    """Load weather data for configured stations.

    CLEANING STEPS:
    1. Matches weather CSV file names (stems) to station names from configuration
    2. Normalizes the date column to timezone-naive, date-only values
    3. Standardizes column names (e.g., "Mean Temp (°C)" -> "Temp")
    4. Converts wind direction from "10s of degrees" to actual degrees (e.g., 25 -> 250°)
    5. Ensures all numeric columns are properly typed and handles missing values

    The ``weather_file`` field in each station config is matched against weather
    CSV file stems (e.g. ``Toronto_City_Centre_Downtown``).

    Columns are renamed and wind direction is converted from "10s of degrees"
    to actual degrees.
    """
    # Build mapping from weather file names to canonical station names
    # File stem is filename without path and extension (e.g., "Toronto_City_Centre_Downtown")
    weather_map: dict[str, str] = {}
    for cfg in station_configs:
        if cfg["weather_file"] and cfg["aq_name"]:  # Only map if both weather file and station name exist
            weather_map[cfg["weather_file"].strip()] = cfg["aq_name"]

    # Find all weather CSV files in the data directory
    weather_files = sorted(
        glob.glob(str(data_dir / "weather" / "weather_data" / "*.csv"))
    )
    print(f"  Scanning {len(weather_files)} Weather file(s)...")

    all_frames: list[pd.DataFrame] = []
    for fpath in weather_files:
        stem = Path(fpath).stem  # Get filename without .csv extension (e.g., "Toronto_City_Centre_Downtown")
        station_name = weather_map.get(stem)  # Look up canonical station name

        if not station_name:
            print(f"    Skipping {stem}.csv — not in stations.txt")
            continue

        try:
            df = pd.read_csv(fpath)
            df["Station"] = station_name  # Add station identifier for merging

            # Normalize date column (different files may use different column names)
            # "Date/Time" is common in some weather data, "Date" in others
            date_col = "Date/Time" if "Date/Time" in df.columns else "Date"
            # Convert to datetime and normalize to date-only (removes times like 00:00:00)
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

            # Standardize column names for consistency across different weather data sources
            # This ensures merging works regardless of the original column name format
            rename_map = {
                "Mean Temp (°C)": "Temp",  # Temperature in Celsius
                "Total Precip (mm)": "Precip",  # Precipitation in millimeters
                "Spd of Max Gust (km/h)": "Wind_Gust",  # Maximum wind gust speed
                "Dir of Max Gust (10s deg)": "Wind_Dir_Raw",  # Raw wind direction in tens of degrees
            }
            df.rename(columns=rename_map, inplace=True)

            # Convert wind direction from "10s of degrees" to actual degrees
            # Example: raw value 25 means 25 * 10 = 250 degrees
            # This is required because the raw data stores degrees in increments of 10
            if "Wind_Dir_Raw" in df.columns:
                df["Wind_Dir"] = pd.to_numeric(df["Wind_Dir_Raw"], errors="coerce") * 10
            else:
                df["Wind_Dir"] = np.nan  # Create NaN column if raw wind data doesn't exist

            # Ensure all numeric columns exist and are properly typed
            # For columns that don't exist, create them as NaN
            # For columns that exist, convert to numeric and handle non-numeric values
            for col in ("Temp", "Precip", "Wind_Gust", "Wind_Dir"):
                if col not in df.columns:
                    df[col] = np.nan  # Create column with missing values
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, replace non-numeric with NaN

            # Keep only the essential columns for merging with other data sources
            keep = ["Date", "Station", "Temp", "Precip", "Wind_Gust", "Wind_Dir"]
            all_frames.append(df[keep])
            print(f"    Loaded {stem}.csv -> {station_name} ({len(df)} rows)")

        except Exception as exc:
            print(f"    Error loading {stem}: {exc}")

    # Handle case where no valid weather files were found
    if not all_frames:
        print("  WARNING: No weather data loaded.")
        return pd.DataFrame(
            columns=["Date", "Station", "Temp", "Precip", "Wind_Gust", "Wind_Dir"]
        )

    # Combine all weather station data into one dataframe
    df_weather = pd.concat(all_frames, ignore_index=True)
    # Sort by station and date for consistent ordering and easier verification
    df_weather.sort_values(["Station", "Date"], inplace=True)
    df_weather.reset_index(drop=True, inplace=True)  # Reset index after sorting

    print(
        f"  Weather loaded: {len(df_weather)} rows, "
        f"{df_weather['Station'].nunique()} station(s)"
    )
    return df_weather


# ===================================================================
# 5. MERGE DATASETS
# ===================================================================
def merge_datasets(
    df_aq: pd.DataFrame,
    df_traffic: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge AQ, Traffic, and Weather on ``(Station, Date)``.

    CLEANING/INTEGRATION STEPS:
    1. Normalizes all date columns to timezone-naive, date-only values
       (prevents date mismatch issues when merging)
    2. Performs outer joins to retain all observations from each source
       (even if not all stations have all three data types on all dates)
    3. Sorts by station and date for consistent ordering

    All date columns are first normalised to timezone-naive, date-only values
    to prevent mismatches.  An outer join is used so that all available
    observations are retained.
    """
    print("  Normalising date columns...")
    # Ensure all date columns are timezone-naive and contain only date (no time component)
    for df in (df_aq, df_traffic, df_weather):
        if df.empty:  # Skip empty dataframes
            continue
        # Normalize to date-only (00:00:00 time)
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        # Remove timezone info if present (e.g., UTC, EST) to avoid merge conflicts
        if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)

    print("  Merging AQ + Traffic...")
    # Outer join ensures we keep all rows even if some stations have AQ but not traffic data
    # Keys: (Station, Date) pairs identify unique observations
    merged = pd.merge(df_aq, df_traffic, on=["Station", "Date"], how="outer")

    print("  Merging with Weather...")
    # Add weather to the merged AQ+Traffic dataframe using another outer join
    # This preserves all existing rows even if some don't have weather data
    merged = pd.merge(merged, df_weather, on=["Station", "Date"], how="outer")

    # Sort by station and date for consistent, predictable ordering
    merged.sort_values(["Station", "Date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)  # Reset index after sorting

    # Print summary statistics of merged data
    print(
        f"  Merged: {merged.shape[0]} rows x {merged.shape[1]} columns  |  "
        f"Date range: {merged['Date'].min()} – {merged['Date'].max()}  |  "
        f"Stations: {sorted(merged['Station'].dropna().unique())}"
    )
    return merged


# ===================================================================
# 5.5 FILTER DATA BY TIMEFRAME
# ===================================================================
def filter_data_by_date(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Filter dataset to include only data within a specified date range.

    CLEANING STEP (Temporal Filtering):
    - Removes observations outside the desired time period
    - Ensures consistent analysis time window across all stations
    - Reduces memory footprint and speeds up downstream modeling

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a 'Date' column (datetime type)
    start_date : pd.Timestamp
        Beginning of the desired time period (inclusive)
    end_date : pd.Timestamp
        End of the desired time period (inclusive)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only rows with dates between start_date
        and end_date (inclusive), sorted by Station and Date
    """
    # Ensure Date column is datetime type for filtering
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    # Count original rows for reporting
    rows_before = len(df)

    # Filter rows where Date is between start_date and end_date (inclusive)
    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    # Count filtered rows to show impact
    rows_after = len(df_filtered)
    rows_removed = rows_before - rows_after

    # Print filtering summary
    print(
        f"  Filtered to date range [{start_date.date()} – {end_date.date()}]  |  "
        f"Rows: {rows_before} → {rows_after} ({rows_removed} removed)"
    )

    return df_filtered


# ===================================================================
# FILE PATH UTILITIES
# ===================================================================
def get_output_filename(station_name: str) -> Path:
    """Generate output filename for a specific station's cleaned data.
    
    USAGE:
    - Converts station name to lowercase and replaces spaces with underscores
    - Creates a clean, standardized filename for each station
    
    Parameters
    ----------
    station_name : str
        Name of the station (e.g., "Toronto Downtown", "Burlington")
        
    Returns
    -------
    Path
        Absolute path to output CSV file (e.g., cleaned_data_toronto_downtown.csv)
    """
    # Sanitize station name for filename: lowercase and replace spaces with underscores
    sanitized_name = station_name.lower().replace(" ", "_")
    filename = f"cleaned_data_{sanitized_name}.csv"
    return DATA_CLEAN_DIR / filename


# ===================================================================
# 6. MISSING-VALUE REPORTING & MULTIPLE IMPUTATION
# ===================================================================
def report_missing(df: pd.DataFrame, label: str = "") -> pd.Series:
    """Print and return a per-column missing-value summary for numeric columns.
    
    ANALYSIS STEP:
    - Provides visibility into data quality before and after imputation
    - Identifies which variables have the most missing values
    - Helps determine if imputation was successful
    """
    print(f"\n{'='*60}")
    print(f"  Missing Values {label}")
    print(f"{'='*60}")

    # Select only numeric columns (non-numeric columns like Station don't need imputation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing = df[numeric_cols].isnull().sum()  # Count NaN values per column
    total = len(df)  # Total number of rows
    # Calculate missing percentage for each column
    report = pd.DataFrame(
        {
            "Missing": missing,  # Number of missing values
            "Total": total,  # Total rows
            "Pct (%)": (missing / total * 100).round(2),  # Percentage missing
        }
    )
    # Show only columns with at least one missing value, sorted by percentage (highest first)
    report = report[report["Missing"] > 0].sort_values("Pct (%)", ascending=False)

    if report.empty:
        print("  No missing values in numeric columns.")
    else:
        print(report.to_string())
    return missing


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform Multiple Imputation using ``IterativeImputer`` (MICE-style).

    CLEANING STEP:
    - Replaces missing values with estimated values based on relationships
      between variables using iterative chained equations (MICE algorithm)
    - Applies physical constraints (e.g., no negative concentrations)
    - Improves data completeness for downstream analysis and modeling

    Only numeric columns participate in imputation.  Categorical columns
    (``Station``) and datetime columns (``Date``) are excluded automatically,
    then re-attached after imputation.

    Post-processing ensures physically meaningful bounds (e.g. no negative
    NO2 concentrations or traffic counts).
    """
    # Extract numeric columns for imputation (excludes Station and Date)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("  No numeric columns found — skipping imputation.")
        return df

    print(f"  Numeric columns for imputation: {numeric_cols}")

    # Initialize IterativeImputer with MICE (Multivariate Imputation by Chained Equations)
    # MICE iteratively estimates missing values by modeling each feature with missing values
    # as a function of other features, then cycles through this process multiple times
    imputer = IterativeImputer(
        max_iter=10,  # Number of iterations for the algorithm to converge
        random_state=42,  # Seed for reproducibility (ensures same results across runs)
        initial_strategy="mean",  # Start with mean values as initial estimates
        sample_posterior=False,  # Use mean estimates rather than drawing from posterior distribution
    )

    df_out = df.copy()
    # Fit the imputer on numeric columns and replace missing values with estimates
    imputed_array = imputer.fit_transform(df_out[numeric_cols])
    df_out[numeric_cols] = imputed_array

    # Post-process: Apply physical/domain constraints to ensure realistic values
    # NO2 concentrations and traffic counts cannot be negative
    non_negative_cols = ["NO2_Mean", "Traffic_Count", "Precip", "Wind_Gust"]
    for col in non_negative_cols:
        if col in df_out.columns:
            # Clip values to be >= 0 (replace any negative values with 0)
            df_out[col] = df_out[col].clip(lower=0)

    # Wind direction should be between 0 and 360 degrees
    if "Wind_Dir" in df_out.columns:
        df_out["Wind_Dir"] = df_out["Wind_Dir"].clip(lower=0, upper=360)

    # Traffic counts should be integers (you can't have fractional vehicles)
    if "Traffic_Count" in df_out.columns:
        df_out["Traffic_Count"] = df_out["Traffic_Count"].round(0).astype(int)

    print("  Imputation complete.")
    return df_out


# ===================================================================
# 7. STATION-BY-STATION PROCESSING
# ===================================================================
def process_single_station(
    df_merged: pd.DataFrame,
    station_name: str,
) -> pd.DataFrame:
    """Process cleaning and imputation for a single station's data.

    PROCESSING STEPS:
    1. Filters merged dataset to rows for the specified station
    2. Reports missing values before imputation
    3. Performs multiple imputation on the filtered data
    4. Reports missing values after imputation
    5. Saves cleaned data to station-specific CSV file
    
    Parameters
    ----------
    df_merged : pd.DataFrame
        Combined dataset with all stations (from merge_datasets)
    station_name : str
        Name of the station to process (e.g., "Toronto Downtown")
        
    Returns
    -------
    pd.DataFrame
        Cleaned and imputed data for the specified station
    """
    print(f"\n  Processing station: {station_name}")
    
    # Filter to just this station's data
    df_station = df_merged[df_merged["Station"] == station_name].copy()
    
    if df_station.empty:
        print(f"    WARNING: No data found for station {station_name}")
        return pd.DataFrame()
    
    n_rows = len(df_station)
    print(f"    Found {n_rows} rows for {station_name}")
    
    # Report missing values before imputation
    missing_before = report_missing(df_station, f"(Before Imputation) — {station_name}")
    
    # Perform multiple imputation for this station
    df_imputed = impute_data(df_station)
    
    # Report missing values after imputation
    missing_after = report_missing(df_imputed, f"(After Imputation) — {station_name}")
    
    # Generate output filename for this station
    output_file = get_output_filename(station_name)
    
    # Save cleaned data to CSV file specific to this station
    df_imputed.to_csv(output_file, index=False)
    print(f"    Saved to: {output_file}")
    print(f"    Final shape: {df_imputed.shape[0]} rows × {df_imputed.shape[1]} columns")
    
    return df_imputed


# ===================================================================
# 8. MAIN PIPELINE ORCHESTRATION
# ===================================================================
def main() -> None:
    """Main pipeline: processes stations separately and saves individual cleaned datasets.
    
    COMPLETE WORKFLOW:
    1. Parse configuration file (stations.txt) to link data sources
    2. Load all Air Quality, Traffic, and Weather data
    3. Merge all datasets on (Station, Date) using outer joins
    4. Filter combined data to specified timeframe (2022/02/02 – 2024/12/31)
    5. FOR EACH STATION:
       - Filter to station-specific rows
       - Report missing values before imputation
       - Perform multiple imputation on station data
       - Report missing values after imputation
       - Save cleaned data to station-specific CSV file
    """
    print("=" * 60)
    print("  Traffic & Air Quality — Data Cleaning Pipeline")
    print("  (Separate files per station)")
    print("=" * 60)

    # ---- Step 1: Parse station configurations ----
    # This configuration file links three independent data sources to the same logical station
    print("\n[1/5] Parsing stations.txt ...")
    station_configs = parse_stations(STATIONS_FILE)
    print(f"  Found {len(station_configs)} station configuration(s):")
    for i, cfg in enumerate(station_configs, start=1):
        print(
            f"    #{i}  AQ = {cfg['aq_name']!r}  |  "
            f"Traffic = {cfg['traffic_id']!r}  |  "
            f"Weather = {cfg['weather_file']!r}"
        )

    # ---- Step 2–4: Load each data source ----
    # Each source is loaded and pre-processed independently, then merged later
    print("\n[2/5] Loading Air Quality data ...")
    df_aq = load_air_quality(DATA_DIR, station_configs)

    print("\n[3/5] Loading Traffic data ...")
    df_traffic = load_traffic(DATA_DIR, station_configs)

    print("\n[4/5] Loading Weather data ...")
    df_weather = load_weather(DATA_DIR, station_configs)

    # ---- Step 5: Merge ----
    # All three dataframes are combined on common keys: Station and Date
    # Outer joins ensure no data is lost if a source is missing for a particular station-date
    print("\n[5/5] Merging datasets ...")
    df_merged = merge_datasets(df_aq, df_traffic, df_weather)

    # ---- Filter by timeframe ----
    # Restrict data to the specified date range for consistent analysis window
    print("\n[5.1/5] Filtering data to specified timeframe ...")
    df_merged = filter_data_by_date(df_merged, START_DATE, END_DATE)

    # ---- Process each station separately ----
    # Extract unique station names and process each one individually
    stations = sorted(df_merged["Station"].dropna().unique())
    
    if not stations:
        print("\nERROR: No stations found in merged data.")
        return
    
    print(f"\n[5.2/5] Processing {len(stations)} station(s) separately ...")
    print("=" * 60)
    
    for station_name in stations:
        process_single_station(df_merged, station_name)
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\nGenerated {len(stations)} cleaned data file(s):")
    for station_name in stations:
        output_file = get_output_filename(station_name)
        print(f"  - {output_file.name}")
    print()


# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    # Execute the main pipeline when script is run directly
    main()
