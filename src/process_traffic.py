import pandas as pd
import numpy as np
import re
import os

def process_traffic():
    # --- Configuration ---
    # Paths relative to the project root
    INPUT_PATH = "traffic_dataset/tf-ft-eng.csv"
    OUTPUT_PATH = "traffic_dataset/filtered_roads_with_sums.csv"
    
    # The specific roads you selected in your notebook
    ROADS_KEEP = [
        "LAWRENCE AVE E / KENNEDY RD",                              # East
        "STEELES AVE W / DUFFERIN ST",                              # North
        "FRONT ST W / JOHN ST / PRIVATE ACCESS",                    # Downtown
        "ISLINGTON AVE / 401 C W ISLINGTON N RAMP / ALLENBY AVE"    # West
    ]

    print("--- Starting Traffic Data Processing ---")
    
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found at: {INPUT_PATH}")

    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded raw data: {len(df)} rows")

    # 2. Identify Date Columns (xYYYY_MM_DD)
    date_col_pattern = re.compile(r"^x\d{4}_\d{2}_\d{2}$")
    traffic_cols = [c for c in df.columns if date_col_pattern.match(str(c))]

    # 3. Filter to Selected Roads
    if "camera_road" not in df.columns:
        raise KeyError("Expected a 'camera_road' column but it was not found.")

    # Normalize strings (strip whitespace) and filter
    mask = df["camera_road"].astype(str).str.strip().isin(ROADS_KEEP)
    df_filt = df[mask].copy()
    print(f"Filtered to {len(df_filt)} rows for selected roads")

    # 4. Compute Statistics (Sums and NA counts)
    df_filt["traffic_sum"] = df_filt[traffic_cols].sum(axis=1, skipna=True)
    df_filt["na_count"] = df_filt[traffic_cols].isna().sum(axis=1)

    # 5. Compute First/Last Dates (Keeping your exact logic)
    def _first_last_dates(row):
        non_na = row.dropna()
        if non_na.empty:
            return pd.NaT, pd.NaT
        first_col = non_na.index[0].lstrip('x')
        last_col = non_na.index[-1].lstrip('x')
        return pd.to_datetime(first_col, format='%Y_%m_%d'), pd.to_datetime(last_col, format='%Y_%m_%d')

    # Apply the date logic
    print("Calculating date ranges...")
    first_last = df_filt[traffic_cols].apply(_first_last_dates, axis=1, result_type='expand')
    df_filt['first_date'] = first_last[0]
    df_filt['last_date'] = first_last[1]

    # 6. Save the Result
    df_filt.to_csv(OUTPUT_PATH, index=False)
    print(f"-> SAVED: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_traffic()