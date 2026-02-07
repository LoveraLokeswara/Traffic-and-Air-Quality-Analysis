import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Define the Root Path relative to this notebook
def get_project_root():
    path = Path.cwd()
    while path.name:
        if (path / 'pyproject.toml').exists(): return path
        path = path.parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"

# Mappings
TRAFFIC_TO_AQ_MAP = {
    'LAWRENCE AVE E / KENNEDY RD': 'Toronto East',
    'STEELES AVE W / DUFFERIN ST': 'Toronto North',
    'FRONT ST W / JOHN ST / PRIVATE ACCESS': 'Toronto Downtown',
    'ISLINGTON AVE / 401 C W ISLINGTON N RAMP / ALLENBY AVE': 'Toronto West'
}

AQ_STATION_NAMES = {
    'Toronto East (33003)': 'Toronto East',
    'Toronto North (34021)': 'Toronto North',
    'Toronto Downtown (31129)': 'Toronto Downtown',
    'Toronto West (35125)': 'Toronto West'
}


def load_and_clean_aq(data_dir):
    all_data = []
    files = glob.glob(str(data_dir / "air_quality" / "aq_data" / "**" / "*.csv"), recursive=True)
    
    print(f"Loading {len(files)} Air Quality files...")
    
    for file_path in files:
        try:
            # Metadata extraction
            with open(file_path, 'r', encoding='latin1') as f:
                meta_lines = [next(f) for _ in range(10)]
                raw_name = meta_lines[1].split(',')[1].strip()
            
            station_name = AQ_STATION_NAMES.get(raw_name, raw_name)
            
            # Load Data
            df = pd.read_csv(file_path, skiprows=10, index_col=False)
            df.replace([9999, -999, 9999.0, -999.0], np.nan, inplace=True)
            
            # Date Parsing
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            
            # Calculate Mean
            hour_cols = [c for c in df.columns if c.startswith('H') and c[1:].isdigit()]
            df['NO2_Mean'] = df[hour_cols].mean(axis=1)
            
            # Keep essentials
            df = df[['Date', 'NO2_Mean']].copy()
            df['Station'] = station_name
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {Path(file_path).name}: {e}")

    # Combine
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 1. Group by Station/Date to merge split rows
    df_clean = full_df.groupby(['Station', 'Date'], as_index=False)['NO2_Mean'].mean()
    
    # 2. Reindex to handle MISSING DATES
    min_date, max_date = df_clean['Date'].min(), df_clean['Date'].max()
    full_range = pd.date_range(min_date, max_date, freq='D')
    
    def fill_gaps(group):
        # FIX: The station name is the 'name' of the group, not a column inside it
        station_name = group.name
        
        # Reindex
        group = group.set_index('Date').reindex(full_range)
        
        # Assign station name to all rows (including the new gap rows)
        group['Station'] = station_name
        
        return group.reset_index().rename(columns={'index': 'Date'})

    # Apply reindexing
    df_clean = df_clean.groupby('Station', group_keys=False).apply(fill_gaps)

    # 3. Impute Missing Values (Forward Fill, limit 7 days)
    df_clean['NO2_Mean'] = df_clean.groupby('Station')['NO2_Mean'].ffill(limit=7)
    # Drop remaining NaNs (e.g. leading missing values)
    df_clean.dropna(subset=['NO2_Mean'], inplace=True)

    
    print(f"✅ AQ Data Loaded: {len(df_clean)} rows across {df_clean['Station'].nunique()} stations.")
    return df_clean


def load_traffic(data_dir):
    traffic_path = data_dir / "traffic" / "filtered_roads_with_sums.csv"
    print(f"Loading Traffic Data from {traffic_path.name}...")
    
    df = pd.read_csv(traffic_path)
    
    # Identify date columns
    date_col_pattern = re.compile(r"^x\d{4}_\d{2}_\d{2}$")
    date_cols = [c for c in df.columns if date_col_pattern.match(c)]
    
    # Melt
    df_long = df.melt(
        id_vars=['camera_road'], 
        value_vars=date_cols, 
        var_name='Date_Str', 
        value_name='Traffic_Count'
    )
    
    # Process Dates
    df_long['Date'] = pd.to_datetime(df_long['Date_Str'].str[1:], format='%Y_%m_%d')
    df_long['Station'] = df_long['camera_road'].map(TRAFFIC_TO_AQ_MAP)
    
    # Filter valid stations
    df_final = df_long.dropna(subset=['Station', 'Traffic_Count'])
    
    # Aggregate (Sum volume if multiple cameras -> 1 Station)
    df_final = df_final.groupby(['Station', 'Date'], as_index=False)['Traffic_Count'].sum()
    
    # --- IMPUTATION STRATEGY: FILL GAPS WITH DAY-OF-WEEK AVERAGE ---
    
    # 1. Reindex to create the missing rows (The "Month Long Gaps")
    min_date, max_date = df_final['Date'].min(), df_final['Date'].max()
    full_range = pd.date_range(min_date, max_date, freq='D')
    
    def reindex_traffic(group):
        station_name = group.name
        group = group.set_index('Date').reindex(full_range)
        group['Station'] = station_name
        return group.reset_index().rename(columns={'index': 'Date'})
    
    df_final = df_final.groupby('Station', group_keys=False).apply(reindex_traffic)
    
    # 2. Calculate Day of Week for every row (including the empty ones)
    df_final['DayOfWeek'] = df_final['Date'].dt.dayofweek
    
    # 3. Impute: "If this is a missing Tuesday at Toronto East, fill it with the average Toronto East Tuesday"
    # We group by [Station, DayOfWeek] so Monday averages don't mix with Sunday averages
    df_final['Traffic_Count'] = df_final['Traffic_Count'].fillna(
        df_final.groupby(['Station', 'DayOfWeek'])['Traffic_Count'].transform('mean')
    )
    
    # 4. Clean up
    df_final.drop(columns=['DayOfWeek'], inplace=True)
    df_final.dropna(subset=['Traffic_Count'], inplace=True) # Drop if a station has NO data for a specific day-of-week ever
    
    print(f"✅ Traffic Data Loaded: {len(df_final)} rows.")
    return df_final

def load_weather_with_direction(data_dir):
    all_weather = []
    weather_files = glob.glob(str(data_dir / "weather" / "weather_data" / "*.csv"))
    
    filename_map = {
        'Toronto_City_Centre_Downtown': 'Toronto Downtown',
        'Toronto_City_Ontario_East': 'Toronto East',
        'Toronto_Pearson_Intl_West': 'Toronto West',
        'Toronto_York_North': 'Toronto North'
    }

    for fpath in weather_files:
        fname = Path(fpath).stem
        station_name = filename_map.get(fname)
        
        # Robust Mapping Fallback
        if not station_name:
            if 'East' in fname: station_name = 'Toronto East'
            elif 'North' in fname: station_name = 'Toronto North'
            elif 'West' in fname: station_name = 'Toronto West'
            elif 'Downtown' in fname: station_name = 'Toronto Downtown'

        if station_name:
            try:
                df = pd.read_csv(fpath)
                df['Station'] = station_name
                
                # Normalize Date
                date_col = 'Date/Time' if 'Date/Time' in df.columns else 'Date'
                df['Date'] = pd.to_datetime(df[date_col]).dt.normalize()
                
                # Map Columns (Adding 'Dir of Max Gust')
                # 'Dir of Max Gust (10s deg)' means '18' = 180 degrees
                rename_map = {
                    'Mean Temp (°C)': 'Temp', 
                    'Total Precip (mm)': 'Precip', 
                    'Spd of Max Gust (km/h)': 'Wind_Gust',
                    'Dir of Max Gust (10s deg)': 'Wind_Dir_Raw' 
                }
                df.rename(columns=rename_map, inplace=True)
                
                # Convert '10s of degrees' to actual degrees (multiply by 10)
                if 'Wind_Dir_Raw' in df.columns:
                    df['Wind_Dir'] = df['Wind_Dir_Raw'] * 10
                else:
                    df['Wind_Dir'] = np.nan

                # Force columns to exist
                for c in ['Temp', 'Precip', 'Wind_Gust', 'Wind_Dir']:
                    if c not in df.columns: df[c] = np.nan
                
                cols = ['Date', 'Station', 'Temp', 'Precip', 'Wind_Gust', 'Wind_Dir']
                all_weather.append(df[cols])
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    df_weather = pd.concat(all_weather, ignore_index=True)
    df_weather = df_weather.sort_values(['Station', 'Date'])
    
    # --- IMPUTATION ---
    # 1. Fill small gaps
    for col in ['Temp', 'Precip']:
        df_weather[col] = df_weather.groupby('Station')[col].transform(lambda x: x.ffill().bfill())
        
    # 2. Regional Imputation for Wind (Speed AND Direction)
    # If East/North are missing wind, grab the daily average from West/Downtown
    for col in ['Wind_Gust', 'Wind_Dir']:
        regional_avg = df_weather.groupby('Date')[col].transform('mean')
        df_weather[col] = df_weather[col].fillna(regional_avg)
    
    # 3. Final Cleanup
    df_weather['Wind_Gust'] = df_weather['Wind_Gust'].fillna(0.0)
    df_weather['Wind_Dir'] = df_weather['Wind_Dir'].fillna(0.0) # Assume North if completely unknown
    
    df_weather = df_weather.dropna(subset=['Temp'])
    print(f"✅ Weather Data Loaded with Direction: {len(df_weather)} rows")
    return df_weather

df_weather = load_weather_with_direction(DATA_DIR)