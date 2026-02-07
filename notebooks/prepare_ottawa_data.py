"""
Helper script to prepare Ottawa data for the Stacking Ensemble notebook
Run this once to create clean, processed CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("PREPARING OTTAWA DATA FOR STACKING ENSEMBLE")
print("="*70)

# ========== 1. Process Air Quality Data ==========
print("\n1. Processing Air Quality Data...")
# Read lines manually to avoid column parsing issues
with open('../data/air_quality/aq_data/station_51001_data.csv', 'r') as f:
    lines = f.readlines()

# Parse data starting from line 11 (0-indexed line 11 is row 12 in file)
data_lines = []
total_days = 0
days_with_missing = 0
days_rejected = 0

for line in lines[11:]:  # Skip first 11 lines (metadata + header)
    total_days += 1
    parts = line.strip().split(',')
    if len(parts) >= 27:  # Station ID, Pollutant, Date, 24 hours
        # Extract: station_id (0), pollutant (1-2 joined), date (2), hours (3-26)
        station_id = parts[0]
        date_str = parts[2]
        hours = parts[3:27]  # 24 hours
        
        # Convert hours to numeric, replacing 9999 and -999 with NaN
        hour_values = []
        for h in hours:
            try:
                val = float(h) if h.strip() else np.nan
                # Handle both positive and negative missing value indicators
                if val == 9999 or val == -999 or val < 0:
                    val = np.nan
                hour_values.append(val)
            except:
                hour_values.append(np.nan)
        
        # Compute mean (only if we have at least 12 valid hours - 50% data coverage)
        valid_hours = [h for h in hour_values if not np.isnan(h)]
        if len(valid_hours) < 24:
            days_with_missing += 1
        
        if len(valid_hours) >= 12:  # At least 12 hours of data
            no2_mean = np.mean(valid_hours)
            if no2_mean >= 0:  # Final sanity check - NO2 cannot be negative
                data_lines.append({'Date': date_str, 'NO2_Mean': no2_mean})
            else:
                days_rejected += 1
        else:
            days_rejected += 1

# Create DataFrame
aq_data = pd.DataFrame(data_lines)
aq_data['Date'] = pd.to_datetime(aq_data['Date'], errors='coerce')
aq_data = aq_data.dropna(subset=['Date', 'NO2_Mean'])

print(f"   Data Quality Summary:")
print(f"     Total days processed: {total_days}")
print(f"     Days with some missing hours: {days_with_missing}")
print(f"     Days rejected (< 12 hours or invalid): {days_rejected}")
print(f"     Days kept: {len(aq_data)}")
print(f"     NO2 range: {aq_data['NO2_Mean'].min():.2f} to {aq_data['NO2_Mean'].max():.2f} ppb")
print(f"     NO2 mean: {aq_data['NO2_Mean'].mean():.2f} ppb")

if (aq_data['NO2_Mean'] < 0).any():
    print(f"     ⚠️ WARNING: {(aq_data['NO2_Mean'] < 0).sum()} negative values still present!")
else:
    print(f"     ✓ All NO2 values are positive")

# Save processed data
output_path = '../data/air_quality/ottawa_air_quality_processed.csv'
aq_data.to_csv(output_path, index=False)
print(f"\n   ✓ Saved to: {output_path}")
print(f"   Shape: {aq_data.shape}")
print(f"   Date range: {aq_data['Date'].min()} to {aq_data['Date'].max()}")

# ========== 2. Process Traffic Data ==========
print("\n2. Processing Traffic Data...")
# Traffic data is in wide format with dates as columns
traffic_raw = pd.read_csv('../data/traffic/ottawa_traffic_filtered.csv')

# Get date columns (start with 'x')
date_cols = [col for col in traffic_raw.columns if col.startswith('x')]

# Transform to long format (use second row for HWY-417)
traffic_long = []
for col in date_cols:
    # Extract date from column name: x2022_02_02 -> 2022-02-02
    date_str = col[1:].replace('_', '-')
    try:
        date_obj = pd.to_datetime(date_str)
        # Use the second row (HWY-417 location)
        if len(traffic_raw) > 1:
            traffic_count = traffic_raw[col].iloc[1]
            if pd.notna(traffic_count):
                traffic_long.append({
                    'Date': date_obj,
                    'Traffic_Count': float(traffic_count)
                })
    except Exception as e:
        continue

traffic_data = pd.DataFrame(traffic_long)

# Save processed data
output_path = '../data/traffic/ottawa_traffic_processed.csv'
traffic_data.to_csv(output_path, index=False)
print(f"   ✓ Saved to: {output_path}")
print(f"   Shape: {traffic_data.shape}")
print(f"   Date range: {traffic_data['Date'].min()} to {traffic_data['Date'].max()}")

# ========== 3. Process Weather Data ==========
print("\n3. Processing Weather Data...")
weather_raw = pd.read_csv('../data/weather/weather_data/Ottawa_CDA_RCS.csv')

# Rename and select columns
weather_data = weather_raw.rename(columns={
    'Date/Time': 'Date',
    'Mean Temp (°C)': 'Mean_Temp',
    'Total Precip (mm)': 'Total_Precip',
    'Spd of Max Gust (km/h)': 'Wind_Speed'
})

weather_data = weather_data[['Date', 'Mean_Temp', 'Total_Precip', 'Wind_Speed']].copy()
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data['Wind_Dir'] = 180.0  # Placeholder (South)

# Fill missing values
weather_data['Total_Precip'] = weather_data['Total_Precip'].fillna(0)
weather_data['Wind_Speed'] = weather_data['Wind_Speed'].fillna(weather_data['Wind_Speed'].mean())

# Save processed data
output_path = '../data/weather/ottawa_weather_processed.csv'
weather_data.to_csv(output_path, index=False)
print(f"   ✓ Saved to: {output_path}")
print(f"   Shape: {weather_data.shape}")
print(f"   Date range: {weather_data['Date'].min()} to {weather_data['Date'].max()}")

# ========== 4. Create Merged Dataset ==========
print("\n4. Creating merged dataset...")
# Merge all three datasets
merged = aq_data.merge(traffic_data, on='Date', how='inner')
merged = merged.merge(weather_data, on='Date', how='inner')

print(f"   ✓ Merged shape: {merged.shape}")
print(f"   Date range: {merged['Date'].min()} to {merged['Date'].max()}")
print(f"   Missing values: {merged.isnull().sum().sum()}")

print("\n" + "="*70)
print("✅ ALL OTTAWA DATA PROCESSED SUCCESSFULLY!")
print("="*70)
print("\nYou can now run the Ottawa_Stacking_Ensemble.ipynb notebook!")
print("\nProcessed files created:")
print("  - ../data/air_quality/ottawa_air_quality_processed.csv")
print("  - ../data/traffic/ottawa_traffic_processed.csv")
print("  - ../data/weather/ottawa_weather_processed.csv")
print("="*70)
