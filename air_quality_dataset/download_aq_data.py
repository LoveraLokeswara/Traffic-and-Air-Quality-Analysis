#!/usr/bin/env python3
"""
Air Quality Data Downloader
Downloads NO2 data from Air Quality Ontario for Toronto stations
"""

import pandas as pd
import requests
import os

# Read station information
print("Reading station information...")
station_info = pd.read_csv('station_info.csv')

# Get station IDs for the target Toronto stations
target_stations = ['TORONTO DOWNTOWN', 'TORONTO EAST', 'TORONTO NORTH', 'TORONTO WEST']
station_ids = station_info[
    station_info['STATION NAME'].isin(target_stations) & 
    (station_info['Year'] == 2024)
]["ID"].unique()

print(f"Found {len(station_ids)} stations: {list(station_ids)}")

# Create folder for data
folder_name = "aq_data"
os.makedirs(folder_name, exist_ok=True)
print(f"Created/verified folder: {folder_name}")

# Download data for each station
# pollutant_id = 36 for NO2
# pollutant_id = 124 for PM2.5
pollutant_id = 36
date_range = "2022-01-01 to 2024-12-31"

print(f"\nDownloading NO2 data for date range: {date_range}")
print("=" * 60)

for sid in station_ids:
    url = f"https://www.airqualityontario.com/history/searchResults.php?page=CSV&s_categoryId=Academic&s_stationId={sid}&s_pollutantId={pollutant_id}&s_startDate=2022-01-01&s_endDate=2024-12-31&s_reportType=CSV"
    file_path = os.path.join(folder_name, f"station_{sid}_data.csv")
    
    print(f"Downloading station {sid}...", end=" ")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✓ Saved: {file_path}")
    else:
        print(f"✗ Failed (HTTP {response.status_code})")

print("\n" + "=" * 60)
print("Download complete!")
print(f"\nFiles saved in: {os.path.abspath(folder_name)}")
