import os
import requests
import pandas as pd

def collect_air_quality():
    # --- Configuration ---
    # Paths relative to the project root
    STATION_INFO_PATH = "air_quality_dataset/station_info.csv"
    OUTPUT_FOLDER = "air_quality_dataset/aq_data"
    
    # Paraeters of interest
    TARGET_STATIONS = ['TORONTO DOWNTOWN', 'TORONTO EAST', 'TORONTO NORTH', 'TORONTO WEST']
    POLLUTANT_ID = 36  # 36 = NO2
    YEAR = 2024

    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- Starting Air Quality Collection ---")

    # --- Step 1: Get Target Station IDs ---
    # We use pandas just to look up the IDs from your metadata file
    if not os.path.exists(STATION_INFO_PATH):
        raise FileNotFoundError(f"Could not find {STATION_INFO_PATH}. Are you running from the project root?")

    station_info = pd.read_csv(STATION_INFO_PATH)
    
    # Filter for specific stations and year
    mask = (station_info['STATION NAME'].isin(TARGET_STATIONS)) & (station_info['Year'] == YEAR)
    station_ids = station_info.loc[mask, "ID"].unique()
    
    print(f"Found {len(station_ids)} stations: {station_ids}")

    # --- Step 2: Download Data for Each Station ---
    for sid in station_ids:
        # Construct the URL
        url = (
            f"https://www.airqualityontario.com/history/searchResults.php?page=CSV"
            f"&s_categoryId=Academic&s_stationId={sid}&s_pollutantId={POLLUTANT_ID}"
            f"&s_startDate=2022-01-01&s_endDate=2024-12-31&s_reportType=CSV"
        )
        
        file_path = os.path.join(OUTPUT_FOLDER, f"station_{sid}_data.csv")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {file_path}")
            else:
                print(f"Failed to download Station {sid} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading Station {sid}: {e}")

    print("--- Download Complete ---")

if __name__ == "__main__":
    collect_air_quality()