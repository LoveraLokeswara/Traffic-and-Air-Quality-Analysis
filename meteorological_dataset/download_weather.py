import pandas as pd
import requests
from io import StringIO
import os

# Configuration
STATIONS = {
    48549: 'Toronto_City_Centre_Downtown',
    26953: 'Toronto_York_North',
    31688: 'Toronto_City_Ontario_East',
    51459: 'Toronto_Pearson_Intl_West'  
}

START_YEAR = 2022
END_YEAR = 2026

def get_station_data(station_id, station_name, start_year, end_year):
    base_url = 'https://climate.weather.gc.ca/climate_data/bulk_data_e.html'
    frames = []

    print(f'Fetching {station_name} (ID: {station_id})...')

    for year in range(start_year, end_year + 1):
        params = {
            'format': 'csv',
            'stationID': station_id,
            'Year': year,
            'Month': 1,
            'Day': 1,
            'timeframe': 2,
            'submit': 'Download+Data'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            if 'Date/Time' not in response.text[:300]:
                print(f'  - No data found for {year}')
                continue

            csv_data = StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_data)
            
            df['Station ID'] = station_id
            df['Station Name'] = station_name
            
            frames.append(df)
            print(f'  - Downloaded {year}')

        except Exception as e:
            print(f'  - Error on {year}: {e}')

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

output_folder = 'weather_data'
os.makedirs(output_folder, exist_ok=True)

for stn_id, stn_name in STATIONS.items():
    df = get_station_data(stn_id, stn_name, START_YEAR, END_YEAR)
    
    if not df.empty:
        df.columns = [c.replace('"', '').strip() for c in df.columns]
        filename = f'{output_folder}/{stn_name}.csv'
        df.to_csv(filename, index=False)
        print(f'-> SAVED: {filename}\n')
    else:
        print(f'-> SKIPPED: {stn_name} (No data found)\n')

print('All downloads complete.')
