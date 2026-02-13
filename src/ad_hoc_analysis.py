import pandas as pd
import os

# This script performs an ad-hoc analysis to calculate the average traffic count for each day of the week for each town, and saves the results to a CSV file.
# Define the folder containing the datasets
data_folder = 'data/data_clean/'

# Define the folder to save the results
results_folder = 'results/'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Initialize an empty DataFrame to store all results
all_results = pd.DataFrame()

# Loop through each CSV file
for file in csv_files:
    # Load the dataset
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)
    
    # Extract the town name from the file name (e.g., "cleaned_data_townname.csv")
    town_name = file.replace('cleaned_data_', '').replace('.csv', '').replace('_', ' ').title()
    
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add a new column for the day of the week (e.g., "Monday", "Tuesday")
    df['Day_of_Week'] = df['Date'].dt.day_name()
    
    # Group by the day of the week and calculate the average traffic count
    average_traffic_by_day = df.groupby('Day_of_Week')['Traffic_Count'].mean()
    
    # Sort the days of the week in order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    average_traffic_by_day = average_traffic_by_day.reindex(day_order)
    
    # Add the town name as a column
    average_traffic_by_day = average_traffic_by_day.reset_index()
    average_traffic_by_day['Town'] = town_name
    
    # Append to the results DataFrame
    all_results = pd.concat([all_results, average_traffic_by_day], ignore_index=True)

# Save the results to a CSV file in the results folder
output_file = os.path.join(results_folder, 'average_traffic_by_town.csv')
all_results.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")