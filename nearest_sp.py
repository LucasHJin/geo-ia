import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import re

# Load the data (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('sound_pedestrian_3.csv')
df.columns = df.columns.str.strip()  # Strip any extra spaces from column names

# Clean function to remove non-numeric characters (for decibels and pedestrian counts)
def clean_value(value):
    if isinstance(value, str):
        # Remove non-numeric characters (like $, commas, etc.)
        value = re.sub(r'[^\d.]', '', value)
    try:
        return float(value) if value else np.nan
    except ValueError:
        return np.nan

# Clean necessary columns (Decibel readings and Pedestrian counts)
df['Decibel reading #1 (maximum)'] = df['Decibel reading #1 (maximum)'].apply(clean_value)
df['Decibel reading #1 (minimum)'] = df['Decibel reading #1 (minimum)'].apply(clean_value)
df['Decibel reading #2 (maximum)'] = df['Decibel reading #2 (maximum)'].apply(clean_value)
df['Decibel reading #2 (minimum)'] = df['Decibel reading #2 (minimum)'].apply(clean_value)

df['Pedestrian count #1'] = df['Pedestrian count #1'].apply(clean_value)
df['Pedestrian count #2'] = df['Pedestrian count #2'].apply(clean_value)


# Filter rows with valid pedestrian counts and decibel readings
pedestrian_data = df[['ObjectID', 'GlobalID', 'Pedestrian count #1', 'Pedestrian count #2', 'x', 'y']].dropna(subset=['Pedestrian count #1', 'Pedestrian count #2'])
decibel_data = df[['ObjectID', 'GlobalID', 'Decibel reading #1 (maximum)', 'Decibel reading #1 (minimum)', 'Decibel reading #2 (maximum)', 'Decibel reading #2 (minimum)', 'x', 'y']].dropna(subset=['Decibel reading #1 (maximum)', 'Decibel reading #1 (minimum)', 'Decibel reading #2 (maximum)', 'Decibel reading #2 (minimum)'])


# Calculate distances between pedestrian and decibel locations
pedestrian_coords = pedestrian_data[['x', 'y']].values
decibel_coords = decibel_data[['x', 'y']].values
distances = cdist(pedestrian_coords, decibel_coords, metric='euclidean')

# Initialize a list to store the results
result = []

# Pair up the closest decibel and pedestrian rows
for i in range(len(pedestrian_data)):
    closest_decibel_index = np.argmin(distances[i, :])  # Find the nearest decibel row
    decibel_row = decibel_data.iloc[closest_decibel_index]
    
    # Calculate the distance between the paired rows
    distance = distances[i, closest_decibel_index]

    # Calculate the average pedestrian count values
    avg_pedestrian_count = np.nanmean([pedestrian_data.iloc[i]['Pedestrian count #1'], pedestrian_data.iloc[i]['Pedestrian count #2']])
    
    # Calculate the average of all four decibel readings (max and min for both readings)
    avg_decibel = np.nanmean([
        decibel_row['Decibel reading #1 (maximum)'], 
        decibel_row['Decibel reading #1 (minimum)'], 
        decibel_row['Decibel reading #2 (maximum)'], 
        decibel_row['Decibel reading #2 (minimum)']
    ])

    # Append the results for this pair
    result.append({
        'ObjectID_pedestrian': pedestrian_data.iloc[i]['ObjectID'],
        'GlobalID_pedestrian': pedestrian_data.iloc[i]['GlobalID'],
        'Avg_Pedestrian_Count': avg_pedestrian_count,
        'ObjectID_decibel': decibel_row['ObjectID'],
        'GlobalID_decibel': decibel_row['GlobalID'],
        'Avg_Decibel': avg_decibel,  # Use the combined average of decibel readings
        'Distance': distance
    })


# Convert the result to a DataFrame
result_df = pd.DataFrame(result)

# Save the result to a new CSV file
result_df.to_csv('paired_sound_pedestrian_3.csv', index=False)

print("Processing complete! The paired results with averaged values have been saved to 'paired_pedestrian_decibel.csv'.")
