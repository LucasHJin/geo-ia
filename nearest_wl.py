import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import re

# Load the data (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('water_litter_3.csv')
df.columns = df.columns.str.strip()  # Strip any extra spaces from column names

# Function to clean and convert the cost/litter data into numeric values
def clean_value(value):
    if isinstance(value, str):
        # Remove any non-numeric characters (like $, commas, etc.)
        value = re.sub(r'[^\d.]', '', value)
    try:
        return float(value) if value else np.nan
    except ValueError:
        return np.nan

# Clean the 'Cost of a common good - price' column
df['Cost of a common good - price'] = df['Cost of a common good - price'].apply(clean_value)

# Clean the 'Environmental quality index - litter' column
df['Environmental quality index - litter'] = df['Environmental quality index - litter'].apply(clean_value)

# Filter rows with a value in the 'Cost of a common good - price' or 'Environmental quality index - litter' columns
common_goods = df[df['Cost of a common good - price'].notna()][['ObjectID', 'GlobalID', 'Cost of a common good - price', 'x', 'y']]
litter = df[df['Environmental quality index - litter'].notna()][['ObjectID', 'GlobalID', 'Environmental quality index - litter', 'x', 'y']]

# Calculate distances between common goods and litter locations
common_goods_coords = common_goods[['x', 'y']].values
litter_coords = litter[['x', 'y']].values

# Compute the Euclidean distance matrix
distances = cdist(common_goods_coords, litter_coords, metric='euclidean')

# Initialize lists to store results
result = []

# Ensure the loop only iterates within the bounds of the common_goods DataFrame
for i in range(len(common_goods)):
    # Find the index of the closest litter row (this gives us the correct index for the second category)
    closest_litter_index = np.argmin(distances[i, :])  # Ensure we're using the right dimension

    # Get the corresponding litter row
    litter_row = litter.iloc[closest_litter_index]
    
    # Get the distance between the two points
    distance = distances[i, closest_litter_index]
    
    # Append the result to the list
    result.append({
        'ObjectID_common_goods': common_goods.iloc[i]['ObjectID'],
        'GlobalID_common_goods': common_goods.iloc[i]['GlobalID'],
        'Cost_of_common_good_price': common_goods.iloc[i]['Cost of a common good - price'],
        'ObjectID_litter': litter_row['ObjectID'],
        'GlobalID_litter': litter_row['GlobalID'],
        'Environmental_quality_index_litter': litter_row['Environmental quality index - litter'],
        'Distance': distance
    })

# Convert the result to a DataFrame
result_df = pd.DataFrame(result)

# Save the result to a new CSV file
result_df.to_csv('paired_water_litter_3.csv', index=False)

print("Processing complete! The paired results have been saved to 'paired_water_litter_1.csv'.")
