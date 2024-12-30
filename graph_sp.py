import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
df = pd.read_csv('paired_sound_pedestrian_3.csv')

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Plot the data using seaborn's regplot to include a trendline
sns.regplot(x='Avg_Pedestrian_Count', y='Avg_Decibel', data=df, scatter_kws={'s': 50}, line_kws={'color': 'red', 'lw': 2})

# Set the plot title and labels
plt.title('Average Pedestrian Count vs. Average Decibel Reading', fontsize=16)
plt.xlabel('Average Pedestrian Count', fontsize=14)
plt.ylabel('Average Decibel Reading', fontsize=14)

# Show the plot
plt.show()
