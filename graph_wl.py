import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed CSV file (replace 'paired_water_litter_1.csv' with your actual file path)
df = pd.read_csv('paired_water_litter_1.csv')

# Plotting the relationship between common good price and litter with a trendline
plt.figure(figsize=(8, 6))

# Use seaborn's regplot to create the scatter plot with a trendline
sns.regplot(x='Cost_of_common_good_price', y='Environmental_quality_index_litter', data=df, scatter_kws={'alpha':0.7}, line_kws={'color':'red', 'lw':2})

plt.title('Common Good Price vs Litter with Trendline')
plt.xlabel('Cost of a Common Good - Price')
plt.ylabel('Environmental Quality Index - Litter')
plt.grid(True)
plt.show()
