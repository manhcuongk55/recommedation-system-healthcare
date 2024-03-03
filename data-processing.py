import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


# Load the data from the CSV file
file_path = 'data/skintype_v3.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand the structure
print(df.head())

# Data processing - Convert 'dateTime' column to datetime format
df['dateTime'] = pd.to_datetime(df['dateTime'])
df['date'] = pd.to_datetime(df['date'])

# Data visualization - Example: Line plot of 'Hydration' over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='dateTime', y='Hydration', data=df, marker='o', label='Hydration')
plt.title('Hydration Over Time')
plt.xlabel('Date and Time')
plt.ylabel('Hydration Level')
plt.xticks(rotation=45)
plt.legend()

# Create a folder named 'visualizations' if it doesn't exist
output_folder = 'visualizations'
os.makedirs(output_folder, exist_ok=True)

# Save the image to the 'visualizations' folder
output_path = os.path.join(output_folder, 'hydration_over_time.png')
plt.savefig(output_path)

plt.show()


