import pandas as pd
import numpy as np

# File path
file_path = 'data/user.csv'

# Read CSV into a Pandas DataFrame
df = pd.read_csv(file_path)

# Convert DataFrame to NumPy array
numpy_array = df.to_numpy()

print(numpy_array)
