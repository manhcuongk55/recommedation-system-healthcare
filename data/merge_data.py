import pandas as pd

# Read the CSV files into pandas DataFrames
user_df = pd.read_csv("user.csv")
skin_profile_df = pd.read_csv("skin_profile.csv")

# Merge the DataFrames using the 'user' column from skin_profile_df and '_id' column from user_df
merged_df = pd.merge(user_df, skin_profile_df, left_on='_id', right_on='user', how='inner')

# Save the merged DataFrame to a CSV file
merged_df.to_csv("user_profile.csv", index=False)
