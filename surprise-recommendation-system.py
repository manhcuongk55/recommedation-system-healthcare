import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load user interactions data
df_users = pd.read_csv("data/skincare.csv")
df_products = pd.read_csv("data/acie_product.csv")

# Merge data to get user-product interactions
df_interactions = pd.merge(df_users, df_products, left_on='_id', right_on='_id')

# If you have explicit ratings, use this block
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df_interactions[['user', 'product_id', 'your_rating_column']], reader)

# If you don't have explicit ratings, use this block
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_interactions[['user', '_id']], reader)

# Check if there are enough ratings for the specified test_size
if len(df_interactions) <= 1:
    raise ValueError("Not enough ratings for train-test split. Ensure there are enough interactions.")

# Split the data into training and testing sets
test_size = min(0.2, len(df_interactions) - 1)  # Set a maximum test_size to avoid the error
trainset, testset = train_test_split(data, test_size=test_size, random_state=42)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use a basic collaborative filtering algorithm (User-Based Collaborative Filtering)
sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
print("RMSE on the test set:", rmse(predictions))

# Function to get product recommendations for a user
def get_user_recommendations(user_id, n=5):
    # Get products that the user has not interacted with
    user_products = df_products[~df_products['_id'].isin(df_interactions[df_interactions['user'] == user_id]['product_id'])]['_id']
    
    # Make predictions for the user on the products not interacted with
    user_predictions = [model.predict(user_id, prod) for prod in user_products]

    # Sort predictions by estimated rating
    user_predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_n_recommendations = user_predictions[:n]

    # Extract product IDs from recommendations
    recommended_product_ids = [int(pred.iid) for pred in top_n_recommendations]

    # Get product information from acie_product.csv
    recommended_products = df_products[df_products['_id'].isin(recommended_product_ids)]

    return recommended_products

# Example usage
user_id = 'your_user_id'  # Replace with the desired user ID
user_recommendations = get_user_recommendations(user_id)
print(f"Top recommendations for user {user_id}:")
print(user_recommendations)
