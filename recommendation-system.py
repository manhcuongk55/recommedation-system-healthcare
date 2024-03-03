import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load user-product interaction data
df_users = pd.read_csv("acie_user.csv")
df_ratings = pd.read_csv("acie_ratings.csv")

# Merge data to get user-product interactions
df_interactions = pd.merge(df_ratings, df_users, on='_id')

# Load data into Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_interactions[['_id', 'product_id', 'rating']], reader)

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
    user_products = df_interactions[df_interactions['_id'] == user_id]['product_id']
    all_products = df_interactions['product_id'].unique()
    products_to_predict = list(set(all_products) - set(user_products))

    # Make predictions for the user on the products not interacted with
    user_predictions = [model.predict(user_id, prod) for prod in products_to_predict]

    # Sort predictions by estimated rating
    user_predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_n_recommendations = user_predictions[:n]

    # Extract product IDs from recommendations
    recommended_product_ids = [int(pred.iid) for pred in top_n_recommendations]

    # Get product names from product IDs
    recommended_products = df_products[df_products['_id'].isin(recommended_product_ids)]['productInfo.name']

    return recommended_products

# Example usage
user_id = '649da2e0fadf9ea81aa9ff9a'  # Replace with the desired user ID
user_recommendations = get_user_recommendations(user_id)
print(f"Top recommendations for user {user_id}:")
print(user_recommendations)
