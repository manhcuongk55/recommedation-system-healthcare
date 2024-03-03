import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the product and user data
df_products = pd.read_csv("data/acie_product.csv")
df_users = pd.read_csv("data/user_profile.csv")

# For simplicity, let's create a user-product interaction matrix with random interactions
import numpy as np

# Create a matrix with random interactions (1 for interaction, 0 for no interaction)
interaction_matrix = np.random.choice([0, 1], size=(len(df_users), len(df_products)), p=[0.8, 0.2])

# Assuming each row corresponds to a user and each column corresponds to a product
df_interactions = pd.DataFrame(interaction_matrix, columns=df_products['_id'].values)

# Calculate item-item similarity (cosine similarity)
item_similarity = cosine_similarity(df_interactions.T)

# Function to get product recommendations for a given user
def get_user_recommendations(user_id, item_similarity=item_similarity, num_recommendations=5):
    user_idx = df_users[df_users['_id_x'] == user_id].index
    if not user_idx.empty:
        user_idx = user_idx[0]
        user_scores = item_similarity[:, user_idx]
        recommended_products = list(df_products['_id'].iloc[np.argsort(user_scores)[::-1][:num_recommendations]])
        return df_products[df_products['_id'].isin(recommended_products)]
    else:
        print(f"User ID {user_id} not found.")
        return None

# Example usage
user_id = '64a8dd1dd6a510dd7e1b3ad1'  # Replace with an actual user ID from your dataset
user_recommendations = get_user_recommendations(user_id)
print(f"Recommendations for user {user_id}:")
print(user_recommendations)
