import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the product and user data
df_products = pd.read_csv("data/acie_product.csv")
df_users = pd.read_csv("data/skincare.csv")

# For simplicity, let's create a user-product interaction matrix with random interactions
import numpy as np

# Create a matrix with random interactions (1 for interaction, 0 for no interaction)
interaction_matrix = np.random.choice([0, 1], size=(len(df_users), len(df_products)), p=[0.8, 0.2])

# Assuming each row corresponds to a user and each column corresponds to a product
df_interactions = pd.DataFrame(interaction_matrix, columns=df_products['_id'].values)

# Calculate item-item similarity (cosine similarity)
item_similarity = cosine_similarity(df_interactions.T)

# Function to get product recommendations for a given product
def get_product_recommendations(product_id, item_similarity=item_similarity, num_recommendations=5):
    product_idx = df_products[df_products['_id'] == product_id].index
    if not product_idx.empty:
        product_idx = product_idx[0]
        similar_scores = item_similarity[product_idx]
        similar_products = list(df_products['_id'].iloc[np.argsort(similar_scores)[::-1][1:num_recommendations + 1]])
        return df_products[df_products['_id'].isin(similar_products)]
    else:
        print(f"Product ID {product_id} not found.")
        return None


# Example usage
product_id = '60e9f52eb5363725ef87e1b4'  # Replace with an actual product ID from your dataset
recommendations = get_product_recommendations(product_id)
print(f"Recommendations for product {product_id}:")
print(recommendations)
