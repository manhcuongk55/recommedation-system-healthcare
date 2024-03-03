import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the skincare product data
df_products = pd.read_csv("acie_product.csv")

# Preprocess the data
df_products['productInfo.categories'] = df_products['productInfo.categories'].apply(eval)
df_products['productInfo.name'] = df_products['productInfo.name'].str.lower()

# Combine relevant information into a single column for TF-IDF
df_products['combined_info'] = df_products['productInfo.categories'] + ' ' + df_products['productInfo.name'] + ' ' + df_products['productOverview.notableEffectsAndIngredients']

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(df_products['combined_info'])

# Calculate the cosine similarity between products
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on product name
def get_recommendations(product_name, cosine_sim=cosine_sim):
    idx = df_products.index[df_products['productInfo.name'] == product_name.lower()].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get the top 5 similar products (excluding the product itself)
    product_indices = [i[0] for i in sim_scores]
    return df_products['productInfo.name'].iloc[product_indices]

# Example usage
product_name = 'Soothing Relief Moisture Cream, Fragrance Free'
recommendations = get_recommendations(product_name)
print(f"Recommendations for {product_name}:")
print(recommendations)
