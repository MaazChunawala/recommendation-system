import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Amazon dataset (adjust path if necessary)
df = pd.read_csv("/Users/alichunawala/Downloads/Amazon Product Reviews/Reviews.csv")

# Assuming that the 'product_description' column exists
# You can adjust column names based on your dataset
product_descriptions = df['product_description'].fillna('')

# Convert text descriptions into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(product_descriptions)

# Calculate cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get similar products based on a product index
def get_recommendations(product_idx, cosine_sim=cosine_sim):
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar products
    product_indices = [i[0] for i in sim_scores]
    return df['product_name'].iloc[product_indices]

# Test: Get recommendations for a specific product
product_index = 10  # Adjust this index as needed
recommended_products = get_recommendations(product_index)
print("Recommended Products:")
print(recommended_products)