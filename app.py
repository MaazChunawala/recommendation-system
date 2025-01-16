import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the MovieLens and Amazon datasets
movie_df = pd.read_csv("/Users/alichunawala/Downloads/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
amazon_df = pd.read_csv("/Users/alichunawala/Downloads/Amazon Product Reviews/Reviews.csv")

# Collaborative Filtering using Surprise
def collaborative_filtering(user_id, item_id):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(movie_df[["user_id", "item_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    predicted_rating = model.predict(user_id, item_id).est
    return predicted_rating

# Content-Based Filtering using Product Descriptions
def content_based_filtering(product_idx):
    product_descriptions = amazon_df['product_description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_descriptions)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar products
    product_indices = [i[0] for i in sim_scores]
    return amazon_df['product_name'].iloc[product_indices]

# Streamlit UI
st.title('E-Commerce Product Recommendation System')

# Add custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
        body {
            font-family: 'Roboto', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.title("Recommendation Options")
recommendation_type = st.sidebar.radio(
    "Choose Recommendation Type:",
    ("Collaborative Filtering", "Content-Based Filtering")
)

# Input for Collaborative Filtering
if recommendation_type == "Collaborative Filtering":
    st.sidebar.subheader("Collaborative Filtering Inputs")
    user_id = st.sidebar.number_input("Enter user ID for Collaborative Filtering", min_value=1, step=1)
    item_id = st.sidebar.number_input("Enter item ID for Collaborative Filtering", min_value=1, step=1)

# Input for Content-Based Filtering
if recommendation_type == "Content-Based Filtering":
    st.sidebar.subheader("Content-Based Filtering Inputs")
    product_idx = st.sidebar.number_input("Enter product index for Content-Based Filtering", min_value=0, step=1)

# Show recommendations based on selected option
if recommendation_type == "Collaborative Filtering":
    st.subheader("Collaborative Filtering Recommendations")
    if user_id and item_id:
        predicted_rating = collaborative_filtering(user_id, item_id)
        st.write(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating}")

elif recommendation_type == "Content-Based Filtering":
    st.subheader("Content-Based Filtering Recommendations")
    if product_idx:
        recommendations = content_based_filtering(product_idx)
        st.write("Recommended Products:")
        st.write(recommendations)

# Use columns to organize the layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Recommendations")
    if recommendation_type == "Collaborative Filtering" and user_id and item_id:
        predicted_rating = collaborative_filtering(user_id, item_id)
        st.write(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating}")
    elif recommendation_type == "Content-Based Filtering" and product_idx:
        recommendations = content_based_filtering(product_idx)
        st.write("Recommended Products:")
        st.write(recommendations)

with col2:
    st.subheader("Product Details")
    st.markdown("Here, you can display additional details about the recommended products.")