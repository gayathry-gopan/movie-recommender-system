import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---- Load DataFrames ----
MOVIELENS_DIR = "ml-latest-small"
MOVIES_CSV = os.path.join(MOVIELENS_DIR, "movies.csv")
RATINGS_CSV = os.path.join(MOVIELENS_DIR, "ratings.csv")

ratings = pd.read_csv(RATINGS_CSV)
movies = pd.read_csv(MOVIES_CSV)

# ---- Load Model Artifacts ----
with open("content_model.pkl", "rb") as f:
    content_data = pickle.load(f)
tfidf = content_data['tfidf']
cosine_sim = content_data['cosine_sim']
movies_content = content_data['movies']

with open("cf_model.pkl", "rb") as f:
    cf_data = pickle.load(f)
predicted_ratings_df = cf_data['predicted_ratings_df']
movies_cf = cf_data['movies']

st.title("Hybrid Movie Recommender System")

# ------------------ Hybrid Recommendation Function ------------------
def get_content_recommendations(title, top_n=10):
    indices = pd.Series(movies_content.index, index=movies_content['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip itself
    movie_indices = [i[0] for i in sim_scores]
    return movies_content['movieId'].iloc[movie_indices].tolist()

def get_cf_recommendations(user_id, top_n=10):
    if user_id not in predicted_ratings_df.index:
        return []
    user_row = predicted_ratings_df.loc[user_id]
    rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'])
    recommendations = user_row.drop(labels=rated_movies).sort_values(ascending=False).head(top_n)
    return recommendations.index.tolist()

def get_hybrid_recommendations(user_id, fav_movie, final_n=10):
    cf_list = get_cf_recommendations(user_id, top_n=final_n)
    content_list = get_content_recommendations(fav_movie, top_n=final_n)
    combined = cf_list + content_list
    # Remove duplicates while preserving order
    seen = set()
    hybrid = []
    for mid in combined:
        if mid not in seen:
            hybrid.append(mid)
            seen.add(mid)
        if len(hybrid) == final_n:
            break
    # Get titles for display
    hybrid_titles = movies[movies['movieId'].isin(hybrid)][['movieId', 'title']]
    # To preserve order in hybrid list
    hybrid_titles = hybrid_titles.set_index('movieId').loc[hybrid].reset_index()
    return hybrid_titles

# ------------------ Streamlit UI ------------------
st.header("Hybrid Recommendation Feature")

# Movie selection
movie_titles = movies_content['title'].drop_duplicates().sort_values().tolist()
selected_movie = st.selectbox("Select your favorite movie:", movie_titles)

# User ID input
min_user = int(predicted_ratings_df.index.min())
max_user = int(predicted_ratings_df.index.max())
user_id = st.number_input(f"Enter your User ID (between {min_user} and {max_user}):", min_value=min_user, max_value=max_user, step=1)

if st.button("Show Hybrid Recommendations"):
    if selected_movie and user_id:
        hybrid_recs = get_hybrid_recommendations(int(user_id), selected_movie, final_n=10)
        if not hybrid_recs.empty:
            st.subheader(f"Top 10 Hybrid Recommendations for User {user_id} and '{selected_movie}':")
            for i, row in hybrid_recs.iterrows():
                st.write(f"{i+1}. {row['title']}")
        else:
            st.warning("No recommendations found. Please check your inputs.")
    else:
        st.warning("Please select a movie and enter a valid user ID.")

st.markdown("---")
st.caption("Hybrid: Combines top 10 collaborative filtering and top 10 content-based, removes duplicates, and re-ranks.")