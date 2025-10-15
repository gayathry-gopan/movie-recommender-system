import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title("Dual Movie Recommender System")

# ---- Helper: Try multiple possible file locations ----
def find_file(filename, search_dirs=[".", "ml-latest-small"]):
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None

# ---- Load files with fallback ----
MOVIES_CSV = find_file("movies.csv")
RATINGS_CSV = find_file("ratings.csv")
CONTENT_MODEL = find_file("content_model.pkl")
CF_MODEL = find_file("cf_model.pkl")

if not MOVIES_CSV or not RATINGS_CSV:
    st.error(
        "Cannot find movies.csv and/or ratings.csv. Please upload these files to the root or 'ml-latest-small' folder of your repository."
    )
    st.stop()
if not CONTENT_MODEL or not CF_MODEL:
    st.error(
        "Cannot find content_model.pkl and/or cf_model.pkl. Please upload these model files to the root or 'ml-latest-small' folder of your repository."
    )
    st.stop()

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

movies = load_csv(MOVIES_CSV)
ratings = load_csv(RATINGS_CSV)
content_data = load_pickle(CONTENT_MODEL)
cf_data = load_pickle(CF_MODEL)

tfidf = content_data['tfidf']
cosine_sim = content_data['cosine_sim']
movies_content = content_data['movies']

predicted_ratings_df = cf_data['predicted_ratings_df']
movies_cf = cf_data['movies']

# ------------------ Content-Based Section ------------------
st.header("Section 1: Content-Based Recommendations")

movie_titles = movies_content['title'].drop_duplicates().sort_values().tolist()
selected_movie = st.selectbox("Select a movie for content-based recommendations:", movie_titles)

if selected_movie:
    indices = pd.Series(movies_content.index, index=movies_content['title']).drop_duplicates()
    idx = indices[selected_movie]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Skip itself, get top 5
    movie_indices = [i[0] for i in sim_scores]
    recommended = movies_content['title'].iloc[movie_indices].values

    st.subheader("Top 5 movies similar to:")
    st.write(f"**{selected_movie}**")
    for i, title in enumerate(recommended, 1):
        st.write(f"{i}. {title}")

# ------------------ Collaborative Filtering Section ------------------
st.header("Section 2: Collaborative Filtering Recommendations")

min_user = int(predicted_ratings_df.index.min())
max_user = int(predicted_ratings_df.index.max())
user_id = st.number_input(
    f"Enter a User ID (between {min_user} and {max_user}):",
    min_value=min_user,
    max_value=max_user,
    step=1
)

if user_id:
    user_id = int(user_id)
    if user_id in predicted_ratings_df.index:
        user_row = predicted_ratings_df.loc[user_id]
        rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'])
        recs = user_row.drop(labels=rated_movies, errors='ignore').sort_values(ascending=False).head(5)
        recommended_movie_ids = recs.index
        recommended_titles = movies_cf[movies_cf['movieId'].isin(recommended_movie_ids)]['title'].values

        st.subheader(f"Top 5 movie recommendations for User {user_id}:")
        for i, title in enumerate(recommended_titles, 1):
            st.write(f"{i}. {title}")
    else:
        st.warning("User ID not found in the ratings data. Please try another ID.")

st.markdown("---")
st.caption("Content-based: TF-IDF on genres; Collaborative Filtering: Matrix factorization (SVD)")
