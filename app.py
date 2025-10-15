import streamlit as st
import pandas as pd
import numpy as np

# File paths
RATINGS_CSV = "ratings.csv"
MOVIES_CSV = "movies.csv"  # Optional: comment this line if you don't have movies.csv

# Load the ratings data
@st.cache_data
def load_ratings():
    return pd.read_csv(RATINGS_CSV)

# Optional: Load the movies data
@st.cache_data
def load_movies():
    try:
        return pd.read_csv(MOVIES_CSV)
    except FileNotFoundError:
        return None

def main():
    st.title("Movie Recommender System")

    ratings = load_ratings()
    movies = load_movies()

    st.header("Dataset Preview")
    st.subheader("Ratings")
    st.write(ratings.head())
    if movies is not None:
        st.subheader("Movies")
        st.write(movies.head())
    else:
        st.info("movies.csv not found. Showing only ratings-based recommendations.")

    st.header("Get Top Rated Movies")

    # Recommend top movies by average rating (with at least 10 ratings)
    if movies is not None:
        merged = pd.merge(ratings, movies, on='movieId')
        movie_stats = merged.groupby('title').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['average_rating', 'rating_count']
        filtered = movie_stats[movie_stats['rating_count'] >= 10]
        top_n = st.slider("How many recommendations?", 5, 30, 10)
        st.write(filtered.sort_values('average_rating', ascending=False).head(top_n))
    else:
        # If only ratings.csv is available
        movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['average_rating', 'rating_count']
        filtered = movie_stats[movie_stats['rating_count'] >= 10]
        top_n = st.slider("How many recommendations?", 5, 30, 10)
        st.write(filtered.sort_values('average_rating', ascending=False).head(top_n))

    st.header("Find Movies by User")
    user_ids = ratings['userId'].unique()
    user_id = st.selectbox("Select userId", user_ids)
    user_ratings = ratings[ratings['userId'] == user_id]
    st.write("Movies rated by user:")
    if movies is not None:
        user_movies = pd.merge(user_ratings, movies, on='movieId')
        st.write(user_movies[['title', 'rating']])
    else:
        st.write(user_ratings[['movieId', 'rating']])

if __name__ == '__main__':
    main()
