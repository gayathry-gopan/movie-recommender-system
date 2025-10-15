# MovieLens Content-Based and Collaborative Filtering Model Trainer & Saver

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import pickle

# 1. Download MovieLens dataset if not present
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_ZIP = "ml-latest-small.zip"
DATASET_DIR = "ml-latest-small"

if not os.path.exists(DATASET_ZIP):
    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)
if not os.path.exists(DATASET_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")

# 2. Load data
movies = pd.read_csv(f"{DATASET_DIR}/movies.csv")
ratings = pd.read_csv(f"{DATASET_DIR}/ratings.csv")

# 3. Content-Based: Create 'content soup' for each movie
movies['content_soup'] = movies['genres'].apply(lambda x: ' '.join(x.lower().replace('-', '').split('|')))

# 4. Vectorize using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content_soup'])

# 5. Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 6. Save content-based artifacts
with open('content_model.pkl', 'wb') as f:
    pickle.dump({'tfidf': tfidf, 'cosine_sim': cosine_sim, 'movies': movies}, f)
print("Saved content-based model to content_model.pkl")

# 7. Collaborative Filtering (SVD via numpy/scipy)
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
R_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, VT = svds(R_demeaned, k=20)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), VT) + user_ratings_mean.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# 8. Save collaborative filtering artifacts
with open('cf_model.pkl', 'wb') as f:
    pickle.dump({'predicted_ratings_df': predicted_ratings_df, 'movies': movies}, f)
print("Saved collaborative filtering model to cf_model.pkl")

# OPTIONAL: Print sample recommendations
def get_recommendations(title, top_n=5):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return f"Movie '{title}' not found in database."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

def recommend_movies(pred_ratings_df, original_ratings, movies_df, user_id, n=5):
    user_row = pred_ratings_df.loc[user_id]
    rated_movies = set(original_ratings[original_ratings['userId']==user_id]['movieId'])
    recommendations = user_row.drop(labels=rated_movies).sort_values(ascending=False).head(n)
    return movies_df[movies_df['movieId'].isin(recommendations.index)][['movieId','title']]

print("\nTop 5 movies similar to 'Toy Story (1995)':")
print(get_recommendations('Toy Story (1995)'))

print("\nTop 5 recommendations for user 1:")
print(recommend_movies(predicted_ratings_df, ratings, movies, user_id=1, n=5))