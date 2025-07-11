from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset safely
try:
    df = pd.read_csv("movies.csv")  # Ensure the file path is correct
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame(columns=['title', 'genres', 'keywords', 'overview', 'vote_average'])

# Ensure required columns exist
required_columns = {'title', 'genres', 'keywords', 'overview', 'vote_average'}
if not required_columns.issubset(df.columns):
    raise ValueError("Dataset is missing required columns.")

# Fill missing values
df.fillna({'genres': '', 'keywords': '', 'overview': '', 'vote_average': 0}, inplace=True)

# Create a lowercase title column for case-insensitive search
df['title_lower'] = df['title'].str.lower()

# Combine features for similarity
df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(df['combined_features'])

# Function to generate Google search link for the movie
def get_movie_link(title):
    return f"https://www.google.com/search?q={title.replace(' ', '+')}+movie"

# Function to recommend movies
def recommend_movies(movie_name, num_recommendations=10):
    movie_name = movie_name.lower()

    # Find movie index based on title
    matching_movies = df[df['title_lower'] == movie_name]
    if matching_movies.empty:
        return []

    index_of_the_movie = matching_movies.index[0]

    # Compute similarity scores
    similarity_scores = cosine_similarity(feature_vectors[index_of_the_movie], feature_vectors).flatten()

    # Get top recommendations (excluding itself)
    top_indices = similarity_scores.argsort()[::-1][1:num_recommendations+1]
    recommended_movies = df.iloc[top_indices][['title', 'vote_average']].copy()

    # Assign Google search links
    recommended_movies['link'] = recommended_movies['title'].apply(get_movie_link)

    return recommended_movies.rename(columns={'vote_average': 'rating'}).to_dict(orient="records")

# Flask route for recommendations
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    movie_name = request.args.get('movie_name', '').strip()
    if not movie_name:
        return jsonify({"error": "Movie name is required"}), 400

    recommendations = recommend_movies(movie_name)

    if not recommendations:
        return jsonify({"error": "Movie not found or no recommendations available"}), 404

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
