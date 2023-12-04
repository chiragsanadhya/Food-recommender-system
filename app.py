from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data outside of the route functions for better performance
data = pd.read_csv("cuisines.csv")
df = data.dropna()

# Load the TF-IDF vectorizer during app initialization
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['ingredients'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def make_recommendation():
    print("Request data:", request.data)
    print("Content type:", request.content_type)

    if request.method == 'POST' and request.is_json:
        user_input = request.get_json().get("ingredients", "")
        user_input = user_input.split(',')

        # Use the loaded components to make recommendations
        user_input_vector = tfidf_vectorizer.transform([' '.join(user_input)])
        similarities = cosine_similarity(user_input_vector, tfidf_matrix)
        top_n = 10
        top_indices = similarities.argsort()[0, ::-1][:top_n]

        # Assuming df is accessible or loaded within this scope
        recommendations = [df['name'].iloc[idx] for idx in top_indices]

        # Return a JSON response with recommendations
        return jsonify({"recommendations": recommendations})

    # Return an error response if the request is not valid
    return jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(debug=True)
