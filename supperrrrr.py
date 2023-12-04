import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

data = pd.read_csv("/Users/chira/Desktop/desktop/projects/Food recommender system/cuisines.csv")

df = data.copy()
df = df.dropna()

# ... (Your data preprocessing steps)
df['ingredients'].replace('\t', '', regex=True, inplace=True)
df['ingredients'].replace('\n', '', regex=True, inplace=True)
df['ingredients'].replace('[0-9]+', '', regex=True, inplace=True)
df['ingredients'].replace('[^\w\s]', '', regex=True, inplace=True)
df['ingredients'].replace('cup', '', regex=True, inplace=True)
df['ingredients'].replace('teaspoon', '', regex=True, inplace=True)
df['ingredients'].replace('pinch', '', regex=True, inplace=True)


def separate_words_by_commas(text):
    words = text.split()
    return ', '.join(words)
df['ingredients'] = df['ingredients'].apply(separate_words_by_commas)
df['tags'] = df['instructions'].apply(separate_words_by_commas)
df['upvotes'] = 0


def increment(recipe):
    row_index = df[df['name'] == recipe].index[0]
    df.loc[row_index, 'upvotes'] += 1
    print(f'Upvoted {recipe}')


df['prep_time'].replace('Total in', '', regex=True, inplace=True)
df['prep_time'].replace('M', '', regex=True, inplace=True)


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['ingredients'])

# Save the TF-IDF vectorizer for future use during inference
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

user_input = input("Enter your preferred ingredients (separated by commas): ")
user_input = user_input.split(',')
user_input_vector = tfidf_vectorizer.transform([' '.join(user_input)])
similarities = cosine_similarity(user_input_vector, tfidf_matrix)
top_n = 10
top_indices = similarities.argsort()[0, ::-1][:top_n]
print("Recommended Recipes:")
result_1 = []
for idx in top_indices:
    print(df['name'].iloc[idx])
    result_1.append(df['name'].iloc[idx])

cv = CountVectorizer(max_features=50000)
vectors = cv.fit_transform(df['ingredients']).toarray()
similarity = cosine_similarity(vectors)

# Save the CountVectorizer and similarity matrix for future use during inference
joblib.dump(cv, 'count_vectorizer.pkl')
joblib.dump(similarity, 'cosine_similarity_matrix.pkl')


def recommend(recipe):
    # Load the CountVectorizer and similarity matrix used during training
    cv = joblib.load('count_vectorizer.pkl')
    similarity = joblib.load('cosine_similarity_matrix.pkl')

    if not df[df['name'] == recipe].empty:
        index = df[df['name'] == recipe].index[0]
        distances = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)
        similar_recipes = []
        for i, _ in distances[1:10]:
            similar_recipes.append(df.iloc[i]['name'])
        return similar_recipes
    else:
        return f"Recipe '{recipe}' not found in the dataset."


recommendation_result = recommend('Homemade Rice Puttu Recipe | Kerala Matta Rice or Basmati Rice')
print(recommendation_result)

# Save the recommendation function for future use during inference
joblib.dump(recommend, 'recommendation_function.pkl')


