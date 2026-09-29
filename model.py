import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("movies.csv")

# Combine title and description
data["text"] = data["title"] + " " + data["description"]

# Create ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(data["text"], data["genre"])

# Save model
joblib.dump(model, "movie_genre_model.pkl")

print("Model trained successfully!")