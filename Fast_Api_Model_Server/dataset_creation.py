import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

custom_stopwords = [
    "in", "on", "at", "to", "from",
    "and", "or", "but",
    "the", "a", "an",
    "of", "for", "with",
    "by", "as", "is", "are", "be"
]

# Load your CSV
df = pd.read_csv("data/PROMISE_exp.csv")

# Extract your texts and classes
texts = df["RequirementText"].astype(str)
labels = df["_class_"]          # or whatever your label column is called

# Build TF-IDF matrix
vectorizer = TfidfVectorizer(
    min_df=5,      # ignore words appearing in < 5 requirements
    max_df=0.90,   # ignore words appearing in > 90% requirements
    stop_words=custom_stopwords
)
tfidf_matrix = vectorizer.fit_transform(texts)

# Convert TF-IDF to DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Add RequirementText and Class
tfidf_df.insert(0, "RequirementText", texts)
tfidf_df["Class"] = labels

# Save the full structured CSV
tfidf_df.to_csv("data/requirements_tfidf_full_dataset.csv", index=False)

print("Done! CSV saved as requirements_tfidf_full_dataset.csv")

