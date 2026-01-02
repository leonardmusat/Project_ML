import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib, os

custom_stopwords = [
    "in","on","at","to","from","and","or","but","the","a","an",
    "of","for","with","by","as","is","are","be"
]

df = pd.read_csv("data/new_dataset.csv")
texts = df["RequirementText"].astype(str)
labels = df["_class_"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    stop_words=custom_stopwords
)

X_tfidf = vectorizer.fit_transform(texts)   # sparse matrix

# --- Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer_ngram_range.pkl")

# --- Convert reduced output to DataFrame
reduced_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

reduced_tfidf_df.insert(0, "RequirementText", texts)
reduced_tfidf_df["Class"] = labels

reduced_tfidf_df.to_csv("data/requirements_tfidf_ngram_range_dataset.csv", index=False)
print("Done! CSV saved as requirements_tfidf_ngram_range_dataset.csv")