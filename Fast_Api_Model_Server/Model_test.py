import joblib

# === Load TF-IDF vectorizer ===
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === Load encoder ===
encoder = joblib.load("models/label_encoder.pkl")

# === Load model ===
model = joblib.load("models/svm_classifier.pkl")
model_all = joblib.load("models/svm_classifier_all.pkl")

# === Predict on a new requirement using the model tested on 80% from dataset ===
new_req = "The user interface shall respond to touch inputs within 150 milliseconds."
new_vec = vectorizer.transform([new_req])
new_pred = model.predict(new_vec)
label = encoder.inverse_transform(new_pred)

print("\nPrediction for:")
print(new_req)
print("->", label[0])

# === Predict on a new requirement using the model tested on 100% from dataset ===
new_req = "The user interface shall respond to touch inputs within 150 milliseconds."
new_vec = vectorizer.transform([new_req])
new_pred = model.predict(new_vec)
label = encoder.inverse_transform(new_pred)

print("\nPrediction for:")
print(new_req)
print("->", label[0])
