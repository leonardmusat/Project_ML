import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# === Load matrices ===
X_train = np.load("data/matrix/X_train.npy")
X_test = np.load("data/matrix/X_test.npy")
y_train = np.load("data/matrix/y_train.npy", allow_pickle=True)
y_test = np.load("data/matrix/y_test.npy", allow_pickle=True)
X_train_all = np.load("data/matrix/X_total.npy")
y_train_all = np.load("data/matrix/y_total.npy", allow_pickle=True)

# === Load TF-IDF vectorizer ===
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === Load encoder ===
encoder = joblib.load("models/label_encoder.pkl")

print("Data loaded successfully.")

# === Train SVM model ===
model = LinearSVC()
model.fit(X_train, y_train)

print("Model trained: 80% of data used for training.")

# === Evaluate model ===
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification report:")
print(classification_report(y_test, preds))

# === Save the classifier ===
joblib.dump(model, "models/svm_classifier.pkl")
print("Model saved as svm_classifier.pkl")

# === Train SVM model on 100% data ===
model_all = LinearSVC()
model_all.fit(X_train_all, y_train_all)
print("Model trained: 100% of data used for training.")

joblib.dump(model_all, "models/svm_classifier_all.pkl")
print("Model saved as svm_classifier.pkl")

