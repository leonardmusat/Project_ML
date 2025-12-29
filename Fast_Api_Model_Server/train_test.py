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
X_train_nonfunc = np.load("data/matrix/X_train_nonfunc.npy")
X_test_nonfunc = np.load("data/matrix/X_test_nonfunc.npy")
y_train_nonfunc = np.load("data/matrix/y_train_nonfunc.npy", allow_pickle=True)
y_test_nonfunc = np.load("data/matrix/y_test_nonfunc.npy", allow_pickle=True)

# === Load TF-IDF vectorizer ===
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === Load svd ===
svd = joblib.load("models/tfidf_svd_100.pkl")

# === Load encoder ===
binary_encoder = joblib.load("models/binary_encoder.pkl")
nonfunctional_encoder = joblib.load("models/nonfunctional_encoder.pkl")

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
print(binary_encoder.classes_)

# === Save the classifier ===
joblib.dump(model, "models/svm_classifier.pkl")
print("Model saved as svm_classifier.pkl")

# === Train SVM model on non-functional only ===
model_nonfunc = LinearSVC()
model_nonfunc.fit(X_train_nonfunc, y_train_nonfunc)
print("Model trained: Non-functional requirements only. 80% of data used for training")

# === Evaluate non-functional model ===
preds_nonfunc = model_nonfunc.predict(X_test_nonfunc)
print("Non-Functional Accuracy:", accuracy_score(y_test_nonfunc, preds_nonfunc))
print("\nNon-Functional Classification report:")
print(classification_report(y_test_nonfunc, preds_nonfunc))
print(nonfunctional_encoder.classes_)

# === Save the non-functional classifier ===
joblib.dump(model_nonfunc, "models/svm_classifier_nonfunctional.pkl")
print("Non-Functional Model saved as svm_classifier_nonfunctional.pkl")

# === Train SVM model on 100% data ===
model_all = LinearSVC()
model_all.fit(X_train_all, y_train_all)
print("Model trained: 100% of data used for training.")

joblib.dump(model_all, "models/svm_classifier_all.pkl")
print("Model saved as svm_classifier.pkl")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:")
print(cm)

