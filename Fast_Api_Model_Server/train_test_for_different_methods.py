import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KnnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier

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
X_train_all_nonfunc = np.load("data/matrix/X_total_nonfunc.npy")
y_train_all_nonfunc = np.load("data/matrix/y_total_nonfunc.npy", allow_pickle=True)

# === Load TF-IDF vectorizer ===
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === Load svd ===
svd = joblib.load("models/tfidf_svd_100.pkl")

# === Load encoder ===
binary_encoder = joblib.load("models/binary_encoder.pkl")
nonfunctional_encoder = joblib.load("models/nonfunctional_encoder.pkl")

print("Data loaded successfully.")

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

def create_and_evaluate_model_80_20(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print("Model trained: 80% of data used for training.")

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)

    return model, preds

def create_and_evaluate_model_100_crossval(model, X_train_all, y_train_all):
    model.fit(X_train_all, y_train_all)
    print("Model trained: 100% of data used for training.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X_train_all, y_train_all, cv=cv, scoring=scoring)

    print("Cross-validation results:")
    print("Accuracy:", cv_results['test_accuracy'].mean())
    print("Precision:", cv_results['test_precision_weighted'].mean())
    print("Recall:", cv_results['test_recall_weighted'].mean())
    print("F1-score:", cv_results['test_f1_weighted'].mean())

    return model, preds


print("Data for 80%/20% binary classification:")
print("Clasifier: SVM")
model = LinearSVC()
model, preds = create_and_evaluate_model_80_20(model, X_train, y_train, X_test, y_test)
print(binary_encoder.classes_)
joblib.dump(model, "models/svm_classifier.pkl")
print("Model saved as svm_classifier.pkl")

print("\n\n")
print("Clasifier: KNN")
model_knn = KnnClassifier()
model_knn, preds_knn = create_and_evaluate_model_80_20(model_knn, X_train, y_train, X_test, y_test)
print(binary_encoder.classes_)
joblib.dump(model_knn, "models/knn_classifier.pkl")
print("KNN Model saved as knn_classifier.pkl")  

print("\n\n")
print("Clasifier: Logistic Regression") 
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
model_lr, preds_lr = create_and_evaluate_model_80_20(model_lr, X_train, y_train, X_test, y_test)
print(binary_encoder.classes_)
joblib.dump(model_lr, "models/logistic_regression_classifier.pkl")
print("Model saved as logistic_regression_classifier.pkl")

print("\n\n")
print("Clasifier: SGD")
model_sgd = SGDClassifier(max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42, average=True)
model_sgd, preds_sgd = create_and_evaluate_model_80_20(model_sgd, X_train, y_train, X_test, y_test)
print(binary_encoder.classes_)
joblib.dump(model_sgd, "models/sgd_classifier.pkl")
print("Model saved as sgd_classifier.pkl") 

print("\n\n")
print("Data for 80%/20% non-functional only classification:")

print("\n\n")
print("Clasifier: SVM")
model_nonfunc = LinearSVC()
model_nonfunc, preds_nonfunc = create_and_evaluate_model_80_20(model_nonfunc, X_train_nonfunc, y_train_nonfunc, X_test_nonfunc, y_test_nonfunc)
print(nonfunctional_encoder.classes_)
joblib.dump(model_nonfunc, "models/svm_classifier_nonfunctional.pkl")
print("Non-Functional Model saved as svm_classifier_nonfunctional.pkl")

print("\n\n")
print("Clasifier: KNN")
model_knn_nonfunc = KnnClassifier()
model_knn_nonfunc, preds_knn_nonfunc = create_and_evaluate_model_80_20(model_knn_nonfunc, X_train_nonfunc, y_train_nonfunc, X_test_nonfunc, y_test_nonfunc)
print(nonfunctional_encoder.classes_)
joblib.dump(model_knn_nonfunc, "models/knn_classifier_nonfunctional.pkl")
print("Non-Functional KNN Model saved as knn_classifier_nonfunctional.pkl")  

print("\n\n")
print("Clasifier: Logistic Regression") 
model_lr_nonfunc = LogisticRegression(max_iter=1000, class_weight='balanced')
model_lr_nonfunc, preds_lr_nonfunc = create_and_evaluate_model_80_20(model_lr_nonfunc, X_train_nonfunc, y_train_nonfunc, X_test_nonfunc, y_test_nonfunc)
print(nonfunctional_encoder.classes_)
joblib.dump(model_lr_nonfunc, "models/logistic_regression_classifier_nonfunctional.pkl")
print("Non-Functional Model saved as logistic_regression_classifier_nonfunctional.pkl")

print("\n\n")
print("Clasifier: SGD Classifier")
model_sgd_nonfunc = SGDClassifier(max_iter=1000, tol=1e-3, class_weight='balanced', random_state=41, average= True)
model_sgd_nonfunc, preds_sgd_nonfunc = create_and_evaluate_model_80_20(model_sgd_nonfunc, X_train_nonfunc, y_train_nonfunc, X_test_nonfunc, y_test_nonfunc)
print(nonfunctional_encoder.classes_)
joblib.dump(model_sgd_nonfunc, "models/sgd_classifier_nonfunctional.pkl")
print("Non-Functional Model saved as sgd_classifier_nonfunctional.pkl") 

print("\n\n")
print("Data for 100% binary classification:")

print("\n\n")
print("Clasifier: SVM")
model_all = LinearSVC()
model_all, preds_all = create_and_evaluate_model_100_crossval(model_all, X_train_all, y_train_all)
joblib.dump(model_all, "models/svm_classifier_all.pkl")
print("Model saved as svm_classifier_all.pkl")

print("\n\n")
print("Clasifier: Knn")
model_knn_all = KnnClassifier()
model_knn_all, preds_knn_all = create_and_evaluate_model_100_crossval(model_knn_all, X_train_all, y_train_all)
joblib.dump(model_knn_all, "models/knn_classifier_all.pkl")
print("KNN Model saved as knn_classifier_all.pkl") 

print("\n\n")
print("Clasifier: Logistic Regression") 
model_lr_all = LogisticRegression(max_iter=1000, class_weight='balanced')
model_lr_all, preds_lr_all = create_and_evaluate_model_100_crossval(model_lr_all, X_train_all, y_train_all)
joblib.dump(model_lr_all, "models/logistic_regression_classifier_all.pkl")
print("Model saved as logistic_regression_classifier_all.pkl")

print("\n\n")
print("Clasifier: SGD Classifier")

model_sgd_all = SGDClassifier(max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42, average = True)
model_sgd_all, preds_sgd_all = create_and_evaluate_model_100_crossval(model_sgd_all, X_train_all, y_train_all)
joblib.dump(model_sgd_all, "models/sgd_classifier_all.pkl")
print("Model saved as sgd_classifier_all.pkl")

print("\n\n")
print("Data for 100% non-functional only classification:")

print("\n\n")
print("Clasifier: SVM")
model_all_nonfunc = LinearSVC()
model_all_nonfunc, preds_all_nonfunc = create_and_evaluate_model_100_crossval(model_all_nonfunc, X_train_all_nonfunc, y_train_all_nonfunc)
joblib.dump(model_all_nonfunc, "models/svm_classifier_all_nonfunctional.pkl")
print("Non-Functional Model saved as svm_classifier_all_nonfunctional.pkl")

print("\n\n")
print("Clasifier: KNN")
model_knn_all_nonfunc = KnnClassifier()
model_knn_all_nonfunc, preds_knn_all_nonfunc = create_and_evaluate_model_100_crossval(model_knn_all_nonfunc, X_train_all_nonfunc, y_train_all_nonfunc)
joblib.dump(model_knn_all_nonfunc, "models/knn_classifier_all_nonfunctional.pkl")
print("Non-Functional KNN Model saved as knn_classifier_all_nonfunctional.pkl")

print("\n\n")
print("Clasifier: Logistic Regression") 
model_lr_all_nonfunc = LogisticRegression(max_iter=1000, class_weight='balanced')
model_lr_all_nonfunc, preds_lr_all_nonfunc = create_and_evaluate_model_100_crossval(model_lr_all_nonfunc, X_train_all_nonfunc, y_train_all_nonfunc)
joblib.dump(model_lr_all_nonfunc, "models/logistic_regression_classifier_all_nonfunctional.pkl")
print("Non-Functional Model saved as logistic_regression_classifier_all_nonfunctional.pkl")

print("\n\n")
print("Clasifier: SGD Classifier")
model_sgd_all_nonfunc = SGDClassifier(max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42, average = True)
model_sgd_all_nonfunc, preds_sgd_all_nonfunc = create_and_evaluate_model_100_crossval(model_sgd_all_nonfunc, X_train_all_nonfunc, y_train_all_nonfunc)
joblib.dump(model_sgd_all_nonfunc, "models/sgd_classifier_all_nonfunctional.pkl")
print("Non-Functional Model saved as sgd_classifier_all_nonfunctional.pkl")

voter_model_binary = VotingClassifier(
    estimators=[
        ("svm", model_all),
        ("lr", model_lr_all),
        ("sgd", model_sgd_all),
    ],
    voting="hard"
)

print("\n\n")
voter_model_binary, preds_voter = create_and_evaluate_model_100_crossval(voter_model_binary, X_train_all, y_train_all)
print("\n\n")

voter_model_nonfunc = VotingClassifier(
    estimators=[
        ("svm", model_all_nonfunc),
        ("lr", model_lr_all_nonfunc),
        ("sgd", model_sgd_all_nonfunc),
    ],
    voting="hard"
)

voter_model_nonfunc, preds_voter_nonfunc = create_and_evaluate_model_100_crossval(voter_model_nonfunc, X_train_all_nonfunc, y_train_all_nonfunc)