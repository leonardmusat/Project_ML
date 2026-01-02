from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow your Next.js app (localhost:3000)
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load TF-IDF vectorizer ===
vectorizer = joblib.load("models/tfidf_vectorizer_ngram_range.pkl")

# === Load encoder ===
binary_encoder = joblib.load("models/binary_encoder.pkl")
nonfunctional_encoder = joblib.load("models/nonfunctional_encoder.pkl")

# === Load model ===
model_svm_binary = joblib.load("models/svm_classifier_all.pkl")
model_svm_nonfunc = joblib.load("models/svm_classifier_all_nonfunctional.pkl")
model_logistic_regression_binary = joblib.load("models/logistic_regression_classifier_all.pkl")
model_logistic_regression_nonfunc = joblib.load("models/logistic_regression_classifier_all_nonfunctional.pkl")
model_sgd_binary = joblib.load("models/sgd_classifier_all.pkl")
model_sgd_nonfunc = joblib.load("models/sgd_classifier_all_nonfunctional.pkl")  
model_voter_classifier_binary = joblib.load("models/voter_classifier_all.pkl")
model_voter_classifier_nonfunc = joblib.load("models/voter_classifier_all_nonfunctional.pkl")
print("Models loaded successfully.")

class Requirement(BaseModel):
    requirement_text: str

class Model(BaseModel):
    name: str

dict_model_mapping = {
    "SVM": [model_svm_binary, model_svm_nonfunc],
    "Logistic Regression": [model_logistic_regression_binary, model_logistic_regression_nonfunc],
    "SGD": [model_sgd_binary, model_sgd_nonfunc],
    "All Models": [model_voter_classifier_binary, model_voter_classifier_nonfunc],
}

@app.post("/predict")
def predict_requirement(req: Requirement, model: Model):
    requirement_text = req.requirement_text
    model_name = model.name

    # === Predict on the new requirement ===
    new_vec = vectorizer.transform([requirement_text])
    new_pred = dict_model_mapping[model_name][0].predict(new_vec)
    label = binary_encoder.inverse_transform(new_pred)
    if label[0] == "Non-Functional":
        new_pred_nonfunc = dict_model_mapping[model_name][1].predict(new_vec)
        label = nonfunctional_encoder.inverse_transform(new_pred_nonfunc)

    return {"requirement": requirement_text, "predicted_class": label[0]}