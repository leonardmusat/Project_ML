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
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === Load encoder ===
encoder = joblib.load("models/label_encoder.pkl")

# === Load model ===
model = joblib.load("models/svm_classifier.pkl")

class Requirement(BaseModel):
    requirement_text: str

@app.post("/predict")
def predict_requirement(req: Requirement):
    requirement_text = req.requirement_text

    # === Predict on the new requirement ===
    new_vec = vectorizer.transform([requirement_text])
    new_pred = model.predict(new_vec)
    label = encoder.inverse_transform(new_pred)

    return {"requirement": requirement_text, "predicted_class": label[0]}