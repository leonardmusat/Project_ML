import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

dictionary_requirements = {
    "F": "Funtional",
    "A": "Availability",
    "L": "Legal",
    "LF": "Look and Feel",
    "MN": "Maintainability",
    "O": "Operability",
    "PE": "Performance",
    "SC": "Scalability",
    "SE": "Security",
    "US": "Usability",
    "FT": "Fault Tolerance",
    "PO": "Portability"
}

# Load processed TF-IDF dataset
df = pd.read_csv("data/requirements_tfidf_full_dataset.csv")

# Split features and labels
X_df = df.drop(columns=["RequirementText", "Class"])
y = df["Class"]
for clas in y:
    if clas in dictionary_requirements:
        y.replace(clas, dictionary_requirements[clas], inplace=True)  

# Encode labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Save encoder
joblib.dump(encoder, "models/label_encoder.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_df.to_numpy(), y_enc,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# Save numpy matrices
os.makedirs("data/matrix", exist_ok=True)
np.save("data/matrix/X_train.npy", X_train)
np.save("data/matrix/X_test.npy", X_test)
np.save("data/matrix/y_train.npy", y_train)
np.save("data/matrix/y_test.npy", y_test)
np.save("data/matrix/X_total.npy", X_df.to_numpy())
np.save("data/matrix/y_total.npy", y_enc)

print("Done. Train/Test matrices saved.")
