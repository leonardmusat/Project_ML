import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def rename_classes(array, mapping):
    for clas in array:
        if clas in mapping:
            array.replace(clas, mapping[clas], inplace=True)
    return array


dictionary_requirements = {
    "F": "Functional",
    "A": "Availability",
    "LF": "Look and Feel",
    "O": "Operability",
    "PE": "Performance",
    "SE": "Security",
    "US": "Usability",
    "NF": "Non-Functional"
}

nonfunctinal_classes = ["A", "LF", "O", "PE", "SE", "US"]

# Load processed TF-IDF dataset
df = pd.read_csv("data/requirements_tfidf_reduced_dataset.csv")

# Split features and labels
X_df = df.drop(columns=["RequirementText", "Class"])
y = df["Class"].copy()

for cls in y:
    if cls in nonfunctinal_classes:
        y.replace(cls, "Non-Functional", inplace=True)

df_nonfunc = df[df["Class"] != "F"]
print(df_nonfunc)
X_df_nonfunc = df_nonfunc.drop(columns=["RequirementText", "Class"])
y_nonfunc = df_nonfunc["Class"]

# Rename classes
y = rename_classes(y, dictionary_requirements)
y_nonfunc = rename_classes(y_nonfunc, dictionary_requirements)
print("Classes after renaming (all data):", y.unique())
print("Classes after renaming (non-functional only):", y_nonfunc.unique())

# Encode labels
binary_encoder= LabelEncoder()
nonfunctilnal_encoder = LabelEncoder()
y_enc = binary_encoder.fit_transform(y)
y_enc_nonfunc = nonfunctilnal_encoder.fit_transform(y_nonfunc)

print("Classes found (all data):", binary_encoder.classes_)
print("Classes found (non-functional only):", nonfunctilnal_encoder.classes_)

# Save encoder
joblib.dump(binary_encoder, "models/binary_encoder.pkl")
joblib.dump(nonfunctilnal_encoder, "models/nonfunctional_encoder.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_df.to_numpy(), y_enc,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# Split non-functional only
X_train_nonfunc, X_test_nonfunc, y_train_nonfunc, y_test_nonfunc = train_test_split(
    X_df_nonfunc.to_numpy(), y_enc_nonfunc,
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
np.save("data/matrix/X_train_nonfunc.npy", X_train_nonfunc)
np.save("data/matrix/X_test_nonfunc.npy", X_test_nonfunc)
np.save("data/matrix/y_train_nonfunc.npy", y_train_nonfunc)
np.save("data/matrix/y_test_nonfunc.npy", y_test_nonfunc)
np.save("data/matrix/y_total_nonfunc.npy", y_enc_nonfunc)
np.save("data/matrix/X_total_nonfunc.npy", X_df_nonfunc.to_numpy())

print("Done. Train/Test matrices saved.")
