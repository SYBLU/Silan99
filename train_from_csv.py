import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

X = []
y = []

print("🔍 Loading CSV landmark files...")

for file in os.listdir(DATASET_DIR):
    if file.endswith(".csv"):
        label = file.replace(".csv", "").replace("data_", "")
        path = os.path.join(DATASET_DIR, file)

        df = pd.read_csv(path, header=None)   # ← IMPORTANT: no header
        df = df.astype(float)                 # ← ensure all numeric

        features = df.values.tolist()

        X.extend(features)
        y.extend([label] * len(df))

        print(f"Loaded {file}: {len(df)} samples")

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n🎉 Training Complete")
print("Model Accuracy:", round(acc * 100, 2), "%")

# Save files
pickle.dump(model, open(f"{MODEL_DIR}/gesture_model_v2.pkl", "wb"))
pickle.dump(scaler, open(f"{MODEL_DIR}/scaler_v2.pkl", "wb"))
pickle.dump(le, open(f"{MODEL_DIR}/label_encoder_v2.pkl", "wb"))

print("\nSaved:")
print(f"  {MODEL_DIR}/gesture_model_v2.pkl")
print(f"  {MODEL_DIR}/scaler_v2.pkl")
print(f"  {MODEL_DIR}/label_encoder_v2.pkl")
