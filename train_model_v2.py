import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Correct path to features.pkl
FEATURES_PATH = FEATURES_PATH = r"D:\python baby girl\PythonProject\models\features.pkl"

print("Loading features from:", FEATURES_PATH)

with open(FEATURES_PATH, "rb") as f:
    data = pickle.load(f)

X = data["features"]
y = data["labels"]

print("Features loaded:", len(X))
print("Labels loaded:", len(y))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(acc * 100, 2), "%")

# Save outputs
SAVE_DIR = r"/PythonProject/models"

with open(os.path.join(SAVE_DIR, "gesture_model_v2.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(SAVE_DIR, "scaler_v2.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(SAVE_DIR, "label_encoder_v2.pkl"), "wb") as f:
    pickle.dump(le, f)

print("\nSaved model, scaler, label encoder successfully!")
