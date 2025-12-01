from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
import requests
import os
import pickle
from flask import Flask

app = Flask(__name__)

def download_file(url, filename):
    """Download from Google Drive if not already downloaded."""
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"{filename} downloaded successfully!")
    else:
        print(f"{filename} already exists. Skipping download.")


# ----------- YOUR GOOGLE DRIVE DIRECT LINKS -----------
file1_url = "https://drive.google.com/uc?export=download&id=1hjIMhfuXs4nb-jq2F6wmplG-WJjnivBu"
file2_url = "https://drive.google.com/uc?export=download&id=1BPgZaviKms93Qk1Rq_WHVuSUdY7UDre0"
file3_url = "https://drive.google.com/uc?export=download&id=1hxLgzFPD4EcnW9U29shplGj25bVEDCNi"

# ----------- DOWNLOAD THE FILES -----------
download_file(file1_url, "gesture_model_v2.pkl")
download_file(file2_url, "label_encoder_v2.pkl")
download_file(file3_url, "scaler_v2.pkl")

# ----------- LOAD YOUR MODELS / FILES -----------
model1 = pickle.load(open("gesture_model_v2.pkl", "rb"))
model2 = pickle.load(open("label_encoder_v2.pkl", "rb"))
model3 = pickle.load(open("scaler_v2.pkl", "rb"))

# Example route
@app.route("/")
def home():
    return "Files downloaded & models loaded successfully!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

app = Flask(__name__)
CORS(app)  # allow frontend requests

# Load ML models (same as your code)
model = joblib.load(open(gesture_model_v2.pkl, "rb"))
scaler = joblib.load(open(scaler_v2.pkl, "rb"))
label_encoder = joblib.load(open(label_encoder_v2.pkl, "rb"))

# 63 features (x,y,z for 21 landmarks)
feature_columns = [str(i) for i in range(63)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)


def extract_landmarks(img):
    """Extract hand landmarks and return 63 features."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    landmarks = []

    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    if len(landmarks) != 63:
        return None

    df = pd.DataFrame([landmarks], columns=feature_columns)
    scaled = scaler.transform(df)
    probs = model.predict_proba(scaled)[0]
    label = label_encoder.inverse_transform([np.argmax(probs)])[0]

    return label.upper()


def decode_base64_image(base64_string):
    """Convert base64 to cv2 image."""
    header, data = base64_string.split(",", 1)
    decoded = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    return img


import base64

@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json()
        img_b64 = req.get("image", None)

        if img_b64 is None:
            return jsonify({"error": "No image received"}), 400

        img = decode_base64_image(img_b64)
        label = extract_landmarks(img)

        if label is None:
            return jsonify({"prediction": ""})  # hand not detected

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Backend OK"})


if __name__ == "__main__":
    app.run(port=5000)





