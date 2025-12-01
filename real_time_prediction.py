import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import pandas as pd
import pickle
from collections import deque

# Load encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# PyTorch Model (same architecture as training)
class GestureModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load checkpoint
input_size = 63  # 21 landmarks * 3 coords
num_classes = len(label_encoder.classes_)
model = GestureModel(input_size, num_classes)
model.load_state_dict(torch.load("gesture_model.pt", map_location="cpu"))
model.eval()

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Prediction Smoothing
BUFFER = 15
pred_buffer = deque(maxlen=BUFFER)

print("🟢 PyTorch Sign Language Model Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    prediction = "nothing"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # Extract x,y,z (21*3 = 63)
            lm = []
            for p in hand.landmark:
                lm.extend([p.x, p.y, p.z])

            lm = torch.tensor(lm, dtype=torch.float32).unsqueeze(0)
            logits = model(lm)
            probs = torch.softmax(logits, dim=1).detach().numpy()[0]

            idx = np.argmax(probs)
            prediction = label_encoder.inverse_transform([idx])[0]

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    pred_buffer.append(prediction)

    # Majority Vote Smooth
    if len(pred_buffer) == BUFFER:
        prediction = max(set(pred_buffer), key=pred_buffer.count)

    cv2.putText(frame, f"Prediction: {prediction}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("PyTorch Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
