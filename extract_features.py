import cv2
import mediapipe as mp
import os
import numpy as np
import joblib

# Initialize MediaPipe hands with better detection sensitivity
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,         # Keep True for images
    max_num_hands=1,
    min_detection_confidence=0.3    # Lower threshold so fewer are skipped
)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # Skip broken or unreadable images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        vector = []
        for lm in landmarks.landmark:
            vector.extend([lm.x, lm.y, lm.z])
        return vector
    else:
        return None  # No hand detected

X = []
y = []

dataset_path = 'dataset'
total_images = 0
used_images = 0
skipped_images = 0

label_stats = {}  # To store per-label counts

for label in sorted(os.listdir(dataset_path)):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        label_total = 0
        label_used = 0
        label_skipped = 0

        print(f"\n📂 Processing label: '{label}'")
        for file_name in os.listdir(label_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                label_total += 1

                image_path = os.path.join(label_path, file_name)
                features = extract_landmarks(image_path)
                if features:
                    X.append(features)
                    y.append(label)
                    used_images += 1
                    label_used += 1
                else:
                    skipped_images += 1
                    label_skipped += 1

        label_stats[label] = (label_total, label_used, label_skipped)

# Save features
X = np.array(X)
y = np.array(y)
os.makedirs("models", exist_ok=True)
joblib.dump((X, y), 'models/features.pkl')

print(f"\n✅ Feature extraction complete!")
print(f"Total images: {total_images}")
print(f"Used (with hands detected): {used_images}")
print(f"Skipped (no hands or failed): {skipped_images}")

print("\n📊 Per-label stats:")
for label, (total, used, skipped) in label_stats.items():
    print(f"{label}: total={total}, used={used}, skipped={skipped}")
