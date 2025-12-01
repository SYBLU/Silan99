import cv2, os, numpy as np, joblib, mediapipe as mp
from tqdm import tqdm

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

# === Dataset Paths ===
dataset_dir = r"D:\python baby girl\PythonProject\dataset"
out_dir = r"D:\python baby girl\PythonProject\models"
os.makedirs(out_dir, exist_ok=True)

X = []
y = []

def extract_landmarks(path):
    img = cv2.imread(path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]
    vec = []
    for p in lm.landmark:
        vec.extend([p.x, p.y, p.z])

    if len(vec) == 63:
        return vec
    return None


print("\n=== SCANNING DATASET ===")

labels = sorted(os.listdir(dataset_dir))
total_images = 0

# Count total images first
for label in labels:
    folder = os.path.join(dataset_dir, label)
    if os.path.isdir(folder):
        total_images += len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

print("Total images found:", total_images)
print("Starting extraction...\n")

processed = 0
skipped = 0

for label in labels:
    folder = os.path.join(dataset_dir, label)
    if not os.path.isdir(folder):
        continue

    print(f"\nProcessing label: {label}")

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in tqdm(images, desc=f"{label}"):
        path = os.path.join(folder, img_name)
        vec = extract_landmarks(path)

        if vec is None:
            skipped += 1
        else:
            X.append(vec)
            y.append(label)

        processed += 1

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Save features
out_file = os.path.join(out_dir, "features.pkl")
joblib.dump((X, y), out_file)

print("\n=== DONE ===")
print("Processed:", processed)
print("Used (hands detected):", len(X))
print("Skipped:", skipped)
print("Saved to:", out_file)
