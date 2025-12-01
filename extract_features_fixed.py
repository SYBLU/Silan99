# extract_features_fixed.py
import cv2, os, numpy as np, joblib
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

dataset_dir = r"D:\python baby girl\PythonProject\dataset"
out_dir     = r"D:\python baby girl\PythonProject\models"
os.makedirs(out_dir, exist_ok=True)

X, y = [], []
label_stats = {}

def extract_landmarks_from_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    vec = []
    for p in lm.landmark:
        vec.extend([p.x, p.y, p.z])
    return vec if len(vec) == 63 else None

total = used = skipped = 0

for label in sorted(os.listdir(dataset_dir)):
    lab_path = os.path.join(dataset_dir, label)
    if not os.path.isdir(lab_path):
        continue

    label_total = label_used = label_skipped = 0

    for fn in os.listdir(lab_path):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total += 1
        label_total += 1

        p = os.path.join(lab_path, fn)
        vec = extract_landmarks_from_image(p)

        if vec is None:
            skipped += 1
            label_skipped += 1
        else:
            X.append(vec)
            y.append(label)
            used += 1
            label_used += 1

    label_stats[label] = (label_total, label_used, label_skipped)

X, y = np.array(X), np.array(y)
joblib.dump((X, y), os.path.join(out_dir, "features.pkl"))

print("=== Extraction Summary ===")
print(f"Total images scanned: {total}")
print(f"Used (hands detected): {used}")
print(f"Skipped: {skipped}")
print("\nPer-label stats:")
for lbl, (t, u, s) in label_stats.items():
    print(f"{lbl}: total={t}, used={u}, skipped={s}")

print("\nSaved:", os.path.join(out_dir, "features.pkl"))
print("X shape:", X.shape)
