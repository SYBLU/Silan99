import os

dataset_path = 'dataset'

for label in sorted(os.listdir(dataset_path)):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        count = len([
            f for f in os.listdir(label_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"{label}: {count} images")

