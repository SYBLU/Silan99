# organize_dataset.py

import pandas as pd
import os

# Path to your dataset folder
dataset_folder = 'dataset/'

# List to hold each individual CSV as a DataFrame
dataframes = []

# Loop through each file in the dataset folder
for file in os.listdir(dataset_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(dataset_folder, file)

        try:
            df = pd.read_csv(file_path)

            # Optional sanity check: make sure 'label' column exists
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                print(f"Found {len(df)} samples for label: {label}")
                dataframes.append(df)
            else:
                print(f"⚠️ Skipping {file} — no 'label' column found.")
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

# Concatenate all dataframes into one
if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Shuffle the dataset (optional but recommended)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)

    # Save to a new CSV
    merged_df.to_csv('gesture_dataset_processed.csv', index=False)
    print("✅ Merged dataset saved as gesture_dataset_processed.csv")
else:
    print("❌ No valid CSV files found to merge.")
