import pandas as pd
import glob

files = glob.glob("dataset/*.csv")
dfs = [pd.read_csv(f) for f in files]

full = pd.concat(dfs, axis=0)
full.to_csv("hand_landmarks_dataset.csv", index=False)

print("✔ Dataset created as hand_landmarks_dataset.csv")
