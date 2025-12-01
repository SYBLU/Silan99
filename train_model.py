import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset CSV (landmarks + label column)
df = pd.read_csv("hand_landmarks_dataset.csv")

X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

class HandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

train_data = HandDataset(X_train, y_train)
test_data  = HandDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32)

# Neural Network Architecture
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

input_size = X.shape[1]
num_classes = len(np.unique(y))

model = GestureModel(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train
for epoch in range(40):
    model.train()
    for landmarks, labels in train_loader:
        optimizer.zero_grad()
        out = model(landmarks)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/40 — Loss: {loss.item():.4f}")

# Save PyTorch Model
torch.save(model.state_dict(), "gesture_model.pt")

print("Training Complete ✔")
