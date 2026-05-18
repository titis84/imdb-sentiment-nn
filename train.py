import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# -------------------------------
# 1. Load and prepare data
# -------------------------------
print("Loading IMDB dataset...")
dataset = load_dataset("imdb", split="train+test")  # combine train and test for variety
df = pd.DataFrame(dataset)
df = df.sample(frac=0.2, random_state=42)  # use 20% for faster training

X = df["text"].values
y = df["label"].values  # 0 for negative, 1 for positive

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 2. TF-IDF vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_val_vec = vectorizer.transform(X_val).toarray()

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_vec, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

# -------------------------------
# 3. Define the neural network
# -------------------------------
class SentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = X_train_vec.shape[1]
model = SentimentNN(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 4. Training loop
# -------------------------------
epochs = 5
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model/best_model.pt")

# -------------------------------
# 5. Save artifacts
# -------------------------------
# Save vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save config
config = {
    "input_dim": input_dim,
    "hidden_dim": 128,
    "output_dim": 2,
    "max_features": 5000,
    "best_val_acc": best_val_acc
}
with open("model/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Save metrics
metrics = {
    "best_validation_accuracy": best_val_acc,
    "epochs": epochs,
    "batch_size": batch_size
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Training complete. Artifacts saved to ./model/")