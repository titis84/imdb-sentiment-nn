import pickle
import torch
import torch.nn as nn
import sys

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

def predict(text):
    # Load vectorizer
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    # Load config
    import json
    with open("model/config.json", "r") as f:
        config = json.load(f)
    
    # Load model
    model = SentimentNN(config["input_dim"], config["hidden_dim"])
    model.load_state_dict(torch.load("model/best_model.pt", map_location=torch.device("cpu")))
    model.eval()
    
    # Vectorize and predict
    vec = vectorizer.transform([text]).toarray()
    tensor = torch.tensor(vec, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    return sentiment

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a movie review: ")
    print(f"Sentiment: {predict(text)}")