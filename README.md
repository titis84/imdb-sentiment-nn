# IMDB Sentiment Analysis with CI/CD

This project trains a neural network on IMDB movie reviews to classify sentiment (positive/negative). The model is automatically trained and uploaded to Hugging Face Hub whenever code is pushed to GitHub.

## How to Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train.py`
3. Run prediction: `python predict.py "I loved this movie!"`

## CI/CD Pipeline

- On every push to `main` or `master`, GitHub Actions:
  - Installs dependencies
  - Trains the model
  - Uploads the model artifacts to Hugging Face Hub

## Model Artifacts

- `best_model.pt`: PyTorch model weights
- `vectorizer.pkl`: TF-IDF vectorizer
- `config.json`: Model configuration
- `metrics.json`: Validation accuracy and training parameters