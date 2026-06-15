import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, "data", "fake_news.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "dataset_processed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
