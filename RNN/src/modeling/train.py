import os
import numpy as np
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data
from src.features.build_features import preprocess
from src.modeling.model_builder import build_model
from src.utils.config import EPOCHS, BATCH_SIZE, MODEL_PATH

def run_training():
    print("Loading data...")
    raw_data = load_raw_data()
    cleaned_data = clean_data(raw_data)
    X, y = preprocess(cleaned_data)
    
    # Model specific input shapes
    workspace = 'RNN'
    if workspace == 'ANN':
        model = build_model(X.shape[1])
    elif workspace == 'CNN':
        X = X.reshape(-1, 28, 28, 1)
        model = build_model((28, 28, 1))
    elif workspace in ['RNN', 'LSTM', 'GRU']:
        X = X.reshape(-1, 1, X.shape[1])
        model = build_model((1, X.shape[2]))
    elif workspace == 'BiLSTM':
        model = build_model(1000, 50, 10)
        X = np.random.randint(0, 1000, (100, 50))
        y = np.random.randint(0, 10, (100, 50))
    elif workspace in ['Sentiment_LSTM', 'BERT']:
        model = build_model(1000, 50)
        X = np.random.randint(0, 1000, (100, 50))
        y = np.random.randint(0, 2, (100,))
    elif workspace == 'Transformer':
        model = build_model(1000, 1000, 50)
        X = np.random.randint(0, 1000, (100, 50))
        y = np.random.randint(0, 1000, (100,))
    elif workspace == 'GPT':
        model = build_model(100, 20)
        X = np.random.randint(0, 100, (100, 20))
        y = np.random.randint(0, 100, (100, 20))

    print("Training model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Training complete.")

if __name__ == "__main__":
    run_training()
