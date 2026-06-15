from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding
def build_model(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
