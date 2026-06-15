from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
def build_model(input_shape):
    model = Sequential([
        SimpleRNN(32, input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
