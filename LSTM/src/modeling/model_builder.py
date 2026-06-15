from keras.models import Sequential
from keras.layers import LSTM, Dense
def build_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dense(2) # temp and humidity
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
