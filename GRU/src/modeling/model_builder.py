from keras.models import Sequential
from keras.layers import GRU, Dense
def build_model(input_shape):
    model = Sequential([
        GRU(32, input_shape=input_shape, return_sequences=False),
        Dense(1) # pm2.5
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
