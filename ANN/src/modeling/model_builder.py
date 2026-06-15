from keras.models import Sequential
from keras.layers import Dense, Dropout
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) # regression for wine quality
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
