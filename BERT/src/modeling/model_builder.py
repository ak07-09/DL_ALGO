import keras
from keras import layers
def build_model(vocab_size, max_len):
    inputs = layers.Input(shape=(max_len,))
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=64)(inputs)
    positions = keras.ops.arange(start=0, stop=max_len, step=1)
    pos_embedding = layers.Embedding(input_dim=max_len, output_dim=64)(positions)
    x = embedding + pos_embedding
    
    # Simple Encoder Block
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = layers.Dense(64, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) # Binary classification (Real/Fake)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
