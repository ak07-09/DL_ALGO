import keras
from keras import layers
def build_model(vocab_size, max_len):
    inputs = layers.Input(shape=(max_len,))
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=64)(inputs)
    positions = keras.ops.arange(start=0, stop=max_len, step=1)
    pos_embedding = layers.Embedding(input_dim=max_len, output_dim=64)(positions)
    x = embedding + pos_embedding
    
    # Causal Attention Block
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x, use_causal_mask=True)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = layers.Dense(64, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    outputs = layers.Dense(vocab_size, activation="softmax")(x) # Next char prediction
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
