import keras
from keras import layers
def build_model(vocab_size_en, vocab_size_es, max_len):
    # A toy seq2seq transformer setup is complex, using a simpler classification approach or simplified Transformer encoder for demonstration
    inputs = layers.Input(shape=(max_len,))
    embedding = layers.Embedding(input_dim=vocab_size_en, output_dim=64)(inputs)
    # Positional embedding
    positions = keras.ops.arange(start=0, stop=max_len, step=1)
    pos_embedding = layers.Embedding(input_dim=max_len, output_dim=64)(positions)
    x = embedding + pos_embedding
    
    # Transformer Block
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = layers.Dense(64, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(vocab_size_es, activation="softmax")(x) # Simplification: Predict target vocabulary distribution
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
