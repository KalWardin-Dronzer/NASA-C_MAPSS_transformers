"""
The hybrid CNN + BiLSTM + Transformer-encoder regressor, as in the original
notebook: two 1-D convolutions for local degradation patterns, a bidirectional
LSTM for sequence dependencies, sinusoidal positional encoding and two
Transformer encoder blocks, then a small dense head.
"""

import numpy as np
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="cmapss")
class PositionalEncoding(layers.Layer):
    """Adds the fixed sinusoidal encoding of Vaswani et al. (2017)."""

    def build(self, input_shape):
        seq_len, d_model = input_shape[1], input_shape[2]
        positions = np.arange(seq_len)[:, None]
        dims = np.arange(d_model)[None, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
        encoding = np.zeros((seq_len, d_model), np.float32)
        encoding[:, 0::2] = np.sin(angles[:, 0::2])
        encoding[:, 1::2] = np.cos(angles[:, 1::2])
        self.encoding = keras.ops.convert_to_tensor(encoding[None])

    def call(self, x):
        return x + self.encoding


def _encoder_block(x, d_model: int, heads: int, head_size: int, ff_dim: int, dropout: float):
    attention = layers.MultiHeadAttention(num_heads=heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dropout)(attention))
    ff = layers.Conv1D(ff_dim, 1, activation="relu")(x)
    ff = layers.Conv1D(d_model, 1)(layers.Dropout(dropout)(ff))
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_hybrid(window: int, n_features: int, dropout: float = 0.2, learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=(window, n_features))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(n_features, 1)(x)              # project to the Transformer width
    x = PositionalEncoding()(x)
    for _ in range(2):
        x = _encoder_block(x, n_features, heads=2, head_size=16, ff_dim=128, dropout=dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="rul")(x)
    model = keras.Model(inputs, outputs, name="cnn_bilstm_transformer")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse",
                  metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def mc_dropout_predict(model: keras.Model, X: np.ndarray, passes: int = 50, scale: float = 1.0,
                       batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over `passes` forward passes with dropout left on."""
    draws = []
    for _ in range(passes):
        outs = [np.asarray(model(X[i:i + batch_size], training=True)).ravel()
                for i in range(0, len(X), batch_size)]
        draws.append(np.concatenate(outs) * scale)
    draws = np.stack(draws)
    return draws.mean(axis=0), draws.std(axis=0)
