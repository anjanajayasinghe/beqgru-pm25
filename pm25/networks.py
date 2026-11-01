"""Recurrent network architectures and the quantile (pinball) loss.

Benchmark networks (Section 2.4.3) are trained with the log-cosh loss.
The proposed BEQ-GRU (Section 2.5) is trained with the quantile loss at tau = 0.95.
"""
from __future__ import annotations

import keras
from keras import layers, ops

import config as C


# --------------------------------------------------------------------------- losses
def quantile_loss(tau: float = C.TAU):
    """Pinball loss rho_tau(u) = max(tau*u, (tau-1)*u), u = y - y_hat (Eq 12)."""

    def loss(y_true, y_pred):
        u = y_true - y_pred
        return ops.mean(ops.maximum(tau * u, (tau - 1.0) * u), axis=-1)

    loss.__name__ = f"quantile_loss_{tau:g}"
    return loss


def _compile(model, loss, learning_rate=None):
    opt = keras.optimizers.Nadam(learning_rate=learning_rate) if learning_rate else keras.optimizers.Nadam()
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


# --------------------------------------------------------------------------- benchmark networks
def stacked_rnn(input_shape, horizon, cell="GRU", bidirectional=False, units=128, dropout=0.2, learning_rate=None):
    """Two recurrent layers + dense head: LSTM, GRU, Bi-LSTM and Bi-GRU benchmarks."""
    rnn = layers.GRU if cell == "GRU" else layers.LSTM
    inputs = keras.Input(shape=input_shape)
    first = rnn(units, return_sequences=True)
    x = (layers.Bidirectional(first) if bidirectional else first)(inputs)
    x = layers.Dropout(dropout)(x)
    x = rnn(units)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(horizon)(x)
    return _compile(keras.Model(inputs, outputs), keras.losses.LogCosh(), learning_rate)


def attention_rnn(input_shape, horizon, cell="GRU", units=128, dropout=0.2, heads=4, key_dim=64, learning_rate=None):
    """Bidirectional layer + multi-head self-attention with a residual connection (MHA-Bi-LSTM / MHA-Bi-GRU)."""
    rnn = layers.GRU if cell == "GRU" else layers.LSTM
    inputs = keras.Input(shape=input_shape)
    x = layers.Bidirectional(rnn(units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout)(x)
    att = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)
    att = layers.LayerNormalization()(att)
    att = layers.Dropout(dropout)(att)
    x = layers.Add()([att, x])
    x = layers.Bidirectional(rnn(units))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(horizon)(x)
    return _compile(keras.Model(inputs, outputs), keras.losses.LogCosh(), learning_rate)


# --------------------------------------------------------------------------- BEQ-GRU
def residual_gru_block(x, units, dropout=C.BLOCK_DROPOUT, return_sequences=True):
    """GRU + dropout with an additive skip connection (Section 2.5.3).

    The skip path is projected with a time-distributed dense layer when the
    widths differ and is the identity otherwise. When `return_sequences` is
    False, the block's final hidden state is added to every time step of the
    skip path (broadcast over time), so the block still returns a sequence.
    """
    shortcut = x
    h = layers.GRU(units, return_sequences=return_sequences)(x)
    h = layers.Dropout(dropout)(h)
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units)(shortcut)          # applied identically at every time step
    if not return_sequences:
        h = layers.Reshape((1, units))(h)          # (batch, 1, units), broadcast over time
    return layers.Add()([h, shortcut])


def beq_gru(input_shape, horizon, units=128, dropout=0.2, learning_rate=None, block_units=None,
            n_blocks=C.N_RES_BLOCKS, tau=C.TAU):
    """Bidirectional enhanced quantile GRU.

    Bi-GRU encoder -> dropout -> `n_blocks` residual GRU blocks -> GRU read-out
    -> dropout -> dense layer giving the `horizon`-step forecast. Trained with the
    quantile loss at level `tau`.
    """
    block_units = block_units or units
    inputs = keras.Input(shape=input_shape)
    x = layers.Bidirectional(layers.GRU(units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout)(x)
    for i in range(n_blocks):
        x = residual_gru_block(x, block_units, return_sequences=(i < n_blocks - 1))
    x = layers.GRU(units)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(horizon)(x)
    return _compile(keras.Model(inputs, outputs, name="BEQ_GRU"), quantile_loss(tau), learning_rate)


# --------------------------------------------------------------------------- registry
def build(name, input_shape, horizon, **kw):
    """Build any model in the paper by name."""
    if name == "LSTM":
        return stacked_rnn(input_shape, horizon, "LSTM", False, **kw)
    if name == "GRU":
        return stacked_rnn(input_shape, horizon, "GRU", False, **kw)
    if name == "Bi-LSTM":
        return stacked_rnn(input_shape, horizon, "LSTM", True, **kw)
    if name == "Bi-GRU":
        return stacked_rnn(input_shape, horizon, "GRU", True, **kw)
    if name == "MHA-Bi-LSTM":
        return attention_rnn(input_shape, horizon, "LSTM", **kw)
    if name == "MHA-Bi-GRU":
        return attention_rnn(input_shape, horizon, "GRU", **kw)
    if name == "BEQ-GRU":
        return beq_gru(input_shape, horizon, **kw)
    raise ValueError(f"unknown model {name}")


MODELS = ["LSTM", "GRU", "Bi-LSTM", "Bi-GRU", "MHA-Bi-LSTM", "MHA-Bi-GRU", "BEQ-GRU"]


def settings(station, name, horizon, tuned=True):
    """Hyperparameters for one model: tuned values if available, else defaults."""
    if tuned and (station, name, horizon) in C.TUNED:
        return dict(C.TUNED[(station, name, horizon)])
    return dict(C.DEFAULTS[name])
