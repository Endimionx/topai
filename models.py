import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from automl import find_optimal_window

def window_data(data, pos, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        seq = [d[pos] for d in data[i:i+window_size]]
        target = data[i+window_size][pos]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def build_lstm_block(input_shape):
    inputs = Input(shape=input_shape, name="lstm_input")
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)
    out = Dense(10, activation='softmax')(x)
    return Model(inputs, out, name="LSTM_Model")

def build_transformer_block(input_shape):
    inputs = Input(shape=input_shape, name="trf_input")
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)
    return Model(inputs, out, name="Transformer_Model")

def full_prediction_pipeline(data):
    result_preds = []
    result_confs = []

    for pos in range(4):
        ws = find_optimal_window(data, pos, (10, 30))
        X, y = window_data(data, pos, ws)
        if len(X) < 5:
            raise ValueError(f"Data terlalu sedikit untuk posisi {pos}")
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build models
        lstm = build_lstm_block((ws,1))
        trf = build_transformer_block((ws,1))

        lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        trf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        lstm.fit(X, y, epochs=5, verbose=0)
        trf.fit(X, y, epochs=5, verbose=0)

        last_input = X[-1].reshape(1, ws, 1)
        pred_lstm = lstm.predict(last_input)[0]
        pred_trf = trf.predict(last_input)[0]

        # Kalibrasi
        combined = (pred_lstm + pred_trf) / 2
        combined /= combined.sum()

        top3_idx = np.argsort(combined)[-3:][::-1]
        top3_conf = combined[top3_idx]

        result_preds.append(top3_idx.tolist())
        result_confs.append(top3_conf.tolist())

    return result_preds, result_confs        X = X.reshape((X.shape[0], X.shape[1], 1))

        lstm = build_lstm_block((ws,1))
        transformer = build_transformer_block((ws,1))

        lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        lstm.fit(X, y, epochs=5, verbose=0)
        transformer.fit(X, y, epochs=5, verbose=0)

        last_input = X[-1].reshape(1, ws, 1)
        pred_lstm = lstm.predict(last_input)[0]
        pred_trf = transformer.predict(last_input)[0]

        combined_pred = (pred_lstm + pred_trf) / 2
        top3_idx = np.argsort(combined_pred)[-3:][::-1]
        top3_conf = combined_pred[top3_idx]

        result_preds.append(top3_idx.tolist())
        result_confs.append(top3_conf.tolist())

    return result_preds, result_confs
