import numpy as np
from models import build_lstm_block, build_transformer_block, window_data
from markov_model import top6_markov_hybrid
from automl import find_optimal_window
from meta_learner import MetaLearner
from pattern_filter import filter_top_combinations

def simulate_model_accuracy(X, y, model, last_input):
    model.fit(X, y, epochs=3, verbose=0)
    preds = model.predict(X, verbose=0)
    top3_hits = sum(y[i] in np.argsort(preds[i])[-3:] for i in range(len(y)))
    last_pred = model.predict(last_input, verbose=0)[0]
    return top3_hits / len(y), last_pred

def final_prediction_pipeline(data):
    result_preds = []
    result_confs = []
    meta_inputs = []

    for pos in range(4):
        ws = find_optimal_window(data, pos, (10, 30))
        X, y = window_data(data, pos, ws)
        if len(X) < 5:
            raise ValueError(f"Data terlalu sedikit untuk posisi {pos}")
        X = X.reshape((X.shape[0], X.shape[1], 1))
        last_input = X[-1].reshape(1, ws, 1)

        # Build & train models
        lstm = build_lstm_block((ws,1))
        trf = build_transformer_block((ws,1))
        lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        trf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        acc_lstm, pred_lstm = simulate_model_accuracy(X, y, lstm, last_input)
        acc_trf, pred_trf = simulate_model_accuracy(X, y, trf, last_input)
        _, pred_markov = top6_markov_hybrid(data, pos)
        pred_markov = pred_markov / pred_markov.sum()

        # Meta input = concatenated pred vectors
        meta_input = np.concatenate([pred_lstm, pred_trf, pred_markov])
        meta_inputs.append(meta_input)

    # Meta prediction (model should be trained offline for real use)
    meta = MetaLearner()
    preds, confs = meta.predict_top3(meta_inputs)

    result_preds = preds
    result_confs = confs

    # Kombinasi 4D terbaik dari pattern filter
    top10_combinations = filter_top_combinations(result_preds, top_k=10)

    return {
        "top3_per_posisi": result_preds,
        "confidences": result_confs,
        "top10_kombinasi": top10_combinations
    }
