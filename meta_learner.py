import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MetaLearner:
    def __init__(self):
        self.models = [RandomForestClassifier(n_estimators=50) for _ in range(4)]

    def train(self, data, labels):
        # data: list of shape [n_samples, 4, n_features]
        # labels: list of shape [n_samples, 4]
        data = np.array(data)
        labels = np.array(labels)
        for i in range(4):
            X = data[:, i, :]
            y = labels[:, i]
            self.models[i].fit(X, y)

    def predict_top3(self, input_features):
        # input_features: list of 4 arrays [LSTM, TRF, MARKOV] per posisi
        result_preds, result_confs = [], []
        for i in range(4):
            feat = np.array(input_features[i]).reshape(1, -1)
            prob = self.models[i].predict_proba(feat)[0]
            prob = prob / prob.sum()
            top3 = np.argsort(prob)[-3:][::-1]
            result_preds.append(top3.tolist())
            result_confs.append(prob[top3].tolist())
        return result_preds, result_confs
