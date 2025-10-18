import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

class BaselineModel:
    def __init__(self):
        base = LogisticRegression(solver='liblinear', max_iter=1000)
        self.model = OneVsRestClassifier(base, n_jobs=4)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)
        except Exception:
            dec = self.model.decision_function(X)
            return 1/(1+np.exp(-dec))
