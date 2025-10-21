import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

class BaselineModel:
    def __init__(self):
        base = LogisticRegression(max_iter=200, solver='liblinear')
        self.model = OneVsRestClassifier(base)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict_proba(self, X):
        probas = [clf.predict_proba(X)[:, 1] for clf in self.model.estimators_]
        return np.vstack(probas).T  # shape (n_samples, n_classes)
