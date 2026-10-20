"""
models.py
=========
Operator models for the similarity experiment.

All models share the same MLP backbone (sklearn MLPRegressor) and identical
hyperparameters. The ONLY difference is which feature encoding they receive:
  - NaiveOperator       <- features_naive   (log nu, log L separately)
  - SimilarityOperator  <- features_similarity (log Re)

This isolates the inductive bias as the sole variable.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .dataset import features_naive, features_similarity

# Shared MLP hyperparameters — identical for both operators
_MLP_KWARGS = dict(
    hidden_layer_sizes=(128, 128),
    activation="tanh",
    solver="adam",
    alpha=1e-6,
    max_iter=1200,
    tol=1e-7,
    n_iter_no_change=30,
    random_state=0,
)


class NaiveOperator:
    """
    Baseline neural operator conditioned on (nu, L) separately.
    Must learn the dynamical-similarity invariance from data.
    """

    def __init__(self):
        self.mlp = MLPRegressor(**_MLP_KWARGS)
        self.xs = StandardScaler()
        self.ys = StandardScaler()

    def fit(self, Xic, nu, L, Y):
        X = features_naive(Xic, nu, L)
        self.xs.fit(X)
        self.ys.fit(Y)
        self.mlp.fit(self.xs.transform(X), self.ys.transform(Y))
        return self

    def predict(self, Xic, nu, L):
        X = features_naive(Xic, nu, L)
        return self.ys.inverse_transform(self.mlp.predict(self.xs.transform(X)))

    def relative_l2(self, Xic, nu, L, Y_true):
        pred = self.predict(Xic, nu, L)
        return (np.linalg.norm(pred - Y_true, axis=1)
                / np.linalg.norm(Y_true, axis=1))


class SimilarityOperator:
    """
    Similarity-invariant neural operator conditioned only on Re = L/nu.
    The exact dynamical-similarity symmetry is baked in as the sole scalar.
    """

    def __init__(self):
        self.mlp = MLPRegressor(**_MLP_KWARGS)
        self.xs = StandardScaler()
        self.ys = StandardScaler()

    def fit(self, Xic, Re, Y):
        X = features_similarity(Xic, Re)
        self.xs.fit(X)
        self.ys.fit(Y)
        self.mlp.fit(self.xs.transform(X), self.ys.transform(Y))
        return self

    def predict(self, Xic, Re):
        X = features_similarity(Xic, Re)
        return self.ys.inverse_transform(self.mlp.predict(self.xs.transform(X)))

    def relative_l2(self, Xic, Re, Y_true):
        pred = self.predict(Xic, Re)
        return (np.linalg.norm(pred - Y_true, axis=1)
                / np.linalg.norm(Y_true, axis=1))
