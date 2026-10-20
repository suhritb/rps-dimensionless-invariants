"""
src/compressible_models.py
==========================
Operator models for the compressible-flow similarity experiment.

Both classes share an identical MLP backbone and identical hyperparameters.
The ONLY difference is the feature encoding:

  NaiveCompressibleOperator      <- features_naive     (log mu, log L, log c0)
  SimilarityCompressibleOperator <- features_similarity (log Re, log Ma)

Loss function
-------------
Solution amplitudes differ substantially across (Re, Ma), so an unweighted
squared-error loss would be dominated by the largest-amplitude samples.
Per-sample weights give a relative-L2-style objective in which every
(Re, Ma) sample contributes comparably despite amplitude differences.

Targets are standardized per output column by StandardScaler (mean m_j,
std sigma_j), and the weights are passed via
MLPRegressor.fit(..., sample_weight=w) (scikit-learn >= 1.7). Written in
original units, the training loss is

    L  ∝  sum_i  w_i * sum_j (yhat_ij - y_ij)^2 / sigma_j^2,
    w_i = 1 / ||Y_i||^2   (||Y_i||^2 floored at its 10th percentile),

i.e. an amplitude-weighted squared error that approximates a per-sample
relative L2 loss (exact only if all sigma_j were equal).

Normalization guarantee
-----------------------
  - The TARGET Y is the true solution field; it is never rescaled per
    sample. No analytic decay factor, no IC norm, no per-sample output
    division appears anywhere in the target or predictions.
  - w_i = 1/||Y_i||^2 is used ONLY as a training-time loss weight;
    at PREDICTION TIME only the input X is used, Y is never accessed.
  - The per-column StandardScaler on Y is applied for numerical stability
    and is inverted at prediction time.
  - Both operators use identical MLP kwargs and identical sw formula.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .compressible_dataset import features_naive, features_similarity


_MLP_KWARGS = dict(
    hidden_layer_sizes=(128, 128),
    activation="tanh",
    solver="adam",
    alpha=1e-6,
    max_iter=5000,
    tol=1e-7,
    n_iter_no_change=50,
    random_state=0,
)


def _amplitude_weights(Y):
    """
    Per-sample loss weights: sw_i = 1 / ||Y_i||^2.
    Approximates a per-sample relative-L2 training loss via sklearn sample_weight.
    Y is the unmodified true solution field (not scaled, not divided).

    Floor: clip ||Y_i||^2 at max(p10_sq, 1e-6) where p10_sq is the 10th
    percentile of the squared norms.  This prevents any near-zero sample
    (e.g. an IC whose dominant modes have fully decayed at T_FINAL) from
    receiving a weight orders of magnitude above its neighbours, which would
    collapse the Adam step size and stop training after ~150 iterations.
    The floor does not affect samples with typical amplitudes.
    """
    norms_sq = np.sum(Y**2, axis=1)                  # ||Y_i||^2, shape (n,)
    p10_sq   = float(np.percentile(norms_sq, 10))
    floor    = max(p10_sq, 1e-6)
    return 1.0 / np.maximum(norms_sq, floor)


# ── Operator classes ──────────────────────────────────────────────────────────

class NaiveCompressibleOperator:
    """
    Baseline operator conditioned on raw dimensional parameters
    (mu, L, c0) separately via log(mu), log(L), log(c0) scalar features.
    Must discover from data that only (L/mu, 1/c0) = (Re, Ma) govern flow.

    Target Y: true unmodified solution field.  Loss weighted by 1/||Y_i||^2
    so low-Re (small amplitude) and high-Re (large amplitude) contribute
    comparably to the training objective.
    """

    def __init__(self):
        self.mlp = MLPRegressor(**_MLP_KWARGS)
        self.xs  = StandardScaler()
        self.ys  = StandardScaler()

    def fit(self, u_ic, mu, L, c0, Y):
        X  = features_naive(u_ic, mu, L, c0)
        sw = _amplitude_weights(Y)                   # from unmodified Y
        self.xs.fit(X)
        self.ys.fit(Y)
        self.mlp.fit(
            self.xs.transform(X),
            self.ys.transform(Y),
            sample_weight=sw,                        # only weighting changes
        )
        return self

    def predict(self, u_ic, mu, L, c0):
        X    = features_naive(u_ic, mu, L, c0)
        Y_sc = self.mlp.predict(self.xs.transform(X))
        return self.ys.inverse_transform(Y_sc)

    def relative_l2(self, u_ic, mu, L, c0, Y_true):
        pred = self.predict(u_ic, mu, L, c0)
        return (np.linalg.norm(pred - Y_true, axis=1)
                / np.linalg.norm(Y_true, axis=1))


class SimilarityCompressibleOperator:
    """
    Similarity-invariant operator conditioned on (Re, Ma) directly via
    log(Re) and log(Ma) — the exact dynamical-similarity invariants.

    Same loss, same architecture, same normalization as NaiveCompressibleOperator.
    The only difference is the feature encoding.
    """

    def __init__(self):
        self.mlp = MLPRegressor(**_MLP_KWARGS)
        self.xs  = StandardScaler()
        self.ys  = StandardScaler()

    def fit(self, u_ic, Re, Ma, Y):
        X  = features_similarity(u_ic, Re, Ma)
        sw = _amplitude_weights(Y)
        self.xs.fit(X)
        self.ys.fit(Y)
        self.mlp.fit(
            self.xs.transform(X),
            self.ys.transform(Y),
            sample_weight=sw,
        )
        return self

    def predict(self, u_ic, Re, Ma):
        X    = features_similarity(u_ic, Re, Ma)
        Y_sc = self.mlp.predict(self.xs.transform(X))
        return self.ys.inverse_transform(Y_sc)

    def relative_l2(self, u_ic, Re, Ma, Y_true):
        pred = self.predict(u_ic, Re, Ma)
        return (np.linalg.norm(pred - Y_true, axis=1)
                / np.linalg.norm(Y_true, axis=1))
