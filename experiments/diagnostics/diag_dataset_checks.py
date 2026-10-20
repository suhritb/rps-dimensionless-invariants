"""
experiments/diagnostics/diag_dataset_checks.py
==============================================
Three checks behind statements in the paper, on the exact data of the MLP
compressible scale sweep (seeds 0-4, Re >= 500 grid):

1. Degenerate targets. Counts the (IC, Re, Ma) solves with ||Y|| < 1e-3, per
   seed; every training and test row, at every length scale, reuses one of
   these solves. (Printed last, after the per-seed fits.)
   (Section 5 / Appendix C; the effect of the few remaining degenerate rows on
   every reported gap is measured by diag_degenerate_robustness.py.)

2. Feature dynamic ranges of the positive-control feature maps at k = 5:
   for each conditioning feature, max|x| / min|x| over the training rows
   (reported as "n/a (crosses 0)" when the feature changes sign).
   (Section 4.3: naive-mix within 1.8-3.4x; naive-raw's mu spans 60x.)

3. Convergence of the MLP backbone. Fits the naive and similarity operators
   of the scale sweep (k = 1, 2, 3, 5) for every seed and reports the number
   of Adam iterations used, against the max_iter = 5000 budget. A fit that
   stops before max_iter met the tol / n_iter_no_change criterion.

Output: results/dataset_checks.txt
Run from the repository root:
    python experiments/diagnostics/diag_dataset_checks.py
"""
import sys, os, time, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset,
    features_naive, features_similarity,
)
from src.compressible_models import _MLP_KWARGS, _amplitude_weights

N         = 64
Re_values = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values = np.array([0.3, 0.5, 0.7])
n_ic      = 20
SEEDS     = [0, 1, 2, 3, 4]
L_pool    = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test    = np.array([0.4, 3.0])
ks        = [1, 2, 3, 5]
DEGEN     = 1e-3


def fit_iters(featfn, feats, Y, seed):
    """Same preprocessing and weights as the paper's operators; returns n_iter_."""
    X = featfn(*feats)
    xs, ys = StandardScaler().fit(X), StandardScaler().fit(Y)
    mlp = MLPRegressor(**{**_MLP_KWARGS, "random_state": seed})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(xs.transform(X), ys.transform(Y), sample_weight=_amplitude_weights(Y))
    return mlp.n_iter_


def dyn_range(x):
    a = np.abs(x)
    if x.min() < 0 < x.max() or a.min() == 0:
        return "n/a (crosses 0)"
    return f"{a.max() / a.min():.2f}x"


out = []
def say(s=""):
    print(s, flush=True); out.append(s)

t0 = time.time()
say("DATASET CHECKS - MLP compressible scale sweep data (Re >= 500 grid)")
say(f"seeds={SEEDS}  n_ic={n_ic}  Re={Re_values.tolist()}  Ma={Ma_values.tolist()}")
say(f"L_pool={L_pool.tolist()}  L_test={L_test.tolist()}  degenerate: ||Y|| < {DEGEN}")
say()

iters, degen_rows, ranges_done = [], [], False
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    u_perts = [make_ic_velocity(rng, N) for _ in range(n_ic)]
    cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=False)

    # 1. degenerate targets (one solve per (IC, Re, Ma); every L reuses it)
    norms = np.array([np.linalg.norm(v) for v in cache.values()])
    n_deg = int((norms < DEGEN).sum())
    degen_rows.append((seed, n_deg, len(norms), float(norms.min())))

    # 2. feature dynamic ranges at k = 5 (identical across seeds: they depend only on the grid)
    if not ranges_done:
        _, mu, L, c0, Re, Ma, _ = assemble_dataset(u_perts, Re_values, Ma_values, L_pool, cache)
        arms = {"naive-log": [("log mu", np.log(mu)), ("log L", np.log(L)), ("log c0", np.log(c0))],
                "naive-mix": [("log mu", np.log(mu)), ("L", L), ("log c0", np.log(c0))],
                "naive-mu":  [("mu", mu), ("log L", np.log(L)), ("log c0", np.log(c0))],
                "naive-raw": [("mu", mu), ("L", L), ("c0", c0)]}
        say("2. Feature dynamic range at k=5 (max|x| / min|x| over training rows)")
        for arm, feats in arms.items():
            say(f"   {arm:<10} " + "  ".join(f"{n}: {dyn_range(v)}" for n, v in feats))
        say()
        ranges_done = True

    # 3. convergence
    for k in ks:
        tr = assemble_dataset(u_perts, Re_values, Ma_values, L_pool[:k], cache)
        u_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = tr
        n_n = fit_iters(features_naive, (u_tr, mu_tr, L_tr, c0_tr), Y_tr, seed)
        n_s = fit_iters(features_similarity, (u_tr, Re_tr, Ma_tr), Y_tr, seed)
        iters.append((seed, k, n_n, n_s))
        print(f"   seed {seed} k={k}: naive {n_n} iters, similarity {n_s} iters  ({time.time()-t0:.0f}s)", flush=True)

say("1. Degenerate targets (||Y|| < 1e-3) per seed, over all (IC, Re, Ma) solves")
say("   (train and test rows at every L reuse these solves)")
for seed, n_deg, n, mn in degen_rows:
    say(f"   seed {seed}: {n_deg}/{n} degenerate   min ||Y|| = {mn:.2e}")
say()
say(f"3. MLP iterations used (budget max_iter = {_MLP_KWARGS['max_iter']}; "
    f"tol = {_MLP_KWARGS['tol']}, n_iter_no_change = {_MLP_KWARGS['n_iter_no_change']})")
for seed, k, n_n, n_s in iters:
    say(f"   seed {seed} k={k}: naive {n_n:>4}  similarity {n_s:>4}")
hit = [(s, k) for s, k, a, b in iters if max(a, b) >= _MLP_KWARGS["max_iter"]]
say(f"   fits that reached max_iter: {len(hit)} of {2 * len(iters)}" + (f"  {hit}" if hit else ""))
say()
say(f"runtime: {time.time()-t0:.0f}s")

res = os.path.join(ROOT, "results"); os.makedirs(res, exist_ok=True)
with open(os.path.join(res, "dataset_checks.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
