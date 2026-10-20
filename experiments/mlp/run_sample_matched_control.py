"""
experiments/mlp/run_sample_matched_control.py
=============================================
Sample-matched version of the MLP compressible scale sweep.

In the main scale sweep the number of training rows grows with k:
n_train = n_ic x 15 (Re, Ma) configurations x k length scales (20 x 15 x k).
So a reader can ask whether the advantage collapses at k >= 2 simply because
the k >= 2 training sets are larger.

This control holds the training-set size fixed at ~300 rows for every k by
using fewer initial conditions as k grows:

    k = 1 : 20 ICs x 15 x 1 = 300 rows   (identical to the main sweep's k = 1)
    k = 2 : 10 ICs x 15 x 2 = 300 rows
    k = 3 :  7 ICs x 15 x 3 = 315 rows
    k = 5 :  4 ICs x 15 x 5 = 300 rows

Everything else is the main sweep's protocol: same seeds (0-4), same ICs (the
first n of the seed's IC sequence), same (Re, Ma) grid, L pool, test L, loss,
MLP settings and feature maps. As in the main sweep, the test set uses the same
ICs as training at the held-out length scales L in {0.4, 3.0}.

If the k >= 2 gap stays near 1 with the training-set size held at k = 1's level,
the collapse is not a sample-count effect.

Output: results/sample_matched_control.{txt,json}
Run from the repository root:
    python experiments/mlp/run_sample_matched_control.py
"""
import sys, os, json, time, platform
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import sklearn, scipy
from scipy import stats
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
N_IC_FOR_K = {1: 20, 2: 10, 3: 7, 5: 4}    # ~300 training rows at every k


def make_op(featfn, seed):
    """Identical to the operator used in run_multiseed_robustness.py / run_positive_control.py."""
    mlp = MLPRegressor(**{**_MLP_KWARGS, "random_state": seed})
    xs, ys = StandardScaler(), StandardScaler()
    class _Op:
        def fit(self, feats, Y):
            X = featfn(*feats); sw = _amplitude_weights(Y)
            xs.fit(X); ys.fit(Y)
            mlp.fit(xs.transform(X), ys.transform(Y), sample_weight=sw); return self
        def score(self, feats, Y):
            X = featfn(*feats)
            pred = ys.inverse_transform(mlp.predict(xs.transform(X)))
            return float((np.linalg.norm(pred - Y, axis=1) / np.linalg.norm(Y, axis=1)).mean())
    return _Op()


t0 = time.time()
rows = {}
data = {}
for seed in SEEDS:
    print(f"\n=== seed {seed}  ({time.time()-t0:.0f}s) ===", flush=True)
    rng     = np.random.default_rng(seed)
    u_perts = [make_ic_velocity(rng, N) for _ in range(n_ic)]
    cache   = precompute_solutions(u_perts, Re_values, Ma_values, verbose=False)
    data[seed] = {}
    for k in ks:
        sub = u_perts[:N_IC_FOR_K[k]]
        # assemble_dataset keys the cache by IC index, so the subset keeps its original indices
        tr = assemble_dataset(sub, Re_values, Ma_values, L_pool[:k], cache)
        te = assemble_dataset(sub, Re_values, Ma_values, L_test, cache)
        u_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = tr
        u_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = te
        rows[k] = len(Y_tr)
        e_n = make_op(features_naive, seed).fit((u_tr, mu_tr, L_tr, c0_tr), Y_tr).score(
            (u_te, mu_te, L_te, c0_te), Y_te)
        e_s = make_op(features_similarity, seed).fit((u_tr, Re_tr, Ma_tr), Y_tr).score(
            (u_te, Re_te, Ma_te), Y_te)
        data[seed][k] = {"naive": e_n, "sim": e_s, "n_train": len(Y_tr), "n_ic": N_IC_FOR_K[k]}
        print(f"  k={k} ({N_IC_FOR_K[k]} ICs, {len(Y_tr)} rows): naive={e_n:.4f} sim={e_s:.4f} "
              f"gap={e_n/e_s:.2f}x", flush=True)

# ── Statistics: geometric mean of per-seed gaps, 95% CI and one-sample t on log-gaps (df=4) ──
tc = float(stats.t.ppf(0.975, len(SEEDS) - 1))
summary = {}
for k in ks:
    lg = np.log([data[s][k]["naive"] / data[s][k]["sim"] for s in SEEDS])
    m, se = lg.mean(), lg.std(ddof=1) / np.sqrt(len(lg))
    p = float(stats.ttest_1samp(lg, 0.0).pvalue)
    summary[k] = {"geo_gap": float(np.exp(m)), "ci_lo": float(np.exp(m - tc * se)),
                  "ci_hi": float(np.exp(m + tc * se)), "p": p,
                  "per_seed_gaps": [float(np.exp(x)) for x in lg]}
pooled_lg = np.concatenate([np.log([data[s][k]["naive"] / data[s][k]["sim"] for s in SEEDS]) for k in ks if k >= 2])
pooled = float(np.exp(pooled_lg.mean()))

versions = {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}
out = os.path.join(ROOT, "results"); os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "sample_matched_control.json"), "w", encoding="utf-8") as fh:
    json.dump({"versions": versions, "seeds": SEEDS, "n_ic_for_k": N_IC_FOR_K, "n_train_rows": rows,
               "per_seed": {str(s): {str(k): v for k, v in d.items()} for s, d in data.items()},
               "summary": {str(k): v for k, v in summary.items()}, "pooled_k_ge_2": pooled}, fh, indent=2)
L = ["SAMPLE-MATCHED SCALE SWEEP (MLP, compressible) — ~300 training rows at every k",
     f"versions: {versions}", f"runtime: {time.time()-t0:.0f}s  seeds={SEEDS}", "",
     f"  {'k':>2} {'ICs':>4} {'rows':>5} {'geo gap':>8} {'95% CI':>16} {'p':>7}   per-seed gaps"]
for k in ks:
    s = summary[k]
    L.append(f"  {k:>2} {N_IC_FOR_K[k]:>4} {rows[k]:>5} {s['geo_gap']:>7.2f}x  [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}] "
             f"{s['p']:>7.3f}   " + " ".join(f"{g:.2f}" for g in s["per_seed_gaps"]))
L += [f"  pooled k>=2 geometric gap = {pooled:.3f}x", "",
      "Per-seed absolute errors (naive, sim):"]
for k in ks:
    L.append(f"  k={k}: " + "  ".join(f"s{s}=({data[s][k]['naive']:.4f}, {data[s][k]['sim']:.4f})" for s in SEEDS))
txt = "\n".join(L)
with open(os.path.join(out, "sample_matched_control.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print("\n" + txt)
