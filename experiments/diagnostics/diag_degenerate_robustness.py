"""
experiments/diagnostics/diag_degenerate_robustness.py
=====================================================
On the Re >= 500 grid a few solves per seed still fall below the degeneracy
threshold ||Y|| < 1e-3 (diag_dataset_checks.py: 2/300 for seed 0, 5/300 for
seed 3, 0/300 for seeds 1, 2, 4). Because the test metric is a per-sample
relative L2 error, such rows can carry large relative errors.

This script re-runs the reported MLP experiments with the code of
experiments/mlp/rerun_mlp_abs_errors.py (same seeds, data, feature maps,
operators, losses) and scores every model twice on the same test set:

  all rows        - the metric used in the paper
  non-degenerate  - test rows with ||Y|| >= 1e-3 only (training unchanged)

and reports every summary gap of the paper both ways.

Output: results/degenerate_robustness.{txt,json}
Run from the repository root:
    python experiments/diagnostics/diag_degenerate_robustness.py
Takes roughly 15-30 minutes on a laptop CPU.
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
    make_ic_velocity, precompute_solutions,
    assemble_dataset, assemble_dataset_paired, features_similarity,
)
from src.compressible_models import _MLP_KWARGS, _amplitude_weights

# Configuration: identical to experiments/mlp/rerun_mlp_abs_errors.py
N         = 64
Re_values = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values = np.array([0.3, 0.5, 0.7])
n_ic      = 20
SEEDS     = [0, 1, 2, 3, 4]
L_pool    = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test_l  = np.array([0.4, 3.0])
ks        = [1, 2, 3, 5]
L_train_c = np.array([0.8, 1.2, 1.6])
L_test_c  = np.array([0.4, 3.0])
m_values  = [1, 2, 4, 8]
CONFIG_ORDER = [
    (2000., 0.5), (10000., 0.7), (500., 0.3), (500., 0.7), (10000., 0.3),
    (1000., 0.5), (5000., 0.5), (2000., 0.3), (2000., 0.7), (1000., 0.3),
    (5000., 0.7), (1000., 0.7), (5000., 0.3), (10000., 0.5), (500., 0.5),
]
DEGEN = 1e-3

def f_log(u_ic, mu, L, c0):
    return np.hstack([u_ic, np.log(mu)[:, None], np.log(L)[:, None], np.log(c0)[:, None]])
def f_mix(u_ic, mu, L, c0):
    return np.hstack([u_ic, np.log(mu)[:, None], L[:, None], np.log(c0)[:, None]])
def f_mu(u_ic, mu, L, c0):
    return np.hstack([u_ic, mu[:, None], np.log(L)[:, None], np.log(c0)[:, None]])
def f_raw(u_ic, mu, L, c0):
    return np.hstack([u_ic, mu[:, None], L[:, None], c0[:, None]])
ARMS = [("log", f_log), ("mix", f_mix), ("mu", f_mu), ("raw", f_raw)]


def make_op(featfn, seed):
    """Same fit as rerun_mlp_abs_errors.py; score returns (all rows, non-degenerate rows)."""
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
            nrm = np.linalg.norm(Y, axis=1)
            rel = np.linalg.norm(pred - Y, axis=1) / nrm
            keep = nrm >= DEGEN
            return float(rel.mean()), float(rel[keep].mean()), int((~keep).sum())
    return _Op()


tc = float(stats.t.ppf(0.975, len(SEEDS) - 1))
def geo(gaps):
    lg = np.log(gaps); m, se = lg.mean(), lg.std(ddof=1) / np.sqrt(len(lg))
    return np.exp(m), np.exp(m - tc * se), np.exp(m + tc * se)


t0 = time.time()
data = {"scale": {}, "config": {}}
for seed in SEEDS:
    print(f"\n=== seed {seed}  ({time.time()-t0:.0f}s) ===", flush=True)
    rng     = np.random.default_rng(seed)
    u_perts = [make_ic_velocity(rng, N) for _ in range(n_ic)]
    cache   = precompute_solutions(u_perts, Re_values, Ma_values, verbose=False)

    te = assemble_dataset(u_perts, Re_values, Ma_values, L_test_l, cache)
    u_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = te
    data["scale"][seed] = {}
    for k in ks:
        u_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
            u_perts, Re_values, Ma_values, L_pool[:k], cache)
        row = {"sim": make_op(features_similarity, seed).fit(
            (u_tr, Re_tr, Ma_tr), Y_tr).score((u_te, Re_te, Ma_te), Y_te)}
        for name, fn in ARMS:
            row[name] = make_op(fn, seed).fit(
                (u_tr, mu_tr, L_tr, c0_tr), Y_tr).score((u_te, mu_te, L_te, c0_te), Y_te)
        data["scale"][seed][k] = row
        print(f"  k={k}: degenerate test rows={row['sim'][2]}  " +
              "  ".join(f"{n}={row[n][0]:.4f}/{row[n][1]:.4f}" for n in ["sim"] + [a for a, _ in ARMS]), flush=True)

    data["config"][seed] = {}
    for m in m_values:
        tr_pairs, te_pairs = CONFIG_ORDER[:m], CONFIG_ORDER[m:]
        u_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset_paired(
            u_perts, tr_pairs, L_train_c, cache)
        u_t2, mu_t2, L_t2, c0_t2, Re_t2, Ma_t2, Y_t2 = assemble_dataset_paired(
            u_perts, te_pairs, L_test_c, cache)
        e_n = make_op(f_log, seed).fit((u_tr, mu_tr, L_tr, c0_tr), Y_tr).score(
            (u_t2, mu_t2, L_t2, c0_t2), Y_t2)
        e_s = make_op(features_similarity, seed).fit((u_tr, Re_tr, Ma_tr), Y_tr).score(
            (u_t2, Re_t2, Ma_t2), Y_t2)
        data["config"][seed][m] = {"naive": e_n, "sim": e_s}
        print(f"  m={m}: degenerate test rows={e_s[2]}  naive={e_n[0]:.4f}/{e_n[1]:.4f}  "
              f"sim={e_s[0]:.4f}/{e_s[1]:.4f}", flush=True)

# ── Summary: every gap both ways ──
L = ["DEGENERATE-ROW ROBUSTNESS - MLP compressible experiments",
     f"Test rows with ||Y|| < {DEGEN} excluded in the second column; training unchanged.",
     "Gap = geometric mean of per-seed ratios naive/similarity, 95% CI Student t on log-gaps (df=4).", ""]
L.append("Degenerate test rows per seed (scale sweep test set, of 600; config sweep test set at m=1/2/4/8):")
for s in SEEDS:
    L.append(f"  seed {s}: scale {data['scale'][s][1]['sim'][2]}   config " +
             "/".join(str(data['config'][s][m]['sim'][2]) for m in m_values))
L.append("")
summary = {}
def block(title, keys, getter):
    L.append(title)
    L.append(f"  {'level':<8} {'all rows':>24}   {'non-degenerate rows':>24}")
    for key in keys:
        g_all = np.array([getter(s, key, 0) for s in SEEDS])
        g_nd  = np.array([getter(s, key, 1) for s in SEEDS])
        a, n = geo(g_all), geo(g_nd)
        summary[f"{title}|{key}"] = {"all": a, "nondegenerate": n}
        L.append(f"  {str(key):<8} {a[0]:>6.2f}x [{a[1]:.2f}, {a[2]:.2f}]     {n[0]:>6.2f}x [{n[1]:.2f}, {n[2]:.2f}]")
    L.append("")
for arm, _ in ARMS:
    block(f"Scale sweep, naive-{arm} vs similarity", ks,
          lambda s, k, i, arm=arm: data["scale"][s][k][arm][i] / data["scale"][s][k]["sim"][i])
block("Configuration sweep, naive-log vs similarity", m_values,
      lambda s, m, i: data["config"][s][m]["naive"][i] / data["config"][s][m]["sim"][i])
for arm, _ in ARMS:
    for i, lab in [(0, "all rows"), (1, "non-degenerate")]:
        lg = np.concatenate([np.log([data["scale"][s][k][arm][i] / data["scale"][s][k]["sim"][i] for s in SEEDS])
                             for k in ks if k >= 2])
        L.append(f"  pooled k>=2 naive-{arm} ({lab}): {np.exp(lg.mean()):.3f}x")
L.append("")
versions = {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}
L += [f"versions: {versions}", f"runtime: {time.time()-t0:.0f}s"]
txt = "\n".join(L)
print("\n" + txt)
out = os.path.join(ROOT, "results"); os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "degenerate_robustness.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
with open(os.path.join(out, "degenerate_robustness.json"), "w", encoding="utf-8") as fh:
    json.dump({"versions": versions, "degenerate_threshold": DEGEN,
               "per_seed": {kind: {str(s): {str(k): v for k, v in d.items()} for s, d in data[kind].items()}
                            for kind in data},
               "summary": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in summary.items()}},
              fh, indent=2)
    fh.write("\n")
