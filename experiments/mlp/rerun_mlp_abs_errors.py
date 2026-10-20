"""
experiments/mlp/rerun_mlp_abs_errors.py
=======================================
Recovers the per-seed ABSOLUTE test errors for every MLP experiment in the paper.
The original scripts printed them to the screen but saved only means and ratios.

It re-runs, with identical seeds, data, models and losses:
  * run_multiseed_robustness.py  -> scale sweep (k = 1,2,3,5) + config sweep (m = 1,2,4,8)
  * run_positive_control.py      -> naive-log / mix / mu / raw arms vs similarity

Nothing is re-implemented: the feature maps, operator classes, MLP settings,
sample weights, data grids and seed handling are imported from (or copied
verbatim out of) those two scripts. The only change is that every raw error is
stored.

Built-in check: every recomputed ratio is compared with the value saved by the
original runs (results/mlp/positive_control.txt, 3 decimals;
results/mlp/multiseed_robustness.txt, 2-decimal per-seed gaps and 4-decimal
means). The script reports whether the rerun reproduced the originals.

Outputs (written to results/; the reported copy is in results/mlp/):
  mlp_per_seed_abs_errors.json   all raw errors, per seed
  mlp_per_seed_abs_errors.txt    readable tables + reproduction check + versions

Run from the repo root:
    python experiments/mlp/rerun_mlp_abs_errors.py
Takes roughly 15-30 minutes on a laptop CPU (the PDE solves dominate).
"""
import sys, os, re, json, time, platform
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import sklearn, scipy
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions,
    assemble_dataset, assemble_dataset_paired,
    features_naive, features_similarity,
)
from src.compressible_models import _MLP_KWARGS, _amplitude_weights

# ── Config: identical to run_multiseed_robustness.py / run_positive_control.py ──
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

# ── Feature maps: verbatim from run_positive_control.py ──
def f_log(u_ic, mu, L, c0):
    return np.hstack([u_ic, np.log(mu)[:, None], np.log(L)[:, None], np.log(c0)[:, None]])
def f_mix(u_ic, mu, L, c0):
    return np.hstack([u_ic, np.log(mu)[:, None], L[:, None], np.log(c0)[:, None]])
def f_mu(u_ic, mu, L, c0):
    return np.hstack([u_ic, mu[:, None], np.log(L)[:, None], np.log(c0)[:, None]])
def f_raw(u_ic, mu, L, c0):
    return np.hstack([u_ic, mu[:, None], L[:, None], c0[:, None]])
ARMS = [("log", f_log), ("mix", f_mix), ("mu", f_mu), ("raw", f_raw)]

# ── Operator: verbatim from run_positive_control.py (same fit/score as multiseed) ──
def make_op(featfn, seed):
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

# features_naive must equal f_log, otherwise the shared-arm shortcut is invalid
_chk = np.random.default_rng(0)
_a = [_chk.random((3, N)), _chk.random(3) + .1, _chk.random(3) + .1, _chk.random(3) + .1]
assert np.array_equal(features_naive(*_a), f_log(*_a)), "features_naive != f_log"

# ── Run ──
t0 = time.time()
data = {"scale": {}, "config": {}}
for seed in SEEDS:
    print(f"\n=== seed {seed}  ({time.time()-t0:.0f}s elapsed) ===", flush=True)
    rng     = np.random.default_rng(seed)
    u_perts = [make_ic_velocity(rng, N) for _ in range(n_ic)]
    cache   = precompute_solutions(u_perts, Re_values, Ma_values, verbose=False)

    # scale sweep: similarity + four naive arms (naive-log is the paper's naive operator)
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
        print(f"  k={k}: sim={row['sim']:.4f}  " +
              "  ".join(f"{n}={row[n]:.4f}" for n, _ in ARMS), flush=True)

    # config sweep: naive (= f_log) vs similarity, paired configs
    data["config"][seed] = {}
    for m in m_values:
        tr_pairs, te_pairs = CONFIG_ORDER[:m], CONFIG_ORDER[m:]
        u_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset_paired(
            u_perts, tr_pairs, L_train_c, cache)
        u_te2, mu_te2, L_te2, c0_te2, Re_te2, Ma_te2, Y_te2 = assemble_dataset_paired(
            u_perts, te_pairs, L_test_c, cache)
        e_n = make_op(f_log, seed).fit((u_tr, mu_tr, L_tr, c0_tr), Y_tr).score(
            (u_te2, mu_te2, L_te2, c0_te2), Y_te2)
        e_s = make_op(features_similarity, seed).fit((u_tr, Re_tr, Ma_tr), Y_tr).score(
            (u_te2, Re_te2, Ma_te2), Y_te2)
        data["config"][seed][m] = {"naive": e_n, "sim": e_s}
        print(f"  m={m}: naive={e_n:.4f}  sim={e_s:.4f}", flush=True)

# ── Reproduction check against the files the original runs saved ──
res = os.path.join(ROOT, "results", "mlp")      # saved originals, read for the reproduction check
out = os.path.join(ROOT, "results")             # where this run writes (same convention as the other scripts)
checks = []

def _cmp(label, new, old, tol):
    ok = abs(new - old) <= tol
    checks.append((label, new, old, ok))

pc = open(os.path.join(res, "positive_control.txt"), encoding="utf-8").read()
for name, _ in ARMS:
    for k in ks:
        m_ = re.search(rf"^\s*{name} k={k}: \[(.*?)\]", pc, re.M)
        old = [float(x.strip(" '")) for x in m_.group(1).split(",")]
        for s, o in zip(SEEDS, old):
            r = data["scale"][s][k]
            _cmp(f"posctrl {name} k={k} seed{s} gap", r[name] / r["sim"], o, 0.0005 + 1e-9)

ms = open(os.path.join(res, "multiseed_robustness.txt"), encoding="utf-8").read()
def _block(title):
    return ms.split(title, 1)[1]
for key, label, levels, title_tab, title_seed in [
        ("scale", "k", ks, "L-SCALE SWEEP", "L-sweep per-seed gaps:"),
        ("config", "m", m_values, "CONFIG-COUNT SWEEP", "Config-sweep per-seed gaps:")]:
    tab = _block(title_tab)
    for lv in levels:
        nk = "log" if key == "scale" else "naive"
        ns = [data[key][s][lv][nk] for s in SEEDS]; ss = [data[key][s][lv]["sim"] for s in SEEDS]
        row = re.search(rf"^\s*{lv}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", tab, re.M)
        _cmp(f"multiseed {label}={lv} naive_mean", float(np.mean(ns)), float(row.group(1)), 0.00005 + 1e-9)
        _cmp(f"multiseed {label}={lv} naive_std",  float(np.std(ns)),  float(row.group(2)), 0.00005 + 1e-9)
        _cmp(f"multiseed {label}={lv} sim_mean",   float(np.mean(ss)), float(row.group(3)), 0.00005 + 1e-9)
        _cmp(f"multiseed {label}={lv} sim_std",    float(np.std(ss)),  float(row.group(4)), 0.00005 + 1e-9)
        line = re.search(rf"^\s*{label}={lv}: (.*)$", _block(title_seed), re.M).group(1)
        old = [float(g) for g in re.findall(r"s\d=([\d.]+)x", line)]
        for s, o in zip(SEEDS, old):
            _cmp(f"multiseed {label}={lv} seed{s} gap", ns[s] / ss[s], o, 0.005 + 1e-9)

n_ok = sum(c[3] for c in checks)
verdict = ("REPRODUCED: every recomputed value matches the originally saved results."
           if n_ok == len(checks) else
           f"NOT FULLY REPRODUCED: {len(checks)-n_ok} of {len(checks)} values differ (see list).")

# ── Write outputs ──
versions = {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform()}
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "mlp_per_seed_abs_errors.json"), "w", encoding="utf-8") as fh:
    json.dump({"versions": versions, "seeds": SEEDS,
               "scale_sweep": {str(s): {str(k): v for k, v in d.items()} for s, d in data["scale"].items()},
               "config_sweep": {str(s): {str(m): v for m, v in d.items()} for s, d in data["config"].items()},
               "reproduction": {"checked": len(checks), "matched": n_ok, "verdict": verdict}},
              fh, indent=2)

L = []
L.append("MLP PER-SEED ABSOLUTE TEST ERRORS (relative L2, out-of-band test set)")
L.append(f"versions: {versions}")
L.append(f"runtime: {time.time()-t0:.0f}s   seeds={SEEDS}")
L.append("")
L.append("SCALE SWEEP — similarity + four naive arms (naive-log is the paper's naive operator)")
L.append(f"  {'k':>2} {'seed':>4} {'sim':>8} " + " ".join(f"{n:>8}" for n, _ in ARMS))
for k in ks:
    for s in SEEDS:
        r = data["scale"][s][k]
        L.append(f"  {k:>2} {s:>4} {r['sim']:>8.4f} " + " ".join(f"{r[n]:>8.4f}" for n, _ in ARMS))
L.append("")
L.append("CONFIG SWEEP — naive vs similarity")
L.append(f"  {'m':>2} {'seed':>4} {'naive':>8} {'sim':>8}")
for m in m_values:
    for s in SEEDS:
        r = data["config"][s][m]
        L.append(f"  {m:>2} {s:>4} {r['naive']:>8.4f} {r['sim']:>8.4f}")
L.append("")
L.append(f"REPRODUCTION CHECK: {n_ok}/{len(checks)} values match the original saved results")
L.append(f"  {verdict}")
for lab, new, old, ok in checks:
    if not ok:
        L.append(f"  MISMATCH {lab}: rerun={new:.6f} saved={old}")
txt = "\n".join(L)
with open(os.path.join(out, "mlp_per_seed_abs_errors.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print("\n" + txt)
