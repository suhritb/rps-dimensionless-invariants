"""
experiments/mlp/run_multiseed_robustness.py
========================================
Multi-seed robustness check: runs both the L-scale sweep and the corrected
config-count sweep across seeds [0, 1, 2, 3, 4].

Each seed independently controls:
  - IC generation : np.random.default_rng(seed)
  - MLP init      : MLPRegressor(random_state=seed)

The sol_cache is shared between the two sweeps within each seed (same Re, Ma
grid, same ICs).  No cross-seed sharing.

Saves full output to results/multiseed_robustness.txt.

Run from the project root:
    python experiments/mlp/run_multiseed_robustness.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions,
    assemble_dataset, assemble_dataset_paired,
    features_naive, features_similarity,
)
from src.compressible_models import _MLP_KWARGS, _amplitude_weights

# ── Config ────────────────────────────────────────────────────────────────────
N         = 64
Re_values = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values = np.array([0.3, 0.5, 0.7])
n_ic      = 20
SEEDS     = [0, 1, 2, 3, 4]

# L-sweep
L_pool   = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test_l = np.array([0.4, 3.0])
ks       = [1, 2, 3, 5]

# Config-sweep (paired, corrected)
L_train_c = np.array([0.8, 1.2, 1.6])
L_test_c  = np.array([0.4, 3.0])
m_values  = [1, 2, 4, 8]

CONFIG_ORDER = [
    (2000.,  0.5),
    (10000., 0.7),
    (500.,   0.3),
    (500.,   0.7),
    (10000., 0.3),
    (1000.,  0.5),
    (5000.,  0.5),
    (2000.,  0.3),
    (2000.,  0.7),
    (1000.,  0.3),
    (5000.,  0.7),
    (1000.,  0.7),
    (5000.,  0.3),
    (10000., 0.5),
    (500.,   0.5),
]
assert len(CONFIG_ORDER) == len(Re_values) * len(Ma_values) == 15


def make_naive(seed):
    kwargs = {**_MLP_KWARGS, "random_state": seed}
    mlp = MLPRegressor(**kwargs)
    xs  = StandardScaler()
    ys  = StandardScaler()

    class _Op:
        def fit(self, u_ic, mu, L, c0, Y):
            X  = features_naive(u_ic, mu, L, c0)
            sw = _amplitude_weights(Y)
            xs.fit(X); ys.fit(Y)
            mlp.fit(xs.transform(X), ys.transform(Y), sample_weight=sw)
            return self

        def score(self, u_ic, mu, L, c0, Y):
            X    = features_naive(u_ic, mu, L, c0)
            pred = ys.inverse_transform(mlp.predict(xs.transform(X)))
            return float(
                (np.linalg.norm(pred - Y, axis=1) /
                 np.linalg.norm(Y, axis=1)).mean()
            )

    return _Op()


def make_sim(seed):
    kwargs = {**_MLP_KWARGS, "random_state": seed}
    mlp = MLPRegressor(**kwargs)
    xs  = StandardScaler()
    ys  = StandardScaler()

    class _Op:
        def fit(self, u_ic, Re, Ma, Y):
            X  = features_similarity(u_ic, Re, Ma)
            sw = _amplitude_weights(Y)
            xs.fit(X); ys.fit(Y)
            mlp.fit(xs.transform(X), ys.transform(Y), sample_weight=sw)
            return self

        def score(self, u_ic, Re, Ma, Y):
            X    = features_similarity(u_ic, Re, Ma)
            pred = ys.inverse_transform(mlp.predict(xs.transform(X)))
            return float(
                (np.linalg.norm(pred - Y, axis=1) /
                 np.linalg.norm(Y, axis=1)).mean()
            )

    return _Op()


# ── Results storage ───────────────────────────────────────────────────────────
lsweep  = {k: {"naive": [], "sim": []} for k in ks}
csweep  = {m: {"naive": [], "sim": []} for m in m_values}

# ── Main loop ─────────────────────────────────────────────────────────────────
for seed in SEEDS:
    print(f"\n{'='*68}")
    print(f"SEED {seed}")
    print(f"{'='*68}")

    rng       = np.random.default_rng(seed)
    u_perts   = [make_ic_velocity(rng, N) for _ in range(n_ic)]
    sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)

    # ── L-sweep ───────────────────────────────────────────────────────────────
    print(f"\n  [L-sweep seed={seed}]")
    u_ic_te_l, mu_te_l, L_te_l, c0_te_l, Re_te_l, Ma_te_l, Y_te_l = assemble_dataset(
        u_perts, Re_values, Ma_values, L_test_l, sol_cache)

    for k in ks:
        L_tr_k = L_pool[:k]
        u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
            u_perts, Re_values, Ma_values, L_tr_k, sol_cache)

        n_op = make_naive(seed).fit(u_ic_tr, mu_tr, L_tr, c0_tr, Y_tr)
        s_op = make_sim(seed).fit(u_ic_tr, Re_tr, Ma_tr, Y_tr)

        e_n = n_op.score(u_ic_te_l, mu_te_l, L_te_l, c0_te_l, Y_te_l)
        e_s = s_op.score(u_ic_te_l, Re_te_l, Ma_te_l, Y_te_l)
        lsweep[k]["naive"].append(e_n)
        lsweep[k]["sim"].append(e_s)
        print(f"    k={k}: naive={e_n:.4f}  sim={e_s:.4f}  gap={e_n/max(e_s,1e-9):.2f}x")

    # ── Config-sweep ──────────────────────────────────────────────────────────
    print(f"\n  [config-sweep seed={seed}]")
    for m in m_values:
        tr_pairs = CONFIG_ORDER[:m]
        te_pairs = CONFIG_ORDER[m:]
        assert set(tr_pairs).isdisjoint(set(te_pairs))

        u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset_paired(
            u_perts, tr_pairs, L_train_c, sol_cache)
        u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset_paired(
            u_perts, te_pairs, L_test_c, sol_cache)

        n_op = make_naive(seed).fit(u_ic_tr, mu_tr, L_tr, c0_tr, Y_tr)
        s_op = make_sim(seed).fit(u_ic_tr, Re_tr, Ma_tr, Y_tr)

        e_n = n_op.score(u_ic_te, mu_te, L_te, c0_te, Y_te)
        e_s = s_op.score(u_ic_te, Re_te, Ma_te, Y_te)
        csweep[m]["naive"].append(e_n)
        csweep[m]["sim"].append(e_s)
        print(f"    m={m}: naive={e_n:.4f}  sim={e_s:.4f}  gap={e_n/max(e_s,1e-9):.2f}x")


# ── Build output ──────────────────────────────────────────────────────────────
def _gap(n, s):
    return n / max(s, 1e-9)

lines = []
lines.append("=" * 72)
lines.append("MULTI-SEED ROBUSTNESS CHECK")
lines.append(f"Seeds       : {SEEDS}")
lines.append( "Seed scope  : IC generation (np.random.default_rng(seed))"
             " + MLP random_state=seed")
lines.append("Re range    : {500, 1000, 2000, 5000, 10000}  T_final=1.5")
lines.append("n_ic        : 20 per seed  |  N=64")
lines.append("=" * 72)
lines.append("")

# ── L-sweep summary table ─────────────────────────────────────────────────────
lines.append("L-SCALE SWEEP  (mean ± std across 5 seeds)")
lines.append(
    f"  {'k':>2}  {'naive_mean':>10}  {'naive_std':>9}  "
    f"{'sim_mean':>8}  {'sim_std':>8}  {'gap_mean':>8}  {'gap_std':>7}"
)
lines.append("  " + "-" * 66)
for k in ks:
    ns = lsweep[k]["naive"]
    ss = lsweep[k]["sim"]
    gs = [_gap(n, s) for n, s in zip(ns, ss)]
    lines.append(
        f"  {k:>2}  {np.mean(ns):>10.4f}  {np.std(ns):>9.4f}  "
        f"{np.mean(ss):>8.4f}  {np.std(ss):>8.4f}  "
        f"{np.mean(gs):>8.3f}  {np.std(gs):>7.3f}"
    )

lines.append("")
lines.append("L-sweep per-seed gaps:")
for k in ks:
    ns = lsweep[k]["naive"]
    ss = lsweep[k]["sim"]
    gs = [_gap(n, s) for n, s in zip(ns, ss)]
    seed_str = "  ".join(f"s{s}={g:.2f}x" for s, g in zip(SEEDS, gs))
    lines.append(f"  k={k}: {seed_str}")

lines.append("")

# ── Config-sweep summary table ────────────────────────────────────────────────
lines.append("CONFIG-COUNT SWEEP (corrected, paired)  (mean ± std across 5 seeds)")
lines.append(
    f"  {'m':>2}  {'naive_mean':>10}  {'naive_std':>9}  "
    f"{'sim_mean':>8}  {'sim_std':>8}  {'gap_mean':>8}  {'gap_std':>7}"
)
lines.append("  " + "-" * 66)
for m in m_values:
    ns = csweep[m]["naive"]
    ss = csweep[m]["sim"]
    gs = [_gap(n, s) for n, s in zip(ns, ss)]
    lines.append(
        f"  {m:>2}  {np.mean(ns):>10.4f}  {np.std(ns):>9.4f}  "
        f"{np.mean(ss):>8.4f}  {np.std(ss):>8.4f}  "
        f"{np.mean(gs):>8.3f}  {np.std(gs):>7.3f}"
    )

lines.append("")
lines.append("Config-sweep per-seed gaps:")
for m in m_values:
    ns = csweep[m]["naive"]
    ss = csweep[m]["sim"]
    gs = [_gap(n, s) for n, s in zip(ns, ss)]
    seed_str = "  ".join(f"s{s}={g:.2f}x" for s, g in zip(SEEDS, gs))
    lines.append(f"  m={m}: {seed_str}")

lines.append("")
lines.append("=" * 72)

summary = "\n".join(lines)
print("\n\n" + summary + "\n")

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results",
                        "multiseed_robustness.txt")
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(summary + "\n")
print(f"Saved to {out_path}")
