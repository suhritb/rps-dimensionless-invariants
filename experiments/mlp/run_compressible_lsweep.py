"""
experiments/mlp/run_compressible_lsweep.py
=======================================
Data-efficiency sweep over the number of distinct training SCALES (L values),
holding the (Re, Ma) grid fixed in both training and test.

Single-seed exploratory run (rng 42); console output only, not a paper number.

This is the direct compressible analog of experiments/burgers/run_burgers_sweep.py:
  - Train both operators on k = 1, 2, 3, 5 distinct L values.
  - Evaluate on out-of-band L_test = [0.4, 3.0] at ALL (Re, Ma).
  - Report naive vs similarity relative-L2 error and the gap at each k.

Architecture: identical 2-layer MLP (128x128 tanh) for both operators.
Only the feature encoding differs:
  Naive      : [u_ic | log(mu) | log(L) | log(c0)]
  Similarity : [u_ic | log(Re) | log(Ma)]

Every sample is a solve of the validated compressible NS solver
(src/compressible_solver.py, checked by tests/validate_compressible.py).

Run from the project root:
    python experiments/mlp/run_compressible_lsweep.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset,
    T_FINAL, CFL,
)
from src.compressible_models import (
    NaiveCompressibleOperator, SimilarityCompressibleOperator,
)

RNG = np.random.default_rng(42)

# ── Config ────────────────────────────────────────────────────────────────────
N          = 64
Re_values  = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values  = np.array([0.3, 0.5, 0.7])
n_ic       = 20
L_pool     = np.array([0.8, 1.2, 1.6, 2.0, 2.4])   # candidate training scales
L_test     = np.array([0.4, 3.0])                    # out-of-band test scales
ks         = [1, 2, 3, 5]                             # sweep points

# ── ICs and solutions ─────────────────────────────────────────────────────────
print("=" * 68)
print("Re range: {500, 1000, 2000, 5000, 10000}  T_final=1.5")
print("Original random multi-mode ICs (k in {1..5}), no forced modes.")
print("Relative-L2 loss: sw_i = 1/||Y_i||^2 (percentile-floored).")
print("No sample exclusion from training or metric.")
print("=" * 68)
print("Generating ICs and pre-computing solutions ...")
print("=" * 68)
u_perts  = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)
print()

# ── Fixed test set ────────────────────────────────────────────────────────────
u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset(
    u_perts, Re_values, Ma_values, L_test, sol_cache)
# ── Degenerate-sample check ───────────────────────────────────────────────────
DEGEN = 1e-3
norms_te   = np.linalg.norm(Y_te, axis=1)
n_degen_te = int((norms_te < DEGEN).sum())
print(f"Test set: {Y_te.shape[0]} samples  "
      f"(L in {L_test.tolist()}, all Re and Ma)")
print(f"Degenerate test samples (||Y|| < {DEGEN}): {n_degen_te}  "
      f"(expect 0 for Re >= 500)")
if n_degen_te > 0:
    print("  WARNING: unexpected degenerate samples — check Re/T settings")
print()

# ── L-scale sweep ─────────────────────────────────────────────────────────────
print("=" * 68)
print("L-SCALE DATA-EFFICIENCY SWEEP")
print("  Varying number of distinct training L values (k).")
print("  Both operators see ALL (Re, Ma) combinations at each k.")
print("  Test: out-of-band L in {0.4, 3.0}, ALL (Re, Ma).")
print("=" * 68)
print(f"  {'k':>3}  {'L_train':>20}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 60)

naive_curve = []
sim_curve   = []

for k in ks:
    L_train_k = L_pool[:k]

    u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
        u_perts, Re_values, Ma_values, L_train_k, sol_cache)

    naive_m = NaiveCompressibleOperator()
    naive_m.fit(u_ic_tr, mu_tr, L_tr, c0_tr, Y_tr)
    e_naive = naive_m.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te).mean()

    sim_m = SimilarityCompressibleOperator()
    sim_m.fit(u_ic_tr, Re_tr, Ma_tr, Y_tr)
    e_sim = sim_m.relative_l2(u_ic_te, Re_te, Ma_te, Y_te).mean()

    gap = e_naive / max(e_sim, 1e-9)
    naive_curve.append(float(e_naive))
    sim_curve.append(float(e_sim))

    L_str = str(L_train_k.tolist())
    print(f"  {k:>3}  {L_str:>20}  {e_naive:>10.4f}  {e_sim:>12.4f}  {gap:>7.1f}x")

# ── Per-(Re, Ma) breakdown at k=1 ────────────────────────────────────────────
print()
print("=" * 68)
print("PER-(Re, Ma) BREAKDOWN  at k=1 (hardest case for naive)")
print("=" * 68)

L_train_1 = L_pool[:1]
u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Re_tr1, Ma_tr1, Y_tr1 = assemble_dataset(
    u_perts, Re_values, Ma_values, L_train_1, sol_cache)

naive_1 = NaiveCompressibleOperator().fit(u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Y_tr1)
sim_1   = SimilarityCompressibleOperator().fit(u_ic_tr1, Re_tr1, Ma_tr1, Y_tr1)

err_naive_1 = naive_1.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te)
err_sim_1   = sim_1.relative_l2(u_ic_te, Re_te, Ma_te, Y_te)

print(f"  {'Re':>6}  {'Ma':>5}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 46)
for Re in Re_values:
    for Ma in Ma_values:
        mask   = (Re_te == Re) & (Ma_te == Ma)
        en     = err_naive_1[mask].mean()
        es     = err_sim_1[mask].mean()
        g      = en / max(es, 1e-9)
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {en:>10.4f}  {es:>12.4f}  {g:>7.1f}x")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("SUMMARY")
print("=" * 68)
print(f"  k   naive    similarity   gap")
for k, en, es in zip(ks, naive_curve, sim_curve):
    g = en / max(es, 1e-9)
    print(f"  {k:>2}  {en:.4f}   {es:.4f}      {g:.1f}x")
print()
k1_gap = naive_curve[0] / max(sim_curve[0], 1e-9)
k5_gap = naive_curve[-1] / max(sim_curve[-1], 1e-9)
print(f"  Gap at k=1: {k1_gap:.1f}x")
print(f"  Gap at k=5: {k5_gap:.1f}x")
if k1_gap > 2.0:
    print("  Similarity operator leads at low scale diversity (k=1).")
else:
    print("  NOTE: gap at k=1 is small — similarity advantage not dominant here.")
if k5_gap < 1.5:
    print("  Naive operator catches up at k=5 (invariance learnable from data).")
print()
print("=" * 68)
