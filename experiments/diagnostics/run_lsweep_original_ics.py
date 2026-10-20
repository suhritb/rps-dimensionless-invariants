"""
experiments/diagnostics/run_lsweep_original_ics.py
========================================
Same experiment as run_compressible_lsweep.py but with:
  - ORIGINAL IC distribution (k drawn from {1..5}, no k=1 guarantee)
  - Percentile-floored sample weights  sw_i = 1/||Y_i||^2  for training
  - Degenerate test samples (||Y|| < DEGEN_THRESHOLD) excluded from metric;
    they remain in the training set so the optimizer sees them.

Purpose: isolate the contribution of the loss fix alone, without the IC
change that was reverted.  Compare side-by-side with the forced-k=1 run.

Run from the project root:
    python experiments/diagnostics/run_lsweep_original_ics.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset,
)
from src.compressible_models import (
    NaiveCompressibleOperator, SimilarityCompressibleOperator,
)

RNG = np.random.default_rng(42)   # same seed as other runs

N          = 64
Re_values  = np.array([50., 100., 200., 500., 1000.])
Ma_values  = np.array([0.3, 0.5, 0.7])
n_ic       = 20
L_pool     = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test     = np.array([0.4, 3.0])
ks         = [1, 2, 3, 5]

# Samples with ||Y|| below this are physically near-equilibrium (all modes
# decayed to machine precision).  The relative-L2 metric is undefined for
# them (numerator / epsilon).  Excluded from metric; kept in training set.
DEGEN_THRESHOLD = 1e-3

# ── ICs and solutions ─────────────────────────────────────────────────────────
print("=" * 68)
print("Generating ICs — ORIGINAL distribution (k random from {1..5})")
print("=" * 68)
u_perts   = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)
print()

# ── Test set + degenerate mask ────────────────────────────────────────────────
u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset(
    u_perts, Re_values, Ma_values, L_test, sol_cache)

norms_te     = np.linalg.norm(Y_te, axis=1)
non_degen    = norms_te >= DEGEN_THRESHOLD
n_total      = len(Y_te)
n_excluded   = int((~non_degen).sum())

print(f"Test set: {n_total} samples total")
print(f"Excluded from metric (||Y|| < {DEGEN_THRESHOLD}): "
      f"{n_excluded}/{n_total} samples")
print()
print(f"  {'Re':>6}  {'Ma':>5}  {'n_eval':>6}  {'n_excl':>6}  "
      f"{'||Y|| min (non-degen)':>22}")
print("  " + "-" * 55)
for Re in Re_values:
    for Ma in Ma_values:
        m     = (Re_te == Re) & (Ma_te == Ma)
        good  = m & non_degen
        n_ev  = int(good.sum())
        n_ex  = int(m.sum()) - n_ev
        if n_ev > 0:
            min_norm = float(norms_te[good].min())
            print(f"  {Re:>6.0f}  {Ma:>5.2f}  {n_ev:>6}  {n_ex:>6}  {min_norm:>22.6f}")
        else:
            print(f"  {Re:>6.0f}  {Ma:>5.2f}  {n_ev:>6}  {n_ex:>6}  {'ALL EXCLUDED':>22}")
print()

# ── Sweep ─────────────────────────────────────────────────────────────────────
print("=" * 68)
print("L-SCALE SWEEP — original ICs, loss-weighting + percentile floor,")
print(f"                metric on non-degenerate test samples (||Y|| >= {DEGEN_THRESHOLD})")
print("=" * 68)
print(f"  {'k':>3}  {'L_train':>20}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 60)

naive_curve, sim_curve = [], []

for k in ks:
    L_train_k = L_pool[:k]
    u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
        u_perts, Re_values, Ma_values, L_train_k, sol_cache)

    naive_m = NaiveCompressibleOperator().fit(u_ic_tr, mu_tr, L_tr, c0_tr, Y_tr)
    sim_m   = SimilarityCompressibleOperator().fit(u_ic_tr, Re_tr, Ma_tr, Y_tr)

    # Full-set predictions, then mask to non-degenerate for the mean
    e_n_all = naive_m.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te)
    e_s_all = sim_m.relative_l2(u_ic_te, Re_te, Ma_te, Y_te)

    e_naive = float(e_n_all[non_degen].mean())
    e_sim   = float(e_s_all[non_degen].mean())
    gap     = e_naive / max(e_sim, 1e-9)
    naive_curve.append(e_naive)
    sim_curve.append(e_sim)

    L_str = str(L_train_k.tolist())
    print(f"  {k:>3}  {L_str:>20}  {e_naive:>10.4f}  {e_sim:>12.4f}  {gap:>7.1f}x")

# ── Per-(Re,Ma) breakdown at k=1 ─────────────────────────────────────────────
print()
print("=" * 68)
print("PER-(Re, Ma) BREAKDOWN at k=1  (non-degenerate test samples only)")
print("=" * 68)

L_train_1 = L_pool[:1]
u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Re_tr1, Ma_tr1, Y_tr1 = assemble_dataset(
    u_perts, Re_values, Ma_values, L_train_1, sol_cache)

naive_1 = NaiveCompressibleOperator().fit(u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Y_tr1)
sim_1   = SimilarityCompressibleOperator().fit(u_ic_tr1, Re_tr1, Ma_tr1, Y_tr1)

err_n1 = naive_1.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te)
err_s1 = sim_1.relative_l2(u_ic_te, Re_te, Ma_te, Y_te)

print(f"  {'Re':>6}  {'Ma':>5}  {'n_eval':>6}  {'n_excl':>6}  "
      f"{'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 65)
for Re in Re_values:
    for Ma in Ma_values:
        m    = (Re_te == Re) & (Ma_te == Ma)
        mask = m & non_degen
        n_ev = int(mask.sum())
        n_ex = int(m.sum()) - n_ev
        if n_ev == 0:
            print(f"  {Re:>6.0f}  {Ma:>5.2f}  {n_ev:>6}  {n_ex:>6}  "
                  f"{'ALL EXCLUDED':>24}")
            continue
        en = float(err_n1[mask].mean())
        es = float(err_s1[mask].mean())
        g  = en / max(es, 1e-9)
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {n_ev:>6}  {n_ex:>6}  "
              f"{en:>10.4f}  {es:>12.4f}  {g:>7.1f}x")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("SUMMARY")
print("=" * 68)
print(f"  {'k':>2}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
for i, k in enumerate(ks):
    g = naive_curve[i] / max(sim_curve[i], 1e-9)
    print(f"  {k:>2}  {naive_curve[i]:>10.4f}  {sim_curve[i]:>12.4f}  {g:>7.1f}x")
print()
k1_gap = naive_curve[0] / max(sim_curve[0], 1e-9)
k5_gap = naive_curve[-1] / max(sim_curve[-1], 1e-9)
print(f"  Gap at k=1: {k1_gap:.1f}x  (metric on non-degenerate subset)")
print(f"  Gap at k=5: {k5_gap:.1f}x")
print()
print(f"  Degenerate excluded: {n_excluded}/{n_total} test samples (||Y|| < {DEGEN_THRESHOLD})")
print("=" * 68)
