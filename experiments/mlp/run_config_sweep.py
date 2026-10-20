"""
experiments/mlp/run_config_sweep.py
================================
Configuration-count sweep: vary the number of distinct (Re, Ma) training
configurations and test on held-out (Re, Ma) pairs not seen in training.

Single-seed exploratory run (rng 42); console output only, not a paper number.

This is the two-invariant test.  The L-scale sweep showed that naive can
discover the single invariant Re = L/mu given enough L values.  Here the L
coverage is fixed and the (Re, Ma) coverage varies instead, asking: does knowing
(Re, Ma) directly give a durable advantage when the operator must generalise
to new points in the invariant plane?

Design
------
  All configs:  5 Re × 3 Ma = 15 (Re, Ma) pairs
  CONFIG_ORDER: a fixed space-filling order of all 15 pairs (see below),
                so the first m entries spread over the invariant plane.
  Sweep m = 1, 2, 4, 8 training configs.
  Training:     m (Re, Ma) pairs × L_train {0.8, 1.2, 1.6} × 20 ICs
  Test:         held-out (15 - m) (Re, Ma) pairs × L_test {0.4, 3.0} × 20 ICs
                  — new L values AND new (Re, Ma): both generalisation axes.

Re range:  {500, 1000, 2000, 5000, 10000}.  T_final = 1.5.
ICs:       original random multi-mode (k in {1..5}).  No forced modes.
Loss:      sw_i = 1/||Y_i||^2 (percentile-floored).  Target Y unmodified.
Exclusion: none.

Run from the project root:
    python experiments/mlp/run_config_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset,
    assemble_dataset_paired,
)
from src.compressible_models import (
    NaiveCompressibleOperator, SimilarityCompressibleOperator,
)

RNG = np.random.default_rng(42)

N         = 64
Re_values = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values = np.array([0.3, 0.5, 0.7])
n_ic      = 20
L_train   = np.array([0.8, 1.2, 1.6])
L_test    = np.array([0.4, 3.0])
m_values  = [1, 2, 4, 8]

# ── Space-filling CONFIG_ORDER ────────────────────────────────────────────────
# Ordered so that the first m entries spread over the (log Re, Ma) square
# for each m in {1, 2, 4, 8}.
# m=1: center (2000, 0.5)
# m=2: add the high-Re / high-Ma corner (10000, 0.7)
# m=4: add the low-Re / low-Ma corner (500, 0.3) and the low-Re / high-Ma
#      corner (500, 0.7)
# m=8: add the high-Re / low-Ma corner (10000, 0.3), the Ma=0.5 edge
#      mid-points (1000, 0.5) and (5000, 0.5), and (2000, 0.3)
CONFIG_ORDER = [
    (2000.,  0.5),   #  1  center
    (10000., 0.7),   #  2  high-Re / high-Ma corner
    (500.,   0.3),   #  3  low-Re  / low-Ma  corner
    (500.,   0.7),   #  4  low-Re  / high-Ma corner
    (10000., 0.3),   #  5  high-Re / low-Ma  corner
    (1000.,  0.5),   #  6  low-mid edge
    (5000.,  0.5),   #  7  high-mid edge
    (2000.,  0.3),   #  8  center bottom
    # remaining 7 for completeness
    (2000.,  0.7),   #  9  center top
    (1000.,  0.3),   # 10
    (5000.,  0.7),   # 11
    (1000.,  0.7),   # 12
    (5000.,  0.3),   # 13
    (10000., 0.5),   # 14
    (500.,   0.5),   # 15
]
assert len(CONFIG_ORDER) == len(Re_values) * len(Ma_values) == 15

# ── ICs and full solution cache ───────────────────────────────────────────────
print("=" * 68)
print("CONFIG-COUNT SWEEP — (Re, Ma) generalisation test")
print("Re in {500, 1000, 2000, 5000, 10000}  T_final=1.5")
print("Original random ICs (k in {1..5}).  No forced modes.  No exclusions.")
print("L_train = [0.8, 1.2, 1.6]   L_test = [0.4, 3.0]")
print("=" * 68)
print("Generating ICs and pre-computing solutions ...")
u_perts   = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)
print()

# Degenerate check (expect 0 for Re >= 500 at T=1.5)
_, _, _, _, Re_all, Ma_all, Y_all = assemble_dataset(
    u_perts, Re_values, Ma_values, L_train, sol_cache)
norms_all = np.linalg.norm(Y_all, axis=1)
n_degen   = int((norms_all < 1e-3).sum())
print(f"Training pool: {len(Y_all)} samples | degenerate (||Y||<1e-3): {n_degen}")
if n_degen > 0:
    print("  WARNING: unexpected degenerate samples")
print()

# ── Sweep ─────────────────────────────────────────────────────────────────────
print("=" * 68)
print("SWEEP: m = number of (Re, Ma) training configs (test on held-out rest)")
print("=" * 68)
print(f"  {'m':>2}  {'n_train':>7}  {'n_test':>7}  "
      f"{'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 55)

naive_curve, sim_curve = [], []

for m in m_values:
    train_configs = CONFIG_ORDER[:m]
    test_configs  = CONFIG_ORDER[m:]

    # Verify no overlap between train and test pairs
    train_set = set(train_configs)
    test_set  = set(test_configs)
    assert train_set.isdisjoint(test_set), \
        f"m={m}: train/test overlap: {train_set & test_set}"

    # Assemble training set — one row per (IC, pair, L), no Cartesian product
    u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset_paired(
        u_perts, train_configs, L_train, sol_cache)

    # Assemble test set — held-out (Re,Ma) pairs only
    u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset_paired(
        u_perts, test_configs, L_test, sol_cache)

    naive_m = NaiveCompressibleOperator().fit(u_ic_tr, mu_tr, L_tr, c0_tr, Y_tr)
    sim_m   = SimilarityCompressibleOperator().fit(u_ic_tr, Re_tr, Ma_tr, Y_tr)

    e_naive = float(naive_m.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te).mean())
    e_sim   = float(sim_m.relative_l2(u_ic_te, Re_te, Ma_te, Y_te).mean())
    gap     = e_naive / max(e_sim, 1e-9)
    naive_curve.append(e_naive)
    sim_curve.append(e_sim)

    tr_cfgs = ", ".join(f"({int(r)},{ma:.1f})" for r, ma in train_configs)
    print(f"  {m:>2}  {len(Y_tr):>7}  {len(Y_te):>7}  "
          f"{e_naive:>10.4f}  {e_sim:>12.4f}  {gap:>7.1f}x")
    print(f"      train configs: {tr_cfgs}")

# ── Per-(Re,Ma) breakdown at m=1 ─────────────────────────────────────────────
print()
print("=" * 68)
print("PER-(Re, Ma) BREAKDOWN at m=1")
print("  Training config: (Re=2000, Ma=0.5)  |  Test: all 14 other configs")
print("=" * 68)

m1_train = CONFIG_ORDER[:1]
m1_test  = CONFIG_ORDER[1:]

u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Re_tr1_arr, Ma_tr1_arr, Y_tr1 = assemble_dataset_paired(
    u_perts, m1_train, L_train, sol_cache)
u_ic_te1, mu_te1, L_te1, c0_te1, Re_te1_arr, Ma_te1_arr, Y_te1 = assemble_dataset_paired(
    u_perts, m1_test, L_test, sol_cache)

naive_1 = NaiveCompressibleOperator().fit(u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Y_tr1)
sim_1   = SimilarityCompressibleOperator().fit(u_ic_tr1, Re_tr1_arr, Ma_tr1_arr, Y_tr1)

err_n1 = naive_1.relative_l2(u_ic_te1, mu_te1, L_te1, c0_te1, Y_te1)
err_s1 = sim_1.relative_l2(u_ic_te1, Re_te1_arr, Ma_te1_arr, Y_te1)

print(f"  {'Re':>6}  {'Ma':>5}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 46)
for Re, Ma in m1_test:
    mask = (Re_te1_arr == Re) & (Ma_te1_arr == Ma)
    en   = float(err_n1[mask].mean())
    es   = float(err_s1[mask].mean())
    g    = en / max(es, 1e-9)
    print(f"  {Re:>6.0f}  {Ma:>5.2f}  {en:>10.4f}  {es:>12.4f}  {g:>7.1f}x")

# ── Per-(Re,Ma) breakdown at m=8 ─────────────────────────────────────────────
print()
print("=" * 68)
print("PER-(Re, Ma) BREAKDOWN at m=8")
print("  Test: 7 held-out configs")
print("=" * 68)

m8_train = CONFIG_ORDER[:8]
m8_test  = CONFIG_ORDER[8:]

u_ic_tr8, mu_tr8, L_tr8, c0_tr8, Re_tr8_arr, Ma_tr8_arr, Y_tr8 = assemble_dataset_paired(
    u_perts, m8_train, L_train, sol_cache)
u_ic_te8, mu_te8, L_te8, c0_te8, Re_te8_arr, Ma_te8_arr, Y_te8 = assemble_dataset_paired(
    u_perts, m8_test, L_test, sol_cache)

naive_8 = NaiveCompressibleOperator().fit(u_ic_tr8, mu_tr8, L_tr8, c0_tr8, Y_tr8)
sim_8   = SimilarityCompressibleOperator().fit(u_ic_tr8, Re_tr8_arr, Ma_tr8_arr, Y_tr8)

err_n8 = naive_8.relative_l2(u_ic_te8, mu_te8, L_te8, c0_te8, Y_te8)
err_s8 = sim_8.relative_l2(u_ic_te8, Re_te8_arr, Ma_te8_arr, Y_te8)

print(f"  {'Re':>6}  {'Ma':>5}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 46)
for Re, Ma in m8_test:
    mask = (Re_te8_arr == Re) & (Ma_te8_arr == Ma)
    en   = float(err_n8[mask].mean())
    es   = float(err_s8[mask].mean())
    g    = en / max(es, 1e-9)
    print(f"  {Re:>6.0f}  {Ma:>5.2f}  {en:>10.4f}  {es:>12.4f}  {g:>7.1f}x")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("SUMMARY")
print("=" * 68)
print(f"  {'m':>2}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
for i, m in enumerate(m_values):
    g = naive_curve[i] / max(sim_curve[i], 1e-9)
    print(f"  {m:>2}  {naive_curve[i]:>10.4f}  {sim_curve[i]:>12.4f}  {g:>7.1f}x")
print()
m1_gap = naive_curve[0]  / max(sim_curve[0],  1e-9)
m8_gap = naive_curve[-1] / max(sim_curve[-1], 1e-9)
print(f"  Gap at m=1: {m1_gap:.1f}x   Gap at m=8: {m8_gap:.1f}x")
print()
print("=" * 68)
