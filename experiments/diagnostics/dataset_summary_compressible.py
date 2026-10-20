"""
experiments/diagnostics/dataset_summary_compressible.py
===========================================
Prints a summary of the compressible dataset on the LEGACY (pre-paper) Re grid
{50, 100, 200, 500, 1000}. This grid is kept deliberately: it is the one used to
diagnose the degenerate-sample pitfall (near-zero ||Y|| at Re=50; see
docs/development_log.md). The paper uses Re in {500, 1000, 2000, 5000, 10000}.

Contents:
  - Configuration (N, Re values, Ma values, n_ic, total solves, T_final)
  - Orbit invariance check: same (Re, Ma), different L -> identical non-dim solution
  - Solution range sanity check: min/max velocity at T_final across all (Re, Ma)
  - Training / test set sizes for the full L_pool

Run from the project root:
    python experiments/diagnostics/dataset_summary_compressible.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    make_ic_velocity, solve_config, precompute_solutions,
    assemble_dataset, features_naive, features_similarity,
    T_FINAL, CFL, AMP,
)

RNG = np.random.default_rng(42)

# ── Experiment grid ───────────────────────────────────────────────────────────
N          = 64
Re_values  = np.array([50., 100., 200., 500., 1000.])
Ma_values  = np.array([0.3, 0.5, 0.7])
n_ic       = 20
L_pool     = np.array([0.8, 1.2, 1.6, 2.0, 2.4])   # training scales
L_test     = np.array([0.4, 3.0])                    # out-of-band test scales
L_orbit_chk = np.array([0.5, 1.0, 2.0, 3.5])        # orbit-check L values

n_Re_Ma  = len(Re_values) * len(Ma_values)
n_solves = n_ic * n_Re_Ma

print("=" * 68)
print("COMPRESSIBLE DATASET SUMMARY (legacy pre-paper Re grid; degenerate-sample diagnosis)")
print("=" * 68)
print(f"  Grid size N        : {N}")
print(f"  Re values          : {Re_values.tolist()}")
print(f"  Ma values          : {Ma_values.tolist()}")
print(f"  n_ic               : {n_ic}")
print(f"  Distinct (Re,Ma)   : {n_Re_Ma}")
print(f"  Total solves       : {n_solves}  (one per (IC, Re, Ma))")
print(f"  T_final            : {T_FINAL}")
print(f"  CFL                : {CFL}")
print(f"  IC amplitude       : {AMP}")
print(f"  L_pool (train)     : {L_pool.tolist()}")
print(f"  L_test             : {L_test.tolist()}")
print()

# ── Generate ICs ──────────────────────────────────────────────────────────────
print("Generating ICs ...")
u_perts = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
print(f"  IC amplitude range: [{min(u.min() for u in u_perts):.4f}, "
      f"{max(u.max() for u in u_perts):.4f}]")
print()

# ── Orbit invariance check ────────────────────────────────────────────────────
# For a fixed (Re, Ma), different L labels attach to the SAME non-dim solve.
# The non-dim solver receives only (Re, Ma), not L, so orbit members are
# computationally identical.  We verify this explicitly.
print("=" * 68)
print("ORBIT INVARIANCE CHECK")
print("  (same Re, Ma, IC) at different L -> identical non-dim solution")
print("  Error should be exactly 0.0 (same computation, not an approximation)")
print("=" * 68)

ic_chk = u_perts[0]
print(f"  {'Re':>6}  {'Ma':>5}  {'max orbit rel-L2':>18}")
print("  " + "-" * 35)
all_orbit_clean = True
for Re in [50., 200., 1000.]:
    for Ma in [0.3, 0.7]:
        ref_sol = solve_config(ic_chk, Re, Ma)
        errs = []
        for L in L_orbit_chk:
            # L does not enter solve_config; this is the same computation
            sol_L = solve_config(ic_chk, Re, Ma)
            rel = np.linalg.norm(sol_L - ref_sol) / (np.linalg.norm(ref_sol) + 1e-30)
            errs.append(rel)
        max_err = max(errs)
        if max_err > 1e-14:
            all_orbit_clean = False
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {max_err:>18.2e}")

tag = "PASS" if all_orbit_clean else "FAIL"
print(f"\n  [{tag}]  Orbit invariance: all errors < 1e-14\n")

# ── Pre-compute solutions ─────────────────────────────────────────────────────
print("=" * 68)
print(f"Pre-computing {n_solves} solutions ...")
print("=" * 68)
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)
print()

# ── Solution range sanity check ───────────────────────────────────────────────
print("=" * 68)
print("SOLUTION RANGE CHECK  (velocity at T_final)")
print(f"  {'Re':>6}  {'Ma':>5}  {'min u':>10}  {'max u':>10}  {'max|u| / amp':>14}")
print("  " + "-" * 48)
any_large = False
for Re in Re_values:
    for Ma in Ma_values:
        vals = np.array([sol_cache[(i, float(Re), float(Ma))]
                         for i in range(n_ic)])
        lo = vals.min(); hi = vals.max()
        ratio = max(abs(lo), abs(hi)) / AMP
        flag = "  <-- LARGE" if ratio > 5.0 else ""
        if ratio > 5.0:
            any_large = True
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {lo:>10.4f}  {hi:>10.4f}  {ratio:>14.3f}{flag}")

if not any_large:
    print("  All solution amplitudes < 5 x IC amplitude  (no-shock regime confirmed)")
print()

# ── Dataset sizes ─────────────────────────────────────────────────────────────
print("=" * 68)
print("DATASET SIZES")
print("=" * 68)

# Full training set (all L_pool scales)
u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
    u_perts, Re_values, Ma_values, L_pool, sol_cache)

# Test set (out-of-band L values, same Re and Ma)
u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset(
    u_perts, Re_values, Ma_values, L_test, sol_cache)

print(f"  Training (all {len(L_pool)} L_pool scales): {Y_tr.shape[0]} samples")
print(f"    u_ic  : {u_ic_tr.shape}")
print(f"    Y     : {Y_tr.shape}")
print(f"    Re range: [{Re_tr.min():.0f}, {Re_tr.max():.0f}]")
print(f"    Ma range: [{Ma_tr.min():.2f}, {Ma_tr.max():.2f}]")
print(f"    L  range: [{L_tr.min():.2f}, {L_tr.max():.2f}]")
print()
print(f"  Test (out-of-band L, all Re and Ma): {Y_te.shape[0]} samples")
print(f"    L  values: {np.unique(L_te).tolist()}")
print()

# Feature encoding shapes
X_naive = features_naive(u_ic_tr[:1], mu_tr[:1], L_tr[:1], c0_tr[:1])
X_sim   = features_similarity(u_ic_tr[:1], Re_tr[:1], Ma_tr[:1])
print(f"  Naive feature dim   : {X_naive.shape[1]}  (N + 3 dimensional scalars)")
print(f"  Similarity feat dim : {X_sim.shape[1]}   (N + 2 invariants)")
print(f"  Output dim          : {Y_tr.shape[1]}   (N-point velocity field)")
print()

print("=" * 68)
print("SUMMARY: dataset module ready.")
print(f"  {n_solves} solves cached.")
print(f"  Orbit invariance exact to machine precision by construction.")
print("=" * 68)
