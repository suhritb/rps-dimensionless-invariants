"""
experiments/diagnostics/diag_forced_k1_artifact.py
=======================================
Reproduces the "forced-k=1 artifact" described in the methodological history
of docs/development_log.md.

Background
----------
During early development the L-sweep was run with a modified IC generator that
guaranteed every IC contained a k=1 Fourier mode with amplitude >= 0.01.  The
motivation was to eliminate degenerate outputs (||Y|| ~ 1e-15 at Re=50) caused
by ICs composed only of k>=3 modes, which completely decay by T=1.5 at low Re.
This produced a large apparent similarity advantage (~6x at k=1).  The result
was identified as an artifact: the IC change shifted the data distribution
rather than fixing the loss, making the problem artificially easier for the
similarity operator.  The IC generator was reverted.

This script reconstructs the artifact run to make the claim reproducible.

Design
------
- Implements forced-k=1 IC generation LOCALLY in this file only.
  src/compressible_dataset.make_ic_velocity is NOT modified.
- Uses the original Re range [50, 100, 200, 500, 1000] (the range that was
  active when the artifact was observed; Re=50 produces degenerate outputs
  under the original IC distribution, which the forcing was designed to hide).
- All other settings (solver, loss, MLP architecture) are identical to the
  main pipeline: same models, same sample_weight = 1/||Y||^2 loss, same seed.
- Runs the L-scale sweep (k=1,2,3,5 training L values) and reports the gap.

The gap at k=1 is the quantity of interest.  The reproducible value is the
one produced by this script (3.9x; results/diagnostics/diag_forced_k1_output.txt),
not the non-reproducible development-time figure of ~6.1x.

Run from the project root:
    python experiments/diagnostics/diag_forced_k1_artifact.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    precompute_solutions, assemble_dataset, T_FINAL, CFL, AMP,
)
from src.compressible_models import (
    NaiveCompressibleOperator, SimilarityCompressibleOperator,
)

# ── Local forced-k=1 IC generator ────────────────────────────────────────────
# This is the reconstructed version of the modified make_ic_velocity.
# It guarantees the first mode is always k=1 with |amplitude| >= MIN_K1_AMP.
# src/compressible_dataset.make_ic_velocity is NOT touched.

_MIN_K1_AMP = 0.01

def make_ic_velocity_forced_k1(rng, N, amp=AMP, n_modes_max=4):
    """
    Forced-k=1 IC: first mode always k=1 with |a| >= _MIN_K1_AMP.
    Remaining modes are random k in {1..5} as in the original.
    """
    x       = np.linspace(0.0, 1.0, N, endpoint=False)
    u_pert  = np.zeros(N)
    n_modes = rng.integers(1, n_modes_max + 1)

    for i in range(n_modes):
        a   = rng.uniform(-amp, amp)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        if i == 0:
            # Force k=1 with enforced minimum amplitude
            k = 1
            if abs(a) < _MIN_K1_AMP:
                a = _MIN_K1_AMP * (1.0 if a >= 0.0 else -1.0)
        else:
            k = rng.integers(1, 6)
        u_pert += a * np.cos(2.0 * np.pi * k * x + phi)
    return u_pert

# ── Config ────────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

N          = 64
Re_values  = np.array([50., 100., 200., 500., 1000.])   # ORIGINAL Re range
Ma_values  = np.array([0.3, 0.5, 0.7])
n_ic       = 20
L_pool     = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test     = np.array([0.4, 3.0])
ks         = [1, 2, 3, 5]

print("=" * 68)
print("FORCED-k=1 ARTIFACT DIAGNOSTIC")
print("Reconstructs the early-development run that produced a spurious")
print("similarity advantage before the IC generator was reverted.")
print()
print("IC generator: forced-k=1 (local to this script; main pipeline")
print("              src/compressible_dataset.make_ic_velocity unchanged)")
print(f"MIN_K1_AMP   = {_MIN_K1_AMP}")
print(f"Re range     : {Re_values.tolist()}  (original range)")
print(f"Ma range     : {Ma_values.tolist()}")
print(f"T_final      = {T_FINAL}")
print(f"n_ic         = {n_ic}  (RNG seed = 42)")
print("=" * 68)

# ── ICs and solutions ─────────────────────────────────────────────────────────
print("Generating forced-k=1 ICs and pre-computing solutions ...")
u_perts   = [make_ic_velocity_forced_k1(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)
print()

# ── IC summary ────────────────────────────────────────────────────────────────
ic_norms = [np.linalg.norm(u) for u in u_perts]
print(f"IC norms (forced-k=1): min={min(ic_norms):.4f}  "
      f"max={max(ic_norms):.4f}  median={np.median(ic_norms):.4f}")
print()

# ── Degenerate check ──────────────────────────────────────────────────────────
u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset(
    u_perts, Re_values, Ma_values, L_test, sol_cache)
norms_te  = np.linalg.norm(Y_te, axis=1)
n_degen   = int((norms_te < 1e-3).sum())
print(f"Test set: {len(Y_te)} samples")
print(f"Degenerate (||Y|| < 1e-3): {n_degen}/{len(Y_te)} "
      f"({'reduced vs original ICs' if n_degen < 79 else 'unchanged'})")
print()

# Per-(Re, Ma) degenerate counts
print("Degenerate test samples per (Re, Ma):")
print(f"  {'Re':>6}  {'Ma':>5}  {'n_degen':>8}  {'n_total':>8}")
print("  " + "-" * 36)
for Re in Re_values:
    for Ma in Ma_values:
        mask   = (Re_te == Re) & (Ma_te == Ma)
        n_tot  = int(mask.sum())
        n_deg  = int((norms_te[mask] < 1e-3).sum())
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {n_deg:>8}  {n_tot:>8}")
print()

# ── L-scale sweep (no metric exclusion — same as the artifact run) ────────────
print("=" * 68)
print("L-SCALE SWEEP — forced-k=1 ICs, no degenerate exclusion from metric")
print("(replicates the artifact: metric includes degenerate samples)")
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

    e_naive = float(naive_m.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te).mean())
    e_sim   = float(sim_m.relative_l2(u_ic_te, Re_te, Ma_te, Y_te).mean())
    gap     = e_naive / max(e_sim, 1e-9)
    naive_curve.append(e_naive)
    sim_curve.append(e_sim)

    L_str = str(L_train_k.tolist())
    print(f"  {k:>3}  {L_str:>20}  {e_naive:>10.4f}  {e_sim:>12.4f}  {gap:>7.1f}x")

# ── Per-(Re,Ma) at k=1 ────────────────────────────────────────────────────────
print()
print("=" * 68)
print("PER-(Re, Ma) BREAKDOWN at k=1")
print("=" * 68)

L_train_1 = L_pool[:1]
u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Re_tr1, Ma_tr1, Y_tr1 = assemble_dataset(
    u_perts, Re_values, Ma_values, L_train_1, sol_cache)

naive_1 = NaiveCompressibleOperator().fit(u_ic_tr1, mu_tr1, L_tr1, c0_tr1, Y_tr1)
sim_1   = SimilarityCompressibleOperator().fit(u_ic_tr1, Re_tr1, Ma_tr1, Y_tr1)

err_n1 = naive_1.relative_l2(u_ic_te, mu_te, L_te, c0_te, Y_te)
err_s1 = sim_1.relative_l2(u_ic_te, Re_te, Ma_te, Y_te)

print(f"  {'Re':>6}  {'Ma':>5}  {'naive':>10}  {'similarity':>12}  {'gap':>8}")
print("  " + "-" * 46)
for Re in Re_values:
    for Ma in Ma_values:
        mask = (Re_te == Re) & (Ma_te == Ma)
        en   = float(err_n1[mask].mean())
        es   = float(err_s1[mask].mean())
        g    = en / max(es, 1e-9)
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {en:>10.4f}  {es:>12.4f}  {g:>7.1f}x")

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
print(f"  Gap at k=1: {k1_gap:.1f}x  (the artifact gap — original claim was ~6.1x)")
print()
print("  Note: this is a reconstruction of the forced-k=1 run.")
print("  The exact IC sequence differs from the original (reverted) code")
print("  because the original modified make_ic_velocity was not saved.")
print("  The gap magnitude is indicative, not an exact replay.")
print()
print("=" * 68)
