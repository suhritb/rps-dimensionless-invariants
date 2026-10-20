"""
experiments/diagnostics/diag_norm_distribution.py
======================================
Measures the distribution of solution norms ||Y|| across the full
(Re, Ma, IC) grid at T_final = 1.5 with the ORIGINAL IC distribution.

No models.  No exclusions.  Just the raw solver output statistics.

Run from the project root:
    python experiments/diagnostics/diag_norm_distribution.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, T_FINAL,
)
from src.compressible_solver import (
    solve_compressible, primitives, conserved_from_primitives, GAMMA, PR,
)

RNG       = np.random.default_rng(42)
N         = 64
Re_values = np.array([50., 100., 200., 500., 1000.])
Ma_values = np.array([0.3, 0.5, 0.7])
n_ic      = 20

DEGEN     = 1e-3

print("=" * 70)
print(f"Original IC distribution (k random in {{1..5}}), T_final = {T_FINAL}")
print("=" * 70)
u_perts   = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)
print()

# ── IC amplitudes ─────────────────────────────────────────────────────────────
ic_norms = np.array([np.linalg.norm(u) for u in u_perts])
print(f"IC norms:  min={ic_norms.min():.4f}  max={ic_norms.max():.4f}  "
      f"median={np.median(ic_norms):.4f}")
print()

# ── Per-(Re, Ma) norm distribution ───────────────────────────────────────────
print("=" * 70)
print("||Y(T=1.5)|| distribution per (Re, Ma) across 20 ICs")
print(f"  degen = ||Y|| < {DEGEN}")
print("=" * 70)
print(f"  {'Re':>6}  {'Ma':>5}  {'min':>10}  {'p25':>10}  "
      f"{'median':>10}  {'max':>10}  {'n_degen':>10}")
print("  " + "-" * 70)

all_norms = {}
for Re in Re_values:
    for Ma in Ma_values:
        norms = np.array([
            np.linalg.norm(sol_cache[(i, float(Re), float(Ma))])
            for i in range(n_ic)
        ])
        all_norms[(Re, Ma)] = norms
        n_d = int((norms < DEGEN).sum())
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  "
              f"{norms.min():>10.2e}  {np.percentile(norms, 25):>10.2e}  "
              f"{np.median(norms):>10.2e}  {norms.max():>10.2e}  "
              f"{n_d:>5}/{n_ic}")
print()

# ── Per-IC norms at Re=50 ─────────────────────────────────────────────────────
print("=" * 70)
print("Per-IC norms at Re=50 (shows which ICs hit machine precision)")
print("=" * 70)
print(f"  {'IC':>3}  {'Ma=0.3':>10}  {'Ma=0.5':>10}  {'Ma=0.7':>10}")
print("  " + "-" * 42)
for i in range(n_ic):
    row = []
    is_degen = False
    for Ma in Ma_values:
        v = np.linalg.norm(sol_cache[(i, 50., float(Ma))])
        row.append(f"{v:>10.2e}")
        if v < DEGEN:
            is_degen = True
    flag = "  DEGEN" if is_degen else ""
    print(f"  {i:>3}  {'  '.join(row)}{flag}")
print()

# ── Degenerate summary ────────────────────────────────────────────────────────
print("=" * 70)
print("Degenerate samples per Re (summed over 3 Ma values, out of 60 each)")
print("=" * 70)
for Re in Re_values:
    n_d   = sum(int((all_norms[(Re, Ma)] < DEGEN).sum()) for Ma in Ma_values)
    total = n_ic * len(Ma_values)
    print(f"  Re={Re:>6.0f}:  {n_d:>2}/{total}  ({100.*n_d/total:.0f}%)")
grand_d = sum(v for v in (int((all_norms[(Re, Ma)] < DEGEN).sum())
                           for Re in Re_values for Ma in Ma_values))
grand_t = n_ic * len(Re_values) * len(Ma_values)
print(f"  Grand total: {grand_d}/{grand_t}  ({100.*grand_d/grand_t:.0f}%)")
print()

# ── Non-degenerate amplitude spread ───────────────────────────────────────────
nd_norms = np.array([v for (Re, Ma), arr in all_norms.items()
                     for v in arr if v >= DEGEN])
print(f"Non-degenerate amplitude range (||Y|| >= {DEGEN}):")
print(f"  n={len(nd_norms)}/{grand_t}  "
      f"min={nd_norms.min():.4f}  max={nd_norms.max():.4f}  "
      f"ratio max/min = {nd_norms.max()/nd_norms.min():.1f}x")
print()

# ── Single-mode decay test ────────────────────────────────────────────────────
x = np.linspace(0.0, 1.0, N, endpoint=False)

def single_mode_survival(Re, Ma, k, T, amp=0.05):
    u = amp * np.cos(2.0 * np.pi * k * x)
    Q0 = conserved_from_primitives(np.ones(N), u, np.ones(N), Ma, GAMMA)
    Qf = solve_compressible(Q0, Re, Ma, T, cfl=0.4, gamma=GAMMA, Pr=PR)
    _, u_fin, _, _ = primitives(Qf, Ma, GAMMA)
    return np.linalg.norm(u_fin) / np.linalg.norm(u)

print("=" * 70)
print("Single-mode survival  exp(-sigma_k * T)  at T=1.5")
print("=" * 70)
for Re in [50., 100., 200., 500., 1000.]:
    row = []
    for k in range(1, 6):
        s = single_mode_survival(Re, 0.3, k, T_FINAL)
        row.append(f"{s:.3e}")
    print(f"  Re={Re:>6.0f}:  " + "  ".join(f"k={k+1}:{row[k]}" for k in range(5)))
print()

print("=" * 70)
print("Single-mode survival at Re=50 for various T_final values")
print("(looking for T where ALL k=1..5 survive above 0.01 = 1%)")
print("=" * 70)
print(f"  {'T':>5}  " + "  ".join(f"{'k='+str(k):>9}" for k in range(1, 6)))
print("  " + "-" * 60)
for T_test in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
    row = []
    for k in range(1, 6):
        s = single_mode_survival(50., 0.3, k, T_test)
        row.append(s)
    flags = "  ok" if all(s > 0.01 for s in row) else ("  k>=2 degen" if row[0] > 0.01 else "  k=1 degen")
    print(f"  {T_test:>5.2f}  " + "  ".join(f"{s:>9.4f}" for s in row) + flags)
print()
