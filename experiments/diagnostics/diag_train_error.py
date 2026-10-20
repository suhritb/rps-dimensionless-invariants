"""
experiments/diagnostics/diag_train_error.py
===========================================
Diagnostic: TRAINING error (not test error) per (Re, Ma) for the similarity
model at k=1, on the legacy (pre-paper) Re grid {50, 100, 200, 500, 1000}.

Also prints per-(Re, Ma) solution amplitudes and the 1/||Y||^2 sample-weight
ratios relative to Re=1000. Used during development to diagnose the
degenerate-sample pitfall (see docs/development_log.md):
  - If training error at Re=50 is also very large, the optimizer is failing
    to fit those samples (an optimization problem).
  - If training error at Re=50 is low but test error is high, the cause cannot
    be the similarity encoding, since its test features equal its training
    features (Re, Ma do not change with L).

Run from the project root:
    python experiments/diagnostics/diag_train_error.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset,
)
from src.compressible_models import SimilarityCompressibleOperator

RNG = np.random.default_rng(42)
N         = 64
Re_values = np.array([50., 100., 200., 500., 1000.])
Ma_values = np.array([0.3, 0.5, 0.7])
n_ic      = 20
L_pool    = np.array([0.8])

print("Running 300 solves ...")
u_perts  = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=False)

u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
    u_perts, Re_values, Ma_values, L_pool, sol_cache)

print(f"Training set: {len(Y_tr)} samples")
print()

# --- solution amplitude diagnostics ---
print("=== Solution amplitude stats per (Re, Ma) ===")
for Re in Re_values:
    for Ma in Ma_values:
        m = (Re_tr == Re) & (Ma_tr == Ma)
        norms = np.linalg.norm(Y_tr[m], axis=1)
        sw    = 1.0 / norms**2
        print(f"  Re={Re:>5.0f}  Ma={Ma:.1f}  ||Y|| mean={norms.mean():.5f}  "
              f"sw mean={sw.mean():.3e}")
norms_all = np.linalg.norm(Y_tr, axis=1)
sw_all    = 1.0 / norms_all**2
m1000     = (Re_tr == 1000.)
sw_re1000 = sw_all[m1000].mean()
print()
print("=== Sample weight ratios (relative to Re=1000) ===")
for Re in Re_values:
    m = (Re_tr == Re)
    ratio = sw_all[m].mean() / sw_re1000
    print(f"  Re={Re:>5.0f}: sw_ratio = {ratio:.1f}x")

print()
print("=== Training model ===")
sim = SimilarityCompressibleOperator()
sim.fit(u_ic_tr, Re_tr, Ma_tr, Y_tr)
print(f"Training done.  n_iter_ = {sim.mlp.n_iter_}")

print()
print("=== TRAINING error per (Re, Ma) ===")
print(f"  {'Re':>6}  {'Ma':>5}  {'train_rel_L2':>14}  {'||Y|| mean':>12}")
for Re in Re_values:
    for Ma in Ma_values:
        m = (Re_tr == Re) & (Ma_tr == Ma)
        e = sim.relative_l2(u_ic_tr[m], Re_tr[m], Ma_tr[m], Y_tr[m]).mean()
        nrm = np.linalg.norm(Y_tr[m], axis=1).mean()
        print(f"  {Re:>6.0f}  {Ma:>5.2f}  {e:>14.4f}  {nrm:>12.6f}")
