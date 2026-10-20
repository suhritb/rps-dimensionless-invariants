"""
experiments/diagnostics/diag_degenerate_count_diagnosis_ics.py
==============================================================
Counts degenerate targets (||Y|| < 1e-3) on the IC set used to diagnose the
first methodological pitfall (np.random.default_rng(42), 20 ICs), on the
original Re grid {50, ..., 1000} and on the paper's Re >= 500 grid
(paper Section 5: 79/300 vs 0/300).

Run from the repository root:
    python experiments/diagnostics/diag_degenerate_count_diagnosis_ics.py > results/diagnostics/degenerate_count_diagnosis_ics.txt
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np
from src.compressible_dataset import make_ic_velocity, precompute_solutions

rng = np.random.default_rng(42)
u = [make_ic_velocity(rng, 64) for _ in range(20)]
print("Degenerate targets (||Y|| < 1e-3) on the diagnosis IC set "
      "(np.random.default_rng(42), 20 ICs, Ma in {0.3,0.5,0.7}, T_final=1.5)")
for name, grid in (("original grid", [50., 100., 200., 500., 1000.]),
                   ("paper grid", [500., 1000., 2000., 5000., 10000.])):
    c = precompute_solutions(u, np.array(grid), np.array([0.3, 0.5, 0.7]), verbose=False)
    n = np.array([np.linalg.norm(v) for v in c.values()])
    print(f"  {name} Re={grid}: {(n < 1e-3).sum()}/{len(n)} degenerate, min ||Y|| = {n.min():.2e}", flush=True)
