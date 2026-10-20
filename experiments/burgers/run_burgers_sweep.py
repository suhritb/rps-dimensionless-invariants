"""
experiments/burgers/run_burgers_sweep.py
========================
Original single-seed Burgers sweep (rng=1234); superseded by run_burgers_multiseed.py.

Hypothesis
----------
An operator that receives the similarity-invariant coordinate log(Re) reaches
low out-of-band-scale error from a SINGLE training scale. A naive operator
given (log nu, log L) separately must sample multiple scales before it
discovers the invariance from data.

The exact dynamical-similarity symmetry of Burgers PRESERVES Reynolds number
(Re = L/nu); it cannot manufacture new Re values. The benefit is scale-data
efficiency, not Re-extrapolation. Re-extrapolation would require approximate
self-similarity (inertial-range / Kolmogorov), which is outside the scope of
this experiment.

Outputs (saved to results/)
---------------------------
  data_efficiency_curve.png  -- main result figure
  solution_comparison.png    -- example prediction vs truth
  physics_check.png          -- orbit invariance bar chart
  sweep_numbers.txt          -- raw numbers for reproducibility
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src import (make_ic_vector, precompute_solutions, assemble_dataset,
                 NaiveOperator, SimilarityOperator, solve_config,
                 plot_data_efficiency_curve, plot_solution_comparison,
                 plot_physics_check)

RNG = np.random.default_rng(1234)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
N = 64
Re_values = np.array([15., 22., 30., 45., 65.])
n_ic = 60
L_pool = np.array([2.6, 2.9, 3.2, 2.3, 3.5, 2.0, 3.8])   # candidate train scales
L_test = np.array([1.2, 1.6, 4.4, 5.2])                    # out-of-band test scales
ks = [1, 2, 3, 5, 7]                                        # sweep points

# ── Physics check: same Re => same solution vector ─────────────────────────
print("="*68)
print("PHYSICS CHECK: orbit invariance")
print("="*68)
ic_chk = make_ic_vector(RNG, N)
ref_L = 2.8; orbit_errs = []
for Re in Re_values:
    ref_sol = solve_config(ic_chk, ref_L/Re, ref_L)
    errs = []
    for L_alt in [1.5, 2.2, 3.6]:
        sol_alt = solve_config(ic_chk, L_alt/Re, L_alt)
        errs.append(np.linalg.norm(sol_alt - ref_sol) / np.linalg.norm(ref_sol))
    orbit_errs.append(max(errs))
    print(f"  Re={Re:.0f}  max orbit rel-L2 = {max(errs):.2e}")
plot_physics_check(Re_values, orbit_errs,
                   os.path.join(RESULTS_DIR, "physics_check.png"))

# ── Build ICs and pre-compute solutions ────────────────────────────────────
print(f"\nGenerating {n_ic} ICs and pre-computing {n_ic*len(Re_values)} solutions ...")
ics = [make_ic_vector(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(ics, Re_values, ref_L=ref_L)

# Build fixed test set
Xic_te, nu_te, L_te, Re_te, Y_te = assemble_dataset(ics, Re_values, L_test, sol_cache)
print(f"Test set: {Y_te.shape[0]} samples, L in {L_test.tolist()}")

# ── Data-efficiency sweep ──────────────────────────────────────────────────
print("\n"+"="*68)
print("DATA-EFFICIENCY SWEEP")
print("="*68)
naive_curve, sim_curve = [], []

for k in ks:
    L_train_k = L_pool[:k]
    Xic_tr, nu_tr, L_tr, Re_tr, Y_tr = assemble_dataset(
        ics, Re_values, L_train_k, sol_cache)

    naive_m = NaiveOperator()
    naive_m.fit(Xic_tr, nu_tr, L_tr, Y_tr)
    e_naive = naive_m.relative_l2(Xic_te, nu_te, L_te, Y_te).mean()

    sim_m = SimilarityOperator()
    sim_m.fit(Xic_tr, Re_tr, Y_tr)
    e_sim = sim_m.relative_l2(Xic_te, Re_te, Y_te).mean()

    naive_curve.append(float(e_naive))
    sim_curve.append(float(e_sim))
    print(f"  #train scales={k:2d}  naive={e_naive:.3f}  "
          f"similarity={e_sim:.3f}  gap={e_naive/max(e_sim,1e-9):.1f}x")

# ── Plots ──────────────────────────────────────────────────────────────────
plot_data_efficiency_curve(
    ks, naive_curve, sim_curve,
    os.path.join(RESULTS_DIR, "data_efficiency_curve.png"))

# Example solution comparison (1 training scale -- where gap is largest)
Xic_1, nu_1, L_1, Re_1, Y_1 = assemble_dataset(
    ics, Re_values, L_pool[:1], sol_cache)
naive_ex = NaiveOperator().fit(Xic_1, nu_1, L_1, Y_1)
sim_ex = SimilarityOperator().fit(Xic_1, Re_1, Y_1)

idx = 0
x_plot = np.linspace(0.0, L_te[idx], N, endpoint=False)
ic_ex = Xic_te[idx]; y_true = Y_te[idx]
y_naive = naive_ex.predict(Xic_te[[idx]], nu_te[[idx]], L_te[[idx]])[0]
y_sim = sim_ex.predict(Xic_te[[idx]], Re_te[[idx]])[0]
plot_solution_comparison(
    x_plot, ic_ex, y_naive, y_sim, y_true,
    nu=nu_te[idx], L=L_te[idx], Re=Re_te[idx],
    save_path=os.path.join(RESULTS_DIR, "solution_comparison.png"))

# ── Save raw numbers ───────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "sweep_numbers.txt"), "w") as f:
    f.write("# data-efficiency sweep results\n")
    f.write("# k | naive_rel_l2 | sim_rel_l2 | gap\n")
    for k, en, es in zip(ks, naive_curve, sim_curve):
        f.write(f"{k}  {en:.4f}  {es:.4f}  {en/max(es,1e-9):.2f}\n")

print("\nAll results saved to results/")
print("\nInterpretation")
print("--------------")
print("Similarity-invariant operator reaches low error with ONE training scale.")
print("Naive operator needs multiple scales to discover Re = L/nu is all that matters.")
print("Benefit is SCALE-DATA EFFICIENCY, not Re-extrapolation.")
print("(The exact symmetry preserves Re.)")
