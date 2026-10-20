"""
tests/test_all.py
=================
Test suite for the Burgers solver, dataset, and operator code.
Run from the project root:   python tests/test_all.py

Tests
-----
1. Solver accuracy vs Cole-Hopf exact solution
2. Scaling symmetry (exact to machine precision)
3. Scale-invariance of solution vectors (same Re => same solution)
4. Dataset assembly shapes and label consistency
5. Model fit/predict interface (no silent attribute errors)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.solver import solve_burgers, cole_hopf_solution, apply_scaling_symmetry
from src.dataset import (make_ic_vector, solve_config, precompute_solutions,
                         assemble_dataset, features_naive, features_similarity)
from src.models import NaiveOperator, SimilarityOperator

RNG = np.random.default_rng(42)
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}]  {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ============================================================
# 1. Solver accuracy vs Cole-Hopf
# ============================================================
print("\n--- 1. Solver accuracy vs Cole-Hopf exact solution ---")
L = 2 * np.pi; N = 256; x = np.linspace(0, L, N, endpoint=False)
for nu in [0.10, 0.03, 0.01]:
    eps, k0, T = 0.5, 1.0, 1.0
    u0 = cole_hopf_solution(x, 0.0, nu, eps, k0)
    u_num = solve_burgers(u0, nu, L, T, nt=4000)
    u_exact = cole_hopf_solution(x, T, nu, eps, k0)
    rel = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    check(f"nu={nu}: rel L2 < 1e-3", rel < 1e-3, f"rel={rel:.2e}")


# ============================================================
# 2. Scaling symmetry: v(x,t)=u(ax,at) solves Burgers at nu/a
# ============================================================
print("\n--- 2. Scaling symmetry (exact) ---")
Lbig = 8 * np.pi; Nfine = 2048; xf = np.linspace(0, Lbig, Nfine, endpoint=False)
c = Lbig / 2
u0_fine = 0.8 * np.exp(-((xf - c)**2) / (2 * 0.6**2)) * np.sin(2*(xf-c))
for (nu, a) in [(0.08, 1.0), (0.08, 2.0), (0.06, 3.0)]:
    T = 0.6
    snaps = solve_burgers(u0_fine, nu, Lbig, a*T, int(6000*a),
                          save_times=[0.0, a*T])
    v0, Lstd = apply_scaling_symmetry(snaps[0], xf, a, Lbig, 256)
    v_target, _ = apply_scaling_symmetry(snaps[1], xf, a, Lbig, 256)
    v_num = solve_burgers(v0, nu/a, Lstd, T, 6000)
    rel = np.linalg.norm(v_num - v_target) / (np.linalg.norm(v_target) + 1e-12)
    tol = 1e-5 if a == 1.0 else 5e-2
    check(f"symmetry a={a} (nu={nu}->{nu/a:.4f})", rel < tol, f"rel={rel:.2e}")


# ============================================================
# 3. Scale-invariance: same Re => same solution vector
# ============================================================
print("\n--- 3. Scale-invariance: same Re => same sol vector ---")
N_small = 64; TAU = 0.15; NT = 2000
ic = make_ic_vector(RNG, N_small)
ref_L = 2.8
for Re in [20.0, 40.0]:
    ref_sol = solve_config(ic, ref_L/Re, ref_L)
    for L_alt in [1.5, 3.6]:
        sol_alt = solve_config(ic, L_alt/Re, L_alt)
        rel = np.linalg.norm(sol_alt - ref_sol) / np.linalg.norm(ref_sol)
        check(f"Re={Re:.0f} L={L_alt} vs L={ref_L}", rel < 5e-2, f"rel={rel:.2e}")


# ============================================================
# 4. Dataset assembly
# ============================================================
print("\n--- 4. Dataset assembly shapes ---")
Re_values = np.array([20., 35., 55.])
n_ic = 8
ics = [make_ic_vector(RNG, N_small) for _ in range(n_ic)]
cache = precompute_solutions(ics, Re_values, ref_L=ref_L)
check("cache size", len(cache) == n_ic * len(Re_values),
      f"expected {n_ic*len(Re_values)}, got {len(cache)}")

L_list = np.array([2.0, 3.0])
Xic, nu, L, Re, Y = assemble_dataset(ics, Re_values, L_list, cache)
n_exp = n_ic * len(Re_values) * len(L_list)
check("dataset n_samples", Xic.shape[0] == n_exp, f"{Xic.shape[0]} vs {n_exp}")
check("dataset Y shape", Y.shape == (n_exp, N_small))
check("nu consistency", np.allclose(nu, L / Re))

Xf = features_naive(Xic, nu, L)
check("naive features shape", Xf.shape == (n_exp, N_small + 2))
Xs = features_similarity(Xic, Re)
check("sim features shape", Xs.shape == (n_exp, N_small + 1))


# ============================================================
# 5. Model interface (tiny fit/predict smoke test)
# ============================================================
print("\n--- 5. Model interface (smoke test) ---")
Xic_s, nu_s, L_s, Re_s, Y_s = assemble_dataset(ics[:4], Re_values, L_list, cache)
naive_m = NaiveOperator()
naive_m.fit(Xic_s, nu_s, L_s, Y_s)
pred_n = naive_m.predict(Xic_s, nu_s, L_s)
check("NaiveOperator predict shape", pred_n.shape == Y_s.shape)
errs_n = naive_m.relative_l2(Xic_s, nu_s, L_s, Y_s)
check("NaiveOperator rel_l2 finite", np.all(np.isfinite(errs_n)))

sim_m = SimilarityOperator()
sim_m.fit(Xic_s, Re_s, Y_s)
pred_s = sim_m.predict(Xic_s, Re_s)
check("SimilarityOperator predict shape", pred_s.shape == Y_s.shape)
errs_s = sim_m.relative_l2(Xic_s, Re_s, Y_s)
check("SimilarityOperator rel_l2 finite", np.all(np.isfinite(errs_s)))


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
n_pass = sum(results)
n_total = len(results)
status = PASS if n_pass == n_total else FAIL
print(f"  [{status}]  {n_pass}/{n_total} tests passed\n")
if n_pass < n_total:
    sys.exit(1)
