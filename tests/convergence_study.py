"""
tests/convergence_study.py
==========================
Grid-convergence study for the acoustic-decay validation.

Fixed case: Re=200, Ma=0.3, mode=1
Grid: N = 64, 128, 256, 512
CFL:  0.4, 0.2, 0.1

For each (N, CFL):
  - Run single-mode acoustic IC to T_final = max(4*T_osc, ln2/sigma_exact)
  - Measure sigma from log(A0/Af)/T_final
  - Compute rel_err = |sigma_meas - sigma_exact| / sigma_exact

Print error table, then estimate convergence orders.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.compressible_solver import (
    make_acoustic_ic, solve_compressible, primitives,
    exact_acoustic_decay_rate,
)

Re     = 200.0
Ma     = 0.3
mode   = 1
eps    = 1e-5
N_list = [64, 128, 256, 512]
cfl_list = [0.4, 0.2, 0.1]

sigma_exact = exact_acoustic_decay_rate(mode, Re, Ma)
T_osc   = Ma / mode
T_final = max(4.0 * T_osc, np.log(2.0) / sigma_exact)

print(f"Re={Re}, Ma={Ma}, mode={mode}, eps={eps}")
print(f"sigma_exact = {sigma_exact:.8f}")
print(f"T_final     = {T_final:.4f}  (max(4*T_osc={4*T_osc:.4f}, ln2/sigma={np.log(2)/sigma_exact:.4f}))")
print()

# Header
header = f"{'N':>6}  {'CFL':>6}  {'sigma_meas':>14}  {'rel_err':>12}  {'abs_err':>12}"
print(header)
print("-" * len(header))

results = {}
for N in N_list:
    results[N] = {}
    for cfl in cfl_list:
        Q0 = make_acoustic_ic(N, Ma, mode=mode, eps=eps)
        snaps = solve_compressible(Q0, Re, Ma, T_final, cfl=cfl,
                                   save_times=[0.0, T_final])

        def _amp(Q):
            _, u, _, _ = primitives(Q, Ma)
            uh = np.fft.rfft(u)
            return 2.0 * abs(uh[mode]) / N

        A0 = _amp(snaps[0])
        Af = _amp(snaps[1])
        sigma_meas = np.log(A0 / Af) / T_final
        rel_err = abs(sigma_meas - sigma_exact) / abs(sigma_exact)
        abs_err = abs(sigma_meas - sigma_exact)

        results[N][cfl] = dict(sigma_meas=sigma_meas, rel_err=rel_err, abs_err=abs_err)
        print(f"{N:>6}  {cfl:>6.2f}  {sigma_meas:>14.8f}  {rel_err:>12.3e}  {abs_err:>12.3e}")

print()
print("Convergence orders (error ratio as N doubles, fixed CFL):")
print(f"{'CFL':>6}  {'N=64->128':>12}  {'N=128->256':>12}  {'N=256->512':>12}")
print("-" * 50)
for cfl in cfl_list:
    orders = []
    for i in range(len(N_list) - 1):
        e1 = results[N_list[i]][cfl]['rel_err']
        e2 = results[N_list[i+1]][cfl]['rel_err']
        if e2 > 0 and e1 > 0:
            order = np.log2(e1 / e2)
        else:
            order = float('nan')
        orders.append(order)
    print(f"{cfl:>6.2f}  {orders[0]:>12.2f}  {orders[1]:>12.2f}  {orders[2]:>12.2f}")

print()
print("Convergence orders (error ratio as CFL halves, fixed N):")
print(f"{'N':>6}  {'CFL 0.4->0.2':>14}  {'CFL 0.2->0.1':>14}")
print("-" * 40)
for N in N_list:
    orders = []
    for i in range(len(cfl_list) - 1):
        e1 = results[N][cfl_list[i]]['rel_err']
        e2 = results[N][cfl_list[i+1]]['rel_err']
        if e2 > 0 and e1 > 0:
            order = np.log2(e1 / e2)
        else:
            order = float('nan')
        orders.append(order)
    print(f"{N:>6}  {orders[0]:>14.2f}  {orders[1]:>14.2f}")
