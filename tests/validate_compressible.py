"""
tests/validate_compressible.py
==============================
Validates the compressible solver against the analytic acoustic-decay rate
and checks conservation laws.  Run from the project root:

    python tests/validate_compressible.py

MEASUREMENT METHOD NOTE
-----------------------
The original validation used the inviscid isentropic IC (make_acoustic_ic)
and a two-point decay estimate sigma = log(A0/A_T)/T.  A grid-convergence
study (N=64/128/256/512, CFL=0.4/0.2/0.1) showed the relative error was
identically 2.714e-3 at EVERY combination for Re=200, Ma=0.3 -- a fixed
measurement bias, not a discretization error.

Root cause: the isentropic IC uses the Re->inf eigenvector.  At finite Re it
has an O(1/Re) projection onto the backward acoustic mode.  Both acoustic
modes share exactly the same decay rate sigma, so their beat never damps.
A two-point estimate samples two arbitrary beat phases and is biased by
O(1/Re) (~2.7e-3 for Re=200).

Fix: validate_acoustic_decay now uses make_eigenmode_ic (exact finite-Re
eigenvector) plus a 40-point least-squares exponential fit.  With this
method rel_err is ~5e-6 at N=64 (~18x below the 1e-4 tolerance).
The tolerance was NOT loosened; the measurement was corrected.

Audit trail of original flat-error result in tests/ACOUSTIC_VALIDATION_AUDIT.txt.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.compressible_solver import (
    validate_acoustic_decay, check_conservation,
    make_acoustic_ic, solve_compressible,
    acoustic_decay_rate, exact_acoustic_decay_rate,
)

PASS_STR = "\033[92mPASS\033[0m"
FAIL_STR = "\033[91mFAIL\033[0m"

# ── 1. Acoustic decay rate ────────────────────────────────────────────────────
print("=" * 65)
print("1. Acoustic decay rate  (tol = 1e-4 relative to exact eigenvalue)")
print("   Note: approx formula sigma=k^2/(2Re)*(4/3+(g-1)/Pr) is O(1/Re);")
print("   exact eigenvalue from cubic dispersion relation is used as ref.")
print("=" * 65)

results = validate_acoustic_decay(
    N=128, mode=1, eps=1e-5,
    Re_list=(50.0, 200.0, 1000.0),
    Ma_list=(0.3, 0.6),
    cfl=0.4, tol=1e-4, verbose=True,
)

n_pass = sum(r["passed"] for r in results)
n_tot  = len(results)
tag    = PASS_STR if n_pass == n_tot else FAIL_STR
print(f"\n  [{tag}]  {n_pass}/{n_tot} acoustic cases passed\n")

# ── 2. Conservation laws ──────────────────────────────────────────────────────
print("=" * 65)
print("2. Conservation laws  (mean rho, rho*u, E  constant to 1e-8)")
print("=" * 65)

conservation_pass = True
for Re, Ma in [(50.0, 0.3), (200.0, 0.6), (1000.0, 0.3)]:
    Q0 = make_acoustic_ic(128, Ma=Ma, mode=1, eps=0.05)
    Qf = solve_compressible(Q0, Re=Re, Ma=Ma, T_final=1.0, cfl=0.4)
    c  = check_conservation(Q0, Qf, tol=1e-8)
    ok = c["passed"]
    conservation_pass = conservation_pass and ok
    tag_c = PASS_STR if ok else FAIL_STR
    print(f"  [{tag_c}]  Re={Re:.0f}  Ma={Ma:.1f}  "
          f"mass={c['mass']:.1e}  mom={c['momentum']:.1e}  eng={c['energy']:.1e}")

print()

# ── Summary ───────────────────────────────────────────────────────────────────
all_ok = (n_pass == n_tot) and conservation_pass
final  = PASS_STR if all_ok else FAIL_STR
print("=" * 65)
print(f"  [{final}]  All validation checks {'PASSED' if all_ok else 'FAILED'}")
print("=" * 65)

if not all_ok:
    sys.exit(1)
