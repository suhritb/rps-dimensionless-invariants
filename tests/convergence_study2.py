"""
tests/convergence_study2.py
===========================
Diagnostic: does the ~2.7e-3 systematic error shrink when we use
(a) the exact finite-Re eigenvector as IC, and
(b) a least-squares exponential fit over many time points
instead of the inviscid isentropic IC + two-point estimate?

Fixed case: Re=200, Ma=0.3, mode=1
Grid: N = 64, 128, 256
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.compressible_solver import (
    make_acoustic_ic, solve_compressible, primitives,
    exact_acoustic_decay_rate, conserved_from_primitives,
    GAMMA, PR,
)

Re    = 200.0
Ma    = 0.3
mode  = 1
cfl   = 0.4

sigma_exact = exact_acoustic_decay_rate(mode, Re, Ma)
T_osc   = Ma / mode
T_final = max(4.0 * T_osc, np.log(2.0) / sigma_exact)

print(f"Re={Re}, Ma={Ma}, mode={mode}")
print(f"sigma_exact = {sigma_exact:.10f}")
print(f"T_final     = {T_final:.6f}")
print()

# ── Build linearization matrix for +k wavenumber ──────────────────────────────

def linearization_matrix(mode, Re, Ma, gamma=GAMMA, Pr=PR):
    """
    3x3 matrix A such that d/dt [δρ, δu, δE] = A [δρ, δu, δE]
    for Fourier mode k = 2π*mode around base state (ρ,u,T)=(1,0,1).
    """
    k = 2.0 * np.pi * mode
    a = 4.0 * k**2 / (3.0 * Re)
    b = gamma * k**2 / (Re * Pr)
    c = k**2 / (Re * Pr * (gamma - 1.0) * Ma**2)
    A = np.array([
        [0.0,                  -1j * k,                      0.0   ],
        [0.0,                  -a,                  -1j * k * (gamma - 1.0)],
        [c,  -1j * k / ((gamma - 1.0) * Ma**2),   -b       ],
    ], dtype=complex)
    return A


def right_going_eigenvector(mode, Re, Ma, gamma=GAMMA, Pr=PR):
    """
    Returns (eigenvalue, eigenvector) for the right-going acoustic mode.
    Right-going at wavenumber k>0 has Im(λ) < 0 (frequency -k/Ma).
    """
    A = linearization_matrix(mode, Re, Ma, gamma, Pr)
    vals, vecs = np.linalg.eig(A)
    # right-going acoustic: most negative imaginary part
    idx = np.argmin(vals.imag)
    return vals[idx], vecs[:, idx]


def project_ic_onto_modes(mode, Re, Ma, eps=1e-5, gamma=GAMMA, Pr=PR):
    """
    Project the isentropic IC Fourier components onto the three eigenmodes.
    Returns projection amplitudes.
    """
    A = linearization_matrix(mode, Re, Ma, gamma, Pr)
    vals, vecs = np.linalg.eig(A)

    # IC at wavenumber k (δρ, δu, δE) = eps/2 * (1, 1/Ma, 1/((γ-1)Ma²))
    ic_vec = np.array([
        eps / 2.0,
        eps / (2.0 * Ma),
        eps / (2.0 * (gamma - 1.0) * Ma**2),
    ], dtype=complex)

    # Projection coefficients via inverse eigenvector matrix
    coeffs = np.linalg.solve(vecs, ic_vec)

    # Sort by imaginary part of eigenvalue for readability
    order = np.argsort(vals.imag)
    print("  Mode-projection of isentropic IC onto eigenmodes:")
    labels = ["right-going acoustic", "entropy/thermal", "left-going acoustic"]
    for i, (lbl, j) in enumerate(zip(labels, order)):
        print(f"    {lbl:25s}  lam={vals[j]:.6f}   |c|={abs(coeffs[j]):.4e}")
    print()
    return vals, vecs, coeffs


def make_pure_eigenmode_ic(N, Re, Ma, mode=1, eps=1e-5, gamma=GAMMA, Pr=PR):
    """
    IC built from the exact right-going acoustic eigenvector at finite Re.
    Velocity amplitude = eps/Ma (same as make_acoustic_ic).
    State vector: (δρ, δu, δE) with δm = δu since ρ₀=1, u₀=0.
    """
    lam, v = right_going_eigenvector(mode, Re, Ma, gamma, Pr)

    # Normalize so velocity component has amplitude 1
    v = v / abs(v[1])

    x = np.linspace(0.0, 1.0, N, endpoint=False)
    phase = np.exp(1j * 2.0 * np.pi * mode * x)

    # Physical perturbation: 2*Re(scale * v * e^{ikx})
    scale = eps / Ma
    dQ = scale * 2.0 * np.real(v[:, np.newaxis] * phase[np.newaxis, :])
    # dQ[0]=δρ, dQ[1]=δu=δm (ρ₀=1), dQ[2]=δE

    # Base state
    p0 = 1.0 / (gamma * Ma**2)
    E0 = p0 / (gamma - 1.0)
    Q_base = np.array([np.ones(N), np.zeros(N), E0 * np.ones(N)])
    return Q_base + dQ


def amplitude(Q, Ma, mode, gamma=GAMMA):
    """One-sided rfft amplitude of velocity at Fourier mode."""
    _, u, _, _ = primitives(Q, Ma, gamma)
    N = len(u)
    uh = np.fft.rfft(u)
    return 2.0 * abs(uh[mode]) / N


def two_point_sigma(snaps, T_final, Ma, mode, gamma=GAMMA):
    A0 = amplitude(snaps[0], Ma, mode, gamma)
    Af = amplitude(snaps[1], Ma, mode, gamma)
    return np.log(A0 / Af) / T_final


def multipoint_sigma(save_times, snaps, Ma, mode, gamma=GAMMA):
    """Least-squares exponential fit: log A = log A0 - sigma*t."""
    amps = np.array([amplitude(Q, Ma, mode, gamma) for Q in snaps])
    log_amps = np.log(amps)
    slope, _ = np.polyfit(save_times, log_amps, 1)
    return -slope


# ── Show mode projection ───────────────────────────────────────────────────────
print("=" * 65)
print("Mode projection of isentropic IC (Re=200, Ma=0.3)")
print("=" * 65)
project_ic_onto_modes(mode, Re, Ma)

# ── Comparison table ──────────────────────────────────────────────────────────
print("=" * 65)
print("Error comparison: two approaches, N = 64, 128, 256")
print("=" * 65)

n_snap = 40   # time points for multi-point fit
save_times_fit = np.linspace(0.0, T_final, n_snap)

N_list = [64, 128, 256]
eps_vals = [1e-5, 1e-6]

for eps in eps_vals:
    print(f"\neps = {eps:.0e}")
    hdr = f"  {'N':>5}  {'iso 2pt':>12}  {'iso fit':>12}  {'eigen fit':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    iso_2pt_prev = None
    iso_fit_prev = None
    eig_fit_prev = None

    for N in N_list:
        # --- Isentropic IC, two-point ---
        Q0_iso = make_acoustic_ic(N, Ma, mode=mode, eps=eps)
        snaps_2pt = solve_compressible(Q0_iso, Re, Ma, T_final, cfl=cfl,
                                       save_times=[0.0, T_final])
        sig_iso_2pt = two_point_sigma(snaps_2pt, T_final, Ma, mode)
        rel_iso_2pt = abs(sig_iso_2pt - sigma_exact) / sigma_exact

        # --- Isentropic IC, multi-point fit ---
        snaps_fit = solve_compressible(Q0_iso, Re, Ma, T_final, cfl=cfl,
                                       save_times=list(save_times_fit))
        sig_iso_fit = multipoint_sigma(save_times_fit, snaps_fit, Ma, mode)
        rel_iso_fit = abs(sig_iso_fit - sigma_exact) / sigma_exact

        # --- Pure eigenmode IC, multi-point fit ---
        Q0_eig = make_pure_eigenmode_ic(N, Re, Ma, mode=mode, eps=eps)
        snaps_eig = solve_compressible(Q0_eig, Re, Ma, T_final, cfl=cfl,
                                       save_times=list(save_times_fit))
        sig_eig_fit = multipoint_sigma(save_times_fit, snaps_eig, Ma, mode)
        rel_eig_fit = abs(sig_eig_fit - sigma_exact) / sigma_exact

        print(f"  {N:>5}  {rel_iso_2pt:>12.3e}  {rel_iso_fit:>12.3e}  {rel_eig_fit:>12.3e}")

        iso_2pt_prev = rel_iso_2pt
        iso_fit_prev = rel_iso_fit
        eig_fit_prev = rel_eig_fit

print()
print("Columns:")
print("  iso 2pt   : isentropic IC + two-point estimate (original method)")
print("  iso fit   : isentropic IC + 40-point least-squares fit")
print("  eigen fit : exact eigenmode IC + 40-point least-squares fit")
print()
print("If 'eigen fit' error decreases with N => solver correct, IC was biased.")
print("If 'eigen fit' error stays fixed => real solver bug.")
