"""
src/compressible_dataset.py
===========================
Dataset for the compressible-flow similarity operator experiment.

Design
------
The validated non-dim compressible NS solver depends only on
(IC_nondim, Re, Ma); it knows nothing about physical length or
velocity scales.  To create a meaningful naive/similarity comparison
we introduce "dimensional configuration" labels:

    L   : domain-length scale            (varied in training/test)
    c0  : sound-speed scale  = U0/Ma     (U0 = 1 implicit, so c0 = 1/Ma)
    mu  : kinematic viscosity = L/Re     (derived; viscosity ~ L for fixed Re)

The similarity orbit at fixed (Re, Ma) is the set of all (L, mu, c0)
with L/mu = Re and 1/c0 = Ma.  All orbit members share one cached
solve: the non-dim solution is the same by construction.

Feature encodings
-----------------
Naive      : [u_ic | log(mu) | log(L) | log(c0)]
              model must discover Re = L/mu and Ma = 1/c0
Similarity : [u_ic | log(Re) | log(Ma)]
              both exact invariants supplied directly

Both encodings see the full (Re, Ma) grid in training; the difference
is representation only.

Pre-compute logic
-----------------
One solve per (IC index, Re, Ma).  Attaching different L values to
the same (IC, Re, Ma) costs zero additional solves.
"""

import numpy as np
from .compressible_solver import (
    solve_compressible, primitives, conserved_from_primitives, GAMMA, PR,
)

T_FINAL = 1.5    # non-dim integration time (a few acoustic periods; no-shock regime)
CFL     = 0.4    # CFL safety factor
AMP     = 0.05   # max velocity perturbation amplitude (well sub-shock)


# ── IC generation ─────────────────────────────────────────────────────────────

def make_ic_velocity(rng: np.random.Generator, N: int,
                     amp: float = AMP, n_modes_max: int = 4) -> np.ndarray:
    """
    Random smooth multi-mode velocity perturbation on [0,1).

    Superposition of cosine modes with random amplitudes, wavenumbers k in
    {1..5}, and phases.  Amplitude << 1 keeps the flow sub-shock throughout
    the integration interval T_FINAL.

    Returns
    -------
    u_pert : (N,) float array, O(amp) amplitude
    """
    x       = np.linspace(0.0, 1.0, N, endpoint=False)
    u_pert  = np.zeros(N)
    n_modes = rng.integers(1, n_modes_max + 1)
    for _ in range(n_modes):
        a   = rng.uniform(-amp, amp)
        k   = rng.integers(1, 6)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        u_pert += a * np.cos(2.0 * np.pi * k * x + phi)
    return u_pert


def _Q0_from_upert(u_pert: np.ndarray, Ma: float, gamma: float = GAMMA) -> np.ndarray:
    """
    Build 3×N conserved-state IC from velocity perturbation.
    Base state: rho=1, T=1; velocity = u_pert.
    """
    N   = len(u_pert)
    rho = np.ones(N)
    T   = np.ones(N)
    return conserved_from_primitives(rho, u_pert, T, Ma, gamma)


# ── Solver wrapper ────────────────────────────────────────────────────────────

def solve_config(u_pert: np.ndarray, Re: float, Ma: float,
                 T_final: float = T_FINAL, cfl: float = CFL,
                 gamma: float = GAMMA, Pr: float = PR) -> np.ndarray:
    """
    Run the validated compressible solver for velocity IC u_pert at (Re, Ma).

    The L and mu labels are dimensional metadata only; the non-dim solve
    depends solely on (u_pert, Re, Ma, T_final).

    Returns
    -------
    u_final : (N,) velocity field at T_final
    """
    Q0          = _Q0_from_upert(u_pert, Ma, gamma)
    Qf          = solve_compressible(Q0, Re, Ma, T_final, cfl=cfl,
                                     gamma=gamma, Pr=Pr)
    _, u, _, _  = primitives(Qf, Ma, gamma)
    return u


# ── Pre-computation ───────────────────────────────────────────────────────────

def precompute_solutions(u_perts: list, Re_values: np.ndarray,
                         Ma_values: np.ndarray,
                         T_final: float = T_FINAL, cfl: float = CFL,
                         gamma: float = GAMMA, Pr: float = PR,
                         verbose: bool = True) -> dict:
    """
    Pre-compute one solve per (IC index, Re, Ma).

    All dimensional-scale variants (different L) that share the same
    (Re, Ma) reuse the cached result; no duplicate solves are performed.

    Raises RuntimeError if any solve diverges (non-finite values or
    max |u| > 5.0 — well above any physically reasonable amplitude).

    Returns
    -------
    sol_cache : dict  (ic_idx, Re, Ma) -> u_final (N,)
    """
    cache = {}
    total = len(u_perts) * len(Re_values) * len(Ma_values)
    done  = 0

    for i, u in enumerate(u_perts):
        for Re in Re_values:
            for Ma in Ma_values:
                u_fin = solve_config(u, float(Re), float(Ma),
                                     T_final, cfl, gamma, Pr)
                if not np.isfinite(u_fin).all() or np.max(np.abs(u_fin)) > 5.0:
                    raise RuntimeError(
                        f"Solve diverged: IC={i}, Re={Re:.0f}, Ma={Ma:.2f}, "
                        f"max|u|={np.max(np.abs(u_fin)):.3f}"
                    )
                cache[(i, float(Re), float(Ma))] = u_fin
                done += 1
                if verbose and done % 30 == 0:
                    print(f"    {done}/{total} solves complete ...")

    if verbose:
        print(f"    Done: {total} solves total.")
    return cache


# ── Dataset assembly ──────────────────────────────────────────────────────────

def assemble_dataset(u_perts: list, Re_values: np.ndarray,
                     Ma_values: np.ndarray, L_list: np.ndarray,
                     sol_cache: dict):
    """
    Assemble (X, Y) arrays by attaching dimensional labels to cached results.

    For each (IC, Re, Ma, L):
        mu  = L / Re          [kinematic viscosity]
        c0  = 1.0 / Ma        [sound speed, with U0=1]

    No new PDE solves are performed.

    Returns
    -------
    u_ic  (n, N)   velocity IC vectors
    mu    (n,)     kinematic viscosity labels
    L_out (n,)     domain-length labels
    c0    (n,)     sound-speed labels
    Re    (n,)     Reynolds numbers
    Ma    (n,)     Mach numbers
    Y     (n, N)   velocity solutions at T_final
    """
    u_ic_l, mu_l, L_l, c0_l, Re_l, Ma_l, Y_l = [], [], [], [], [], [], []

    for i, u in enumerate(u_perts):
        for Re in Re_values:
            for Ma in Ma_values:
                Y   = sol_cache[(i, float(Re), float(Ma))]
                c0  = 1.0 / float(Ma)
                for L in L_list:
                    mu = float(L) / float(Re)
                    u_ic_l.append(u)
                    mu_l.append(mu)
                    L_l.append(float(L))
                    c0_l.append(c0)
                    Re_l.append(float(Re))
                    Ma_l.append(float(Ma))
                    Y_l.append(Y)

    return (np.array(u_ic_l), np.array(mu_l), np.array(L_l),
            np.array(c0_l),  np.array(Re_l), np.array(Ma_l),
            np.array(Y_l))


def assemble_dataset_paired(u_perts: list, re_ma_pairs: list,
                             L_list: np.ndarray, sol_cache: dict):
    """
    Assemble dataset from an explicit list of (Re, Ma) tuples.

    Unlike assemble_dataset, this does NOT form the Cartesian product of
    Re values × Ma values.  Each (Re, Ma) tuple is used exactly once per
    (IC, L) combination, so n_samples = len(re_ma_pairs) × len(u_perts)
    × len(L_list).

    Parameters
    ----------
    re_ma_pairs : list of (Re, Ma) tuples
        Explicit set of configurations to include.

    Returns
    -------
    Same seven arrays as assemble_dataset.
    """
    u_ic_l, mu_l, L_l, c0_l, Re_l, Ma_l, Y_l = [], [], [], [], [], [], []
    for i, u in enumerate(u_perts):
        for Re, Ma in re_ma_pairs:
            Y  = sol_cache[(i, float(Re), float(Ma))]
            c0 = 1.0 / float(Ma)
            for L in L_list:
                mu = float(L) / float(Re)
                u_ic_l.append(u)
                mu_l.append(mu)
                L_l.append(float(L))
                c0_l.append(c0)
                Re_l.append(float(Re))
                Ma_l.append(float(Ma))
                Y_l.append(Y)
    return (np.array(u_ic_l), np.array(mu_l), np.array(L_l),
            np.array(c0_l),  np.array(Re_l), np.array(Ma_l),
            np.array(Y_l))


# ── Feature encodings ─────────────────────────────────────────────────────────

def features_naive(u_ic: np.ndarray, mu: np.ndarray,
                   L: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """
    Naive encoding: [u_ic | log(mu) | log(L) | log(c0)].

    The model receives three dimensional scalars separately.
    It must discover that only (L/mu, 1/c0) = (Re, Ma) govern the solution.
    """
    return np.hstack([u_ic,
                      np.log(mu)[:, None],
                      np.log(L)[:, None],
                      np.log(c0)[:, None]])


def features_similarity(u_ic: np.ndarray,
                        Re: np.ndarray, Ma: np.ndarray) -> np.ndarray:
    """
    Similarity encoding: [u_ic | log(Re) | log(Ma)].

    Both exact dynamical-similarity invariants are supplied directly.
    """
    return np.hstack([u_ic,
                      np.log(Re)[:, None],
                      np.log(Ma)[:, None]])
