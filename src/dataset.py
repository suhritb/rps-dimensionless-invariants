"""
dataset.py
==========
Dataset generation for the similarity-operator seed experiment.

Key design decisions
--------------------
1. Each "configuration" is a (nu, L) pair with a fixed O(1) IC vector.
2. We evolve each config to T = TAU * L so that ALL configs on the same
   similarity orbit (same Re = L/nu) share identical sampled IC and solution
   vectors -- confirmed to machine precision in tests/.
3. Solutions are PRE-COMPUTED once per (IC_index, Re) and reused, since
   adding more scale replicates costs zero extra PDE solves.
4. Feature encodings expose where the inductive bias lives:
      naive      : [ic_vec | log(nu) | log(L)]   (must discover invariance)
      similarity : [ic_vec | log(Re)]             (invariance built in)
"""

import numpy as np
from .solver import solve_burgers

TAU = 0.15   # non-dim time horizon: T = TAU * L
NT = 2000    # solver steps per simulation


def make_ic_vector(rng: np.random.Generator, N: int = 64) -> np.ndarray:
    """
    Random localized O(1) initial condition as an N-vector.

    Represents a smooth, localized bump: amplitude * Gaussian_envelope * sin.
    Domain-independent (normalised coordinate s in [0,1)).
    """
    s = np.linspace(0.0, 1.0, N, endpoint=False)
    center = rng.uniform(0.35, 0.65)
    width = rng.uniform(0.06, 0.12)
    amp = rng.uniform(0.6, 1.0) * rng.choice([-1.0, 1.0])
    kk = rng.integers(1, 4)
    env = np.exp(-((s - center) ** 2) / (2 * width ** 2))
    return amp * env * np.sin(2 * np.pi * kk * (s - center))


def solve_config(ic_vec: np.ndarray, nu: float, L: float) -> np.ndarray:
    """Instantiate ic_vec on [0,L) and evolve to T = TAU*L."""
    return solve_burgers(ic_vec.copy(), nu, L, TAU * L, NT)


def precompute_solutions(ics: list, Re_values: np.ndarray,
                         ref_L: float = 2.8) -> dict:
    """
    Pre-compute one solution per (IC index, Re).

    Uses ref_L to instantiate: nu = ref_L / Re, then solve on [0, ref_L).
    Since the solution vector depends only on (IC_vec, Re) -- not on the
    specific (nu, L) chosen on the orbit -- this one solve covers ALL scales.

    Returns
    -------
    sol_cache : dict  (ic_index: int, Re: float) -> solution array (N,)
    """
    sol_cache = {}
    for i, icv in enumerate(ics):
        for Re in Re_values:
            nu = ref_L / Re
            sol_cache[(i, float(Re))] = solve_config(icv, nu, ref_L)
    return sol_cache


def assemble_dataset(ics: list, Re_values: np.ndarray,
                     L_list: np.ndarray, sol_cache: dict):
    """
    Assemble (X, Y) arrays by attaching (nu, L) labels to cached solutions.

    A config at scale L on orbit Re has nu = L / Re.
    No new PDE solves are performed here.

    Returns
    -------
    Xic : (n_samples, N)     IC vectors
    nu  : (n_samples,)       viscosities
    L   : (n_samples,)       domain lengths
    Re  : (n_samples,)       Reynolds numbers
    Y   : (n_samples, N)     solution vectors
    """
    Xic, nu_out, L_out, Re_out, Y = [], [], [], [], []
    for i, icv in enumerate(ics):
        for Rev in Re_values:
            sol = sol_cache[(i, float(Rev))]
            for Lv in L_list:
                Xic.append(icv)
                nu_out.append(Lv / Rev)
                L_out.append(Lv)
                Re_out.append(Rev)
                Y.append(sol)
    return (np.array(Xic), np.array(nu_out), np.array(L_out),
            np.array(Re_out), np.array(Y))


# ----- Feature encodings --------------------------------------------------

def features_naive(Xic: np.ndarray, nu: np.ndarray,
                   L: np.ndarray) -> np.ndarray:
    """
    Naive encoding: [IC_vec | log(nu) | log(L)].
    The model must discover that only log(L) - log(nu) matters.
    """
    return np.hstack([Xic, np.log(nu)[:, None], np.log(L)[:, None]])


def features_similarity(Xic: np.ndarray, Re: np.ndarray) -> np.ndarray:
    """
    Similarity encoding: [IC_vec | log(Re)].
    The exact dynamical-similarity invariant is built in as the sole scalar.
    """
    return np.hstack([Xic, np.log(Re)[:, None]])
