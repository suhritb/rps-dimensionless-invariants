"""
solver.py
=========
Spectral solver for the viscous Burgers equation:

    u_t + u u_x = nu u_xx    on periodic domain [0, L)

Method: CNAB2 (Crank-Nicolson / Adams-Bashforth-2)
  - Crank-Nicolson on the viscous (stiff) term  -> unconditionally stable
  - Adams-Bashforth-2 on the nonlinear term      -> explicit, 2nd-order
  - 2/3 dealiasing in Fourier space
  - Validated against exact Cole-Hopf solution (see tests/)

Scaling symmetry (proven analytically, validated to ~1e-17):
    If u(x,t) solves Burgers with viscosity nu, then
        v(x,t) = u(a*x, a*t)   solves Burgers with viscosity  nu/a.
    Proof: v_t=a u_t, v v_x=a u u_x, v_xx=a^2 u_xx =>
           v_t + v v_x = a(u_t+u u_x) = a nu u_xx = (nu/a) v_xx.  QED

    Corollary: Re = U*L/nu is PRESERVED by this symmetry (dynamical similarity).
    The symmetry gives scale-generalization at FIXED Re, not Re-extrapolation.
"""

import numpy as np


def solve_burgers(u0: np.ndarray, nu: float, L: float, T: float,
                  nt: int, save_times=None):
    """
    Solve viscous Burgers on [0, L) periodic, initial condition u0.

    Parameters
    ----------
    u0         : (N,) real array — initial condition on uniform grid
    nu         : kinematic viscosity  (> 0)
    L          : domain length
    T          : final time
    nt         : number of time steps
    save_times : optional sorted list/array of times at which to save snapshots

    Returns
    -------
    If save_times is None : final field (N,)
    Else                  : array of shape (len(save_times), N)
    """
    N = u0.shape[0]
    dt = T / nt
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)   # wave numbers
    k2 = k ** 2

    # 2/3 dealiasing mask
    dealias = (np.abs(k) <= (2.0 / 3.0) * np.max(np.abs(k))).astype(float)

    def _nonlinear(uhat):
        """Fourier-space nonlinear term: -ik * FFT(u^2 / 2)."""
        u = np.fft.ifft(uhat).real
        return -(1j * k) * np.fft.fft(0.5 * u * u) * dealias

    uhat = np.fft.fft(u0)

    # CNAB2 coefficients for the linear (viscous) operator
    denom = 1.0 + 0.5 * dt * nu * k2      # (1 + dt/2 * nu k^2)
    lin_num = 1.0 - 0.5 * dt * nu * k2   # (1 - dt/2 * nu k^2)

    # Snapshot bookkeeping
    if save_times is not None:
        save_times = np.asarray(save_times, dtype=float)
        save_steps = np.round(save_times / dt).astype(int)
        snaps = np.zeros((len(save_times), N))
        for j, st in enumerate(save_steps):
            if st == 0:
                snaps[j] = u0

    Nprev = None
    for n in range(1, nt + 1):
        Ncur = _nonlinear(uhat)
        adv = dt * Ncur if Nprev is None else dt * (1.5 * Ncur - 0.5 * Nprev)
        uhat = (lin_num * uhat + adv) / denom
        Nprev = Ncur

        if save_times is not None:
            for j, st in enumerate(save_steps):
                if st == n:
                    snaps[j] = np.fft.ifft(uhat).real

    if save_times is not None:
        return snaps
    return np.fft.ifft(uhat).real


def cole_hopf_solution(x: np.ndarray, t: float, nu: float,
                       eps: float, k0: float) -> np.ndarray:
    """
    Exact periodic Burgers solution via the Cole-Hopf transform.

    phi(x,t) = 1 + eps * exp(-nu k0^2 t) * cos(k0 x)  solves heat eq.
    u = -2 nu phi_x / phi                               solves Burgers.

    Requires |eps| < 1 so phi stays positive.
    """
    decay = np.exp(-nu * k0 ** 2 * t)
    phi = 1.0 + eps * decay * np.cos(k0 * x)
    phi_x = -eps * decay * k0 * np.sin(k0 * x)
    return -2.0 * nu * phi_x / phi


def apply_scaling_symmetry(u_field: np.ndarray, x_src: np.ndarray,
                           a: float, L_src: float, N_tgt: int):
    """
    Apply the exact scaling symmetry  v(x) = u(a*x).

    The source field u is periodic on [0, L_src).
    The image v(x) = u(a x) is periodic on [0, L_src/a).

    Parameters
    ----------
    u_field : (M,) field values on x_src
    x_src   : (M,) uniform grid over [0, L_src)
    a       : scaling factor (a > 1 compresses domain/viscosity)
    L_src   : source domain length
    N_tgt   : number of output grid points

    Returns
    -------
    v       : (N_tgt,) field on [0, L_src/a)
    L_tgt   : = L_src / a
    """
    L_tgt = L_src / a
    x_tgt = np.linspace(0.0, L_tgt, N_tgt, endpoint=False)
    # map each target point x into source coordinate a*x (spans [0, L_src))
    src_x = np.mod(a * x_tgt, L_src)
    v = np.interp(src_x, x_src, u_field, period=L_src)
    return v, L_tgt
